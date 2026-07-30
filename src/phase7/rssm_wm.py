"""MiniRSSM + TemporalJEPA: Recurrent State-Space Model for grid-maze exploration.

Two complementary architectures:

MiniRSSM (full):
  Maintains a stochastic latent state z_t and deterministic recurrent h_t.
  h_t = GRU([z_{t-1}, a_{t-1}], h_{t-1})
  z_t ~ posterior(h_t, s_t) during training; prior(h_t) during rollouts.
  Decoder predicts next state embedding from z_t.
  Ensemble of 3 RSSMs provides epistemic uncertainty via prediction variance.

TemporalJEPA (simpler fallback):
  GRU maintains hidden state across timesteps.
  3 MLP predictors each take (state_emb, action_emb, h_t) → pred_next_emb.
  Ensemble variance = epistemic signal, same interface as MiniRSSM.

Both share:
  - CPU-friendly learned state encoders (no Qwen dependency)
  - Compatible GridState → feature encoding
  - Same training interface (train_on_transitions)
  - Ensemble epistemic_uncertainty() method
"""

import hashlib
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from phase6.grid_env import GridState
from phase6.maze_generator import ITEMS

# ── Constants ────────────────────────────────────────────

_ACTION_VOCAB = 64  # hash-based action embedding table size
_STATE_FEAT_DIM = 35  # x_oh(10) + y_oh(10) + inv(14) + goal(1)


def _normalize_hash(s: str, mod: int) -> int:
    """Stable hash of a string into [0, mod)."""
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) % mod


# ── Feature Encoding ────────────────────────────────────

def grid_state_features(state: GridState) -> torch.Tensor:
    """Convert GridState to a flat feature vector [1, STATE_FEAT_DIM].

    Layout: x_onehot(10) + y_onehot(10) + inventory_bitmask(14) + goal_reached(1)
    """
    x_oh = torch.zeros(10)
    y_oh = torch.zeros(10)
    x_oh[min(state.x, 9)] = 1.0
    y_oh[min(state.y, 9)] = 1.0

    inv = torch.zeros(len(ITEMS))
    for item in state.inventory:
        for i, known in enumerate(ITEMS):
            if item == known:
                inv[i] = 1.0
                break

    goal = torch.tensor([1.0 if state.goal_reached else 0.0])

    return torch.cat([x_oh, y_oh, inv, goal]).unsqueeze(0)  # [1, 35]


def action_to_idx(action: str, vocab: int = _ACTION_VOCAB) -> int:
    """Map action string to a stable embedding index."""
    return _normalize_hash(action, vocab)


# ── State Encoder / Decoder ──────────────────────────────

class StateEncoder(nn.Module):
    """Learned encoder: raw 35-dim features → state_dim embedding."""

    def __init__(self, state_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(_STATE_FEAT_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim),
        )

    def forward(self, raw_feats: torch.Tensor) -> torch.Tensor:
        """raw_feats: [B, 35] → [B, state_dim]"""
        return self.net(raw_feats)


class ActionEncoder(nn.Module):
    """Learned action embedding via hash bucket."""

    def __init__(self, action_dim: int = 16, vocab: int = _ACTION_VOCAB):
        super().__init__()
        self.embed = nn.Embedding(vocab, action_dim)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """actions: [B] (long indices) → [B, action_dim]"""
        return self.embed(actions)


# ── MiniRSSM ────────────────────────────────────────────

class MiniRSSM(nn.Module):
    """Recurrent State-Space Model (single ensemble member).

    Dynamics:
      h_t = GRU([z_{t-1}, a_{t-1}], h_{t-1})
      posterior: z_t ~ N(μ_post, σ_post) with μ_post,σ_post = f_post([h_t, s_t])
      prior:     z_t ~ N(μ_prior, σ_prior) with μ_prior,σ_prior = f_prior(h_t)
      s_{t+1} = decoder(z_t)
    """

    def __init__(
        self,
        state_dim: int = 32,
        action_dim: int = 16,
        hidden_dim: int = 128,
        latent_dim: int = 16,
        action_vocab: int = _ACTION_VOCAB,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoders
        self.state_encoder = StateEncoder(state_dim)
        self.action_encoder = ActionEncoder(action_dim, action_vocab)

        # RSSM core
        self.rnn = nn.GRUCell(latent_dim + action_dim, hidden_dim)

        # Posterior: q(z_t | h_t, s_t) — uses observed state
        self.posterior = nn.Linear(hidden_dim + state_dim, 2 * latent_dim)

        # Prior: p(z_t | h_t) — predicts from history alone
        self.prior = nn.Linear(hidden_dim, 2 * latent_dim)

        # Decoder: s_{t+1} = f(z_t)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(
        self,
        state_feats: torch.Tensor,
        action_idx: torch.Tensor,
        prev_h: torch.Tensor,
        prev_z: torch.Tensor,
        state_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One RSSM step.

        Args:
            state_feats: [B, 35] raw state features for posterior conditioning
            action_idx: [B] action hash indices
            prev_h: [B, hidden_dim] previous deterministic state
            prev_z: [B, latent_dim] previous stochastic latent
            state_emb: optional [B, state_dim] pre-encoded state (cached)

        Returns:
            pred_state: [B, state_dim] predicted next embedding
            h_t: [B, hidden_dim] new deterministic state
            z_t: [B, latent_dim] sampled posterior latent
            posterior_params: (μ_post, logσ_post)
            prior_params: (μ_prior, logσ_prior)
        """
        B = state_feats.shape[0]

        # Encode action
        act_emb = self.action_encoder(action_idx)  # [B, action_dim]

        # Update deterministic state: h_t = GRU([z_{t-1}, a_{t-1}], h_{t-1})
        rnn_in = torch.cat([prev_z, act_emb], dim=-1)  # [B, latent_dim + action_dim]
        h_t = self.rnn(rnn_in, prev_h)  # [B, hidden_dim]

        # Posterior: z_t ~ q(z_t | h_t, s_t)
        if state_emb is None:
            state_emb = self.state_encoder(state_feats)
        post_in = torch.cat([h_t, state_emb], dim=-1)
        post_params = self.posterior(post_in)
        μ_post, logσ_post = post_params.chunk(2, dim=-1)
        logσ_post = torch.clamp(logσ_post, -10.0, 2.0)
        σ_post = torch.exp(logσ_post)
        z_t = μ_post + torch.randn_like(σ_post) * σ_post

        # Prior: p(z_t | h_t) for KL
        prior_params = self.prior(h_t)
        μ_prior, logσ_prior = prior_params.chunk(2, dim=-1)
        logσ_prior = torch.clamp(logσ_prior, -10.0, 2.0)

        # Decode: predict next state embedding
        pred_state = self.decoder(z_t)  # [B, state_dim]

        return pred_state, h_t, z_t, (μ_post, logσ_post), (μ_prior, logσ_prior)

    @torch.no_grad()
    def forward_prior(
        self,
        action_idx: torch.Tensor,
        prev_h: torch.Tensor,
        prev_z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prior-only forward (no state observation, for rollouts).

        Returns:
            pred_state: [B, state_dim]
            h_t: [B, hidden_dim]
            z_t: [B, latent_dim]
        """
        act_emb = self.action_encoder(action_idx)
        rnn_in = torch.cat([prev_z, act_emb], dim=-1)
        h_t = self.rnn(rnn_in, prev_h)

        prior_params = self.prior(h_t)
        μ_prior, logσ_prior = prior_params.chunk(2, dim=-1)
        logσ_prior = torch.clamp(logσ_prior, -10.0, 2.0)
        σ_prior = torch.exp(logσ_prior)
        z_t = μ_prior + torch.randn_like(σ_prior) * σ_prior

        pred_state = self.decoder(z_t)
        return pred_state, h_t, z_t

    @staticmethod
    def kl_divergence(
        posterior_params: Tuple[torch.Tensor, torch.Tensor],
        prior_params: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """KL[ q(z_t) || p(z_t) ] for one step.

        Returns scalar.
        """
        μ_q, logσ_q = posterior_params
        μ_p, logσ_p = prior_params

        σ_q = torch.exp(logσ_q)
        σ_p = torch.exp(logσ_p)

        # KL(N(μ_q, σ_q) || N(μ_p, σ_p))
        kl = (
            logσ_p - logσ_q
            + (σ_q**2 + (μ_q - μ_p) ** 2) / (2 * σ_p**2)
            - 0.5
        )
        return kl.sum(dim=-1).mean()

    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """Initial hidden state h_0 = zeros."""
        return torch.zeros(batch_size, self.hidden_dim)

    def init_latent(self, batch_size: int = 1) -> torch.Tensor:
        """Initial latent z_0 = zeros."""
        return torch.zeros(batch_size, self.latent_dim)


# ── TemporalJEPA (GRU + MLP ensemble, simpler) ─────────

class TemporalMLPPredictor(nn.Module):
    """MLP that predicts next embedding from (state_emb, action_emb, hidden).

    Architecture:
      concat(state_emb, action_emb, hidden) → Linear(64) → ReLU → Linear(state_dim)
    """

    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        in_dim = state_dim + _ACTION_EMB_DIM + hidden_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim),
        )

    def forward(
        self,
        state_emb: torch.Tensor,
        action_emb: torch.Tensor,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([state_emb, action_emb, hidden], dim=-1)
        return self.net(x)


_ACTION_EMB_DIM = 16  # action embedding dimension used by TemporalJEPA


class TemporalJEPA(nn.Module):
    """GRU + ensemble of MLP predictors.

    h_t = GRU([state_emb_t, a_t], h_{t-1})
    pred_{t+1}^{(i)} = MLP_i(state_emb_t, action_emb_t, h_t)
    epistemic = variance over ensemble predictions

    Simpler than MiniRSSM — no stochastic latent, just temporal context.
    """

    def __init__(
        self,
        state_dim: int = 32,
        action_dim: int = _ACTION_EMB_DIM,
        hidden_dim: int = 64,
        n_ensemble: int = 3,
        action_vocab: int = _ACTION_VOCAB,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.n_ensemble = n_ensemble

        # Encoders
        self.state_encoder = StateEncoder(state_dim)
        self.action_encoder = ActionEncoder(action_dim, action_vocab)

        # GRU for temporal context
        self.rnn = nn.GRUCell(state_dim + action_dim, hidden_dim)

        # Ensemble of MLP predictors
        self.predictors = nn.ModuleList([
            TemporalMLPPredictor(state_dim, hidden_dim)
            for _ in range(n_ensemble)
        ])

    def forward_step(
        self,
        state_feats: torch.Tensor,
        action_idx: torch.Tensor,
        prev_h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One step: compute hidden, then predict with all ensemble members.

        Returns:
            preds: [n_ensemble, B, state_dim] predictions
            h_t: [B, hidden_dim] new hidden state
            state_emb: [B, state_dim] encoded state
        """
        state_emb = self.state_encoder(state_feats)
        act_emb = self.action_encoder(action_idx)
        rnn_in = torch.cat([state_emb, act_emb], dim=-1)
        h_t = self.rnn(rnn_in, prev_h)
        preds = torch.stack([p(state_emb, act_emb, h_t) for p in self.predictors])
        return preds, h_t, state_emb

    @torch.no_grad()
    def epistemic_uncertainty(
        self, state_feats: torch.Tensor, action_idx: torch.Tensor, h_t: torch.Tensor
    ) -> float:
        """Ensemble prediction variance = epistemic uncertainty."""
        preds, _, _ = self.forward_step(state_feats, action_idx, h_t)
        mean_pred = preds.mean(dim=0)
        epistemic = ((preds - mean_pred.unsqueeze(0)) ** 2).mean().item()
        return epistemic

    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim)

    def compute_loss(
        self,
        preds: torch.Tensor,
        target_state_emb: torch.Tensor,
    ) -> torch.Tensor:
        """MSE loss averaged over ensemble members."""
        return F.mse_loss(preds, target_state_emb.unsqueeze(0).expand_as(preds))


# ── SimpleJEPA (no GRU, for ablation comparison) ────────

class SimpleJEPAPredictor(nn.Module):
    """MLP predicting next embedding from (state_emb, action_emb)."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        in_dim = state_dim + action_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim),
        )

    def forward(self, state_emb: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state_emb, action_emb], dim=-1)
        return self.net(x)


class SimpleJEPA(nn.Module):
    """MLP ensemble (no temporal context) — ablation baseline.

    Used to compare against TemporalJEPA to measure the value of recurrence.
    """

    def __init__(
        self,
        state_dim: int = 32,
        action_dim: int = _ACTION_EMB_DIM,
        n_ensemble: int = 3,
        action_vocab: int = _ACTION_VOCAB,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_ensemble = n_ensemble

        self.state_encoder = StateEncoder(state_dim)
        self.action_encoder = ActionEncoder(action_dim, action_vocab)

        self.predictors = nn.ModuleList([
            SimpleJEPAPredictor(state_dim, action_dim)
            for _ in range(n_ensemble)
        ])

    def forward(
        self, state_feats: torch.Tensor, action_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns preds [n_ensemble, B, state_dim] and state_emb [B, state_dim]."""
        state_emb = self.state_encoder(state_feats)
        act_emb = self.action_encoder(action_idx)
        preds = torch.stack([p(state_emb, act_emb) for p in self.predictors])
        return preds, state_emb

    @torch.no_grad()
    def epistemic_uncertainty(
        self, state_feats: torch.Tensor, action_idx: torch.Tensor
    ) -> float:
        """Ensemble prediction variance."""
        preds, _ = self.forward(state_feats, action_idx)
        mean_pred = preds.mean(dim=0)
        return ((preds - mean_pred.unsqueeze(0)) ** 2).mean().item()

    def compute_loss(
        self, preds: torch.Tensor, target_emb: torch.Tensor
    ) -> torch.Tensor:
        return F.mse_loss(preds, target_emb.unsqueeze(0).expand_as(preds))


# ── Training helpers (shared) ────────────────────────────

def rssm_training_step(
    rssm: MiniRSSM,
    transitions: List[Tuple[GridState, str, GridState]],
    kl_weight: float = 0.1,
    lr: float = 1e-3,
) -> float:
    """Train a MiniRSSM on a list of transitions.

    Each transition: (state, action_str, next_state).
    Returns average loss (reconstruction + β * KL).
    """
    if not transitions:
        return 0.0

    optimizer = torch.optim.Adam(rssm.parameters(), lr=lr)
    optimizer.zero_grad()

    total_loss = 0.0
    h = rssm.init_hidden()
    z = rssm.init_latent()

    loss_sum = torch.tensor(0.0)

    for state, action_str, next_state in transitions:
        feats = grid_state_features(state)
        next_feats = grid_state_features(next_state)
        next_emb = rssm.state_encoder(next_feats)
        action_idx = torch.tensor([action_to_idx(action_str)])

        pred_emb, h, z, post_params, prior_params = rssm.forward(
            feats, action_idx, h, z
        )

        # Reconstruction loss
        recon_loss = F.mse_loss(pred_emb, next_emb.detach())

        # KL loss
        kl_loss = MiniRSSM.kl_divergence(post_params, prior_params)

        loss = recon_loss + kl_weight * kl_loss
        total_loss += loss.item()
        loss_sum = loss_sum + loss

        # Detach RNN state to prevent backprop through entire sequence
        h = h.detach()
        z = z.detach()

    loss_sum.backward()
    optimizer.step()
    return total_loss / max(len(transitions), 1)


def temporal_jepa_training_step(
    tjepa: TemporalJEPA,
    transitions: List[Tuple[GridState, str, GridState]],
    lr: float = 1e-3,
) -> float:
    """Train TemporalJEPA on transitions.

    Returns average MSE loss across ensemble.
    """
    if not transitions:
        return 0.0

    optimizer = torch.optim.Adam(tjepa.parameters(), lr=lr)
    optimizer.zero_grad()

    total_loss = 0.0
    h = tjepa.init_hidden()

    loss_sum = torch.tensor(0.0)

    for state, action_str, next_state in transitions:
        feats = grid_state_features(state)
        next_feats = grid_state_features(next_state)
        action_idx = torch.tensor([action_to_idx(action_str)])

        preds, h, _ = tjepa.forward_step(feats, action_idx, h)
        target_emb = tjepa.state_encoder(next_feats)

        loss = tjepa.compute_loss(preds, target_emb.detach())
        total_loss += loss.item()
        loss_sum = loss_sum + loss

        # Detach hidden to prevent BPTT through entire sequence
        h = h.detach()

    loss_sum.backward()
    optimizer.step()
    return total_loss / max(len(transitions), 1)


def simple_jepa_training_step(
    sjepa: SimpleJEPA,
    transitions: List[Tuple[GridState, str, GridState]],
    lr: float = 1e-3,
) -> float:
    """Train SimpleJEPA on transitions.

    Returns average MSE loss across ensemble.
    """
    if not transitions:
        return 0.0

    optimizer = torch.optim.Adam(sjepa.parameters(), lr=lr)
    optimizer.zero_grad()

    total_loss = 0.0

    for state, action_str, next_state in transitions:
        feats = grid_state_features(state)
        next_feats = grid_state_features(next_state)
        action_idx = torch.tensor([action_to_idx(action_str)])

        preds, _ = sjepa.forward(feats, action_idx)
        target_emb = sjepa.state_encoder(next_feats)

        loss = sjepa.compute_loss(preds, target_emb.detach())
        total_loss += loss.item()
        loss.backward(retain_graph=False)

    optimizer.step()
    return total_loss / max(len(transitions), 1)
