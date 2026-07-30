"""Goal-Conditioned JEPA: embedding-space prediction conditioned on goal.

Extends JEPAEnsemble by conditioning predictors on a goal description.
Each ensemble member takes (state_emb || goal_emb || action_emb) → predicted next_emb.

Epistemic uncertainty is scaled by predicted distance-to-goal reduction,
giving discriminatory power: "action A is more uncertain than action B
for reaching the goal."

Architecture:
  (state_text, action, goal_text) → [frozen Qwen 0.5B] → s_emb (768), g_emb (768)
                                                                  ↓
                                              [GoalMLP ensemble x3]   ← a_emb (64)
                                                                  ↓
                                                         predicted next_emb
                                                                  ↓
                                  uncertainty = variance x (1 + progress_scalar)
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class GoalMLP(nn.Module):
    """MLP that predicts next embedding from (state_emb, goal_emb, action_emb).

    The goal embedding makes the predictor goal-aware: the same action in the
    same state will produce different predictions depending on the goal.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        state_emb: torch.Tensor,
        goal_emb: torch.Tensor,
        action_emb: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([state_emb, goal_emb, action_emb], dim=-1)
        return self.net(x)


class GoalJEPAEnsemble:
    """Ensemble of goal-conditioned MLP predictors over frozen Qwen embeddings.

    Epistemic uncertainty = ensemble prediction variance, scaled by predicted
    progress toward the goal (reduction in embedding-space distance to goal).
    """

    def __init__(
        self,
        model_path: str,
        n_ensemble: int = 3,
        action_emb_dim: int = 64,
        device: str = "cuda",
    ):
        self.device = device
        self.action_emb_dim = action_emb_dim

        # --- Shared frozen encoder ---
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.encoder = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
        ).to(device)

        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        self.hidden_size = self.encoder.config.hidden_size

        # --- Learned action projection ---
        # Projects 768-dim encoder embeddings to compact 64-dim representation.
        # Trainable linear layer, no token-id collision problem.
        self.action_projection = nn.Linear(self.hidden_size, action_emb_dim).to(device)

        predictor_input_dim = self.hidden_size * 2 + action_emb_dim

        # --- Ensemble of goal-conditioned MLP predictors ---
        self.predictors = nn.ModuleList()
        for i in range(n_ensemble):
            torch.manual_seed(42 + i)
            predictor = GoalMLP(
                input_dim=predictor_input_dim,
                hidden_dim=256,
                output_dim=self.hidden_size,
            )
            predictor.to(device)
            self.predictors.append(predictor)

        self.optimizers = [
            torch.optim.Adam(p.parameters(), lr=1e-3)
            for p in self.predictors
        ]

        # Encoding caches (avoid redundant Qwen forward passes in same step)
        self._state_cache = {}
        self._goal_cache = {}

    def _cache_clear(self):
        """Clear caches (call on episode reset)."""
        self._state_cache = {}
        self._goal_cache = {}

    def reset_predictors(self):
        """Reinitialize all GoalMLP weights and reset optimizers."""
        for i, predictor in enumerate(self.predictors):
            for layer in predictor.net:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
            # Re-init action projection
            nn.init.xavier_uniform_(self.action_projection.weight)
            if self.action_projection.bias is not None:
                nn.init.zeros_(self.action_projection.bias)
            self.optimizers[i] = torch.optim.Adam(predictor.parameters(), lr=1e-3)

    # ── Encoding ─────────────────────────────────────────

    @torch.no_grad()
    def encode_state(self, state_text: str) -> torch.Tensor:
        """Tokenize -> Qwen forward -> mean-pooled last hidden -> [1, H]."""
        if state_text in self._state_cache:
            return self._state_cache[state_text]
        tokens = self.tokenizer(
            state_text, return_tensors="pt", truncation=True, max_length=256
        ).to(self.device)
        outputs = self.encoder(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]  # [1, seq_len, H]
        mask = tokens.attention_mask.unsqueeze(-1).float()
        emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        self._state_cache[state_text] = emb
        return emb

    @torch.no_grad()
    def encode_goal(self, goal_text: str) -> torch.Tensor:
        """Encode a goal description into the shared embedding space.

        Uses the same frozen Qwen encoder as encode_state so that
        state embeddings and goal embeddings live in the same space.
        Distance in this space measures task-relevance.
        """
        if goal_text in self._goal_cache:
            return self._goal_cache[goal_text]

        tokens = self.tokenizer(
            goal_text, return_tensors="pt", truncation=True, max_length=64
        ).to(self.device)
        outputs = self.encoder(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]
        mask = tokens.attention_mask.unsqueeze(-1).float()
        emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        self._goal_cache[goal_text] = emb
        return emb

    def encode_action(self, action_text: str) -> torch.Tensor:
        """Map action text to compact learned embedding [1, A].

        Uses encoder's embedding layer then projects to 64-dim via trainable
        linear projection. No @torch.no_grad() here because the projection
        needs gradients during training. Callers already add no_grad when needed
        (epistemic_uncertainty, goal_progress_uncertainty are @torch.no_grad()).
        """
        tokens = self.tokenizer(
            action_text, return_tensors="pt", truncation=True, max_length=32
        ).to(self.device)
        # Full encoder embedding layer (768-dim, frozen)
        raw_emb = self.encoder.get_input_embeddings()(tokens.input_ids)
        mask = tokens.attention_mask.unsqueeze(-1).float()
        # Mean-pool over sequence
        pooled = (raw_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        # Project to compact 64-dim (trainable)
        compact = self.action_projection(pooled)
        return compact

    # ── Prediction ──────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        state_emb: torch.Tensor,
        action_emb: torch.Tensor,
        goal_emb: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Run all ensemble predictors. Returns list of [1, H] tensors."""
        return [p(state_emb, goal_emb, action_emb) for p in self.predictors]

    # ── Uncertainty signals ─────────────────────────────

    @torch.no_grad()
    def epistemic_uncertainty(
        self,
        state_text: str,
        action_text: str,
        goal_text: str,
    ) -> float:
        """Raw ensemble prediction variance = epistemic uncertainty.

        Range depends on embedding scale (~0-2+). Does NOT factor in
        goal-progress — use goal_progress_uncertainty for the full signal.
        """
        state_emb = self.encode_state(state_text)
        action_emb = self.encode_action(action_text)
        goal_emb = self.encode_goal(goal_text)

        preds = torch.stack(self.predict(state_emb, action_emb, goal_emb))  # [N, 1, H]
        mean_pred = preds.mean(dim=0)
        epistemic = ((preds - mean_pred.unsqueeze(0)) ** 2).mean().item()
        return epistemic

    @torch.no_grad()
    def goal_progress_uncertainty(
        self,
        state_text: str,
        action_text: str,
        goal_text: str,
    ) -> float:
        """Goal-scaled epistemic uncertainty.

        Scoring function for action selection:
          score = epistemic_variance * (1 + max(0, predicted_progress))

        Where predicted_progress = current_dist_to_goal - predicted_dist_to_goal
        (positive = ensemble predicts this action moves toward the goal).

        This gives high scores to actions that are both uncertain AND
        predicted to make progress — the key discriminatory signal
        that vanilla JEPA lacks.
        """
        state_emb = self.encode_state(state_text)
        action_emb = self.encode_action(action_text)
        goal_emb = self.encode_goal(goal_text)

        preds = torch.stack(self.predict(state_emb, action_emb, goal_emb))  # [N, 1, H]
        mean_pred = preds.mean(dim=0)

        # Epistemic = ensemble variance
        epistemic = ((preds - mean_pred.unsqueeze(0)) ** 2).mean().item()

        # Predicted distance-to-goal reduction
        # Distance in embedding space = how far is current state from goal
        current_dist = torch.norm(state_emb - goal_emb, dim=-1).item()
        predicted_dist = torch.norm(mean_pred - goal_emb, dim=-1).item()
        progress = current_dist - predicted_dist  # positive = moving toward goal

        # Score: high uncertainty + predicted progress = highest
        score = epistemic * (1.0 + max(0.0, progress))
        return score

    # ── Training ─────────────────────────────────────────

    def train_step(
        self,
        transitions: list,
        goal_text: str,
    ) -> float:
        """Train all ensemble members on (state, action, next_state) transitions.

        Args:
            transitions: list of (state_obj, action_str, next_state_obj)
            goal_text: the fixed goal description for this episode

        Returns:
            Average MSE loss across ensemble (float).
        """
        if not transitions:
            return 0.0

        goal_emb = self.encode_goal(goal_text)

        total_loss = 0.0
        for i, predictor in enumerate(self.predictors):
            self.optimizers[i].zero_grad()
            losses = []

            for state, action_text, next_state in transitions:
                # Convert to text representation
                s_text = grid_state_to_text(state)
                ns_text = grid_state_to_text(next_state)

                state_emb = self.encode_state(s_text)
                action_emb = self.encode_action(action_text)
                next_emb = self.encode_state(ns_text)

                pred_emb = predictor(state_emb, goal_emb, action_emb)
                loss = F.mse_loss(pred_emb, next_emb)
                losses.append(loss)

            avg_loss = torch.stack(losses).mean()
            avg_loss.backward()
            self.optimizers[i].step()
            total_loss += avg_loss.item()

        return total_loss / len(self.predictors)


# ── State/text helpers ──────────────────────────────────

def grid_state_to_text(state) -> str:
    """Convert a GridState to flat text for JEPA encoding.

    Mirrors grid_env.grid_state_to_text to avoid circular import.
    """
    exit_str = ",".join(
        k for k in ["north", "south", "east", "west"]
        if getattr(state, "exits", {}).get(k)
    )
    inv_str = ",".join(state.inventory) if getattr(state, "inventory", None) else "none"
    return (
        f"x: {getattr(state, 'x', 0)} | y: {getattr(state, 'y', 0)} | "
        f"inventory: {inv_str} | "
        f"room: {getattr(state, 'room_name', '')} | "
        f"exits: {exit_str}"
    )
