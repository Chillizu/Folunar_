"""Phase 7: Lightweight JEPA for Giant Maze Scaling.

Adapted from phase5 JEPAEnsemble with scaling optimizations:
  - Reduced ensemble (2 members, 128 hidden dim) for speed
  - Down-sampled training: only ~5 random rooms per episode
  - State encoding cache reused from phase5 pattern
  - No text generation, only embedding prediction

Hypothesis: At 100x100 scale (~100M states), JEPA's learned embedding
abstraction might compress the state space and provide useful uncertainty
where counting gives uniformly high novelty.
"""

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class GiantMLPPredictor(nn.Module):
    """Reduced MLP predictor for giant maze.

    896 → 128 → 24 → 896  (vs phase5's 896 → 256 → 896 × 3)
    """

    def __init__(self, input_dim: int = 896, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + input_dim, hidden_dim),   # state + action
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 5),          # 128→25 or 256→51
            nn.ReLU(),
            nn.Linear(hidden_dim // 5, input_dim),           # back to hidden
        )

    def forward(self, state_emb: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state_emb, action_emb], dim=-1)
        return self.net(x)


class GiantJEPAEnsemble:
    """Lightweight JEPA ensemble for giant maze scaling experiments.

    Key differences from JEPAEnsemble (phase5):
      - 2 ensemble members (vs 3)
      - 128 hidden dim (vs 256)
      - Down-sampled training: sample N transitions per episode
      - Imports GiantGridState support via giant_state_to_text
    """

    def __init__(
        self,
        model_path: str,
        n_ensemble: int = 2,
        hidden_dim: int = 128,
        device: str = "cuda",
    ):
        self.device = device
        self.n_ensemble = n_ensemble
        self.hidden_dim = hidden_dim

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

        # --- Reduced ensemble ---
        self.predictors = nn.ModuleList()
        for i in range(n_ensemble):
            torch.manual_seed(42 + i)
            predictor = GiantMLPPredictor(self.hidden_size, hidden_dim)
            predictor.to(device)
            self.predictors.append(predictor)

        self.optimizers = [
            torch.optim.Adam(p.parameters(), lr=1e-3)
            for p in self.predictors
        ]

        # State encoding cache
        self._state_cache: dict = {}

    def _cache_clear(self):
        """Clear state encoding cache (call on episode reset)."""
        self._state_cache = {}

    def reset_predictors(self):
        """Reinitialize all MLP weights and reset optimizers."""
        for i, predictor in enumerate(self.predictors):
            for layer in predictor.net:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
            self.optimizers[i] = torch.optim.Adam(
                predictor.parameters(), lr=1e-3
            )

    # ── Encoding ─────────────────────────────────────────

    @torch.no_grad()
    def encode_state(self, state_text: str) -> torch.Tensor:
        """Tokenize → Qwen forward → mean-pooled last hidden → [1, H].

        Results cached by state_text.
        """
        if state_text in self._state_cache:
            return self._state_cache[state_text]
        tokens = self.tokenizer(
            state_text, return_tensors="pt", truncation=True, max_length=128
        ).to(self.device)
        outputs = self.encoder(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]
        mask = tokens.attention_mask.unsqueeze(-1).float()
        emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        self._state_cache[state_text] = emb
        return emb

    @torch.no_grad()
    def encode_action(self, action_text: str) -> torch.Tensor:
        """Action text → embedding layer mean-pool → [1, H]."""
        tokens = self.tokenizer(
            action_text, return_tensors="pt", truncation=True, max_length=32
        ).to(self.device)
        emb = self.encoder.get_input_embeddings()(tokens.input_ids)
        mask = tokens.attention_mask.unsqueeze(-1).float()
        emb = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return emb

    # ── Epistemic signal ─────────────────────────────────

    @torch.no_grad()
    def epistemic_uncertainty(self, state_text: str, action_text: str) -> float:
        """Ensemble prediction variance = epistemic uncertainty.

        Range ~[0, 2+] depending on embedding scale.
        Higher = more uncertain = worth exploring.
        """
        state_emb = self.encode_state(state_text)
        action_emb = self.encode_action(action_text)

        preds = torch.stack([
            p(state_emb, action_emb) for p in self.predictors
        ])  # [N, 1, H]
        mean_pred = preds.mean(dim=0)
        epistemic = ((preds - mean_pred.unsqueeze(0)) ** 2).mean().item()
        return epistemic

    # ── Down-sampled Training ────────────────────────────

    def train_step(
        self,
        transitions: list,
        state_to_text_fn,
        max_samples: int = 5,
    ) -> float:
        """Train ensemble on down-sampled transitions.

        Only trains on max_samples random transitions per episode,
        because encoding every step of a 2000-step episode through
        Qwen 0.5B would be prohibitively slow.

        Args:
            transitions: list of (state_obj, action_str, next_state_obj)
            state_to_text_fn: callable that converts a state to flat text
            max_samples: max transitions to train on (default 5)

        Returns:
            Average MSE loss across ensemble (float).
        """
        if not transitions:
            return 0.0

        # Down-sample: pick up to max_samples random transitions
        if len(transitions) > max_samples:
            indices = np.random.choice(
                len(transitions), max_samples, replace=False
            )
            transitions = [transitions[i] for i in indices]

        total_loss = 0.0
        for i, predictor in enumerate(self.predictors):
            self.optimizers[i].zero_grad()
            losses = []

            for state, action_text, next_state in transitions:
                state_text = state_to_text_fn(state)
                next_text = state_to_text_fn(next_state)

                state_emb = self.encode_state(state_text)
                action_emb = self.encode_action(action_text)
                next_emb = self.encode_state(next_text)

                pred_emb = predictor(state_emb, action_emb)
                loss = nn.functional.mse_loss(pred_emb, next_emb)
                losses.append(loss)

            if losses:
                avg_loss = torch.stack(losses).mean()
                avg_loss.backward()
                self.optimizers[i].step()
                total_loss += avg_loss.item()

        return total_loss / len(self.predictors) if self.predictors else 0.0
