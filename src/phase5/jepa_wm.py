"""JEPA-based World Model: embedding-space prediction with ensemble uncertainty.

Replaces LLM token prediction with embedding-space dynamics.
Uses frozen Qwen 0.5B encoder + trainable MLP ensemble predictors.

Architecture:
  (state_text, action) → [frozen Qwen 0.5B] → state_emb (768-dim)
                                                    ↓
                                        [MLP ensemble × 3]
                                                    ↓
                                           predicted next_emb
                                                    ↓
                               epistemic = variance across predictions
"""

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def _state_to_text(state) -> str:
    """Convert a SandboxState into a flat text string for encoding."""
    files_str = ",".join(sorted(state.files)) if state.files else ""
    output_str = (state.last_output or "")[:100]
    return f"cwd: {state.cwd} | files: {files_str} | last_output: {output_str}"


def _next_state_to_text(next_state) -> str:
    """Convert a next-state/state object to text (same format as _state_to_text)."""
    files_str = ",".join(sorted(next_state.files)) if hasattr(next_state, "files") and next_state.files else ""
    output_str = (getattr(next_state, "last_output", "") or "")[:100]
    cwd = getattr(next_state, "cwd", "/sandbox")
    return f"cwd: {cwd} | files: {files_str} | last_output: {output_str}"


class MLPPredictor(nn.Module):
    """MLP that predicts next embedding from (state_emb, action_emb)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_size),
        )

    def forward(self, state_emb: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state_emb, action_emb], dim=-1)
        return self.net(x)


class JEPAEnsemble:
    """Ensemble of MLP predictors over frozen Qwen embeddings.

    Epistemic uncertainty = variance of predictions across ensemble members.
    No ground-truth next_state needed during action selection.
    """

    def __init__(self, model_path: str, n_ensemble: int = 3, device: str = "cuda"):
        self.device = device

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

        # --- Ensemble of MLP predictors ---
        self.predictors = nn.ModuleList()
        for i in range(n_ensemble):
            torch.manual_seed(42 + i)
            predictor = MLPPredictor(self.hidden_size)
            predictor.to(device)
            self.predictors.append(predictor)

        self.optimizers = [
            torch.optim.Adam(p.parameters(), lr=1e-3)
            for p in self.predictors
        ]

        # State encoding cache (avoids redundant Qwen forward passes in same step)
        self._state_cache = {}

    def _cache_clear(self):
        """Clear the state encoding cache (call on episode reset)."""
        self._state_cache = {}

    def reset_predictors(self):
        """Reinitialize all MLP predictor weights and reset optimizers."""
        for i, predictor in enumerate(self.predictors):
            for layer in predictor.net:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
            self.optimizers[i] = torch.optim.Adam(predictor.parameters(), lr=1e-3)

    # ── Encoding ─────────────────────────────────────────

    @torch.no_grad()
    def encode_state(self, state_text: str) -> torch.Tensor:
        """Tokenize → Qwen forward → mean-pooled last hidden → [1, H].

        Results are cached by state_text to avoid redundant forward passes
        when the same state is evaluated for multiple candidate actions.
        """
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

        preds = torch.stack([p(state_emb, action_emb) for p in self.predictors])  # [N, 1, H]
        mean_pred = preds.mean(dim=0)  # [1, H]
        # Mean squared deviation across ensemble and all dims
        epistemic = ((preds - mean_pred.unsqueeze(0)) ** 2).mean().item()
        return epistemic

    # ── Training ─────────────────────────────────────────

    def train_step(self, transitions: list) -> float:
        """Train all ensemble members on (state, action, next_state) transitions.

        Args:
            transitions: list of (state_obj, action_str, next_state_obj)

        Returns:
            Average MSE loss across ensemble (float).
        """
        if not transitions:
            return 0.0

        total_loss = 0.0
        for i, predictor in enumerate(self.predictors):
            self.optimizers[i].zero_grad()
            losses = []

            for state, action_text, next_state in transitions:
                state_text = _state_to_text(state)
                next_text = _next_state_to_text(next_state)

                state_emb = self.encode_state(state_text)
                action_emb = self.encode_action(action_text)
                next_emb = self.encode_state(next_text)

                pred_emb = predictor(state_emb, action_emb)
                loss = nn.functional.mse_loss(pred_emb, next_emb)
                losses.append(loss)

            avg_loss = torch.stack(losses).mean()
            avg_loss.backward()
            self.optimizers[i].step()
            total_loss += avg_loss.item()

        return total_loss / len(self.predictors)


# ── State/text helpers used externally ──────────────────

def state_to_text(state) -> str:
    return _state_to_text(state)


def next_state_to_text(next_state) -> str:
    return _next_state_to_text(next_state)
