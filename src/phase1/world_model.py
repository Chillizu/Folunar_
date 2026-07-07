"""World Model, ensemble error computer, and learning module for Phase 1."""

import ast
import json
import math
import os
import random
import re
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

from phase1.grid_env import GridWorld, Perception
from phase1.types import Action, ErrorVector, Experience, GridState, PredictedState
from phase2.sandbox_env import SandboxState

# Optional heavy dependencies. The LLM path is enabled when they are installed and
# a model is available; otherwise the deterministic stub path is used.
try:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _LLM_DEPS_AVAILABLE = True
except Exception:  # pragma: no cover - depends on environment
    _LLM_DEPS_AVAILABLE = False


class WorldModel:
    """Three-level predictive model: Level 1 exit code, Level 2 next position, Level 3 summary."""

    DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
    FALLBACK_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    EMERGENCY_MODEL = "microsoft/Phi-3-mini-4k-instruct"

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        use_stub: Optional[bool] = None,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        adapter_path: Optional[str] = None,
    ):
        self.model_name = model_name or self.DEFAULT_MODEL
        if _LLM_DEPS_AVAILABLE and device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device or "cpu"
        self.mode = "llm"
        self.model = None
        self.tokenizer = None
        self.adapter_name = "default"
        self._lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules="all-linear",
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        ) if _LLM_DEPS_AVAILABLE else None

        if use_stub is True:
            self.mode = "stub"
            return

        if use_stub is None:
            if os.environ.get("FOLUNAR_STUB_MODEL", "0") == "1":
                self.mode = "stub"
                return

        if not _LLM_DEPS_AVAILABLE:
            print("[WorldModel] transformers/peft not installed; falling back to deterministic stub.")
            self.mode = "stub"
            return

        self._load_model()
        if adapter_path and self.mode == "llm" and self.model is not None:
            path = Path(adapter_path)
            if path.exists():
                try:
                    self.model.load_adapter(str(path), adapter_name="synthetic")
                    self.model.set_adapter("synthetic")
                    self.adapter_name = "synthetic"
                    print(f"[WorldModel] Loaded adapter from {path}")
                except Exception as exc:
                    print(f"[WorldModel] Failed to load adapter {adapter_path}: {exc}")
            else:
                print(f"[WorldModel] Adapter path not found: {adapter_path}; using default adapter.")

    def _load_model(self) -> None:
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                device_map=None,
            ).to(self.device)
            if self._lora_config is not None:
                self.model = get_peft_model(self.model, self._lora_config)
                self.model.set_adapter(self.adapter_name)
            self.model.eval()
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"[WorldModel] Failed to load {self.model_name}: {exc}")
            print("[WorldModel] Falling back to deterministic stub.")
            self.mode = "stub"
            self.model = None
            self.tokenizer = None

    def _system_message(self) -> str:
        """System instruction with one concise example for grid prediction."""
        return (
            "You predict the next state of a 5x5 grid world. "
            "Given a state and action, output exactly one JSON object with "
            "next_position, exit_code, and summary.\n"
            "exit_code rules: 0 = moved successfully, 1 = hit wall or obstacle, 2 = reached goal.\n\n"
            "Example:\n"
            "State: Agent at (0, 0). Goal at (4, 4). Obstacles at none.\n"
            "Action: RIGHT\n"
            '{"next_position": [1, 0], "exit_code": 0, "summary": "agent moved right"}'
        )

    def _build_prompt(self, state: GridState, action: Optional[Action]) -> str:
        state_text = Perception.render(state)
        action_text = action.name if action else "NONE"
        return (
            f"State: {state_text}\n"
            f"Action: {action_text}\n"
            "Predict next position, exit code, and one-line summary as JSON:"
        )

    def _text_system_message(self) -> str:
        """System instruction for text adventure prediction."""
        return (
            "You predict what happens next in a text adventure game. "
            "Given a state description and action, output exactly one JSON object with "
            "next_room, next_description, next_inventory, exit_code, and summary.\n"
            "exit_code rules: 0 = action succeeded, 1 = action failed or invalid, 2 = game completed.\n\n"
            "Example:\n"
            "State:\n"
            'Location: STUDY.\n'
            'Description: You are in a small study. On the desk is a key. A door leads north.\n'
            'Inventory: nothing.\n'
            'Goal: Find the treasure.\n'
            "Action: take key\n"
            '{"next_room": "study", "next_description": "You take the key from the desk.", '
            '"next_inventory": "key", "exit_code": 0, "summary": "you take the key"}'
        )

    def _sandbox_system_message(self) -> str:
        """System instruction for Linux sandbox state prediction."""
        return (
            "You predict how a Linux shell state changes after a command. "
            "The state is a JSON object with cwd, files, last_command, last_exit_code, last_output. "
            "Given a state and a command, output exactly one JSON object with "
            "cwd, files, last_command, last_exit_code, last_output, exit_code, and summary.\n"
            "exit_code rules: 0 = command succeeded, 1 = command failed, 2 = task completed.\n\n"
            "Example:\n"
            'State: {"cwd": "/sandbox", "files": ["hello.txt", "docs", "data"], "last_command": "", "last_exit_code": 0, "last_output": "", "step": 0}\n'
            "Action: ls\n"
            '{"cwd": "/sandbox", "files": ["hello.txt", "docs", "data"], "last_command": "ls", "last_exit_code": 0, "last_output": "hello.txt\\ndocs\\ndata", "exit_code": 0, "summary": "listed files"}'
        )

    def _build_text_prompt(self, state, action: Optional[Action]) -> str:
        from phase1.grid_env import Perception as P
        state_text = P.render_text(state)
        action_text = action if isinstance(action, str) else (action.name if action else "NONE")
        if hasattr(state, "container_id"):
            return (
                f"State: {state_text}\n"
                f"Action: {action_text}\n"
                "Predict cwd, files, last_command, last_exit_code, last_output, exit_code, and summary as JSON:"
            )
        return (
            f"State:\n{state_text}\n"
            f"Action: {action_text}\n"
            "Predict next_room, next_description, exit_code, and summary as JSON:"
        )

    @staticmethod
    def _parse_generation(text: str) -> Dict[str, Any]:
        """Extract the first JSON object from generated text."""
        # Try to find JSON object {...}
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        # Fallback: try the whole text
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _extract_input_ids(tokenizer_output: Any) -> torch.Tensor:
        """Return the input_ids tensor whether tokenizer output is a Tensor or BatchEncoding."""
        if isinstance(tokenizer_output, torch.Tensor):
            return tokenizer_output
        return tokenizer_output["input_ids"]

    @staticmethod
    def _generation_confidence(scores, generated_ids):
        if not scores or generated_ids.numel() == 0:
            return 0.8
        probs = []
        for token_id, score in zip(generated_ids, scores):
            probs.append(torch.softmax(score, dim=-1)[0, token_id].item())
        return float(sum(probs)) / len(probs)

    def _llm_predict(self, state, action: Optional[Action]) -> PredictedState:
        is_sandbox = hasattr(state, "container_id")
        is_text = hasattr(state, "room") or is_sandbox
        prompt = self._build_text_prompt(state, action) if is_text else self._build_prompt(state, action)
        msg_system = self._sandbox_system_message() if is_sandbox else (self._text_system_message() if hasattr(state, "room") else self._system_message())
        max_tok = 200 if is_sandbox else (120 if is_text else 80)
        messages = [
            {"role": "system", "content": msg_system},
            {"role": "user", "content": prompt},
        ]
        if self.tokenizer.chat_template is not None:
            input_ids = self._extract_input_ids(
                self.tokenizer.apply_chat_template(
                    messages, tokenize=True, return_tensors="pt", add_generation_prompt=True
                )
            ).to(self.device)
            input_len = input_ids.shape[1]
            with torch.no_grad():
                gen_out = self.model.generate(
                    input_ids,
                    max_new_tokens=max_tok,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            generated_ids = gen_out.sequences[0, input_len:]
            generated = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            confidence = self._generation_confidence(gen_out.scores, generated_ids)
        else:
            formatted_prompt = f"{msg_system}\n\n{prompt}"
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
            input_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                gen_out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tok,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            generated_ids = gen_out.sequences[0, input_len:]
            generated = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            confidence = self._generation_confidence(gen_out.scores, generated_ids)
        parsed = self._parse_generation(generated)

        if is_sandbox:
            exit_code = parsed.get("exit_code", 1)
            try:
                exit_code = int(exit_code)
            except (TypeError, ValueError):
                exit_code = 1
            summary = str(parsed.get("summary", ""))[:80]
            pred_json = {k: parsed.get(k, "") for k in ("cwd", "files", "last_command", "last_exit_code", "last_output")}
            l2_text = json.dumps(pred_json, ensure_ascii=False)
            return PredictedState(
                level1_exit_code=exit_code,
                level1_confidence=confidence,
                level2_next_agent=(0, 0),
                level2_confidence=confidence,
                level3_output_summary=summary,
                level3_confidence=confidence,
                level2_text=l2_text,
                epistemic_ratio=1.0 - confidence,
            )
        if is_text:
            next_room = parsed.get("next_room", state.room)
            next_desc = parsed.get("next_description", state.description)
            next_inv = str(parsed.get("next_inventory", "nothing"))
            exit_code = parsed.get("exit_code", 1)
            try:
                exit_code = int(exit_code)
            except (TypeError, ValueError):
                exit_code = 1
            summary = str(parsed.get("summary", ""))[:80]
            return PredictedState(
                level1_exit_code=exit_code,
                level1_confidence=confidence,
                level2_next_agent=(0, 0),
                level2_confidence=confidence,
                level3_output_summary=summary,
                level3_confidence=confidence,
                level2_text=f"Location: {next_room}.\n{next_desc}\nInventory: {next_inv}.",
                epistemic_ratio=1.0 - confidence,
            )
        # Grid state path (original logic)
        next_pos = parsed.get("next_position")
        if isinstance(next_pos, list) and len(next_pos) == 2:
            next_pos = (int(next_pos[0]), int(next_pos[1]))
        else:
            next_pos = state.agent
        exit_code = parsed.get("exit_code")
        try:
            exit_code = int(exit_code)
        except (TypeError, ValueError):
            exit_code = 1
        summary = str(parsed.get("summary", ""))[:80]
        return PredictedState(
            level1_exit_code=exit_code,
            level1_confidence=confidence,
            level2_next_agent=next_pos,
            level2_confidence=confidence,
            level3_output_summary=summary,
            level3_confidence=confidence,
            epistemic_ratio=1.0 - confidence,
        )

    def _stub_predict(self, state: GridState, action: Optional[Action]) -> PredictedState:
        """Deterministic grid-rule predictor for dependency-free testing."""
        # TextState stub (Phase 1.5)
        if hasattr(state, "room"):
            pred_room = state.room
            if action.name in ("go north", "north") and state.room == "study":
                pred_room = "hallway"
            elif action.name in ("go south", "south") and state.room == "hallway":
                pred_room = "study"
            exit_code = 2 if state.victory else 0
            inv_str = ", ".join(state.inventory) if state.inventory else "nothing"
            return PredictedState(
                level1_exit_code=exit_code,
                level1_confidence=1.0,
                level2_next_agent=(0, 0),
                level2_confidence=1.0,
                level3_output_summary=f"you {action.name}",
                level3_confidence=1.0,
                level2_text=f"Location: {pred_room}.\n{state.description}\nInventory: {inv_str}.",
                epistemic_ratio=0.0,
            )

        env = GridWorld(
            width=state.width,
            height=state.height,
            obstacles=state.obstacles,
            goal=state.goal,
            max_steps=state.max_steps,
        )
        if action is None:
            next_pos = state.agent
            exit_code = 1
            summary = "no action provided"
        else:
            next_state, _, _ = env.step(state, action)
            next_pos = next_state.agent
            if next_pos == state.goal:
                exit_code = 2
                summary = f"agent reached goal via {action.name}"
            elif next_pos == state.agent:
                exit_code = 1
                summary = f"agent hit wall/obstacle with {action.name}"
            else:
                exit_code = 0
                summary = f"agent moved {action.name.lower()}"
        return PredictedState(
            level1_exit_code=exit_code,
            level1_confidence=1.0,
            level2_next_agent=next_pos,
            level2_confidence=1.0,
            level3_output_summary=summary,
            level3_confidence=1.0,
            epistemic_ratio=0.0,
        )

    def predict(self, state: GridState, action: Optional[Action] = None) -> PredictedState:
        if self.mode == "stub" or self.model is None:
            return self._stub_predict(state, action)
        return self._llm_predict(state, action)

    def rollout(self, state, action: Action, horizon: int = 2) -> List[PredictedState]:
        """Apply the same candidate action repeatedly to build a short trajectory."""
        trajectory = []
        current_state = state
        for _ in range(horizon):
            pred = self.predict(current_state, action)
            trajectory.append(pred)
            if hasattr(current_state, "container_id"):
                try:
                    j = json.loads(pred.level2_text) if pred.level2_text else {}
                except (json.JSONDecodeError, TypeError):
                    j = {}
                current_state = SandboxState(
                    container_id=current_state.container_id,
                    cwd=j.get("cwd", current_state.cwd),
                    files=j.get("files", current_state.files),
                    last_command=j.get("last_command", ""),
                    last_exit_code=j.get("last_exit_code", 0),
                    last_output=j.get("last_output", "")[:500],
                    step_count=current_state.step_count + 1,
                    max_steps=current_state.max_steps,
                )
                continue
            if hasattr(current_state, "room"):
                # TextState: maintain text fields, update room from prediction
                next_room = current_state.room
                if pred.level2_text and "Location: " in pred.level2_text:
                    next_room = pred.level2_text.split("\n")[0].replace("Location: ", "").strip().rstrip(".")
                current_state = current_state.__class__(
                    room=next_room,
                    description=current_state.description,
                    inventory=list(current_state.inventory),
                    goal=current_state.goal,
                    step=current_state.step + 1,
                    max_steps=current_state.max_steps,
                )
            else:
                current_state = GridState(
                    agent=pred.level2_next_agent,
                    goal=current_state.goal,
                    obstacles=current_state.obstacles,
                    width=current_state.width,
                    height=current_state.height,
                    step=current_state.step + 1,
                    max_steps=current_state.max_steps,
                )
        return trajectory

    def lora_finetune(
        self,
        data: List[Dict[str, str]],
        epochs: int = 1,
        learning_rate: float = 2e-4,
        batch_size: int = 4,
        checkpoint_dir: Optional[Path] = None,
        text_mode: bool = False,
    ) -> None:
        """Batch fine-tune the LoRA adapter on (state, action) -> next-state examples.

        text_mode=True uses text-environment prompt/target format (next_room/next_description)
        instead of grid format (next_position).
        """
        if self.mode == "stub" or self.model is None:
            return

        examples = []
        for ex in data:
            state_text = ex["state_text"]
            action_text = ex["action_name"]
            exit_code = int(ex.get("exit_code", 0))
            summary = str(ex.get("summary", ""))

            if text_mode:
                next_room = ex.get("next_room", "")
                next_desc = ex.get("next_description", "")
                next_inv = ex.get("next_inventory", "nothing")
                target = json.dumps({
                    "next_room": next_room,
                    "next_description": next_desc,
                    "next_inventory": next_inv,
                    "exit_code": exit_code,
                    "summary": summary,
                }, ensure_ascii=False)
                user_content = (
                    f"State:\n{state_text}\n"
                    f"Action: {action_text}\n"
                    "Predict next_room, next_description, next_inventory, exit_code, and summary as JSON:"
                )
                msg_system = self._text_system_message()
            else:
                # Grid format (original logic)
                next_text = ex.get("next_state_text", "")
                next_pos = [0, 0]
                try:
                    import ast
                    parsed = ast.literal_eval(next_text)
                    if isinstance(parsed, (list, tuple)) and len(parsed) == 2:
                        next_pos = [int(parsed[0]), int(parsed[1])]
                except (SyntaxError, ValueError, TypeError):
                    pass
                target = json.dumps({
                    "next_position": next_pos,
                    "exit_code": exit_code,
                    "summary": summary,
                }, ensure_ascii=False)
                user_content = (
                    f"State: {state_text}\n"
                    f"Action: {action_text}\n"
                    "Predict next position, exit code, and one-line summary as JSON:"
                )
                msg_system = self._system_message()

            messages = [
                {"role": "system", "content": msg_system},
                {"role": "user", "content": user_content},
            ]
            if self.tokenizer.chat_template is not None:
                prompt_input_ids = self._extract_input_ids(
                    self.tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True, return_tensors="pt", max_length=256, truncation=True
                    )
                ).squeeze(0)
                full_messages = messages + [{"role": "assistant", "content": target}]
                full_input_ids = self._extract_input_ids(
                    self.tokenizer.apply_chat_template(
                        full_messages, add_generation_prompt=False, return_tensors="pt", max_length=256, truncation=True
                    )
                ).squeeze(0)
            else:
                prompt = f"{msg_system}\n\n{user_content}"
                full = f"{prompt}{target}{self.tokenizer.eos_token}"
                prompt_input_ids = self.tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True)["input_ids"].squeeze(0)
                full_input_ids = self.tokenizer(full, return_tensors="pt", max_length=256, truncation=True)["input_ids"].squeeze(0)
            examples.append((prompt_input_ids, full_input_ids))

        if not examples:
            return

        # (dataset, dataloader, optimizer, training loop — unchanged from original)
        class _GridDataset(torch.utils.data.Dataset):
            def __init__(self, pairs, tokenizer):
                self.pairs = pairs
                self.tokenizer = tokenizer
            def __len__(self):
                return len(self.pairs)
            def __getitem__(self, idx):
                prompt_input_ids, full_input_ids = self.pairs[idx]
                input_ids = full_input_ids
                attention_mask = torch.ones_like(input_ids)
                labels = input_ids.clone()
                prompt_len = prompt_input_ids.shape[0]
                labels[:prompt_len] = -100
                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }

        dataset = _GridDataset(examples, self.tokenizer)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
        optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=learning_rate)
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            for batch_idx, batch in enumerate(dataloader):
                optimizer.zero_grad()
                input_ids = torch.nn.utils.rnn.pad_sequence(
                    [b["input_ids"] for b in batch], batch_first=True, padding_value=self.tokenizer.pad_token_id
                )
                attention_mask = torch.nn.utils.rnn.pad_sequence(
                    [b["attention_mask"] for b in batch], batch_first=True, padding_value=0
                )
                labels = torch.nn.utils.rnn.pad_sequence(
                    [b["labels"] for b in batch], batch_first=True, padding_value=-100
                )
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                if loss is not None:
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    if batch_idx % 20 == 0:
                        print(f"[lora_finetune] epoch {epoch+1}/{epochs} batch {batch_idx} loss={loss.item():.4f}")
                num_batches += 1
            avg_loss = epoch_loss / max(num_batches, 1)
            print(f"[lora_finetune] epoch {epoch+1}/{epochs} avg loss={avg_loss:.4f}")
            if checkpoint_dir is not None:
                epoch_ckpt = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}"
                epoch_ckpt.mkdir(parents=True, exist_ok=True)
                self.model.save_pretrained(str(epoch_ckpt))
                print(f"[lora_finetune] saved checkpoint {epoch_ckpt}")
        self.model.eval()

    def save_lora_checkpoint(self, step: int) -> Path:
        path = Path("checkpoints/phase1") / f"adapter_step_{step}"
        path.mkdir(parents=True, exist_ok=True)
        if self.mode == "llm" and self.model is not None:
            self.model.save_pretrained(path)
        else:
            # Stub marker so the checkpoint directory exists and is discoverable.
            (path / "stub_checkpoint.json").write_text(json.dumps({"step": step, "mode": "stub"}))
        return path

    def predict_with_checkpoint(
        self, state: GridState, action: Optional[Action], ckpt_path: Path
    ) -> PredictedState:
        """Temporarily load a checkpoint adapter, predict, and restore the active adapter."""
        if self.mode == "stub" or self.model is None:
            return self.predict(state, action)
        temp_name = "temp_ckpt"
        self.model.load_adapter(str(ckpt_path), adapter_name=temp_name)
        self.model.set_adapter(temp_name)
        try:
            pred = self.predict(state, action)
        finally:
            self.model.set_adapter(self.adapter_name)
            try:
                self.model.delete_adapter(temp_name)
            except Exception:
                pass
        return pred


class EnsembleErrorComputer:
    """Epistemic/aleatoric decomposition via an ensemble of LoRA checkpoints."""

    def __init__(self, world_model: WorldModel, num_checkpoints: int = 5):
        self.world_model = world_model
        self.num_checkpoints = num_checkpoints
        self.checkpoints: List[Path] = []

    def save_checkpoint(self, step: int) -> Path:
        path = self.world_model.save_lora_checkpoint(step)
        self.checkpoints.append(path)
        if len(self.checkpoints) > self.num_checkpoints:
            self.checkpoints.pop(0)
        return path

    @staticmethod
    def _actual_exit_code(state, actual) -> int:
        if hasattr(actual, "container_id"):
            return actual.last_exit_code if actual.last_exit_code else 0
        if hasattr(actual, "room"):
            return 2 if actual.victory else 0
        if actual.agent == actual.goal:
            return 2
        if actual.agent == state.agent:
            return 1
        return 0
    def _predictions_for(self, state, action: Optional[Action]) -> List[PredictedState]:
        if not self.checkpoints:
            return [self.world_model.predict(state, action)]
        return [
            self.world_model.predict_with_checkpoint(state, action, ckpt)
            for ckpt in self.checkpoints
        ]

    def decompose_error(self, state, action: Optional[Action], actual) -> ErrorVector:
        predictions = self._predictions_for(state, action)
        actual_exit = self._actual_exit_code(state, actual)
        # TextState path
        # SandboxState path
        if hasattr(actual, "container_id"):
            n = len(predictions)
            pred_exits = [p.level1_exit_code for p in predictions]
            pred_files = []
            for p in predictions:
                try:
                    j = json.loads(p.level2_text) if p.level2_text else {}
                    pred_files.append(j.get("files", []))
                except (json.JSONDecodeError, TypeError):
                    pred_files.append([])
            actual_files = actual.files or []
            level1_error = sum(1 for e in pred_exits if e != actual_exit) / n if n else 0
            level2_errors = []
            for pf in pred_files:
                overlap = len(set(pf) & set(actual_files))
                total = max(len(set(pf) | set(actual_files)), 1)
                level2_errors.append(1.0 - overlap / total)
            mean_deviation = sum(level2_errors) / n if n else 0
            if n > 1:
                pw = 0.0
                c = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        d = (1.0 if pred_exits[i] != pred_exits[j] else 0.0)
                        d += (1.0 if set(pred_files[i]) != set(pred_files[j]) else 0.0)
                        pw += d
                        c += 1
                ev = (pw / c) / 2.0 if c else 0.0
            else:
                ev = 0.0
            return ErrorVector(
                total_error=mean_deviation + ev,
                level1_error=level1_error,
                level2_error=mean_deviation,
                level3_error=0.0,
                epistemic_error=ev,
                aleatoric_error=max(0.0, mean_deviation - ev),
                ensemble_variance=ev,
            )
        if hasattr(actual, "room"):
            n = len(predictions)
            pred_exits = [p.level1_exit_code for p in predictions]
            pred_rooms = []
            pred_has_keys = []
            for p in predictions:
                r = "unknown"
                has_key = False
                if p.level2_text:
                    if "Location: " in p.level2_text:
                        r = p.level2_text.split("\n")[0].replace("Location: ", "").strip().rstrip(".")
                    for line in p.level2_text.split("\n"):
                        if line.startswith("Inventory:"):
                            inv_part = line.replace("Inventory:", "").strip().rstrip(".")
                            has_key = "key" in inv_part and inv_part != "nothing"
                pred_rooms.append(r)
                pred_has_keys.append(has_key)
            actual_has_key = "key" in (actual.inventory or [])
            level1_error = sum(1 for e in pred_exits if e != actual_exit) / n if n else 0
            level2_errors = [
                (0.0 if r == actual.room else 1.0) + (0.0 if hk == actual_has_key else 1.0)
                for r, hk in zip(pred_rooms, pred_has_keys)
            ]
            mean_deviation = sum(level2_errors) / n if n else 0
            if n > 1:
                pw = 0.0
                c = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        d = (1.0 if pred_exits[i] != pred_exits[j] else 0.0)
                        d += (1.0 if pred_rooms[i] != pred_rooms[j] else 0.0)
                        d += (1.0 if pred_has_keys[i] != pred_has_keys[j] else 0.0)
                        pw += d
                        c += 1
                ev = (pw / c) / 3.0 if c else 0.0
            else:
                ev = 0.0
            return ErrorVector(
                total_error=mean_deviation + ev,
                level1_error=level1_error,
                level2_error=mean_deviation,
                level3_error=0.0,
                epistemic_error=ev,
                aleatoric_error=max(0.0, mean_deviation - ev),
                ensemble_variance=ev,
            )

        # GridState path (original logic)
        pred_positions = [p.level2_next_agent for p in predictions]
        pred_exits = [p.level1_exit_code for p in predictions]
        n = len(pred_positions)
        level2_errors = [
            math.sqrt((x - actual.agent[0]) ** 2 + (y - actual.agent[1]) ** 2)
            for x, y in pred_positions
        ]
        mean_deviation = sum(level2_errors) / n if n else 0
        if n > 1:
            pairwise = 0.0
            count = 0
            for i in range(n):
                for j in range(i + 1, n):
                    dx = pred_positions[i][0] - pred_positions[j][0]
                    dy = pred_positions[i][1] - pred_positions[j][1]
                    pairwise += dx * dx + dy * dy
                    count += 1
            ensemble_variance = pairwise / count if count else 0.0
        else:
            ensemble_variance = 0.0
        level1_error = sum(1 for e in pred_exits if e != actual_exit) / n if n else 0
        level3_error = 0.0
        epistemic_error = ensemble_variance
        aleatoric_error = max(0.0, mean_deviation - ensemble_variance)
        total_error = mean_deviation + ensemble_variance
        return ErrorVector(
            total_error=total_error,
            level1_error=level1_error,
            level2_error=mean_deviation,
            level3_error=level3_error,
            epistemic_error=epistemic_error,
            aleatoric_error=aleatoric_error,
            ensemble_variance=ensemble_variance,
        )


class SaturationDetector:
    """Detect learning saturation from a recent window of total error."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.errors: deque[float] = deque(maxlen=window_size)

    def add(self, total_error: float) -> None:
        self.errors.append(total_error)

    def is_saturated(self) -> bool:
        if len(self.errors) < self.window_size:
            return False
        half = self.window_size // 2
        old_mean = sum(list(self.errors)[:half]) / half
        new_mean = sum(list(self.errors)[half:]) / half
        if old_mean <= 0:
            return False
        decline = (old_mean - new_mean) / old_mean
        return decline < 0.15

    @property
    def novelty_boost(self) -> float:
        return 0.5 if self.is_saturated() else 0.0


class ExperienceBuffer:
    """Replay buffer with simple prioritized sampling."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer: deque[Experience] = deque(maxlen=max_size)

    def add(self, experience: Experience) -> None:
        self.buffer.append(experience)

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self) -> None:
        self.buffer.clear()

    def sample_prioritized(
        self,
        batch_size: int,
        priority_fn=None,
    ) -> List[Experience]:
        if len(self.buffer) == 0:
            return []
        if len(self.buffer) <= batch_size:
            return list(self.buffer)
        if priority_fn is None:
            def priority_fn(exp: Experience) -> float:
                return exp.error.epistemic_error
        priorities = [max(1e-6, float(priority_fn(exp))) for exp in self.buffer]
        total = sum(priorities)
        probs = [p / total for p in priorities]
        sampled = random.choices(list(self.buffer), weights=probs, k=batch_size)
        return sampled


class LearningModule:
    """Intermittent batch learning from experience replay."""

    def __init__(
        self,
        world_model: WorldModel,
        error_computer: EnsembleErrorComputer,
        buffer_size: int = 1000,
        update_interval: int = 500,
    ):
        self.world_model = world_model
        self.error_computer = error_computer
        self.buffer = ExperienceBuffer(max_size=buffer_size)
        self.update_interval = update_interval
        self.step_count = 0
        self.saturation_detector = SaturationDetector(window_size=100)

    def store_experience(self, experience: Experience) -> None:
        self.buffer.add(experience)
        self.saturation_detector.add(experience.error.total_error)

    def should_update(self) -> bool:
        return len(self.buffer) >= self.update_interval

    def update(self) -> None:
        if not self.should_update():
            return
        samples = self.buffer.sample_prioritized(batch_size=64)
        data = []
        for exp in samples:
            data.append({
                "state_text": Perception.render(exp.state),
                "action_name": exp.action.name,
                "next_state_text": str(exp.next_state.agent),
                "exit_code": exp.exit_code,
                "summary": exp.summary,
            })
        self.world_model.lora_finetune(data, epochs=1, learning_rate=2e-4, batch_size=4)
        self.step_count += 1
        self.error_computer.save_checkpoint(self.step_count)
        self.buffer.clear()
        if self.saturation_detector.is_saturated():
            print("[LearningModule] Saturation detected; novelty boost applied next step.")

    @property
    def saturation_novelty_boost(self) -> float:
        return self.saturation_detector.novelty_boost
