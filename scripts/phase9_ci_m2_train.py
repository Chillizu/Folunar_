#!/usr/bin/env python3
# ruff: noqa: E402
"""M2 learnability — LoRA fine-tune Qwen2.5-0.5B-Instruct on CI v2 transitions.

Reads results/phase9_ci_m2_train.jsonl (200 unique (state, action, next_state)
transitions collected on peda-sandbox:counterintuitive-v2), trains a rank-16
LoRA adapter (3 epochs, lr 2e-4, batch_size 2, CPU) in the world-model
sandbox prompt format (src/phase1/world_model.py), and extracts a STRIPS-style
per-(verb, target-type) rule table as an independent learnability check.

The training loop mirrors WorldModel.lora_finetune(sandbox_mode=True) but with
max_length=640 (384 would truncate twin-rich file lists and the exit_code tail
of the target).

Outputs:
  checkpoints/phase9/ci_m2_lora/            LoRA adapter (deliverable)
  checkpoints/phase9/ci_m2_strips_rules.json  STRIPS rule table (deliverable)
  checkpoints/phase9/ci_m2_train_manifest.json

Usage:
    python scripts/phase9_ci_m2_train.py [--epochs 3] [--batch-size 2] [--lr 2e-4]
"""

import argparse
import collections
import datetime
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402
from phase1.world_model import WorldModel  # noqa: E402

MODEL_PATH = "/home/data/models/Qwen2.5-0.5B-Instruct"
DATA_PATH = REPO_ROOT / "results" / "phase9_ci_m2_train.jsonl"
OUT_DIR = REPO_ROOT / "checkpoints" / "phase9" / "ci_m2_lora"
STRIPS_PATH = REPO_ROOT / "checkpoints" / "phase9" / "ci_m2_strips_rules.json"
MAX_LEN = 640


def load_transitions(subset: int | None = None) -> list[dict]:
    rows = []
    with open(DATA_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "meta" in rec:
                continue
            rows.extend(rec["records"])
    if subset is not None:
        rows = rows[:subset]
    return rows


def build_examples(transitions: list[dict]) -> list[dict]:
    """(state, action) -> world-model JSON target, eval-consistent state_text.

    state_text uses the 5-key reduced format (cwd/files/last_command/
    last_exit_code/last_output) — the same prompt shape the eval harness builds
    — which also keeps sequences short (~250 tok p90) for CPU training.
    summary mirrors output non-emptiness ("" when no output) so the M1 L3 rule
    (output expected = last_output OR summary non-empty) matches training.
    """
    examples = []
    for t in transitions:
        action = t["action"]
        out = (t.get("output") or "")[:200]
        state_text = json.dumps({
            "cwd": t["cwd"],
            "files": t["files"],
            "last_command": "",
            "last_exit_code": 0,
            "last_output": "",
        }, ensure_ascii=False)
        examples.append({
            "state_text": state_text,
            "cwd": t["cwd"],
            "files": t["files"],
            "action_name": action,
            "exit_code": int(t["exit_code"]),
            "summary": out[:60] if out.strip() else "",
            "next_cwd": t["next_cwd"],
            "next_files": t["next_files"],
            "next_last_exit_code": int(t["exit_code"]),
            "next_last_output": out,
        })
    return examples


def make_pairs(wm: WorldModel, examples: list[dict]) -> list[tuple]:
    """Tokenize (prompt, prompt+target) pairs; returns (prompt_ids, full_ids)."""
    system_msg = wm._sandbox_system_message()
    pairs = []
    for ex in examples:
        action_text = ex["action_name"]
        exit_code = int(ex["exit_code"])
        summary = str(ex.get("summary", ""))
        target = json.dumps({
            "cwd": ex.get("next_cwd", ex.get("cwd", "")),
            "files": ex.get("next_files", ex.get("files", [])),
            "last_command": action_text,
            "last_exit_code": int(ex.get("next_last_exit_code", exit_code)),
            "last_output": ex.get("next_last_output", ""),
            "exit_code": exit_code,
            "summary": summary,
        }, ensure_ascii=False)
        user_content = (
            f"State: {ex['state_text']}\n"
            f"Action: {action_text}\n"
            "Predict cwd, files, last_command, last_exit_code, last_output, exit_code, and summary as JSON:"
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ]
        tok = wm.tokenizer
        prompt_input_ids = WorldModel._extract_input_ids(
            tok.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt",
                max_length=MAX_LEN, truncation=True,
            )
        ).squeeze(0)
        full_messages = messages + [{"role": "assistant", "content": target}]
        full_input_ids = WorldModel._extract_input_ids(
            tok.apply_chat_template(
                full_messages, add_generation_prompt=False, return_tensors="pt",
                max_length=MAX_LEN, truncation=True,
            )
        ).squeeze(0)
        pairs.append((prompt_input_ids, full_input_ids))
    return pairs


def train_loop(wm: WorldModel, pairs: list[tuple], epochs: int, batch_size: int,
               lr: float, out_dir: Path, early_stop_loss: float | None = None) -> int:
    """LoRA training; returns the number of epochs actually run."""
    class _Dataset(torch.utils.data.Dataset):
        def __init__(self, ps):
            self.ps = ps

        def __len__(self):
            return len(self.ps)

        def __getitem__(self, idx):
            prompt_ids, full_ids = self.ps[idx]
            labels = full_ids.clone()
            labels[: prompt_ids.shape[0]] = -100
            return {
                "input_ids": full_ids,
                "attention_mask": torch.ones_like(full_ids),
                "labels": labels,
            }

    dataset = _Dataset(pairs)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=lambda x: x, num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        [p for p in wm.model.parameters() if p.requires_grad], lr=lr
    )
    wm.model.train()
    pad_id = wm.tokenizer.pad_token_id
    device = wm.device
    for epoch in range(epochs):
        epoch_loss, n_batches = 0.0, 0
        for batch in loader:
            optimizer.zero_grad()
            input_ids = torch.nn.utils.rnn.pad_sequence(
                [b["input_ids"] for b in batch], batch_first=True, padding_value=pad_id
            ).to(device)
            attention_mask = torch.nn.utils.rnn.pad_sequence(
                [b["attention_mask"] for b in batch], batch_first=True, padding_value=0
            ).to(device)
            labels = torch.nn.utils.rnn.pad_sequence(
                [b["labels"] for b in batch], batch_first=True, padding_value=-100
            ).to(device)
            outputs = wm.model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss
            if loss is not None:
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
                if n_batches % 25 == 0:
                    print(f"  [train] epoch {epoch + 1}/{epochs} batch {n_batches} "
                          f"loss={loss.item():.4f}", flush=True)
        print(f"  [train] epoch {epoch + 1}/{epochs} avg loss="
              f"{epoch_loss / max(n_batches, 1):.4f}", flush=True)
        ckpt = out_dir / f"checkpoint_epoch_{epoch + 1}"
        ckpt.mkdir(parents=True, exist_ok=True)
        wm.model.save_pretrained(str(ckpt))
        if early_stop_loss is not None and epoch_loss / max(n_batches, 1) < early_stop_loss:
            print(f"  [train] early stop: avg loss below {early_stop_loss}", flush=True)
            wm.model.eval()
            return epoch + 1
    wm.model.eval()
    return epochs


# ── STRIPS-style rule table ────────────────────────────────────────────

# classify_target lives in phase9_ci_m2_collect (shared with the eval script).
from phase9_ci_m2_collect import classify_target  # noqa: E402


def stratified_subset(transitions: list[dict], n: int) -> list[dict]:
    """Balanced (verb, target_class) subset: every rule type appears.

    Round-robin over buckets ordered by size (largest first) so rare classes
    (e.g. echo|file reads) are represented even in a small budget.
    """
    buckets: dict = {}
    for t in transitions:
        verb = t["action"].split()[0]
        cls = classify_target(verb, t["action"], t.get("entry_types") or {},
                              list(t.get("files") or []))
        buckets.setdefault(f"{verb}|{cls}", []).append(t)
    keys = sorted(buckets, key=lambda k: -len(buckets[k]))
    out = []
    i = 0
    while len(out) < n:
        for k in keys:
            if len(out) >= n:
                break
            if i < len(buckets[k]):
                out.append(buckets[k][i])
        i += 1
    return out[:n]


def build_strips_table(transitions: list[dict]) -> dict:
    """Per (verb, target_class) majority (exit_code, delta, output) rule."""
    buckets = collections.defaultdict(list)
    for t in transitions:
        verb = t["action"].split()[0]
        cls = classify_target(verb, t["action"], t.get("entry_types") or {},
                              list(t.get("files") or []))
        buckets[(verb, cls)].append({
            "exit_code": int(t["exit_code"]),
            "delta": list(t["next_files"]) != list(t["files"]),
            "output": bool((t.get("output") or "").strip()),
        })
    rules = {}
    for (verb, cls), outs in sorted(buckets.items()):
        counts = collections.Counter((o["exit_code"], o["delta"], o["output"]) for o in outs)
        best, n_best = counts.most_common(1)[0]
        rules[f"{verb}|{cls}"] = {
            "exit_code": best[0],
            "delta": best[1],
            "output": best[2],
            "n": len(outs),
            "n_best": n_best,
            "conf": round(n_best / len(outs), 3),
        }
    # Verb-level fallback rules (used when a (verb, class) is unseen in
    # training, e.g. `ls <file>`: ls has no file-target training examples).
    verb_outs = collections.defaultdict(list)
    for (verb, _cls), outs in buckets.items():
        verb_outs[verb].extend(outs)
    for verb, outs in sorted(verb_outs.items()):
        counts = collections.Counter((o["exit_code"], o["delta"], o["output"]) for o in outs)
        best, n_best = counts.most_common(1)[0]
        rules[f"{verb}|*"] = {
            "exit_code": best[0],
            "delta": best[1],
            "output": best[2],
            "n": len(outs),
            "n_best": n_best,
            "conf": round(n_best / len(outs), 3),
        }
    return rules


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=0,
                        help="0 = auto (8 on CUDA, 2 on CPU)")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--dtype", choices=["auto", "float32", "bfloat16", "float16"],
                        default="auto",
                        help="auto = float16 on CUDA, float32 on CPU (bf16 CPU is emulated-slow)")
    parser.add_argument("--threads", type=int, default=8,
                        help="torch intra-op threads on CPU (caps per-thread arena memory)")
    parser.add_argument("--subset", type=int, default=None,
                        help="train on the first N transitions (slow shared box)")
    parser.add_argument("--stratified", action="store_true",
                        help="balanced (verb, target_class) subset instead of first-N")
    parser.add_argument("--early-stop-loss", type=float, default=0.03,
                        help="stop after an epoch with avg loss below this (deterministic rules)")
    parser.add_argument("--output-dir", default=None,
                        help="override checkpoint dir (default checkpoints/phase9/ci_m2_lora)")
    args = parser.parse_args()

    has_cuda = torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"
    if args.batch_size == 0:
        args.batch_size = 8 if has_cuda else 2
    if args.dtype == "auto":
        args.dtype = "float16" if has_cuda else "float32"
    torch.set_num_threads(args.threads)

    out_dir = Path(args.output_dir) if args.output_dir else Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    transitions = load_transitions(None)
    if args.subset is not None:
        if args.stratified:
            transitions = stratified_subset(transitions, args.subset)
        else:
            transitions = transitions[:args.subset]
    examples = build_examples(transitions)
    print(f"[m2_train] {len(transitions)} transitions, {len(examples)} examples"
          f" (subset={args.subset} stratified={args.stratified})", flush=True)

    # STRIPS rule table (independent learnability check; cheap, no LLM).
    rules = build_strips_table(transitions)
    (Path(STRIPS_PATH)).parent.mkdir(parents=True, exist_ok=True)
    (Path(STRIPS_PATH)).write_text(json.dumps(rules, indent=2))
    print(f"[m2_train] STRIPS rules ({len(rules)}) -> {STRIPS_PATH}", flush=True)

    # LoRA fine-tune.
    t0 = time.time()
    print(f"[m2_train] Loading {args.model} (device={device}, LoRA r=16) ...", flush=True)
    wm = WorldModel(args.model, device=device)
    if wm.mode != "llm" or wm.model is None:
        print("[m2_train] FATAL: model fell back to stub", file=sys.stderr)
        return 1
    if args.dtype == "float16":
        wm.model.half()
        print("[m2_train] cast model to float16", flush=True)
    elif args.dtype == "bfloat16":
        wm.model.bfloat16()
        print("[m2_train] cast model to bfloat16", flush=True)
    else:
        print("[m2_train] model stays float32", flush=True)
    print("[m2_train] Building tokenized pairs ...", flush=True)
    pairs = make_pairs(wm, examples)
    seq_lens = [p[1].shape[0] for p in pairs]
    print(f"[m2_train] pairs={len(pairs)} max_seq_len={max(seq_lens)} "
          f"p90={sorted(seq_lens)[int(0.9 * len(seq_lens))]}", flush=True)
    epochs_run = train_loop(wm, pairs, args.epochs, args.batch_size, args.lr, out_dir,
                            early_stop_loss=args.early_stop_loss)

    wm.model.save_pretrained(str(out_dir))
    manifest = {
        "num_transitions": len(transitions),
        "num_examples": len(examples),
        "epochs": args.epochs,
        "epochs_run": epochs_run,
        "early_stop_loss": args.early_stop_loss,
        "subset": args.subset,
        "stratified": args.stratified,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "lora_r": 16,
        "lora_alpha": 32,
        "max_seq_len": max(seq_lens),
        "model": args.model,
        "dtype": args.dtype,
        "threads": args.threads,
        "data_source": str(DATA_PATH),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "training_seconds": round(time.time() - t0, 1),
    }
    (out_dir / "trained_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[m2_train] adapter -> {out_dir} ({manifest['training_seconds']}s)", flush=True)
    print(f"[m2_train] TRAINING_FINISHED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
