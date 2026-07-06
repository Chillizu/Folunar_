#!/usr/bin/env python3
# ruff: noqa: E402
"""Generate synthetic text-adventure data and fine-tune the World Model LoRA adapter.

Usage:
    python scripts/phase1_5_synthetic_train.py --model Qwen/Qwen2.5-0.5B-Instruct
    python scripts/phase1_5_synthetic_train.py --stub  # smoke test
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
from phase1.grid_env import Perception
from phase1_5.text_env import TextRoomEnv


def generate_text_transitions(
    num_walks: int = 50,
    walk_length: int = 20,
    seed: int = 42,
) -> list[dict]:
    """Generate (state, action, next_state) triples from TextRoomEnv random walks."""
    env = TextRoomEnv()
    rng = random.Random(seed)
    all_actions = TextRoomEnv.all_actions()
    seen: set[str] = set()
    data: list[dict] = []

    def _append(state, action_name, next_s, reward, done):
        state_text = Perception.render_text(state)
        key = state_text + action_name
        if key in seen:
            return
        seen.add(key)
        if next_s.victory:
            exit_code, summary = 2, f"you completed the goal by {action_name}"
        elif next_s.room != state.room:
            exit_code, summary = 0, f"you moved {action_name}"
        elif action_name in ("look", "inventory"):
            exit_code, summary = 0, "you check your surroundings"
        else:
            exit_code, summary = 1, f"you try to {action_name} but nothing happens"
        data.append({
            "state_text": state_text,
            "action_name": action_name,
            "exit_code": exit_code,
            "summary": summary,
            "next_room": next_s.room,
            "next_description": next_s.description,
            "reward": reward,
            "victory": next_s.victory,
        })

    # Exhaustive: from each room, try every action once
    for room in ("study", "hallway"):
        s = env.reset(seed=0)
        s.room = room
        s.description = env._get_description(room)
        s.inventory = []
        for action_name in all_actions:
            next_s, reward, done = env.step(s, action_name)
            _append(s, action_name, next_s, reward, done)

    # Random walks for coverage (diverse inventory/description states)
    for walk in range(num_walks):
        state = env.reset(seed=seed + walk)
        for step in range(walk_length):
            action_name = rng.choice(all_actions)
            next_s, reward, done = env.step(state, action_name)
            _append(state, action_name, next_s, reward, done)
            state = next_s
            if done:
                state = env.reset(seed=seed + walk + (step + 1) * 1000)

    return data


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Phase 1.5 World Model on text-adventure data")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct", help="Base model")
    parser.add_argument("--epochs", type=int, default=3, help="LoRA fine-tuning epochs")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--train-seed", type=int, default=42, help="Seed for data generation")
    parser.add_argument("--output-dir", default="checkpoints/phase1_5/text_adapter_e3", help="Output path")
    parser.add_argument("--num-walks", type=int, default=50, help="Number of random walks")
    parser.add_argument("--walk-length", type=int, default=20, help="Steps per random walk")
    parser.add_argument("--stub", action="store_true", help="Smoke-test mode (no LLM)")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"[phase1_5_train] Generating data from TextRoomEnv ...", flush=True)
    data = generate_text_transitions(
        num_walks=args.num_walks,
        walk_length=args.walk_length,
        seed=args.train_seed,
    )
    print(f"[phase1_5_train] Generated {len(data)} transitions", flush=True)
    if not data:
        print("[phase1_5_train] ERROR: no training data generated!", flush=True)
        sys.exit(1)

    # Save data for reference
    data_path = output_path / "text_training_data.json"
    data_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"[phase1_5_train] Training data saved to {data_path}", flush=True)

    manifest = {
        "num_rooms": 2,
        "num_transitions": len(data),
        "train_seed": args.train_seed,
        "num_walks": args.num_walks,
        "walk_length": args.walk_length,
    }
    manifest_path = output_path / "trained_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[phase1_5_train] Manifest saved to {manifest_path}", flush=True)

    from phase1.world_model import WorldModel

    use_stub = args.stub or os.environ.get("FOLUNAR_STUB_MODEL", "0") == "1"
    if use_stub:
        print("[phase1_5_train] STUB mode: placeholder checkpoints.", flush=True)
        for epoch in range(1, args.epochs + 1):
            p = output_path / f"checkpoint_epoch_{epoch}"
            p.mkdir(parents=True, exist_ok=True)
            (p / "stub_checkpoint.json").write_text(json.dumps({"epoch": epoch, "mode": "stub"}))
        print(f"[phase1_5_train] STUB_DONE at {output_path}", flush=True)
        return

    print(f"[phase1_5_train] Loading model {args.model} ...", flush=True)
    wm = WorldModel(args.model)

    if wm.mode == "stub":
        print("[phase1_5_train] Model fell back to stub.", flush=True)
        return

    print(f"[phase1_5_train] Training LoRA for {args.epochs} epoch(s) (text mode)...", flush=True)
    wm.lora_finetune(
        data,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        checkpoint_dir=output_path,
        text_mode=True,
    )

    print(f"[phase1_5_train] Saving adapter to {output_path} ...", flush=True)
    wm.model.save_pretrained(str(output_path))
    print(f"[phase1_5_train] TRAINING_FINISHED", flush=True)


if __name__ == "__main__":
    main()
