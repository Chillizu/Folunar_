#!/usr/bin/env python3
"""Semantic ensemble probe for Phase 1.5 text adapter.

Parses each checkpoint's prediction into structured fields:
  - room: predicted next room name (str)
  - exit_code: 0=ok, 1=fail, 2=victory
  - has_key: whether agent carries the key after the action (bool or None)

Reports disagreement per field and combined, then compares to threshold.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from phase1.types import Action
from phase1_5.text_env import TextRoomEnv
from phase1.world_model import WorldModel, EnsembleErrorComputer


def _parse_fields(exit_code: int, text: str) -> tuple:
    """Extract (room, exit_code, has_key) from a text prediction."""
    room = "unknown"
    if "Location: " in text:
        room = text.split("\n")[0].replace("Location: ", "").strip().rstrip(".")
    # Infer key ownership from description text
    tl = text.lower()
    if "not carrying anything" in tl or "carrying: nothing" in tl:
        has_key = False
    elif "carrying:" in tl and "key" in tl.split("carrying:")[1].split("\n")[0]:
        has_key = True
    elif "take the key" in tl or "have the key" in tl or "key is already" in tl:
        has_key = True
    else:
        has_key = None  # unknown / not applicable
    return (room, exit_code, has_key)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="checkpoints/phase1_5/text_adapter_e3")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--threshold", type=float, default=0.33)
    args = parser.parse_args()

    wm = WorldModel(args.model, adapter_path=args.adapter)
    ec = EnsembleErrorComputer(wm)
    ec.checkpoints = sorted(Path(args.adapter).glob("checkpoint_epoch_*"))

    env = TextRoomEnv()
    all_actions = TextRoomEnv.all_actions()
    states = []

    s0 = env.reset(seed=42)
    states.append(s0)  # study, no key

    s1 = env.reset(seed=99)
    s1.room = "hallway"
    s1.description = env._get_description("hallway")
    states.append(s1)  # hallway, no key

    s2 = env.reset(seed=1)
    s2.inventory = ["key"]
    s2.description = "study. you have the key. desk empty."
    states.append(s2)  # study, has key

    s3 = env.reset(seed=2)
    s3.room = "hallway"
    s3.inventory = ["key"]
    s3.description = "hallway. you have the key. chest locked."
    states.append(s3)  # hallway, has key

    s4 = env.reset(seed=3)
    s4.room = "hallway"
    s4.inventory = []
    s4.description = "hallway. chest already open. victory."
    s4.victory = True
    states.append(s4)  # hallway, victory state

    total = d_room = d_exit = d_key = d_sem = 0

    for s in states:
        for action_name in all_actions:
            preds = ec._predictions_for(s, Action(action_name))
            fields = [_parse_fields(p.level1_exit_code, p.level2_text) for p in preds]
            rooms = [f[0] for f in fields]
            exits = [f[1] for f in fields]
            keys = [f[2] for f in fields]
            total += 1
            if len(set(rooms)) > 1:
                d_room += 1
            if len(set(exits)) > 1:
                d_exit += 1
            if len(set(keys)) > 1:
                d_key += 1
            if len(set(fields)) > 1:
                d_sem += 1

    print(f"Total probes: {total}")
    print(f"Room disagreement:         {d_room}/{total} = {d_room/total*100:.0f}%")
    print(f"Exit code disagreement:    {d_exit}/{total} = {d_exit/total*100:.0f}%")
    print(f"Has-key disagreement:      {d_key}/{total} = {d_key/total*100:.0f}%")
    print(f"Full semantic disagreement:{d_sem}/{total} = {d_sem/total*100:.0f}%")
    print(f"Threshold: {args.threshold*100:.0f}%")
    passed = d_sem / total >= args.threshold
    print(f"Verdict: {'PASS - epistemic alive' if passed else 'FAIL - need more data'}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
