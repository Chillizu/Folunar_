#!/usr/bin/env python3
"""Generate TextWorld training data for PEDA Phase 1.5.

Generates 3 tiers of TextWorld games, runs random walks to collect
(state, action, next_state, reward, done) transitions, and saves
them as JSONL.

Usage:
    /path/to/.venv_textworld/bin/python scripts/phase1_5_textworld_generate.py

Or from within the venv:
    source .venv_textworld/bin/activate
    python scripts/phase1_5_textworld_generate.py
"""

import json
import random
import sys
import time
from pathlib import Path

# --- project path setup ---
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

# --- imports (TextWorld must be available) ---
from phase1_5.textworld_env import TextWorldEnv, TextWorldState, render_state_text

# Target number of unique transitions to collect across all tiers.
_TARGET_UNIQUE = 1500  # aim high so all 3 tiers contribute

_OUTPUT_PATH = _PROJECT_ROOT / "results" / "phase1_5_textworld_data.jsonl"
_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Number of unique game seeds to generate per tier.
# More seeds = more game variety. Adjust to reach the 500+ transition target.
# Number of unique game seeds to generate per tier.
# More seeds = more game variety.
_TIER_SEEDS = {
    1: list(range(100, 120)),  # 20 simple games
    2: list(range(200, 220)),  # 20 medium games
    3: list(range(300, 315)),  # 15 constrained games
}

_WALK_LENGTH = 50  # max steps per random walk
_MAX_WALKS_PER_GAME = 5  # walks per game (to avoid infinite loops on small games)
_TARGET_UNIQUE = 1500  # aim high so all 3 tiers contribute


def compute_exit_code(
    prev_state: TextWorldState,
    next_state: TextWorldState,
    action: str,
) -> tuple:
    """Compute exit_code and summary for a transition.

    Mirrors the logic in phase1_5_synthetic_train.py.
    """
    if next_state.victory:
        return 2, f"you completed the goal by {action}"
    if next_state.room != prev_state.room:
        return 0, f"you moved {action}"
    if action in ("look", "l", "inventory", "i"):
        return 0, "you check your surroundings"
    return 1, f"you try to {action} but nothing happens"


def run_random_walks(
    env: TextWorldEnv,
    tier: int,
    walk_length: int = _WALK_LENGTH,
    max_walks: int = _MAX_WALKS_PER_GAME,
    seed: int = 0,
) -> list[dict]:
    """Run random walks in a TextWorld game and collect unique transitions.

    Each walk starts from env.reset(). Steps choose randomly from admissible
    commands. Duplicate (state_text, action) pairs are skipped.

    Returns a list of transition dicts.
    """
    transitions: list[dict] = []
    seen: set[str] = set()
    rng = random.Random(seed + 9999)  # separate RNG for action selection

    for walk_idx in range(max_walks):
        state = env.reset(seed=seed)
        steps_taken = 0
        walk_done = False

        for step_idx in range(walk_length):
            if not state.admissible_commands:
                # No valid actions — restart walk
                walk_done = True
                break

            # Pick a random admissible command
            action = rng.choice(state.admissible_commands)

            # Take the action
            next_state, reward, done = env.step(state, action)

            # Compute exit code and summary
            exit_code, summary = compute_exit_code(state, next_state, action)

            # Build transition record
            state_text = render_state_text(state)
            dedup_key = f"{state_text}||{action}"
            if dedup_key not in seen:
                seen.add(dedup_key)
                transitions.append({
                    "state_text": state_text,
                    "action_name": action,
                    "exit_code": exit_code,
                    "summary": summary,
                    "next_room": next_state.room,
                    "next_description": next_state.description,
                    "next_inventory": ", ".join(next_state.inventory)
                    if next_state.inventory else "nothing",
                    "reward": reward,
                    "victory": next_state.victory,
                    "tier": tier,
                    "seed": seed,
                    "step": state.step,
                })

            steps_taken += 1

            if done or next_state.game_over:
                walk_done = True
                break

            state = next_state

        # Small game may have exhausted all transitions — move on
        if steps_taken == 0:
            break

    return transitions


def collect_all_data() -> list[dict]:
    """Collect transitions across all tiers and games."""
    all_data: list[dict] = []
    total_games = 0

    for tier in (1, 2, 3):
        tier_name = TextWorldEnv.tier_name(tier)
        seeds = _TIER_SEEDS[tier]

        print(f"\n{'='*60}")
        print(f"TIER {tier} ({tier_name}): {len(seeds)} games")
        print(f"{'='*60}")

        for seed in seeds:
            env = TextWorldEnv(tier=tier)
            try:
                transitions = run_random_walks(
                    env=env, tier=tier, seed=seed,
                )
                all_data.extend(transitions)
                total_games += 1
                print(
                    f"  seed {seed}: {len(transitions)} transitions "
                    f"(total: {len(all_data)} unique)"
                )
            finally:
                env.close()


    print(f"\nTotal games played: {total_games}")
    return all_data


def compute_stats(data: list[dict]) -> dict:
    """Compute statistics over the collected transitions."""
    if not data:
        return {"total": 0, "unique": 0}

    # Deduplicate by (state_text, action_name)
    seen = set()
    unique = []
    for t in data:
        key = t["state_text"] + t["action_name"]
        if key not in seen:
            seen.add(key)
            unique.append(t)

    # Per-tier stats
    tier_counts: dict = {}
    tier_victories: dict = {}
    for t in unique:
        tier = t["tier"]
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        if t["victory"]:
            tier_victories[tier] = tier_victories.get(tier, 0) + 1

    # Exit code distribution
    exit_codes = {}
    for t in unique:
        ec = t["exit_code"]
        exit_codes[ec] = exit_codes.get(ec, 0) + 1

    return {
        "total_transitions": len(data),
        "unique_transitions": len(unique),
        "duplicate_ratio": round(1 - len(unique) / max(len(data), 1), 4),
        "by_tier": {
            TextWorldEnv.tier_name(t): {
                "count": tier_counts.get(t, 0),
                "victories": tier_victories.get(t, 0),
            }
            for t in (1, 2, 3)
        },
        "by_exit_code": {
            str(ec): exit_codes.get(ec, 0)
            for ec in sorted(exit_codes.keys())
        },
    }


def save_jsonl(data: list[dict], path: Path):
    """Save transitions as JSONL."""
    # Deduplicate before saving
    seen = set()
    unique = []
    for t in data:
        key = t["state_text"] + t["action_name"]
        if key not in seen:
            seen.add(key)
            unique.append(t)

    with open(path, "w") as f:
        for item in unique:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(unique)} unique transitions to {path}")


def main():
    start = time.time()
    print(f"PEDA Phase 1.5 TextWorld Data Generator")
    print(f"Target: {_TARGET_UNIQUE}+ unique transitions")
    print(f"Output: {_OUTPUT_PATH}")
    print(f"Time:   {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Phase 1: Verify TextWorld availability
    print("\n--- Phase 1: Verifying TextWorld ---")
    try:
        import textworld as tw
        print(f"  TextWorld version: {tw.__version__}")
    except ImportError as e:
        print(f"  ERROR: TextWorld not available: {e}")
        print(f"  Activate .venv_textworld and re-run.")
        sys.exit(1)

    # Phase 2: Collect data
    print("\n--- Phase 2: Collecting transitions ---")
    data = collect_all_data()

    if not data:
        print("ERROR: No data collected!")
        sys.exit(1)

    # Phase 3: Statistics
    print("\n--- Phase 3: Statistics ---")
    stats = compute_stats(data)
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    # Phase 4: Save
    save_jsonl(data, _OUTPUT_PATH)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s")

    # Verify
    final_count = sum(1 for _ in open(_OUTPUT_PATH))
    print(f"Final JSONL line count: {final_count}")

    if final_count >= _TARGET_UNIQUE:
        print(f"PASS: {final_count} >= {_TARGET_UNIQUE} unique transitions")
    else:
        print(f"WARNING: Only {final_count} unique transitions (< {_TARGET_UNIQUE})")


if __name__ == "__main__":
    main()
