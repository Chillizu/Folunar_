#!/usr/bin/env python3
"""Merge partial-training evaluation chunks into one report."""
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


def _exploration_metrics(
    episodes: List[Dict[str, Any]], known_cells: Set[Tuple[int, int]], max_steps: int
) -> Dict[str, float]:
    if not episodes:
        return {
            "mean_unknown_fraction": 0.0,
            "mean_unknown_cells_visited": 0.0,
            "mean_steps_before_known": float(max_steps),
        }
    unknown_fractions = []
    unknown_cells_counts = []
    steps_before_known = []
    for ep in episodes:
        traj = [tuple(p) for p in ep["trajectory"]]
        unknown_positions = [p for p in traj if p not in known_cells]
        unknown_fractions.append(len(unknown_positions) / len(traj) if traj else 0.0)
        unknown_cells_counts.append(len(set(unknown_positions)))
        first_known_idx = next(
            (i for i, p in enumerate(traj) if p in known_cells), max_steps
        )
        steps_before_known.append(first_known_idx)
    return {
        "mean_unknown_fraction": sum(unknown_fractions) / len(unknown_fractions),
        "mean_unknown_cells_visited": sum(unknown_cells_counts) / len(unknown_cells_counts),
        "mean_steps_before_known": sum(steps_before_known) / len(steps_before_known),
    }


def _aggregate(
    episodes: List[Dict[str, Any]], max_steps: int
) -> Dict[str, Any]:
    n = len(episodes)
    if n == 0:
        return {
            "success_rate": 0.0,
            "mean_steps": float(max_steps),
            "revisit_rate": 0.0,
            "g1": 0.0,
        }
    return {
        "success_rate": sum(e["success"] for e in episodes) / n,
        "mean_steps": sum(e["steps"] for e in episodes) / n,
        "revisit_rate": sum(e["revisit_rate"] for e in episodes) / n,
        "g1": sum(e["g1"] for e in episodes) / n,
    }


def merge(chunk_paths: List[str], output_path: str) -> None:
    chunks = [json.loads(Path(p).read_text()) for p in chunk_paths]
    if not chunks:
        raise ValueError("No chunk files provided")
    base = chunks[0]
    known_cells = {tuple(c) for c in base["config"]["known_cells"]}
    max_steps = base["config"].get("max_steps", 50)
    merged: Dict[str, Any] = {
        "timestamp": base["timestamp"],
        "config": base["config"],
        "chunks": [c["config"]["start_episode"] for c in chunks],
        "g1_test_set": base["g1_test_set"],
        "conditions": {},
        "raw_results": {},
    }
    for cond in ["goal_known", "goal_unknown"]:
        merged["conditions"][cond] = {}
        merged["raw_results"][cond] = {}
        for agent in ["peda", "pragmatic_only"]:
            episodes = []
            for c in chunks:
                eps = c.get("raw_results", {}).get(cond, {}).get(agent, [])
                episodes.extend(eps)
            merged["raw_results"][cond][agent] = episodes
            agg = _aggregate(episodes, max_steps)
            if agent == "peda":
                agg.update(_exploration_metrics(episodes, known_cells, max_steps))
            merged["conditions"][cond][agent] = agg
    peda_unknown = merged["conditions"]["goal_unknown"]["peda"]["mean_steps"]
    prag_unknown = merged["conditions"]["goal_unknown"]["pragmatic_only"]["mean_steps"]
    merged["verdict"] = {
        "peda_better_in_unknown_goal": peda_unknown < prag_unknown,
        "reason": (
            "PEDA mean_steps < pragmatic_only in goal_unknown condition"
            if peda_unknown < prag_unknown
            else "PEDA did not beat pragmatic_only in goal_unknown condition"
        ),
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, indent=2))
    print(f"Merged {len(chunks)} chunks into {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python merge_partial_eval_chunks.py chunk1.json chunk2.json ... output.json")
        sys.exit(1)
    merge(sys.argv[1:-1], sys.argv[-1])
