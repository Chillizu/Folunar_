#!/usr/bin/env python3
"""Phase 3 Sandbox Experiment: Statistical Analysis.

Reads all 4 condition JSONL files and computes:
- Fisher exact test: goal_unknown success rate (PEDA vs Pragmatic)
- Mann-Whitney U: SCR comparison
- Descriptive statistics
"""

import argparse
import json
import sys
from pathlib import Path


def load_results(path):
    """Load JSONL results."""
    results = []
    if not path.exists():
        print(f"ERROR: {path} not found")
        return results
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def compute_fisher(a_success, a_total, b_success, b_total):
    """Fisher exact test (two-sided)."""
    # Contingency table:
    #       Success  Failure  Total
    # A     a        A-a      a_total
    # B     b        B-b      b_total
    from math import comb, lgamma, exp, log

    def ln_factorial(n):
        return lgamma(n + 1)

    def log_hypergeometric(a, b, c, d):
        n = a + b + c + d
        return (ln_factorial(a + b) + ln_factorial(c + d) +
                ln_factorial(a + c) + ln_factorial(b + d) -
                ln_factorial(n) - ln_factorial(a) -
                ln_factorial(b) - ln_factorial(c) -
                ln_factorial(d))

    a = a_success
    b = b_success
    c = a_total - a_success
    d = b_total - b_success

    # Use scipy if available
    try:
        from scipy.stats import fisher_exact
        odds_ratio, p_value = fisher_exact([[a, b], [c, d]])
    except ImportError:
        # Fallback manual computation
        obs_log_p = log_hypergeometric(a, b, c, d)
        p_value = 0.0
        min_val = min(a_total, b_total)
        for i in range(min_val + 1):
            j = a_success + b_success - i
            k = a_total - i
            l = b_total - j
            if all(x >= 0 for x in [i, j, k, l]):
                lp = log_hypergeometric(i, j, k, l)
                p_value += exp(lp - obs_log_p)
        odds_ratio = (a / max(c, 1)) / (b / max(d, 1)) if c > 0 and d > 0 else float('inf')

    return odds_ratio, min(p_value, 1.0)


def compute_mannwhitney(x, y):
    """Mann-Whitney U test."""
    try:
        from scipy.stats import mannwhitneyu
        stat, p_value = mannwhitneyu(x, y, alternative='two-sided')
        return stat, p_value
    except ImportError:
        # Manual computation
        n1, n2 = len(x), len(y)
        combined = [(v, 0) for v in x] + [(v, 1) for v in y]
        combined.sort(key=lambda t: t[0])
        ranks = []
        i = 0
        while i < len(combined):
            j = i
            while j < len(combined) and combined[j][0] == combined[i][0]:
                j += 1
            rank = (i + j + 1) / 2.0
            for k in range(i, j):
                ranks.append((rank, combined[k][1]))
            i = j
        r1 = sum(r for r, g in ranks if g == 0)
        u1 = r1 - n1 * (n1 + 1) / 2
        u2 = n1 * n2 - u1
        u = min(u1, u2)
        # Approximate p-value using normal approximation
        mu = n1 * n2 / 2
        sigma = ((n1 * n2 * (n1 + n2 + 1)) / 12) ** 0.5
        z = (u - mu) / max(sigma, 0.001)
        from math import erf, sqrt
        p_value = 1 - erf(abs(z) / sqrt(2))
        return u, p_value


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Sandbox Analysis")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # Load all conditions
    conditions = {
        "pragmatic_known": load_results(results_dir / "phase3_sandbox_pragmatic_known.jsonl"),
        "pragmatic_unknown": load_results(results_dir / "phase3_sandbox_pragmatic_unknown.jsonl"),
        "peda_known": load_results(results_dir / "phase3_sandbox_peda_known.jsonl"),
        "peda_unknown": load_results(results_dir / "phase3_sandbox_peda_unknown.jsonl"),
    }

    print("=" * 70)
    print("Phase 3 Sandbox Experiment: Statistical Analysis")
    print("=" * 70)

    # Check all files loaded
    for name, data in conditions.items():
        if not data:
            print(f"  WARNING: {name}: no data loaded!")

    # ── Descriptive Statistics ──
    print("\n## Descriptive Statistics\n")
    print(f"{'Condition':<22} {'N':<4} {'Success':<9} {'Rate':<8} {'SCR(mean)':<10} {'SCR(std)':<10} {'Steps(mean)':<12}")
    print("-" * 75)

    for name in ["pragmatic_known", "pragmatic_unknown", "peda_known", "peda_unknown"]:
        data = conditions[name]
        if not data:
            continue
        n = len(data)
        successes = sum(1 for r in data if r["success"])
        rate = successes / n if n else 0
        scrs = [r["scr"] for r in data]
        steps = [r["steps_count"] for r in data]
        scr_mean = sum(scrs) / len(scrs) if scrs else 0
        scr_std = (sum((s - scr_mean) ** 2 for s in scrs) / len(scrs)) ** 0.5 if len(scrs) > 1 else 0
        steps_mean = sum(steps) / len(steps) if steps else 0
        print(f"{name:<22} {n:<4} {successes}/{n:<5} {rate:.3f}   {scr_mean:.4f}    {scr_std:.4f}    {steps_mean:.1f}")

    # ── Primary Test: Fisher exact on goal_unknown ──
    print("\n## Primary Hypothesis: PEDA > Pragmatic on Goal-Unknown\n")

    peda_unknown = conditions.get("peda_unknown", [])
    prag_unknown = conditions.get("pragmatic_unknown", [])

    if peda_unknown and prag_unknown:
        peda_success = sum(1 for r in peda_unknown if r["success"])
        prag_success = sum(1 for r in prag_unknown if r["success"])
        peda_n = len(peda_unknown)
        prag_n = len(prag_unknown)

        print(f"  PEDA success:     {peda_success}/{peda_n} ({100*peda_success/peda_n:.0f}%)")
        print(f"  Pragmatic success: {prag_success}/{prag_n} ({100*prag_success/prag_n:.0f}%)")

        or_val, p_val = compute_fisher(peda_success, peda_n, prag_success, prag_n)
        print(f"\n  Fisher exact test (one-sided, PEDA > Pragmatic):")
        print(f"    Odds ratio: {or_val:.4f}")
        print(f"    p-value:    {p_val:.4f}")
        if p_val < 0.05:
            print(f"    -> SIGNIFICANT (p < 0.05)")
        else:
            print(f"    -> Not significant (p >= 0.05)")

    # ── Secondary: Mann-Whitney on SCR ──
    print("\n## Secondary: Mann-Whitney U (SCR: PEDA vs Pragmatic)\n")

    for condition_name, cond_label in [("unknown", "Goal-Unknown"), ("known", "Goal-Known")]:
        key = f"peda_{condition_name}"
        pkey = f"pragmatic_{condition_name}"
        peda_data = conditions.get(key, [])
        prag_data = conditions.get(pkey, [])

        if peda_data and prag_data:
            peda_scrs = [r["scr"] for r in peda_data]
            prag_scrs = [r["scr"] for r in prag_data]

            if len(peda_scrs) >= 2 and len(prag_scrs) >= 2:
                u_stat, p_val = compute_mannwhitney(peda_scrs, prag_scrs)
                print(f"  {cond_label}:")
                print(f"    PEDA SCR:     {sum(peda_scrs)/len(peda_scrs):.4f} (n={len(peda_scrs)})")
                print(f"    Pragmatic SCR: {sum(prag_scrs)/len(prag_scrs):.4f} (n={len(prag_scrs)})")
                print(f"    Mann-Whitney U={u_stat:.1f}, p={p_val:.4f}")

    # ── Goal-Known vs Goal-Unknown within each baseline ──
    print("\n## Within-Baseline: Known vs Unknown\n")
    for baseline in ["peda", "pragmatic"]:
        known_data = conditions.get(f"{baseline}_known", [])
        unknown_data = conditions.get(f"{baseline}_unknown", [])

        if known_data and unknown_data:
            known_success = sum(1 for r in known_data if r["success"])
            unk_success = sum(1 for r in unknown_data if r["success"])
            known_n = len(known_data)
            unk_n = len(unknown_data)

            print(f"  {baseline.upper()}:")
            print(f"    Known success:   {known_success}/{known_n} ({100*known_success/known_n:.0f}%)")
            print(f"    Unknown success: {unk_success}/{unk_n} ({100*unk_success/unk_n:.0f}%)")

            if known_success > 0 and unk_success > 0 and known_n > 0 and unk_n > 0:
                or_val, p_val = compute_fisher(known_success, known_n, unk_success, unk_n)
                print(f"    Fisher: OR={or_val:.4f}, p={p_val:.4f}")

    # ── Per-episode detail ──
    print("\n## Per-Episode Details\n")
    for name in ["pragmatic_known", "pragmatic_unknown", "peda_known", "peda_unknown"]:
        data = conditions.get(name, [])
        if not data:
            continue
        print(f"  {name}:")
        for r in data:
            cwd = r.get("cwd", "?")
            succ = "YES" if r["success"] else "NO"
            scr = r["scr"]
            fht = r.get("fht", -1)
            steps = r["steps_count"]
            elapsed = r.get("elapsed", 0)
            print(f"    ep={r['episode']} cwd={cwd} succ={succ} scr={scr:.3f} steps={steps} fht={fht} t={elapsed:.0f}s")
        print()

    print("=" * 70)
    print("Analysis complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
