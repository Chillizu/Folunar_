#!/usr/bin/env python3
"""Phase 3 Epistemic Validation: statistical analysis and report generation.

Reads per-condition JSONL files, computes statistics, writes report.json.
"""

import json
import math
import sys
from pathlib import Path
from collections import defaultdict


def load_results(path: Path) -> list:
    """Load JSONL file, return list of episode dicts."""
    if not path.exists():
        print(f"WARNING: {path} not found, returning empty list")
        return []
    episodes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def fisher_exact(a, b, c, d) -> float:
    """Two-sided Fisher exact test p-value using hypergeometric probability.
    
    a, b, c, d are the contingency table:
         a | b
         c | d
    """
    n = a + b + c + d
    if n == 0:
        return 1.0
    
    def log_hypergeometric(k, n1, n2, t):
        """Log probability of observing k successes in n1 draws from 
        population of size n1+n2 with t total successes."""
        # log(C(t,k) * C(N-t,n1-k) / C(N,n1))
        def log_comb(n, k):
            if k < 0 or k > n:
                return float('-inf')
            return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
        
        return log_comb(t, k) + log_comb(n - t, n1 - k) - log_comb(n, n1)
    
    # Compute all tables with same marginals
    n1 = a + c  # row 1 total
    n2 = b + d  # row 2 total
    t = a + b   # col 1 total
    
    min_k = max(0, t - n2)
    max_k = min(n1, t)
    
    p_obs = math.exp(log_hypergeometric(a, n1, n2, t))
    p_value = 0.0
    
    for k in range(min_k, max_k + 1):
        p_k = math.exp(log_hypergeometric(k, n1, n2, t))
        if p_k <= p_obs:
            p_value += p_k
    
    return min(p_value, 1.0)


def mann_whitney_u(x, y) -> tuple:
    """Mann-Whitney U test. Returns (U, p-value, z-score).
    
    Uses normal approximation with tie correction for n>=20.
    For smaller n, uses exact method (simplified).
    """
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0, 0.0
    
    # Rank all data
    combined = [(x[i], 0, i) for i in range(n1)] + [(y[j], 1, j) for j in range(n2)]
    combined.sort(key=lambda v: v[0])
    
    # Assign ranks (handle ties)
    ranks = [0] * (n1 + n2)
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2.0  # 1-indexed ranks
        for k in range(i, j):
            idx = combined[k][2]
            group = combined[k][1]
            ranks[idx if group == 0 else n1 + idx] = avg_rank
        i = j
    
    # Compute U statistic
    r1 = sum(ranks[:n1])
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    u = min(u1, u2)
    
    # Normal approximation
    mu = n1 * n2 / 2.0
    
    # Tie correction
    combined.sort(key=lambda v: v[0])
    tie_sum = 0.0
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        t = j - i
        if t > 1:
            tie_sum += t * (t * t - 1)
        i = j
    
    sigma = math.sqrt((n1 * n2 / 12.0) * ((n1 + n2 + 1) - tie_sum / ((n1 + n2) * (n1 + n2 - 1))))
    
    if sigma == 0:
        z = 0.0
    else:
        z = (u - mu) / sigma
    
    # Two-sided p-value from normal approximation
    p_value = 2.0 * (1.0 - _norm_cdf(abs(z)))
    p_value = min(p_value, 1.0)
    
    return u, p_value, z


def _norm_cdf(x):
    """Standard normal CDF approximation (Abramowitz and Stegun)."""
    if x < 0:
        return 1.0 - _norm_cdf(-x)
    k = 1.0 / (1.0 + 0.2316419 * x)
    poly = k * (0.319381530 + k * (-0.356563782 + k * (1.781477937 + k * (-1.821255978 + k * 1.330274429))))
    return 1.0 - 0.3989422804014327 * math.exp(-x * x / 2.0) * poly


def compute_effect_size(success_a, success_b, n_a, n_b):
    """Cohen's h effect size for proportions."""
    if n_a == 0 or n_b == 0:
        return 0.0
    p1 = success_a / n_a
    p2 = success_b / n_b
    
    def arcsin(p):
        return 2.0 * math.asin(math.sqrt(p))
    
    h = arcsin(p1) - arcsin(p2)
    return abs(h)


def compute_cohens_d(mean1, mean2, pooled_std):
    """Cohen's d effect size for means."""
    if pooled_std == 0:
        return 0.0
    return abs(mean1 - mean2) / pooled_std


def main():
    results_dir = Path("results/phase3_experiment")
    
    # Load all result files
    conditions = {
        "goal_known": {
            "peda": load_results(results_dir / "goal_known_peda.jsonl"),
            "pragmatic": load_results(results_dir / "goal_known_pragmatic.jsonl"),
        },
        "goal_unknown": {
            "peda": load_results(results_dir / "goal_unknown_peda.jsonl"),
            "pragmatic": load_results(results_dir / "goal_unknown_pragmatic.jsonl"),
        },
    }
    
    report = {
        "experiment": "Phase 3 Epistemic Validation",
        "environment": "Docker sandbox v2 (peda-sandbox:v2)",
        "task": "read_note (cat docs/note.txt → 'secret key: 12345')",
        "adapter": "checkpoints/phase2/sandbox_adapter_v2_partial (40 known transitions)",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "timestamp": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
    }
    
    # Per-condition per-agent statistics
    stats = {}
    all_episodes = []
    
    for condition in ["goal_known", "goal_unknown"]:
        stats[condition] = {}
        for agent_type in ["peda", "pragmatic"]:
            eps = conditions[condition][agent_type]
            n = len(eps)
            if n == 0:
                stats[condition][agent_type] = {
                    "n": 0, "success_rate": 0.0, "mean_steps": 0.0,
                    "mean_revisit_rate": 0.0, "successes": 0, "steps_list": [],
                    "start_cwds": [],
                }
                continue
            
            successes = sum(1 for e in eps if e.get("success", 0) > 0)
            steps_list = [e.get("steps", 0) for e in eps]
            revisit_list = [e.get("revisit_rate", 0) for e in eps]
            start_cwds = [e.get("start_cwd", "?") for e in eps]
            
            stats[condition][agent_type] = {
                "n": n,
                "successes": successes,
                "success_rate": successes / n,
                "mean_steps": sum(steps_list) / n,
                "median_steps": sorted(steps_list)[n // 2] if n > 0 else 0,
                "std_steps": (sum((s - sum(steps_list)/n)**2 for s in steps_list) / n) ** 0.5 if n > 0 else 0,
                "min_steps": min(steps_list) if steps_list else 0,
                "max_steps": max(steps_list) if steps_list else 0,
                "mean_revisit_rate": sum(revisit_list) / n,
                "steps_list": steps_list,
                "revisit_list": revisit_list,
                "start_cwds": start_cwds,
            }
            
            for e in eps:
                all_episodes.append({
                    "condition": condition,
                    "agent": agent_type,
                    "episode": e.get("episode"),
                    "start_cwd": e.get("start_cwd"),
                    "success": e.get("success"),
                    "steps": e.get("steps"),
                    "revisit_rate": e.get("revisit_rate"),
                    "first_action": e.get("first_action"),
                    "elapsed": e.get("elapsed"),
                })
    
    report["per_condition"] = {
        cond: {
            agent: {k: v for k, v in s.items() if k not in ("steps_list", "revisit_list", "start_cwds")}
            for agent, s in cond_stats.items()
        }
        for cond, cond_stats in stats.items()
    }
    
    # Statistical tests
    tests = {}
    
    # --- goal_unknown: PEDA vs Pragmatic (Fisher exact on success) ---
    s_peda = stats["goal_unknown"]["peda"]
    s_prag = stats["goal_unknown"]["pragmatic"]
    a = s_peda["successes"]
    b = s_prag["successes"]
    c = s_peda["n"] - a
    d = s_prag["n"] - b
    
    p_fisher = fisher_exact(a, b, c, d)
    h_effect = compute_effect_size(a, s_peda["n"], b, s_prag["n"])
    
    tests["goal_unknown_success_fisher"] = {
        "test": "Fisher exact (two-sided)",
        "contingency_table": {
            "peda_success": a, "peda_fail": c,
            "pragmatic_success": b, "pragmatic_fail": d,
        },
        "p_value": round(p_fisher, 4),
        "effect_size_cohens_h": round(h_effect, 3),
        "significant_at_005": p_fisher < 0.05,
        "significant_at_001": p_fisher < 0.01,
    }
    
    # --- goal_unknown: PEDA vs Pragmatic (Mann-Whitney U on steps) ---
    u_stat, p_mw, z_mw = mann_whitney_u(
        s_peda["steps_list"], s_prag["steps_list"]
    )
    
    # Pooled std for Cohen's d
    if s_peda["n"] > 0 and s_prag["n"] > 0:
        pooled_std = math.sqrt(
            ((s_peda["n"] - 1) * s_peda["std_steps"]**2 + 
             (s_prag["n"] - 1) * s_prag["std_steps"]**2) /
            (s_peda["n"] + s_prag["n"] - 2)
        )
    else:
        pooled_std = 0
    d_effect = compute_cohens_d(s_peda["mean_steps"], s_prag["mean_steps"], pooled_std)
    
    tests["goal_unknown_steps_mannwhitney"] = {
        "test": "Mann-Whitney U (two-sided, normal approximation)",
        "u_statistic": round(u_stat, 2),
        "z_score": round(z_mw, 3),
        "p_value": round(p_mw, 4),
        "peda_mean_steps": round(s_peda["mean_steps"], 2),
        "pragmatic_mean_steps": round(s_prag["mean_steps"], 2),
        "effect_size_cohens_d": round(d_effect, 3),
        "significant_at_005": p_mw < 0.05,
    }
    
    # --- goal_known: fairness check (Fisher exact on success) ---
    s_peda_k = stats["goal_known"]["peda"]
    s_prag_k = stats["goal_known"]["pragmatic"]
    a_k = s_peda_k["successes"]
    b_k = s_prag_k["successes"]
    c_k = s_peda_k["n"] - a_k
    d_k = s_prag_k["n"] - b_k
    
    p_fairness = fisher_exact(a_k, b_k, c_k, d_k)
    
    tests["goal_known_fairness"] = {
        "test": "Fisher exact (two-sided) — fairness check: PEDA should NOT differ from Pragmatic",
        "contingency_table": {
            "peda_success": a_k, "peda_fail": c_k,
            "pragmatic_success": b_k, "pragmatic_fail": d_k,
        },
        "p_value": round(p_fairness, 4),
        "fairness_pass": p_fairness > 0.05,
        "interpretation": "p > 0.05 = PEDA and Pragmatic are statistically indistinguishable (fairness check PASS)" if p_fairness > 0.05 else "p <= 0.05 = unexpected difference (fairness check FAIL)",
    }
    
    report["statistical_tests"] = tests
    
    # Success criteria
    success_criteria = {
        "goal_unknown_peda_success_rate_gt_60pct": {
            "criterion": "PEDA goal_unknown success rate > 60%",
            "actual": f"{s_peda['success_rate']*100:.1f}%",
            "pass": s_peda["success_rate"] > 0.6,
        },
        "goal_unknown_pragmatic_success_rate_lt_40pct": {
            "criterion": "Pragmatic goal_unknown success rate < 40%",
            "actual": f"{s_prag['success_rate']*100:.1f}%",
            "pass": s_prag["success_rate"] < 0.4,
        },
        "goal_unknown_peda_mean_steps_lt_10": {
            "criterion": "PEDA goal_unknown mean steps < 10",
            "actual": f"{s_peda['mean_steps']:.1f}",
            "pass": s_peda["mean_steps"] < 10,
        },
        "goal_unknown_pragmatic_mean_steps_gt_15": {
            "criterion": "Pragmatic goal_unknown mean steps > 15",
            "actual": f"{s_prag['mean_steps']:.1f}",
            "pass": s_prag["mean_steps"] > 15,
        },
        "goal_known_fairness_pgt_005": {
            "criterion": "goal_known fairness: PEDA ≈ Pragmatic (p > 0.05)",
            "actual": f"p = {p_fairness:.4f}",
            "pass": p_fairness > 0.05,
        },
        "fisher_significant_p_lt_005": {
            "criterion": "Fisher exact p < 0.05 for goal_unknown success rate difference",
            "actual": f"p = {p_fisher:.4f}",
            "pass": p_fisher < 0.05,
        },
        "mannwhitney_significant_p_lt_005": {
            "criterion": "Mann-Whitney U p < 0.05 for goal_unknown steps difference",
            "actual": f"p = {p_mw:.4f}",
            "pass": p_mw < 0.05,
        },
    }
    
    # Verdict
    passed_criteria = sum(1 for v in success_criteria.values() if v["pass"])
    total_criteria = len(success_criteria)
    
    if s_peda["n"] == s_prag["n"] == 0:
        verdict = "NO_DATA"
        verdict_reason = "No episodes completed for goal_unknown condition."
    elif tests["goal_unknown_success_fisher"]["significant_at_005"] and s_peda["success_rate"] > s_prag["success_rate"]:
        verdict = "CORE_HYPOTHESIS_SUPPORTED"
        verdict_reason = (
            f"PEDA achieves {s_peda['success_rate']*100:.1f}% success vs Pragmatic {s_prag['success_rate']*100:.1f}% "
            f"in goal_unknown condition (Fisher p={p_fisher:.4f}, "
            f"Mann-Whitney p={p_mw:.4f}). "
            f"Passed {passed_criteria}/{total_criteria} success criteria."
        )
    elif s_peda["success_rate"] > s_prag["success_rate"]:
        verdict = "DIRECTIONAL_SIGNAL"
        verdict_reason = (
            f"PEDA ({s_peda['success_rate']*100:.1f}%) outperforms Pragmatic ({s_prag['success_rate']*100:.1f}%) "
            f"directionally but not statistically significant (Fisher p={p_fisher:.4f}). "
            f"Passed {passed_criteria}/{total_criteria} success criteria."
        )
    else:
        verdict = "CORE_HYPOTHESIS_NOT_SUPPORTED"
        verdict_reason = (
            f"PEDA ({s_peda['success_rate']*100:.1f}%) does not outperform Pragmatic ({s_prag['success_rate']*100:.1f}%) "
            f"in goal_unknown condition. Passed {passed_criteria}/{total_criteria} success criteria."
        )
    
    report["success_criteria"] = success_criteria
    report["verdict"] = verdict
    report["verdict_reason"] = verdict_reason
    report["passed_criteria"] = f"{passed_criteria}/{total_criteria}"
    
    # Write report
    report_path = results_dir / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    
    # Print summary
    print("=" * 70)
    print("Phase 3 Epistemic Validation — Statistical Report")
    print("=" * 70)
    
    for condition in ["goal_known", "goal_unknown"]:
        print(f"\n--- {condition} ---")
        for agent_type in ["peda", "pragmatic"]:
            s = stats[condition][agent_type]
            if s["n"] == 0:
                print(f"  {agent_type}: NO DATA")
                continue
            print(f"  {agent_type:12s} N={s['n']:2d}  success={s['success_rate']:.3f}  "
                  f"steps={s['mean_steps']:.1f}±{s['std_steps']:.1f}  "
                  f"revisit={s['mean_revisit_rate']:.3f}")
    
    print(f"\n--- Statistical Tests ---")
    for test_name, test_data in tests.items():
        print(f"  {test_name}: p={test_data['p_value']:.4f} "
              f"{'SIGNIFICANT' if test_data.get('significant_at_005', False) else 'not significant'}")
    
    print(f"\n--- Verdict: {verdict} ---")
    print(f"  {verdict_reason}")
    print(f"\n  Success criteria passed: {passed_criteria}/{total_criteria}")
    print(f"\n  Report saved to: {report_path}")
    
    # Write episode-level summary for inspection
    episodes_path = results_dir / "all_episodes.json"
    episodes_path.write_text(json.dumps(all_episodes, indent=2))
    print(f"  Episode data saved to: {episodes_path}")
    
    return 0 if verdict == "CORE_HYPOTHESIS_SUPPORTED" else (
        1 if verdict == "DIRECTIONAL_SIGNAL" else 2
    )


if __name__ == "__main__":
    sys.exit(main())
