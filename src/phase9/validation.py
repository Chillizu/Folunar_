"""Metric helpers for Phase 9 MVP signal validation (V1/V2/V3, F1-F4 gates).

All metrics implemented manually (no sklearn dependency), matching the
pre-registered definitions in PEDA_FINAL/phase9/PHASE9_PLAN.md and
plans/plan-hypothesis-generator.md §4.
"""

import math
from typing import Sequence, Tuple

import numpy as np

# ── AUC (Mann-Whitney U) ────────────────────────────────


def auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """AUC of classifying label=1 vs label=0 from score (higher = more 1).

    Computed as the Mann-Whitney U statistic / (n1 * n0); equivalent to the
    probability that a random positive scores above a random negative.
    """
    if len(labels) != len(scores):
        raise ValueError("labels and scores must be same length")
    pairs = sorted(zip(scores, labels), key=lambda p: (p[0], p[1]))
    n = len(pairs)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg
        i = j
    n1 = sum(1 for label in labels if label == 1)
    n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    sum_ranks_pos = sum(r for r, (_, label) in zip(ranks, pairs) if label == 1)
    u = sum_ranks_pos - n1 * (n1 + 1) / 2.0
    return u / (n1 * n0)


# ── Histograms / KL ─────────────────────────────────────


def laplace_hist(values: Sequence[float], bins: int = 10,
                 lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Laplace-smoothed histogram over [lo, hi] (error values live in 0..1)."""
    hist, _ = np.histogram(values, bins=bins, range=(lo, hi + 1e-9))
    return (hist.astype(np.float64) + 1.0) / (hist.sum() + bins)


def kl_empirical_vs_uniform(values: Sequence[float], bins: int = 10) -> float:
    """KL(empirical error distribution || uniform), Laplace-smoothed, nats."""
    if len(values) < 2:
        return 0.0
    p = laplace_hist(values, bins=bins)
    q = np.full(bins, 1.0 / bins)
    return float(np.sum(p * np.log(p / q)))


def kl_between(values_a: Sequence[float], values_b: Sequence[float],
               bins: int = 10) -> float:
    """KL(P(values_a) || P(values_b)) on shared Laplace-smoothed histograms, nats."""
    if len(values_a) < 2 or len(values_b) < 2:
        return 0.0
    p = laplace_hist(values_a, bins=bins)
    q = laplace_hist(values_b, bins=bins)
    return float(np.sum(p * np.log(p / q)))


# ── Effect sizes / correlation ──────────────────────────


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Cohen's d with pooled sample SD: (mean_a - mean_b) / pooled_sd."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    na, nb = len(a), len(b)
    pooled = math.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if pooled == 0.0:
        return 0.0 if float(a.mean()) == float(b.mean()) else math.copysign(float("inf"), a.mean() - b.mean())
    return float((a.mean() - b.mean()) / pooled)


def _rankdata(x: Sequence[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and arr[order[j]] == arr[order[i]]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation rho."""
    if len(x) != len(y) or len(x) < 3:
        return 0.0
    rx, ry = _rankdata(x), _rankdata(y)
    rx_m, ry_m = rx.mean(), ry.mean()
    num = float(np.sum((rx - rx_m) * (ry - ry_m)))
    den = math.sqrt(float(np.sum((rx - rx_m) ** 2)) * float(np.sum((ry - ry_m) ** 2)))
    if den == 0.0:
        return 0.0
    return num / den


# ── Misc ────────────────────────────────────────────────


def mean_sd(values: Sequence[float]) -> Tuple[float, float]:
    v = np.asarray(values, dtype=np.float64)
    if len(v) == 0:
        return 0.0, 0.0
    return float(v.mean()), float(v.std(ddof=1) if len(v) > 1 else 0.0)


def describe(values: Sequence[float]) -> dict:
    m, sd = mean_sd(values)
    return {
        "n": len(values),
        "mean": round(m, 4),
        "sd": round(sd, 4),
        "min": round(float(min(values)), 4) if values else None,
        "max": round(float(max(values)), 4) if values else None,
    }


def sign_of(v: float, tol: float = 1e-9) -> str:
    if v > tol:
        return "+"
    if v < -tol:
        return "-"
    return "0"
