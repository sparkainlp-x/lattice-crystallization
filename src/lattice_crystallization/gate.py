"""Overlap, gate calibration, and confidence intervals.

Overlap and calibrate_gate are vendored-with-provenance from
sparkainlp-x/microtubule-lattice-memory,
candidate/microtubule-lattice-memory/memory.py (overlap,
calibrate_gate): overlap is the mean product over cells where both
are non-zero; calibration sweeps thresholds at midpoints between
sorted unique scores and picks max Youden's J, ties broken by the
smallest threshold achieving max J (documented, not tuned
afterwards). The gate is then frozen: it is never re-tuned on
held-out probes.

Wilson intervals for TPR/FPR and bootstrap intervals for J and for
the paired delta of J are this package's own evaluation code.
"""

import math
import random


def overlap(a, b):
    """Mean product over cells where both are non-zero (+-1)."""
    pairs = [(p, q) for p, q in zip(a, b) if p != 0 and q != 0]
    if not pairs:
        return 0.0
    return sum(p * q for p, q in pairs) / len(pairs)


def calibrate_gate(true_scores, control_scores):
    """Return (tau, J, TPR, FPR) on the validation scores given."""
    vals = sorted(set(true_scores) | set(control_scores))
    cands = []
    for a, b in zip(vals, vals[1:]):
        cands.append((a + b) / 2.0)
    cands.extend([vals[0] - 0.01, vals[-1] + 0.01])
    best = None
    for tau in cands:
        tpr = sum(1 for s in true_scores if s >= tau) / len(true_scores)
        fpr = sum(1 for s in control_scores if s >= tau) / len(control_scores)
        j = tpr - fpr
        if best is None or j > best[1] + 1e-12:
            best = (tau, j, tpr, fpr)
    return best


def roc_table(true_scores, control_scores, tau):
    tpr = sum(1 for s in true_scores if s >= tau) / len(true_scores)
    fpr = sum(1 for s in control_scores if s >= tau) / len(control_scores)
    return {"tau": tau, "TPR": tpr, "FPR": fpr, "J": tpr - fpr,
            "n_true": len(true_scores), "n_control": len(control_scores)}


def wilson(k, n, z=1.96):
    """Wilson 95% interval for a binomial proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def _j_of(pos, neg, tau):
    tpr = sum(1 for s in pos if s >= tau) / len(pos) if pos else 0.0
    fpr = sum(1 for s in neg if s >= tau) / len(neg) if neg else 0.0
    return tpr - fpr


def bootstrap_delta_ci(pos_a, neg_a, tau_a, pos_b, neg_b, tau_b,
                       reps=2000, seed=999):
    """Bootstrap 95% CI for J_b - J_a, resampling probe indices
    jointly: the same resampled positive and negative probe indices
    are used for both scores, preserving the paired structure
    (same probes, two scores). Returns (lo, hi, mean_delta)."""
    rng = random.Random(seed)
    deltas = []
    for _ in range(reps):
        ipos = [rng.randrange(len(pos_a)) for _ in range(len(pos_a))]
        ineg = [rng.randrange(len(neg_a)) for _ in range(len(neg_a))]
        pa = [pos_a[i] for i in ipos]
        pb = [pos_b[i] for i in ipos]
        na = [neg_a[i] for i in ineg]
        nb = [neg_b[i] for i in ineg]
        deltas.append(_j_of(pb, nb, tau_b) - _j_of(pa, na, tau_a))
    deltas.sort()
    lo = deltas[int(0.025 * reps)]
    hi = deltas[min(reps - 1, int(0.975 * reps))]
    return (lo, hi, sum(deltas) / len(deltas))


def bootstrap_j_ci(pos, neg, tau, reps=2000, seed=999):
    """Bootstrap 95% CI for a single J. Returns (lo, hi)."""
    rng = random.Random(seed)
    js = []
    for _ in range(reps):
        pp = [pos[rng.randrange(len(pos))] for _ in range(len(pos))]
        nn = [neg[rng.randrange(len(neg))] for _ in range(len(neg))]
        js.append(_j_of(pp, nn, tau))
    js.sort()
    return (js[int(0.025 * reps)], js[min(reps - 1, int(0.975 * reps))])
