"""Held-out gate experiment.

Protocol (fixed before running; see README for the bar):

- Splits: calibration families and evaluation families are disjoint
  (different family seeds, different stream seeds, different probe
  RNG seeds). Nothing calibrated on the evaluation split.
- Stored patterns: crystallized top-2 patterns from each family
  (see stream.crystallize_family).
- Probes per stored pattern:
    positives (structured): seam_shift (1), row_slip over rows
      (3, 6, 9) x k (2, 3) (6) -> 7 probes.
    negatives (scattered): for each structured probe, one
      random_flips probe with exactly the same flip count -> 7.
    negatives (random): 7 fresh random patterns.
- Scores per probe: overlap-only, and combined 0.5*overlap +
  0.5*lateral (weights fixed in the lattice PREREG, not tuned).
- Gate: calibrate tau per score on the calibration split by the
  smallest-threshold-max-J rule, freeze it, apply to evaluation.
- Tables: A = structured vs random negatives; B = structured vs
  scattered negatives (the discriminating table: at equal flip
  counts the overlap distributions are identical by construction,
  so any separation in B must come from the lateral term).
- Intervals: Wilson 95% for TPR/FPR; bootstrap 95% (2000 reps,
  seed 999) for J and for the paired delta of J.

Primary bar (README): the lateral term "helps" on table B only if
the held-out bootstrap 95% CI for (J_combined - J_overlap) lies
entirely above 0.05. Otherwise the result is null, reported as
such, and no corruption or score parameter is tuned afterwards.
"""

import random

from .gate import (bootstrap_delta_ci, bootstrap_j_ci, calibrate_gate,
                   overlap, roc_table, wilson)
from .lat import combined_score, lateral_score
from .lattice import (count_flips, random_flips, random_pattern,
                      row_slip, seam_shift)
from .stream import crystallize_family

SLIP_ROWS = (3, 6, 9)
SLIP_KS = (2, 3)


def probes_for(stored, rng):
    """Return (positives, scattered, randoms) probe lists."""
    structured = [seam_shift(stored)]
    for row in SLIP_ROWS:
        for k in SLIP_KS:
            structured.append(row_slip(stored, row=row, k=k))
    scattered = []
    for s in structured:
        scattered.append(random_flips(stored, count_flips(stored, s), rng))
    randoms = [random_pattern(rng) for _ in range(len(structured))]
    return structured, scattered, randoms


def scores_for(stored, probes):
    """Return (overlap_scores, combined_scores) for a probe list."""
    ov = [overlap(stored, p) for p in probes]
    cb = [combined_score(stored, p, overlap) for p in probes]
    return ov, cb


def collect_split(family_seeds, steps, alpha, stream_seed_base,
                  probe_seed_base):
    """Collect scores for one split. Returns a dict with, per score
    type, (positives, scattered, randoms) score lists, plus the
    crystallized sanity values. Families are independent; the only
    shared quantities are the fixed protocol constants."""
    pos_ov, pos_cb = [], []
    scat_ov, scat_cb = [], []
    rand_ov, rand_cb = [], []
    # Per-corruption lateral scores on this split: structured probes
    # are ordered [seam_shift, row_slips...], scattered matched 1:1.
    seam_lat_pos, seam_lat_neg = [], []
    slip_lat_pos, slip_lat_neg = [], []
    sanity = []
    for fi, fseed in enumerate(family_seeds):
        crystallized, san, _ = crystallize_family(
            fseed, steps, alpha, stream_seed_base + fi)
        sanity.extend(san)
        rng = random.Random(probe_seed_base + fi)
        for stored in crystallized:
            structured, scattered, randoms = probes_for(stored, rng)
            for group, out_ov, out_cb in (
                    (structured, pos_ov, pos_cb),
                    (scattered, scat_ov, scat_cb),
                    (randoms, rand_ov, rand_cb)):
                ov, cb = scores_for(stored, group)
                out_ov.extend(ov)
                out_cb.extend(cb)
            for j, (s, c) in enumerate(zip(structured, scattered)):
                ls, lc = lateral_score(stored, s), lateral_score(stored, c)
                if j == 0:
                    seam_lat_pos.append(ls)
                    seam_lat_neg.append(lc)
                else:
                    slip_lat_pos.append(ls)
                    slip_lat_neg.append(lc)
    return {"pos": (pos_ov, pos_cb), "scat": (scat_ov, scat_cb),
            "rand": (rand_ov, rand_cb), "sanity": sanity,
            "lat_by_corruption": {
                "seam_shift": (seam_lat_pos, seam_lat_neg),
                "row_slip": (slip_lat_pos, slip_lat_neg)}}


def evaluate(pos, neg, tau):
    """Held-out ROC numbers for one score type at a frozen tau."""
    t = roc_table(pos, neg, tau)
    tp = sum(1 for s in pos if s >= tau)
    fp = sum(1 for s in neg if s >= tau)
    t["TPR_ci"] = wilson(tp, len(pos))
    t["FPR_ci"] = wilson(fp, len(neg))
    t["J_ci"] = bootstrap_j_ci(pos, neg, tau)
    return t


def run_experiment(cal_families, eval_families, steps, alpha):
    """Run both splits and return the full results dict."""
    cal = collect_split(cal_families, steps, alpha,
                        stream_seed_base=10000, probe_seed_base=20000)
    ev = collect_split(eval_families, steps, alpha,
                       stream_seed_base=30000, probe_seed_base=40000)
    results = {"cal_sanity": cal["sanity"], "eval_sanity": ev["sanity"],
               "tables": {}}
    for table, neg_key in (("A_structured_vs_random", "rand"),
                           ("B_structured_vs_scattered", "scat")):
        cal_pos_ov, cal_pos_cb = cal["pos"]
        cal_neg_ov, cal_neg_cb = cal[neg_key]
        ev_pos_ov, ev_pos_cb = ev["pos"]
        ev_neg_ov, ev_neg_cb = ev[neg_key]
        tau_ov, _, _, _ = calibrate_gate(cal_pos_ov, cal_neg_ov)
        tau_cb, _, _, _ = calibrate_gate(cal_pos_cb, cal_neg_cb)
        row_ov = evaluate(ev_pos_ov, ev_neg_ov, tau_ov)
        row_cb = evaluate(ev_pos_cb, ev_neg_cb, tau_cb)
        dlo, dhi, dmean = bootstrap_delta_ci(
            ev_pos_ov, ev_neg_ov, tau_ov, ev_pos_cb, ev_neg_cb, tau_cb)
        results["tables"][table] = {
            "overlap_only": row_ov, "combined": row_cb,
            "delta_J": dmean, "delta_J_ci": (dlo, dhi),
        }
    # Lateral-score gaps on the evaluation split (raw score means,
    # recomputed from stored probes is not retained, so gaps are
    # reported from the score lists via the combined/overlap
    # identity: lateral = 2*combined - overlap).
    for table, neg_key in (("A_structured_vs_random", "rand"),
                           ("B_structured_vs_scattered", "scat")):
        ev_pos_ov, ev_pos_cb = ev["pos"]
        ev_neg_ov, ev_neg_cb = ev[neg_key]
        pos_lat = [2 * c - o for o, c in zip(ev_pos_ov, ev_pos_cb)]
        neg_lat = [2 * c - o for o, c in zip(ev_neg_ov, ev_neg_cb)]
        results["tables"][table]["mean_lateral_pos"] = sum(pos_lat) / len(pos_lat)
        results["tables"][table]["mean_lateral_neg"] = sum(neg_lat) / len(neg_lat)
        results["tables"][table]["lateral_gap"] = (
            results["tables"][table]["mean_lateral_pos"]
            - results["tables"][table]["mean_lateral_neg"])
    results["lateral_gap_by_corruption"] = {}
    for name, (pos_lat, neg_lat) in ev["lat_by_corruption"].items():
        mp = sum(pos_lat) / len(pos_lat)
        mn = sum(neg_lat) / len(neg_lat)
        results["lateral_gap_by_corruption"][name] = {
            "mean_lateral_pos": mp, "mean_lateral_neg": mn,
            "gap": mp - mn, "n": len(pos_lat)}
    return results
