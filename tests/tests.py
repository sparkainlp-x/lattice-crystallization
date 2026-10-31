"""Tests for lattice-crystallization. Plain asserts, stdlib only.

Run from the package root:  python3 tests/tests.py
"""

import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lattice_crystallization import gate, lat
from lattice_crystallization import lattice as L
from lattice_crystallization import stream as S
from lattice_crystallization.experiment import (SLIP_KS, SLIP_ROWS,
                                                collect_split, probes_for)

PASS = 0


def check(name, cond):
    global PASS
    assert cond, name
    PASS += 1
    print("ok -", name)


# --- lateral score -------------------------------------------------
def test_lateral_identity():
    pat = L.build_pattern(42)
    check("lateral(x=pat) == 1.0", lat.lateral_score(pat, pat) == 1.0)


def test_lateral_pair_count():
    check("299 neighbour pairs", len(lat.PAIRS) == 299)
    check("pairs unique", len(set(lat.PAIRS)) == 299)


def test_lateral_all_plus():
    # pat all +1: lateral = mean over pairs of x_i*x_j.
    pat = [1] * L.N
    x = [1] * L.N
    x[L.idx(0, 0)] = -1
    # Flipping one corner cell flips its pairs: vertical to (1,0),
    # seam pair from (12,0) to (0,0), and horizontal to (0,1):
    # 3 pairs change from +1 to -1, all others +1.
    expect = (299 - 2 * 3) / 299.0
    check("lateral all-plus one flip exact",
          abs(lat.lateral_score(pat, x) - expect) < 1e-12)


def test_lateral_random_near_zero():
    rng = random.Random(7)
    pat = L.build_pattern(42)
    vals = [lat.lateral_score(pat, L.random_pattern(rng)) for _ in range(20)]
    check("random lateral mean |.| < 0.2", abs(sum(vals) / len(vals)) < 0.2)


def test_lateral_skips_zeros():
    pat = L.build_pattern(42)
    x = list(pat)
    x[L.idx(0, 0)] = 0
    check("lateral with unknown cell still in [-1, 1]",
          -1.0 <= lat.lateral_score(pat, x) <= 1.0)
    check("lateral empty probe 0.0",
          lat.lateral_score(pat, [0] * L.N) == 0.0)


def test_combined_weights():
    pat = L.build_pattern(42)
    x = L.row_slip(pat, row=6, k=3)
    ov = gate.overlap(pat, x)
    got = lat.combined_score(pat, x, gate.overlap)
    check("combined = 0.5*overlap + 0.5*lateral",
          abs(got - (0.5 * ov + 0.5 * lat.lateral_score(pat, x))) < 1e-12)


# --- corruptions ----------------------------------------------------
def test_seam_shift_exact():
    pat = L.build_pattern(42)
    out = L.seam_shift(pat)
    rows = L.rows_of(out)
    check("seam row 5 inverted",
          rows[5] == [-c for c in L.rows_of(pat)[5]])
    check("seam row 12 := row 11",
          rows[12] == L.rows_of(pat)[11])
    check("seam other rows unchanged",
          all(rows[r] == L.rows_of(pat)[r]
              for r in range(13) if r not in (5, 12)))


def test_row_slip_exact():
    pat = L.build_pattern(42)
    out = L.row_slip(pat, row=6, k=3)
    rows = L.rows_of(out)
    orig = L.rows_of(pat)[6]
    check("row slip cyclic shift", rows[6] == orig[-3:] + orig[:-3])
    check("row slip other rows unchanged",
          all(rows[r] == L.rows_of(pat)[r] for r in range(13) if r != 6))


def test_flip_count_match():
    pat = L.build_pattern(42)
    rng = random.Random(3)
    structured, scattered, randoms = probes_for(pat, rng)
    check("7 structured probes (1 seam + 6 slips)", len(structured) == 7)
    check("7 scattered probes", len(scattered) == 7)
    check("7 random probes", len(randoms) == 7)
    check("scattered flip counts match structured exactly",
          all(L.count_flips(pat, s) == L.count_flips(pat, c)
              for s, c in zip(structured, scattered)))
    check("slip params used", SLIP_ROWS == (3, 6, 9) and SLIP_KS == (2, 3))


def test_overlap_same_count_identical():
    # At equal flip counts, overlap is identical by construction.
    pat = L.build_pattern(42)
    rng = random.Random(3)
    structured, scattered, _ = probes_for(pat, rng)
    check("overlap equal for matched flip counts",
          all(abs(gate.overlap(pat, s) - gate.overlap(pat, c)) < 1e-12
              for s, c in zip(structured, scattered)))


# --- gate -----------------------------------------------------------
def test_calibrate_smallest_tie():
    # Scores: true {0.9, 0.8}, control {0.7, 0.6}. Max J=1 at any tau
    # in (0.7, 0.8]; candidates midpoints: 0.65, 0.75, 0.85, extremes
    # 0.59, 0.91. Smallest achieving tau is 0.75.
    tau, j, tpr, fpr = gate.calibrate_gate([0.9, 0.8], [0.7, 0.6])
    check("calibrate smallest max-J threshold", abs(tau - 0.75) < 1e-12)
    check("calibrate J=1", abs(j - 1.0) < 1e-12)
    check("calibrate TPR/FPR", tpr == 1.0 and fpr == 0.0)


def test_wilson_bounds():
    lo, hi = gate.wilson(0, 10)
    check("wilson 0/10 sane", 0.0 <= lo <= hi <= 1.0 and hi > 0.0)
    lo, hi = gate.wilson(10, 10)
    check("wilson 10/10 sane", 0.0 <= lo <= hi <= 1.0 and lo < 1.0)


# --- streaming PCA --------------------------------------------------
def test_power_iteration_diagonal():
    M = [[0.0] * 4 for _ in range(4)]
    M[0][0], M[1][1], M[2][2], M[3][3] = 5.0, 3.0, 2.0, 1.0
    v, lam = S.power_iteration(M, seed=1)
    check("power iteration eigenvalue", abs(lam - 5.0) < 1e-6)
    check("power iteration eigenvector", abs(abs(v[0]) - 1.0) < 1e-6)


def test_top_eigenvectors_order():
    M = [[0.0] * 3 for _ in range(3)]
    M[0][0], M[1][1], M[2][2] = 4.0, 2.0, 1.0
    eigs = S.top_eigenvectors(M, 2, seed=1)
    check("top eigenvectors descending",
          eigs[0][1] > eigs[1][1] > 0)
    check("top eigenvectors orthonormal",
          abs(sum(a * b for a, b in zip(eigs[0][0], eigs[1][0]))) < 1e-6)


def test_hebbian_determinism():
    pats = [L.build_pattern(11), L.build_pattern(12), L.build_pattern(13)]
    w1 = S.hebbian_ema(pats, 50, 0.05, seed=5)
    w2 = S.hebbian_ema(pats, 50, 0.05, seed=5)
    check("hebbian EMA deterministic", w1 == w2)
    check("hebbian EMA symmetric",
          all(abs(w1[i][j] - w1[j][i]) < 1e-12
              for i in range(L.N) for j in range(L.N)))


def test_crystallize_determinism_and_sanity():
    c1, s1, _ = S.crystallize_family(500, 60, 0.05, 600, k=2)
    c2, s2, _ = S.crystallize_family(500, 60, 0.05, 600, k=2)
    check("crystallize deterministic", c1 == c2 and s1 == s2)
    check("crystallized patterns are +-1",
          all(c in (-1, 1) for p in c1 for c in p))
    check("crystallized sanity in [0, 1]",
          all(0.0 <= s <= 1.0 for s in s1))


def test_split_disjointness():
    # The protocol's split separation is by construction: calibration
    # families/stream/probe seeds differ from evaluation ones, and
    # collect_split derives all RNG from those bases. Pin the bases.
    from lattice_crystallization.experiment import run_experiment
    import inspect
    src = inspect.getsource(run_experiment)
    check("calibration stream base 10000", "10000" in src)
    check("evaluation stream base 30000", "30000" in src)
    check("calibration probe base 20000", "20000" in src)
    check("evaluation probe base 40000", "40000" in src)
    check("bases all distinct",
          len({10000, 30000, 20000, 40000}) == 4)


def test_collect_split_tiny():
    # Smoke: one family per split, tiny stream; shapes only.
    out = collect_split([900], 20, 0.05, 10000, 20000)
    check("tiny split positives count", len(out["pos"][0]) == 2 * 7)
    check("tiny split sanity length", len(out["sanity"]) == 2)
    check("tiny split per-corruption lateral present",
          "seam_shift" in out["lat_by_corruption"]
          and "row_slip" in out["lat_by_corruption"])


def test_paired_bootstrap_identical():
    # Identical scores for both conditions: paired resampling must
    # give delta exactly 0 (independent resampling would not).
    pos = [0.9, 0.8, 0.7, 0.6]
    neg = [0.3, 0.2, 0.1, 0.0]
    lo, hi, mean = gate.bootstrap_delta_ci(pos, neg, 0.5, pos, neg, 0.5,
                                           reps=50, seed=1)
    check("paired bootstrap identical delta 0",
          lo == 0.0 and hi == 0.0 and mean == 0.0)


def main():
    tests = [test_lateral_identity, test_lateral_pair_count,
             test_lateral_all_plus, test_lateral_random_near_zero,
             test_lateral_skips_zeros, test_combined_weights,
             test_seam_shift_exact, test_row_slip_exact,
             test_flip_count_match, test_overlap_same_count_identical,
             test_calibrate_smallest_tie, test_wilson_bounds,
             test_power_iteration_diagonal, test_top_eigenvectors_order,
             test_hebbian_determinism, test_crystallize_determinism_and_sanity,
             test_split_disjointness, test_collect_split_tiny,
             test_paired_bootstrap_identical]
    for t in tests:
        t()
    print(f"{PASS} checks passed across {len(tests)} test functions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
