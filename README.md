# lattice-crystallization v0.1.0

**SYNTHETIC classical toy only.** This package applies a pattern-specific
lateral score to streaming-PCA ("crystallized") patterns and measures,
on held-out probes, whether the lateral term helps a frozen gate tell
structured corruption apart from scattered corruption. Nothing here is
biology, medicine, quantum hardware, a brain, or a consciousness claim:
the grid is an abstract 13×12 graph, the "patterns" are seeded ±1
vectors, and the "crystallization" is a Hebbian covariance estimate
settling, exactly as in the source repos.

## Provenance

- Lateral score, lattice geometry, seam_shift, row_slip, random_flips,
  overlap, and the smallest-threshold-max-J gate calibration are
  vendored-with-provenance from
  `sparkainlp-x/microtubule-lattice-memory`,
  `candidate/microtubule-lattice-memory/{lat.py, lattice.py, memory.py}`.
  The combined score `S = 0.5·overlap + 0.5·lateral` uses the weights
  fixed in that repo's PREREG.md, not tuned here.
- The streaming-PCA update rule `W ← (1−α)W + α|x⟩⟨x|` is the rule from
  `sparkainlp-x/membrane-crystallization`,
  `membrane_crystallization/stream.py` (Hebbian EMA; Oja 1982).
  Difference stated plainly: that repo extracts eigenvectors with a
  Jacobi solver on 8-dimensional complex patterns; here the patterns
  are 156-dimensional real ±1 lattice patterns and eigenvectors are
  extracted by power iteration with deflation, implemented and tested
  in this package. The extraction method is this package's own.
- Mun Seok Lee is cited as inspiration only for the Digital to Wave
  paradigm. He wrote no line of this package and has not reviewed or
  endorsed it.

## The lattice repo's own negative, kept in context

The lattice repo's own headline result is a failure, and this
experiment does not hide it: on 200 held-out pattern seeds, its
validation-calibrated combined gate reached TPR 0.982 but FPR 0.403
and Youden J 0.579, failing its own FPR ≤ 0.20 bar. Its generic
lateral recall is also falsified: with a generic lattice term,
λ ≥ 2 degrades even clean recall (0.929 vs 1.000 outer-product) and
the corruption-vs-random gap stays ~0. And the earlier
microtubule-crystallize Exp4 was negative for the same structural
reason: outer-product recall is permutation-equivariant, so it only
cares how many cells match, not which cells or where they sit.
This experiment tests the one design that survived that repo's
preregistration: the *pattern-specific* lateral score as a gate
term, not as recall.

## Experiment and preregistered bar

Stored patterns are crystallized patterns: per family, 3 lattice
patterns are streamed as noisy observations (each cell flipped with
probability 0.10, frequencies 0.6/0.3/0.1), the Hebbian EMA runs
(400 steps, α = 0.05), and the top-2 eigenvectors are binarized by
sign. Calibration families (seeds 1000, 1010, 1020, 1030) and
evaluation families (2000, 2010, 2020, 2030) are disjoint in family
seed, stream seed, and probe RNG seed. Nothing is calibrated on the
evaluation split.

Per stored pattern, 7 structured probes (1 seam_shift, 6 row_slips
over rows 3/6/9 × k 2/3), 7 scattered probes (same flip count,
matched 1:1), and 7 random patterns. Gate taus are calibrated per
score on the calibration split and frozen.

- Table A: structured vs random negatives (sanity: both scores
  should separate these easily).
- Table B: structured vs scattered negatives (the discriminating
  table: at equal flip counts the overlap distributions are
  identical by construction, so any separation in B must come from
  the lateral term).

**Bar, fixed before running:** the lateral term "helps" on table B
only if the held-out bootstrap 95% CI (2000 reps, seed 999, paired
resampling) for (J_combined − J_overlap) lies entirely above 0.05.
Otherwise the result is null. No corruption or score parameter was
tuned after seeing results.

## Measured results (demo.py, 13 s)

Sanity — crystallized patterns vs their family's true lattice
patterns (max |overlap|): calibration 0.974–1.000, evaluation
0.808–1.000. The stream did crystallize; the evaluation stored
patterns are not exact copies, which is the honest input to the
gate task.

| Table | Score | tau (frozen) | TPR | FPR | J | J 95% CI |
|---|---|---|---|---|---|---|
| A structured vs random | overlap-only | 0.4359 | 1.000 | 0.000 | 1.000 | (1.000, 1.000) |
| A structured vs random | combined | 0.4019 | 1.000 | 0.000 | 1.000 | (1.000, 1.000) |
| B structured vs scattered | overlap-only | 0.7436 | 1.000 | 1.000 | 0.000 | (0.000, 0.000) |
| B structured vs scattered | combined | 0.8963 | 0.554 | 0.250 | 0.304 | (0.125, 0.464) |

Delta J (combined − overlap-only), held-out: table A 0.000
(0.000, 0.000); **table B 0.307 (0.125, 0.464)**.

**Verdict: the primary bar is met** — the table B delta CI lies
entirely above 0.05, so the lateral term genuinely helps separate
structured from scattered corruption at equal flip counts, on
fresh crystallized patterns the gate never saw. But the absolute
gate is still weak, and that is reported alongside, not buried:
combined J is only 0.304, with FPR 0.250 — it would still fail the
lattice repo's own L2-style bar (J > 0.50, FPR ≤ 0.20). The lateral
term is a real signal, not a good gate yet.

Lateral-score gaps on the evaluation split (structured minus
scattered mean lateral): seam_shift 0.140 (n = 8, above the
lattice repo's 0.10 gap bar); row_slip 0.031 (n = 48, small — the
slip changes only 4–6 cells, so most neighbour products survive in
both the structured and the scattered probe, exactly the small-gap
case the lattice PREREG anticipated for row slip). Overall table B
gap 0.046; table A gap 0.856 (random patterns sit at lateral ~0).

The overlap-only table B result (J = 0.000, TPR = FPR = 1.000) is
not a failure of overlap: with flip counts matched exactly, the
overlap distributions are identical, so no threshold can separate
them. That is the point of the control.

## Run it

```sh
python3 tests/tests.py   # 42 checks across 19 test functions
python3 demo.py          # the J tables above, as JSON
```

Tests pin: the lateral computation (including the 299-pair count
and an exact one-flip value), seam_shift and row_slip exact
outputs, scattered flip-count matching, overlap equality at equal
flip counts, the smallest-threshold tie rule, Wilson bounds, power
iteration on a known diagonal matrix, Hebbian determinism and
symmetry, crystallization determinism, split disjointness by seed
base, and paired-bootstrap correctness (identical scores give
delta exactly 0).

## Limits

- One grid size (13×12), one noise level for the stream (flip
  probability 0.10), one EMA setting (400 steps, α = 0.05), two
  corruption families, 8 stored patterns per split (56 structured
  probes per split). Generalization beyond these settings is
  untested.
- The crystallized patterns are binarized eigenvectors, not the
  lattice patterns themselves; evaluation sanity 0.808–1.000 means
  some stored patterns are imperfect copies, which is part of the
  task, not a defect to tune away.
- The combined weights (0.5/0.5) are fixed from the lattice PREREG.
  Tuning them on the evaluation split would invalidate the bar.
- No biology, no medicine, no quantum hardware, no brain or
  consciousness claim. A graph score on a seeded grid is not
  evidence about microtubules, cells, or brains.
