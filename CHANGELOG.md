# CHANGELOG — lattice-crystallization

## v0.1.0

- Initial release: lateral score and lattice geometry/corruptions
  vendored-with-provenance from microtubule-lattice-memory
  (candidate lat.py, lattice.py, memory.py); streaming-PCA update
  rule from membrane-crystallization (stream.py); eigenvector
  extraction by power iteration with deflation (this package's own).
- Held-out frozen-gate experiment: calibration families
  1000/1010/1020/1030, evaluation families 2000/2010/2020/2030;
  tables A (vs random) and B (vs scattered, discriminating).
- Measured: table B delta J 0.307, bootstrap 95% CI (0.125, 0.464)
  — primary bar met; absolute combined J 0.304, FPR 0.250 —
  still below the lattice repo's L2-style bar, reported plainly.
- Bug caught and fixed before delivery: the first bootstrap delta
  CI resampled the two scores independently, breaking the paired
  structure it claimed; fixed to joint index resampling and pinned
  by a test (identical scores give delta exactly 0).
