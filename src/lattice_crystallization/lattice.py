"""Lattice geometry and corruptions.

Vendored-with-provenance from sparkainlp-x/microtubule-lattice-memory,
candidate/microtubule-lattice-memory/lattice.py (geometry, pattern
construction, seam_shift, row_slip, random_flips, count_flips,
random_pattern). Adapted only by removing the 12x12 cropped-game
helpers and the sonification helper, which this experiment does not
use. The lattice.py docstring's analogy labels ("straight / GTP-like")
are analogy only; nothing here is biology.

Geometry: 13 rows x 12 columns, row-major, N = 156. Row 12 neighbours
row 0 across the seam. Cell state is +-1.
"""

import random

ROWS = 13
COLS = 12
N = ROWS * COLS


def idx(row, col):
    return row * COLS + col


def build_pattern(seed=42):
    """Deterministic lattice pattern. Returns a list of +-1, len 156."""
    rng = random.Random(seed)
    rows = []
    row = [rng.choice((-1, 1)) for _ in range(COLS)]
    rows.append(row[:])
    for _ in range(ROWS - 1):
        nxt = [(-c if rng.random() < 0.15 else c) for c in row]
        rows.append(nxt)
        row = nxt
    # Seam: row 12 is the inverse of row 0 blended with its own chain
    # row (85% inverted), so seam neighbours mostly disagree.
    rows[12] = [(-c if rng.random() < 0.85 else c) for c in rows[0]]
    out = []
    for r in rows:
        out.extend(r)
    return out


def rows_of(x, rows=ROWS):
    return [x[r * COLS:(r + 1) * COLS] for r in range(rows)]


def count_flips(a, b):
    return sum(1 for p, q in zip(a, b) if p != q)


def random_flips(x, k, rng):
    """Flip exactly k distinct cells, chosen at random."""
    out = list(x)
    for i in rng.sample(range(len(x)), k):
        out[i] = -out[i]
    return out


def random_pattern(rng):
    return [rng.choice((-1, 1)) for _ in range(N)]


def seam_shift(x):
    """Move the seam: un-invert row 12, invert row 5 instead.

    Verbatim from lattice.py, including its `(-c if False else c)`
    row-12 copy. Changes exactly 2 rows (24 cells) if row 12 was
    ~85% inverted.
    """
    out = list(x)
    rows = rows_of(out)
    rows[12] = [(-c if False else c) for c in rows[11]]
    rows[5] = [-c for c in rows[5]]
    flat = []
    for r in rows:
        flat.extend(r)
    return flat


def row_slip(x, row=6, k=3):
    """Cyclically shift one row by k columns."""
    rows = rows_of(x)
    r = rows[row]
    rows[row] = r[-k:] + r[:-k]
    flat = []
    for rr in rows:
        flat.extend(rr)
    return flat
