"""Pattern-specific lateral score.

Vendored-with-provenance from sparkainlp-x/microtubule-lattice-memory,
candidate/microtubule-lattice-memory/lat.py (neighbour_pairs,
lateral_score, combined_score), unchanged in definition.

Neighbour pairs are the lattice edges: vertical pairs row r -> r+1
for r = 0..11, the seam pair row 12 -> row 0, and horizontal pairs
col c -> c+1 for c = 0..10. Total 156 + 143 = 299 pairs.

For a stored pattern `pat` (+-1) and a probe `x` (+-1, or 0 for
unknown cells, which are skipped):

  lat(x) = mean over pairs (i, j) of (pat_i * pat_j) * (x_i * x_j)

For x = pat, lat = 1.0 by construction. For a random pattern,
lat ~ 0. For a probe whose flips are structured (a whole row
inverted, a seam moved), many neighbour products are preserved,
so lat falls slower than for scattered flips at the same flip
count. That difference is the lattice-specific signal, and it is
measured, not assumed. This score does NOT use any biology.
"""

from .lattice import COLS, ROWS, idx


def neighbour_pairs():
    pairs = []
    for r in range(ROWS):
        for c in range(COLS):
            i = idx(r, c)
            if r < ROWS - 1:
                pairs.append((i, idx(r + 1, c)))
            else:
                pairs.append((i, idx(0, c)))  # seam edge
            if c < COLS - 1:
                pairs.append((i, idx(r, c + 1)))
    return pairs


PAIRS = neighbour_pairs()


def lateral_score(pat, x):
    vals = []
    for i, j in PAIRS:
        if x[i] == 0 or x[j] == 0:
            continue
        vals.append((pat[i] * pat[j]) * (x[i] * x[j]))
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def combined_score(pat, x, overlap_fn):
    """S = 0.5 * overlap + 0.5 * lateral_score, weights fixed in
    the lattice repo's PREREG.md, not tuned here."""
    return 0.5 * overlap_fn(x, pat) + 0.5 * lateral_score(pat, x)
