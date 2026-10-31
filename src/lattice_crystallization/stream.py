"""Streaming-PCA pattern generation.

The update rule is the streaming-PCA rule from
sparkainlp-x/membrane-crystallization,
membrane_crystallization/stream.py (hebbian_update): an
exponentially weighted running estimate W <- (1 - alpha) W +
alpha |x><x| of the observation covariance, whose top eigenvectors
are its principal components (Oja 1982). "Crystallization" is the
crystallization repo's descriptive label for that estimate
settling, nothing more.

Difference stated plainly: the crystallization repo extracts
eigenvectors with a Jacobi solver (linalg.eigh_hermitian) on
8-dimensional complex patterns. Here the patterns are 156-dim
real +-1 lattice patterns, and eigenvectors are extracted by
power iteration with deflation, implemented and tested in this
package. The extraction method is this package's own; the update
rule is the shared streaming-PCA rule.

Pipeline per family: build 3 lattice patterns (distinct seeds),
stream noisy observations (each cell flipped with probability
noise_p, pattern frequencies 0.6/0.3/0.1 as in the crystallization
repo's sweeps), run the Hebbian EMA, extract the top-k
eigenvectors, binarize by sign (0 -> +1). The binarized vectors
are the crystallized patterns used as stored patterns by the
experiment. Sanity: each crystallized pattern's max |overlap|
against the family's true lattice patterns is reported; a low
sanity value means the stream did not crystallize, and the run
must say so rather than proceed silently.
"""

import math
import random

from .lattice import N, build_pattern


def noisy_flip(pattern, p, rng):
    return [(-c if rng.random() < p else c) for c in pattern]


def hebbian_ema(patterns, steps, alpha, seed, freqs=(0.6, 0.3, 0.1),
                noise_p=0.10):
    """Run the Hebbian EMA stream. Returns the covariance estimate W
    (list of lists, real, symmetric)."""
    rng = random.Random(seed)
    cum = []
    s = 0.0
    for f in freqs:
        s += f
        cum.append(s)
    W = [[0.0] * N for _ in range(N)]
    b = 1.0 - alpha
    for _ in range(steps):
        r = rng.random() * s
        i = next((i for i, c in enumerate(cum) if r < c), len(cum) - 1)
        x = noisy_flip(patterns[i], noise_p, rng)
        for a in range(N):
            xa = alpha * x[a]
            Wa = W[a]
            for c in range(N):
                Wa[c] = b * Wa[c] + xa * x[c]
    return W


def matvec(M, v):
    return [sum(row[j] * v[j] for j in range(len(v))) for row in M]


def norm(v):
    return math.sqrt(sum(c * c for c in v))


def power_iteration(M, seed, tol=1e-10, max_iter=500):
    """Top eigenvector/eigenvalue of a real symmetric matrix by
    power iteration. Deterministic start from seed. Returns (vec, val)."""
    rng = random.Random(seed)
    v = [rng.choice((-1.0, 1.0)) for _ in range(len(M))]
    nv = norm(v)
    v = [c / nv for c in v]
    lam = 0.0
    for _ in range(max_iter):
        w = matvec(M, v)
        nw = norm(w)
        if nw == 0.0:
            break
        w = [c / nw for c in w]
        new_lam = sum(a * b for a, b in zip(w, matvec(M, w)))
        shift = max(abs(a - b) for a, b in zip(w, v))
        v, lam = w, new_lam
        if shift < tol:
            break
    return v, lam


def top_eigenvectors(W, k, seed):
    """Top-k eigenvectors by power iteration with deflation.
    Returns a list of (vec, val) in descending eigenvalue order."""
    M = [row[:] for row in W]
    out = []
    for j in range(k):
        v, lam = power_iteration(M, seed=seed * 1009 + j)
        out.append((v, lam))
        for a in range(len(M)):
            for c in range(len(M)):
                M[a][c] -= lam * v[a] * v[c]
    return out


def binarize(vec):
    return [1 if c >= 0 else -1 for c in vec]


def crystallize_family(family_seed, steps, alpha, stream_seed,
                       freqs=(0.6, 0.3, 0.1), noise_p=0.10, k=2):
    """One family: 3 lattice patterns -> stream -> top-k crystallized
    patterns. Returns (crystallized, sanity, true_patterns) where
    sanity[i] is the max |overlap| of crystallized[i] against the
    family's true lattice patterns."""
    from .gate import overlap
    true_patterns = [build_pattern(family_seed + i) for i in range(3)]
    W = hebbian_ema(true_patterns, steps, alpha, stream_seed,
                    freqs=freqs, noise_p=noise_p)
    eigs = top_eigenvectors(W, k, seed=stream_seed)
    crystallized = [binarize(v) for v, _ in eigs]
    sanity = []
    for cp in crystallized:
        sanity.append(max(abs(overlap(cp, tp)) for tp in true_patterns))
    return crystallized, sanity, true_patterns
