"""
Willshaw feature-attractor engine (pure numpy, no GUI).

This module is the shared numerical core used by both the Demonstration and the
Simulation tabs (and by the multiprocessing workers).  It knows nothing about
Tkinter or drawing -- it only generates sparse patterns, builds the Willshaw
connectivity matrix, runs iterative recall, damages cues and scores recoveries.

A "pattern" / "symbol" is a binary vector of exactly ``S`` active units in a
``D``-dimensional space.  Patterns are represented compactly as a sorted numpy
array of their active indices (dtype int).  The Willshaw matrix is

    W_ij = OR over stored patterns mu of  x_i^mu * x_j^mu ,

i.e. a connection exists between i and j iff they were co-active in any stored
symbol.  Recall computes the analog excitation ``h = W x`` and thresholds it,
iterating until the state stops changing (see :func:`recover`).

References:  thresholds.md (in this folder) and
http://www.scholarpedia.org/article/Confabulation_theory_(computational_intelligence)
"""

import logging

import numpy as np

# Recovery-rate histogram bins (percent).  The last bin (index 8) is the
# "perfect" bucket for symbols recovered 100% correctly; the eight preceding
# bins are the half-open intervals between successive edges.
BIN_EDGES = [0, 10, 25, 50, 60, 70, 80, 90, 100]
BIN_LABELS = ['0-10', '10-25', '25-50', '50-60', '60-70',
              '70-80', '80-90', '90-99', '100']
N_BINS = len(BIN_LABELS)  # 9

DEFAULT_MAX_ITER = 10


# --------------------------------------------------------------------------- #
#  Pattern set and connectivity matrix
# --------------------------------------------------------------------------- #
def make_patterns(D, S, V, rng):
    """
    Generate ``V`` random sparse patterns.

    :param D: dimensionality (number of neurons)
    :param S: sparsity -- number of active units per pattern (1 <= S <= D)
    :param V: number of patterns (vocabulary size)
    :param rng: a numpy Generator (``np.random.default_rng(...)``)
    :return: int array of shape (V, S); each row holds that pattern's sorted
        active indices
    """
    S = int(max(1, min(S, D)))
    patterns = np.empty((V, S), dtype=np.int64)
    for mu in range(V):
        patterns[mu] = np.sort(rng.choice(D, size=S, replace=False))
    return patterns


def build_W(patterns, D):
    """
    Build the binary Willshaw connectivity matrix by OR-ing the outer product
    of every pattern with itself.

    :param patterns: (V, S) int array of active indices
    :param D: dimensionality
    :return: (D, D) bool matrix (symmetric, ones on the co-activation graph)
    """
    W = np.zeros((D, D), dtype=bool)
    for ix in patterns:
        W[np.ix_(ix, ix)] = True
    return W


def matrix_density(W):
    """Fraction of entries of ``W`` that are set (empirical connection density)."""
    return float(W.mean()) if W.size else 0.0


# --------------------------------------------------------------------------- #
#  Recall
# --------------------------------------------------------------------------- #
def excitation(W, active_idx):
    """
    Analog excitation ``h = W x`` for a sparse state, computed by summing only
    the columns of ``W`` that correspond to active units (avoids a dense
    matrix-vector product so it scales to large D).

    :param W: (D, D) bool connectivity matrix
    :param active_idx: 1-D int array of currently-active unit indices
    :return: (D,) int array of excitations
    """
    D = W.shape[0]
    if len(active_idx) == 0:
        return np.zeros(D, dtype=np.int32)
    return W[:, active_idx].sum(axis=1).astype(np.int32)


def threshold_state(h, theta, mode, S):
    """
    Turn an excitation vector into the next binary state.

    :param h: (D,) int excitation
    :param theta: fixed threshold (used when ``mode == 'fixed'``)
    :param mode: 'fixed' -> keep units with ``h >= theta``;
                 'wta'   -> winner-take-all, keep the ``S`` most-excited units
    :param S: target sparsity (number of winners for WTA)
    :return: sorted int array of the newly-active indices
    """
    if mode == 'wta':
        S = int(max(1, S))
        if S >= len(h):
            idx = np.where(h > 0)[0]
            return np.sort(idx)
        # Top-S by excitation; drop any that received zero input (meaningless
        # winners when fewer than S units are excited at all).
        idx = np.argpartition(h, -S)[-S:]
        idx = idx[h[idx] > 0]
        return np.sort(idx)
    # fixed threshold
    theta = max(1, int(round(theta)))
    return np.sort(np.where(h >= theta)[0])


def recover(W, cue_idx, target_idx, theta, mode, S, max_iter=DEFAULT_MAX_ITER):
    """
    Iteratively recall from a (possibly damaged) cue until the state stops
    changing, cycles, or ``max_iter`` is reached.

    :param W: (D, D) bool connectivity matrix
    :param cue_idx: int array of active indices of the starting cue
    :param target_idx: int array of active indices of the true stored pattern
    :param theta: fixed threshold
    :param mode: 'fixed' or 'wta'
    :param S: target sparsity
    :param max_iter: maximum number of recall iterations
    :return: dict with
        'states'      -> list of index arrays [cue, out_1, out_2, ...] (the
                         initial cue followed by each iteration's output),
        'excitations' -> list of (D,) int arrays; excitations[i] is the
                         excitation produced by states[i] (so it aligns with
                         the transition states[i] -> states[i+1]),
        'iters'       -> number of iterations actually run,
        'converged'   -> True if a fixed point equal-to or short-of max_iter
                         was reached (state stopped changing).
    """
    cue_idx = np.asarray(cue_idx, dtype=np.int64)
    states = [np.sort(cue_idx)]
    excitations = []
    seen = [frozenset(cue_idx.tolist())]
    converged = False
    iters = 0

    for it in range(max_iter):
        cur = states[-1]
        h = excitation(W, cur)
        excitations.append(h)
        new_idx = threshold_state(h, theta, mode, S)
        states.append(new_idx)
        iters = it + 1
        new_set = frozenset(new_idx.tolist())
        if new_set == seen[-1]:          # fixed point
            converged = True
            break
        if new_set in seen:              # entered a cycle
            break
        seen.append(new_set)

    return {'states': states, 'excitations': excitations,
            'iters': iters, 'converged': converged}


def to_binary(idx, D):
    """Expand a sorted index array into a dense (D,) uint8 binary vector."""
    v = np.zeros(D, dtype=np.uint8)
    if len(idx):
        v[np.asarray(idx, dtype=np.int64)] = 1
    return v


# --------------------------------------------------------------------------- #
#  Damage model  (shared by the Demo tab and the Simulation)
# --------------------------------------------------------------------------- #
def damage(target_idx, inh_frac, exc_frac, D, rng):
    """
    Corrupt a stored pattern into a noisy cue.

    Inhibition noise turns OFF a fraction of the pattern's valid activations
    ("missing"); excitation noise turns ON random non-pattern units
    ("spurious").

    :param target_idx: int array of the true pattern's active indices
    :param inh_frac: fraction of true units to turn off (0..1)
    :param exc_frac: fraction (of S) of spurious units to turn on (0..1)
    :param D: dimensionality
    :param rng: numpy Generator
    :return: dict with sorted index arrays 'cue', 'correct' (kept-on true
        units), 'missing' (turned-off true units) and 'spurious'
        (turned-on false units)
    """
    target_idx = np.asarray(target_idx, dtype=np.int64)
    S = len(target_idx)
    n_off = int(round(inh_frac * S))
    n_on = int(round(exc_frac * S))

    # Inhibition: choose which true units go missing.
    if n_off > 0:
        off_pos = rng.choice(S, size=min(n_off, S), replace=False)
        missing_mask = np.zeros(S, dtype=bool)
        missing_mask[off_pos] = True
        missing = target_idx[missing_mask]
        correct = target_idx[~missing_mask]
    else:
        missing = np.empty(0, dtype=np.int64)
        correct = target_idx.copy()

    # Excitation: choose spurious non-pattern units to switch on.
    if n_on > 0:
        non_pattern = np.setdiff1d(np.arange(D, dtype=np.int64), target_idx,
                                   assume_unique=True)
        n_on = min(n_on, len(non_pattern))
        spurious = rng.choice(non_pattern, size=n_on, replace=False) \
            if n_on > 0 else np.empty(0, dtype=np.int64)
    else:
        spurious = np.empty(0, dtype=np.int64)

    cue = np.sort(np.concatenate([correct, spurious]).astype(np.int64))
    return {'cue': cue,
            'correct': np.sort(correct),
            'missing': np.sort(missing),
            'spurious': np.sort(spurious)}


# --------------------------------------------------------------------------- #
#  Theory helpers  (see thresholds.md)
# --------------------------------------------------------------------------- #
def p_density(D, S, V):
    """
    Theoretical connection density  p = 1 - exp(-V S^2 / D^2)  -- the expected
    fraction of ones in the Willshaw matrix.
    """
    if D <= 0:
        return 0.0
    return float(1.0 - np.exp(-V * (S ** 2) / (D ** 2)))


def theta_midpoint(c, p):
    """
    Midpoint threshold  theta = c (1 + p) / 2  , halfway between the expected
    true activation ``c`` and the expected false activation ``c p``.

    :param c: number of correct active neurons in the cue
    :param p: connection density (see :func:`p_density`)
    """
    return c * (1.0 + p) / 2.0


def theta_statistical(c, p, alpha=3.0):
    """
    Conservative threshold  theta = c p + alpha * sqrt(c p (1 - p))  -- above
    the upper tail of the false-activation (Binomial) distribution.
    """
    return c * p + alpha * np.sqrt(max(c * p * (1.0 - p), 0.0))


def optimal_sparsity(D):
    """
    Capacity-optimal sparsity.  Willshaw storage capacity peaks when the matrix
    is about half full, which for random sparse codes happens at

        S* ~ log2(D)

    (equivalently k ~ log2 N).  Returned rounded and clamped to >= 1.
    """
    return int(max(1, round(np.log2(max(D, 2)))))


# --------------------------------------------------------------------------- #
#  Scoring  (Simulation)
# --------------------------------------------------------------------------- #
def score(final_idx, target_idx, S):
    """
    Score a recovered state against the true pattern.

    :return: (recovery, incorrect_rate) where
        recovery       = |final ∩ target| / S  (fraction of true units on),
        incorrect_rate = |final \\ target| / S  (spurious units, per S).
    """
    fset = set(np.asarray(final_idx).tolist())
    tset = set(np.asarray(target_idx).tolist())
    if S <= 0:
        return 0.0, 0.0
    correct = len(fset & tset)
    incorrect = len(fset - tset)
    return correct / S, incorrect / S


def bin_index(recovery):
    """
    Map a recovery fraction (0..1) to a histogram bin index (0..N_BINS-1).
    Exactly-perfect recovery lands in the final "100" bin.
    """
    pct = recovery * 100.0
    if pct >= 100.0:
        return N_BINS - 1
    for i in range(len(BIN_EDGES) - 1):
        if BIN_EDGES[i] <= pct < BIN_EDGES[i + 1]:
            return i
    return 0  # pct < 0 shouldn't happen; guard anyway


# --------------------------------------------------------------------------- #
#  Sanity tests
# --------------------------------------------------------------------------- #
def _test_core():
    rng = np.random.default_rng(0)

    # 1) Theory helpers match the thresholds.md worked example.
    p = p_density(1000, 20, 300)
    assert abs(p - 0.1131) < 1e-3, p
    assert abs(theta_midpoint(10, p) - 5.565) < 1e-2
    assert optimal_sparsity(1000) == 10
    logging.info("theory helpers OK (p=%.4f, theta_mid=%.3f, S*=%d)",
                 p, theta_midpoint(10, p), optimal_sparsity(1000))

    # 2) A clean stored pattern is a fixed point.
    D, S, V = 200, 8, 60
    patterns = make_patterns(D, S, V, rng)
    W = build_W(patterns, D)
    tgt = patterns[0]
    # Fixed threshold at S recovers a stored pattern exactly (each true unit is
    # co-active with all S cue units, including itself).
    res = recover(W, tgt, tgt, theta=S, mode='fixed', S=S)
    final = res['states'][-1]
    rec, inc = score(final, tgt, S)
    assert rec == 1.0, (rec, inc)
    assert res['converged']
    logging.info("clean pattern is a fixed point (iters=%d, inc=%.3f)",
                 res['iters'], inc)

    # 3) A lightly-damaged cue recovers to its target.
    n_ok = 0
    for mu in range(20):
        tgt = patterns[mu]
        dmg = damage(tgt, inh_frac=0.25, exc_frac=0.25, D=D, rng=rng)
        theta = round(theta_midpoint(len(dmg['correct']), p_density(D, S, V)))
        res = recover(W, dmg['cue'], tgt, theta=max(1, theta), mode='fixed', S=S)
        rec, inc = score(res['states'][-1], tgt, S)
        if rec >= 0.9:
            n_ok += 1
    logging.info("damaged-cue recovery >=90%%: %d/20 (fixed theta)", n_ok)
    assert n_ok >= 15, n_ok

    # 4) WTA mode returns exactly-S winners on a damaged cue.
    dmg = damage(patterns[0], 0.25, 0.25, D, rng)
    res = recover(W, dmg['cue'], patterns[0], theta=0, mode='wta', S=S)
    assert len(res['states'][-1]) <= S
    rec, inc = score(res['states'][-1], patterns[0], S)
    logging.info("WTA recovery of a damaged cue: rec=%.2f inc=%.2f", rec, inc)

    # 5) Binning edges.
    assert bin_index(1.0) == N_BINS - 1
    assert bin_index(0.95) == 7
    assert bin_index(0.0) == 0
    assert bin_index(0.55) == 3
    logging.info("all core sanity tests passed.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s %(message)s')
    _test_core()
