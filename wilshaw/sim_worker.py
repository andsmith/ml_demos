"""
Multiprocessing worker for the Monte-Carlo Willshaw simulation.

The single entry point :func:`run_chunk` takes one plain-dict "work" item
(only ints / strings / floats, so it pickles cheaply) and returns aggregated
recovery statistics.  It must stay importable at module top level with no GUI
imports so it can be sent to a spawned Pool worker on Windows.

A "chunk" builds ``n_sets`` independent networks and runs ``n_trials`` trials on
each.  Each trial damages every stored symbol with fresh uniform-random noise
(inhibition and excitation fractions drawn in ``[0, rate]``) -- the same damage
model as the Demonstration tab -- recovers it, scores it, and bins the result.
"""

import os
# Keep BLAS single-threaded in workers: we parallelise across processes, so
# per-process thread pools only waste memory (and on Windows the large virtual
# reservation per process can exhaust the page file when many workers start).
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np

import willshaw_core as core


def run_chunk(work):
    """
    Run one chunk of the Monte-Carlo simulation.

    :param work: dict with keys
        D, S, V, theta, mode, max_iter,
        inh_rate, exc_rate  -- ceilings for the uniform-random damage,
        n_sets              -- number of independent networks to build,
        n_trials            -- trials per network,
        net_seed            -- base seed for pattern-set/network generation,
        dmg_seed            -- seed for the damage RNG.
    :return: dict with 'bin_counts', 'bin_inc_sum' (length core.N_BINS),
        'total', 'perfect', 'iter_sum'.
    """
    D, S, V = work['D'], work['S'], work['V']
    theta, mode, mx = work['theta'], work['mode'], work['max_iter']
    inh_r, exc_r = work['inh_rate'], work['exc_rate']
    n_sets, n_trials = work['n_sets'], work['n_trials']

    dmg_rng = np.random.default_rng(work['dmg_seed'])
    counts = np.zeros(core.N_BINS, dtype=np.int64)
    inc_sum = np.zeros(core.N_BINS, dtype=np.float64)
    total = 0
    perfect = 0
    iter_sum = 0

    for s in range(n_sets):
        net_rng = np.random.default_rng(work['net_seed'] + s)
        patterns = core.make_patterns(D, S, V, net_rng)
        W = core.build_W(patterns, D)
        for _ in range(n_trials):
            for mu in range(V):
                tgt = patterns[mu]
                inh = dmg_rng.uniform(0.0, inh_r) if inh_r > 0 else 0.0
                exc = dmg_rng.uniform(0.0, exc_r) if exc_r > 0 else 0.0
                dmg = core.damage(tgt, inh, exc, D, dmg_rng)
                res = core.recover(W, dmg['cue'], tgt, theta, mode, S, mx)
                final = res['states'][-1]
                rec, inc = core.score(final, tgt, S)
                b = core.bin_index(rec)
                counts[b] += 1
                inc_sum[b] += inc
                total += 1
                iter_sum += res['iters']
                if rec >= 0.999 and inc < 1e-9:
                    perfect += 1
        del W

    return {'bin_counts': counts.tolist(),
            'bin_inc_sum': inc_sum.tolist(),
            'total': int(total),
            'perfect': int(perfect),
            'iter_sum': int(iter_sum)}


def _test_worker():
    work = {'D': 200, 'S': 8, 'V': 100, 'theta': 4, 'mode': 'fixed',
            'max_iter': 10, 'inh_rate': 0.3, 'exc_rate': 0.3,
            'n_sets': 2, 'n_trials': 1, 'net_seed': 42, 'dmg_seed': 7}
    r = run_chunk(work)
    assert r['total'] == 200
    assert sum(r['bin_counts']) == 200
    print("sim_worker OK:", r['total'], "symbols,", r['perfect'], "perfect,",
          "bins=", r['bin_counts'])


if __name__ == '__main__':
    _test_worker()
