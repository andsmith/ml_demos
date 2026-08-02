"""
High-level (symbol/probability level, not neurosimulation) confabulation
inference: given up to 5 known words, rank candidates for the 6th word
(absolute mode, fixed sentence positions) or the next word (relative mode,
position-invariant), using the knowledge bases written by
``extract_knowledge_base.py``.

Per ``willshaw.md`` / ``willshaw_dynamics.md``: for each known ("locked
down") neighbor word, its knowledge-base row gives a probability vector
over the target module's vocabulary. Those vectors are combined
(multiplied element-wise, i.e. log-added) across all available neighbors to
get the target module's excitation distribution; the attractor phase here
is simply keeping the top (or top-N) supported word(s).

Also supports **joint** inference of two adjacent unknown words at once
(e.g. the 5th and 6th together, ``--joint``): each unit keeps a
partly-activated top-N candidate list (``N=16`` by default) instead of a
single locked symbol, and a neighbor that is itself still uncertain
contributes a *weighted mixture* over its current candidates (each
candidate's activation weight is a multiplicative weight on the excitation
it sends). The two lists are winnowed synchronously each iteration -- both
recompute using each other's current list, then both are truncated to at
most half their previous size -- until they converge to a single symbol
each. This requires a *reverse* knowledge base (unit 6 back to unit 5, or
relative offset -1) so the later unit can excite the earlier one too; see
``extract_tiny_kbs.sh``.

Usage::

    python confabulation_high_level.py once upon a time there
    python confabulation_high_level.py --relative --top 3 a little boy
    python confabulation_high_level.py --joint once upon a time
    python confabulation_high_level.py --joint --relative a little
    python confabulation_high_level.py --selftest
"""

import argparse
import glob
import math
import os
import re

import numpy as np

import nlp_core as core
from extract_knowledge_base import extract_absolute, extract_relative

_ABS_RE = re.compile(r"kb_abs_(\d+)-(\d+)\.npz$")
_REL_RE = re.compile(r"kb_rel_(-?\d+)\.npz$")
_EPS = 1e-12


# --------------------------------------------------------------------------- #
#  Loading
# --------------------------------------------------------------------------- #
def load_kb_set(mode, kb_dir=core.KB_DIR):
    """
    Load every knowledge base for a mode from ``kb_dir``.

    :param mode: ``"absolute"`` or ``"relative"``
    :param kb_dir: directory containing ``.npz`` files from
        :mod:`extract_knowledge_base`
    :return: ``{(i, j): kb_dict}`` (absolute) or ``{offset: kb_dict}``
        (relative), where ``kb_dict`` is :func:`nlp_core.load_kb`'s return
        value. Raises ``ValueError`` if the loaded KBs don't share an
        identical vocabulary (they must, to combine rows across KBs).
    """
    pattern, regex = {
        "absolute": ("kb_abs_*-*.npz", _ABS_RE),
        "relative": ("kb_rel_*.npz", _REL_RE),
    }[mode]

    kb_set = {}
    shared_vocab = None
    shared_vocab_path = None
    for path in sorted(glob.glob(os.path.join(kb_dir, pattern))):
        m = regex.search(os.path.basename(path))
        if not m:
            continue
        key = (int(m.group(1)), int(m.group(2))) if mode == "absolute" else int(m.group(1))
        kb = core.load_kb(path)
        if shared_vocab is None:
            shared_vocab = kb["vocab"]
            shared_vocab_path = path
        elif kb["vocab"] != shared_vocab:
            raise ValueError(
                f"vocab mismatch between {shared_vocab_path} and {path} -- "
                f"re-run extract_tiny_kbs.sh so all knowledge bases are built "
                f"from the same corpus/vocab")
        kb_set[key] = kb
    return kb_set


# --------------------------------------------------------------------------- #
#  Shared combination machinery
# --------------------------------------------------------------------------- #
def _kb_key(mode, k, n_known, pos, rel_slot=1):
    """
    Knowledge-base key linking known word ``k`` (0-indexed) to a target.

    :param mode: ``"absolute"`` or ``"relative"``
    :param k: 0-indexed position of the known word within ``known_words``
    :param n_known: total number of known words
    :param pos: absolute mode -- the 1-indexed target sentence position
    :param rel_slot: relative mode -- how many words after the last known
        word the target sits (1 = immediately next, 2 = the one after
        that, ...); ``pos`` is unused in relative mode
    :return: an absolute ``(i, j)`` tuple or a relative ``int`` offset,
        suitable as a key into a :func:`load_kb_set` result
    """
    if mode == "absolute":
        return (k + 1, pos)
    return n_known - k + (rel_slot - 1)


def _weighted_row(kb, index, candidates):
    """
    Mix a knowledge-base's rows over an uncertain neighbor's candidates.

    :param kb: a loaded KB dict (as from :func:`nlp_core.load_kb`)
    :param index: ``{token: vocab position}`` map (shared across KBs)
    :param candidates: list of ``(word, weight)`` -- a known fixed word is
        just ``[(word, 1.0)]``; an uncertain neighbor passes its current
        ranked candidate list
    :return: the weight-mixed probability row (V,), or ``None`` if none of
        the candidates are in the vocabulary
    """
    row = None
    for word, weight in candidates:
        wi = index.get(word)
        if wi is None:
            continue
        contribution = weight * kb["probs"][wi]
        row = contribution if row is None else row + contribution
    return row


def _combine_log_excitation(items, kb_set, index):
    """
    Log-add the (mixed) rows from every available neighbor.

    :param items: list of ``(kb_key, candidates)`` -- see
        :func:`_weighted_row` for ``candidates``
    :param kb_set: result of :func:`load_kb_set`
    :param index: ``{token: vocab position}`` map
    :return: ``(log_excitation, n_contrib)`` -- ``log_excitation`` is a
        ``(V,)`` array (zeros if ``n_contrib == 0``)
    """
    V = len(next(iter(kb_set.values()))["vocab"])
    log_excitation = np.zeros(V, dtype=np.float64)
    n_contrib = 0
    for kb_key, candidates in items:
        kb = kb_set.get(kb_key)
        if kb is None:
            print(f"[confab] warning: no KB for neighbor {kb_key!r} -- skipping")
            continue
        row = _weighted_row(kb, index, candidates)
        if row is None:
            print(f"[confab] warning: none of {[c for c, _ in candidates]!r} "
                  f"in vocabulary -- skipping")
            continue
        log_excitation += np.log(row + _EPS)
        n_contrib += 1
    return log_excitation, n_contrib


def _finalize_weights(log_excitation, n_contrib, kb_set, vocab):
    """
    Turn combined log-excitation into a normalized weight vector, falling
    back to the corpus-frequency prior when there was no usable evidence.
    """
    V = len(vocab)
    if n_contrib == 0:
        print("[confab] warning: no usable evidence -- falling back to "
              "corpus-frequency prior")
        target_counts = np.zeros(V, dtype=np.float64)
        for kb in kb_set.values():
            target_counts += kb["counts"].sum(axis=0)
        return (target_counts / target_counts.sum() if target_counts.sum() > 0
                else np.full(V, 1.0 / V))
    log_excitation = log_excitation - log_excitation.max()
    weights = np.exp(log_excitation)
    weights /= weights.sum()
    return weights


def _topn(vocab, weights, n):
    """Top-``n`` ``(word, weight)`` pairs, sorted by descending weight."""
    idx = np.argsort(-weights)[:n]
    return [(vocab[i], float(weights[i])) for i in idx]


# --------------------------------------------------------------------------- #
#  Single-word inference
# --------------------------------------------------------------------------- #
def infer_next_word(known_words, mode="absolute", target_pos=6, top_n=5,
                     kb_dir=core.KB_DIR, kb_set=None):
    """
    Rank candidate words given up to 5 known neighbor words.

    :param known_words: list of 1-5 known word strings. In absolute mode,
        ``known_words[k]`` is the word at (1-indexed) sentence position
        ``k + 1``. In relative mode, ``known_words`` are the words
        immediately preceding the target, in sentence order (the last
        entry is the word right before the target).
    :param mode: ``"absolute"`` or ``"relative"``
    :param target_pos: absolute mode only -- the 1-indexed position being
        predicted (default 6, the 6th word)
    :param top_n: number of ranked candidates to return
    :param kb_dir: knowledge-base directory (used if ``kb_set`` not given)
    :param kb_set: pre-loaded result of :func:`load_kb_set` (loaded fresh
        from ``kb_dir`` if omitted -- pass this in to avoid re-loading from
        disk on every call, e.g. from a GUI)
    :return: list of ``(word, weight)`` tuples, length <= ``top_n``,
        sorted by descending weight (weights sum to 1 across the full
        vocabulary, not just the returned slice)
    """
    if kb_set is None:
        kb_set = load_kb_set(mode, kb_dir)
    if not kb_set:
        raise ValueError(f"no {mode} knowledge bases found in {kb_dir} -- "
                          f"run extract_tiny_kbs.sh first")

    vocab = next(iter(kb_set.values()))["vocab"]
    index = {tok: i for i, tok in enumerate(vocab)}
    n_known = len(known_words)

    items = [(_kb_key(mode, k, n_known, target_pos), [(word, 1.0)])
              for k, word in enumerate(known_words)]
    log_excitation, n_contrib = _combine_log_excitation(items, kb_set, index)
    weights = _finalize_weights(log_excitation, n_contrib, kb_set, vocab)
    return _topn(vocab, weights, top_n)


# --------------------------------------------------------------------------- #
#  Joint two-word inference (winnowing schedule)
# --------------------------------------------------------------------------- #
def infer_joint_pair(known_words, mode="absolute", target_pos=6, list_n=16,
                      report_top_n=5, kb_dir=core.KB_DIR, kb_set=None,
                      max_iters=None):
    """
    Jointly infer two adjacent unknown words (default: the 5th and 6th)
    that mutually excite each other, via a synchronous winnowing schedule.

    Each unit starts with a top-``list_n`` candidate list (seeded from
    known neighbors only). Each iteration, both units recompute their
    excitation -- known neighbors contribute their single KB row as usual,
    and the *other* unit contributes a weighted mixture over its current
    candidate list (each candidate's weight multiplies the excitation it
    sends) -- then both lists are truncated to at most half their previous
    size. This repeats until both converge to a single symbol. If the two
    ever desync (one converges before the other -- not expected with equal
    ``list_n`` and synchronous updates, but handled defensively), the
    schedule stops early for the remaining unit: one final fully-informed
    inference pass is run for it (using the other's now-fixed symbol) and
    its top candidate is taken directly, rather than mechanically halving
    it further.

    :param known_words: list of 1 to ``target_pos - 2`` known words (see
        :func:`infer_next_word` for absolute/relative word-order
        conventions)
    :param mode: ``"absolute"`` or ``"relative"``
    :param target_pos: absolute mode -- 1-indexed position of the *later*
        of the two jointly-inferred words (default 6, so the pair is
        (5, 6)). Ignored in relative mode, where the pair is always the
        immediate next word and the one after that.
    :param list_n: initial per-unit candidate-list size (default 16)
    :param report_top_n: number of candidates to return per unit
    :param kb_dir: knowledge-base directory (used if ``kb_set`` not given)
    :param kb_set: pre-loaded result of :func:`load_kb_set`
    :param max_iters: iteration cap (default: generous, based on
        ``list_n``)
    :return: ``{pos_a: [(word, weight), ...], pos_b: [(word, weight), ...]}``
        -- absolute mode keys are the 1-indexed positions (e.g. 5, 6);
        relative mode keys are 1 (immediate next) and 2 (the one after)
    """
    if kb_set is None:
        kb_set = load_kb_set(mode, kb_dir)
    if not kb_set:
        raise ValueError(f"no {mode} knowledge bases found in {kb_dir} -- "
                          f"run extract_tiny_kbs.sh first")

    vocab = next(iter(kb_set.values()))["vocab"]
    index = {tok: i for i, tok in enumerate(vocab)}
    n_known = len(known_words)

    if mode == "absolute":
        pos_a, pos_b = target_pos - 1, target_pos
        mutual_key_b_from_a = (pos_a, pos_b)  # forward: 5 -> 6
        mutual_key_a_from_b = (pos_b, pos_a)  # reverse: 6 -> 5 (new)
    else:
        pos_a, pos_b = 1, 2  # relative slots: immediate next, next after
        mutual_key_b_from_a = 1    # forward offset +1
        mutual_key_a_from_b = -1   # reverse offset -1 (new)

    known_items_a = [(_kb_key(mode, k, n_known, pos_a, rel_slot=1), [(w, 1.0)])
                      for k, w in enumerate(known_words)]
    known_items_b = [(_kb_key(mode, k, n_known, pos_b, rel_slot=2), [(w, 1.0)])
                      for k, w in enumerate(known_words)]

    def _rank(items, n):
        log_exc, n_contrib = _combine_log_excitation(items, kb_set, index)
        weights = _finalize_weights(log_exc, n_contrib, kb_set, vocab)
        return _topn(vocab, weights, n)

    list_a = _rank(known_items_a, list_n)
    list_b = _rank(known_items_b, list_n)

    if max_iters is None:
        max_iters = 2 * math.ceil(math.log2(max(list_n, 1))) + 2

    for _ in range(max_iters):
        len_a, len_b = len(list_a), len(list_b)
        if len_a == 1 and len_b == 1:
            break
        if len_a == 1 and len_b > 1:
            list_b = _rank(known_items_b + [(mutual_key_b_from_a, list_a)], 1)
            break
        if len_b == 1 and len_a > 1:
            list_a = _rank(known_items_a + [(mutual_key_a_from_b, list_b)], 1)
            break

        # Synchronous (Jacobi) update: both use the *other's pre-iteration* list.
        prev_a, prev_b = list_a, list_b
        new_a = _rank(known_items_a + [(mutual_key_a_from_b, prev_b)], max(1, len_a // 2))
        new_b = _rank(known_items_b + [(mutual_key_b_from_a, prev_a)], max(1, len_b // 2))
        list_a, list_b = new_a, new_b

    return {pos_a: list_a[:report_top_n], pos_b: list_b[:report_top_n]}


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("words", nargs="*",
                     help="known words (sentence-initial order for "
                          "--absolute, most-recent-last for --relative)")
    ap.add_argument("--relative", action="store_true",
                     help="use position-invariant relative KBs instead of "
                          "fixed-position absolute KBs")
    ap.add_argument("--joint", action="store_true",
                     help="jointly infer the two adjacent unknown words "
                          "(default: the 5th and 6th) instead of just one")
    ap.add_argument("--target-pos", type=int, default=6,
                     help="absolute mode: 1-indexed position to predict "
                          "(default: 6); with --joint, the *later* of the "
                          "jointly-inferred pair")
    ap.add_argument("--list-n", type=int, default=16,
                     help="--joint only: initial per-unit candidate-list "
                          "size (default: 16)")
    ap.add_argument("--top", type=int, default=5, help="number of candidates to show")
    ap.add_argument("--kb-dir", default=core.KB_DIR, help=f"default: {core.KB_DIR}")
    ap.add_argument("--selftest", action="store_true",
                     help="run a self-contained correctness check on synthetic "
                          "data and exit (ignores all other arguments)")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    max_known = (args.target_pos - 2) if args.joint else 5
    if not (1 <= len(args.words) <= max(max_known, 1)):
        ap.error(f"provide between 1 and {max(max_known, 1)} known words (or use --selftest)")

    mode = "relative" if args.relative else "absolute"

    if args.joint:
        results = infer_joint_pair(args.words, mode=mode, target_pos=args.target_pos,
                                    list_n=args.list_n, report_top_n=args.top,
                                    kb_dir=args.kb_dir)
        print(f"Given {args.words!r} ({mode}, joint), top predictions:")
        for pos, ranked in sorted(results.items()):
            label = f"word {pos}" if not args.relative else (
                "next word" if pos == 1 else "word after that")
            print(f"  {label}:")
            for word, weight in ranked:
                print(f"    {word:<15} {weight:.4f}")
    else:
        results = infer_next_word(args.words, mode=mode, target_pos=args.target_pos,
                                   top_n=args.top, kb_dir=args.kb_dir)
        label = "next word" if args.relative else f"word {args.target_pos}"
        print(f"Given {args.words!r} ({mode}), top predictions for the {label}:")
        for word, weight in results:
            print(f"  {word:<15} {weight:.4f}")


# --------------------------------------------------------------------------- #
#  Self-test
# --------------------------------------------------------------------------- #
def _selftest():
    """Synthetic-corpus correctness check for single-word and joint inference,
    exercised in-process (no file IO) via extract_knowledge_base's extraction
    functions."""
    # --- absolute: predict position 6 from positions 1-5 ---
    abs_sentences = [["A", "B", "C", "D", "E", "F"]] * 30
    abs_sentences += [["A", "B", "C", "D", "E", "X"]] * 10
    vocab, index = core.build_vocab(abs_sentences)
    abs_kbs = extract_absolute(abs_sentences, vocab, index,
                                [(1, 6), (2, 6), (3, 6), (4, 6), (5, 6)], alpha=1.0)
    abs_kb_set = {k: {**v, "vocab": vocab} for k, v in abs_kbs.items()}

    results = infer_next_word(["A", "B", "C", "D", "E"], mode="absolute",
                               target_pos=6, top_n=3, kb_set=abs_kb_set)
    assert results[0][0] == "F", f"expected top prediction 'F', got {results}"
    print(f"[selftest] absolute: given A B C D E -> {results} OK")

    # --- relative: predict the word right after "M N O" wherever it falls ---
    rel_sentences = [["M", "N", "O", "Z"]] * 20
    rel_sentences += [["P", "M", "N", "O", "Z"]] * 20
    rel_sentences += [["Q", "Q", "M", "N", "O", "Z"]] * 20
    rel_sentences += [["M", "N", "O", "W"]] * 5
    vocab2, index2 = core.build_vocab(rel_sentences)
    rel_kbs = extract_relative(rel_sentences, vocab2, index2, [1, 2, 3], alpha=1.0)
    rel_kb_set = {k: {**v, "vocab": vocab2} for k, v in rel_kbs.items()}

    results2 = infer_next_word(["M", "N", "O"], mode="relative", top_n=3,
                                kb_set=rel_kb_set)
    assert results2[0][0] == "Z", f"expected top prediction 'Z', got {results2}"
    print(f"[selftest] relative: given M N O (position-invariant) -> {results2} OK")

    # --- fallback: no usable evidence should not crash, still returns candidates ---
    results3 = infer_next_word(["totally-unknown-token"], mode="absolute",
                                target_pos=6, top_n=3, kb_set=abs_kb_set)
    assert len(results3) == 3, results3
    print(f"[selftest] unknown-word fallback -> {results3} OK")

    # --- joint: two correlated-but-noisy endings, positions 5 and 6.
    # Independently (ignoring the other), neither position is confident
    # (~54% each). Jointly, the reverse/forward mutual coupling should
    # amplify this into a confident *matching* pick (cat+meows, not
    # e.g. cat+barks), beating the independent single-unit confidence.
    joint_sentences = [["a", "b", "c", "d", "cat", "meows"]] * 30
    joint_sentences += [["a", "b", "c", "d", "dog", "barks"]] * 25
    joint_sentences += [["a", "b", "c", "d", "cat", "barks"]] * 1
    jvocab, jindex = core.build_vocab(joint_sentences)
    j_abs_pairs = [(1, 5), (2, 5), (3, 5), (4, 5), (1, 6), (2, 6), (3, 6), (4, 6),
                   (5, 6), (6, 5)]
    j_abs_kbs = extract_absolute(joint_sentences, jvocab, jindex, j_abs_pairs, alpha=1.0)
    j_abs_kb_set = {k: {**v, "vocab": jvocab} for k, v in j_abs_kbs.items()}

    independent_pos5 = infer_next_word(["a", "b", "c", "d"], mode="absolute",
                                        target_pos=5, top_n=1, kb_set=j_abs_kb_set)
    assert independent_pos5[0][0] == "cat", independent_pos5
    independent_confidence = independent_pos5[0][1]
    assert independent_confidence < 0.85, (
        f"expected a modest independent lean (<0.85), got {independent_pos5}")

    joint = infer_joint_pair(["a", "b", "c", "d"], mode="absolute", target_pos=6,
                              list_n=16, report_top_n=3, kb_set=j_abs_kb_set)
    assert joint[5][0][0] == "cat" and joint[6][0][0] == "meows", (
        f"expected joint top pair (cat, meows), got pos5={joint[5]} pos6={joint[6]}")
    assert joint[5][0][1] > independent_confidence, (
        f"expected joint confidence {joint[5][0][1]} to exceed independent "
        f"{independent_confidence}")
    print(f"[selftest] joint absolute: independent P(cat)={independent_confidence:.4f} "
          f"-> joint pos5={joint[5][0]} pos6={joint[6][0]} OK")

    # --- joint, relative mode: same correlation, but as "current"/"next"
    # words following a variable-length preceding context.
    j_rel_sentences = [["x", "y", "z", "cat", "meows", "w"]] * 30
    j_rel_sentences += [["p", "x", "y", "z", "dog", "barks", "w"]] * 25
    j_rel_sentences += [["z", "cat", "barks", "w"]] * 1
    jrvocab, jrindex = core.build_vocab(j_rel_sentences)
    j_rel_kbs = extract_relative(j_rel_sentences, jrvocab, jrindex,
                                  [1, 2, 3, 4, -1], alpha=1.0)
    j_rel_kb_set = {k: {**v, "vocab": jrvocab} for k, v in j_rel_kbs.items()}

    joint_rel = infer_joint_pair(["x", "y", "z"], mode="relative",
                                  list_n=16, report_top_n=3, kb_set=j_rel_kb_set)
    assert joint_rel[1][0][0] == "cat" and joint_rel[2][0][0] == "meows", (
        f"expected joint relative top pair (cat, meows), got {joint_rel}")
    print(f"[selftest] joint relative: given x y z -> "
          f"next={joint_rel[1][0]} after={joint_rel[2][0]} OK")

    print("[selftest] confabulation_high_level OK")


if __name__ == "__main__":
    main()
