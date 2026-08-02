"""
Learn "knowledge bases" (confabulation-theory co-occurrence probability
tables) from the reduced TinyStories corpus.

A knowledge base is a word-level statistical link between two positions in
a sentence -- see ``nlp_expansion.md`` and ``willshaw_dynamics.md``. Two
kinds are supported (selected by ``--relative``):

* **Absolute** -- ``i-j`` pairs are fixed, 1-indexed sentence positions
  (e.g. ``1-6`` learns "given the 1st word, what's the 6th word").  Useful
  near the start of a sentence, where position itself carries information
  (TinyStories openings are templated: "once upon a time ...").
* **Relative** -- ``i-j`` pairs are reinterpreted as an offset
  ``k = j - i`` (deduplicated across all given pairs) and learned from
  *every* valid ``(t, t+k)`` window in every sentence, position-invariant.
  Useful mid-sentence, predicting the next word from the last few words
  regardless of where they fall in the sentence.

Pairs need not be forward (``i < j``) -- a reverse pair like ``6-5``
learns "given the 6th word, what's the 5th word" (absolute) or offset
``-1`` (relative), needed when two neighboring words are inferred jointly
and have to excite each other in both directions.

Each learned table is saved as one ``.npz`` file under ``--outdir``
(default ``./knowledge_bases``) via ``nlp_core.save_kb``.

Usage::

    python extract_knowledge_base.py 1-6 2-6 3-6 4-6 5-6
    python extract_knowledge_base.py --relative 1-6 2-6 3-6 4-6 5-6
    python extract_knowledge_base.py --selftest
"""

import argparse
import os

import numpy as np

import nlp_core as core


# --------------------------------------------------------------------------- #
#  Pair parsing
# --------------------------------------------------------------------------- #
def parse_pairs(pair_strs):
    """
    Parse ``"i-j"`` CLI tokens into validated ``(i, j)`` int tuples.

    Either order is allowed (``i < j`` for a forward/"later word from
    earlier word" link, ``i > j`` for a reverse link) -- only ``i == j``
    and non-positive positions are rejected.

    :param pair_strs: iterable of strings like ``"1-6"`` or ``"6-5"``
    :return: list of ``(i, j)`` int tuples, 1-indexed, ``i != j``
    """
    pairs = []
    for s in pair_strs:
        parts = s.split("-")
        if len(parts) != 2:
            raise ValueError(f"bad pair {s!r}, expected \"i-j\"")
        i, j = int(parts[0]), int(parts[1])
        if i < 1 or j < 1 or i == j:
            raise ValueError(f"bad pair {s!r}: need i, j >= 1 and i != j")
        pairs.append((i, j))
    return pairs


# --------------------------------------------------------------------------- #
#  Extraction
# --------------------------------------------------------------------------- #
def _smooth(counts, alpha):
    """Row-stochastic Laplace smoothing: ``(counts + alpha) / (row_sum + alpha*V)``."""
    V = counts.shape[0]
    row_sum = counts.sum(axis=1, keepdims=True)
    return (counts + alpha) / (row_sum + alpha * V)


def extract_absolute(sentences, vocab, index, pairs, alpha=1.0):
    """
    Learn one absolute (fixed-position) KB per requested ``(i, j)`` pair.

    :param sentences: list of tokenized sentences
    :param vocab: vocabulary list (row/col order)
    :param index: ``{token: vocab position}`` map
    :param pairs: list of ``(i, j)`` 1-indexed position pairs, ``i != j``
        -- ``i < j`` learns a forward link (later word from earlier word),
        ``i > j`` learns a reverse link (earlier word from later word)
    :param alpha: Laplace smoothing constant
    :return: ``{(i, j): {"counts", "probs", "meta"}}``
    """
    V = len(vocab)
    out = {}
    for i, j in pairs:
        counts = np.zeros((V, V), dtype=np.int64)
        n_used = 0
        required_len = max(i, j)
        for sent in sentences:
            if len(sent) < required_len:
                continue
            counts[index[sent[i - 1]], index[sent[j - 1]]] += 1
            n_used += 1
        probs = _smooth(counts, alpha)
        meta = {
            "mode": "absolute", "i": i, "j": j, "alpha": alpha,
            "n_sentences_used": n_used, "n_pairs": n_used, "vocab_size": V,
        }
        out[(i, j)] = {"counts": counts, "probs": probs, "meta": meta}
    return out


def extract_relative(sentences, vocab, index, offsets, alpha=1.0):
    """
    Learn one relative (sliding-window) KB per requested offset.

    :param sentences: list of tokenized sentences
    :param vocab: vocabulary list (row/col order)
    :param index: ``{token: vocab position}`` map
    :param offsets: iterable of nonzero int offsets ``k`` (deduplicated) --
        ``k > 0`` learns "word ``k`` positions ahead, from the current
        word" (forward), ``k < 0`` learns "word ``|k|`` positions back"
        (reverse)
    :param alpha: Laplace smoothing constant
    :return: ``{k: {"counts", "probs", "meta"}}``
    """
    V = len(vocab)
    out = {}
    for k in sorted(set(offsets)):
        counts = np.zeros((V, V), dtype=np.int64)
        n_sent_used = 0
        n_pairs = 0
        lo, hi_slack = max(0, -k), max(0, k)
        for sent in sentences:
            hi = len(sent) - hi_slack
            if hi <= lo:
                continue
            n_sent_used += 1
            for t in range(lo, hi):
                counts[index[sent[t]], index[sent[t + k]]] += 1
                n_pairs += 1
        probs = _smooth(counts, alpha)
        meta = {
            "mode": "relative", "offset": k, "alpha": alpha,
            "n_sentences_used": n_sent_used, "n_pairs": n_pairs, "vocab_size": V,
        }
        out[k] = {"counts": counts, "probs": probs, "meta": meta}
    return out


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pairs", nargs="*", metavar="i-j",
                     help="1-indexed sentence-position pairs, e.g. 1-6 2-6 ... "
                          "(in --relative mode, reinterpreted as offsets j-i)")
    ap.add_argument("--relative", action="store_true",
                     help="learn position-invariant sliding-window KBs "
                          "(offset = j - i, deduplicated) instead of fixed-position KBs")
    ap.add_argument("--corpus", default=core.CORPUS_PATH,
                     help=f"reduced corpus path (default: {core.CORPUS_PATH})")
    ap.add_argument("--outdir", default=core.KB_DIR,
                     help=f"output directory (default: {core.KB_DIR})")
    ap.add_argument("--alpha", type=float, default=1.0,
                     help="Laplace smoothing constant (default: 1.0)")
    ap.add_argument("--selftest", action="store_true",
                     help="run a self-contained correctness check on synthetic "
                          "data and exit (ignores all other arguments)")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    if not args.pairs:
        ap.error("at least one i-j pair is required (or use --selftest)")

    pairs = parse_pairs(args.pairs)
    sentences = core.load_sentences(args.corpus)
    vocab, index = core.build_vocab(sentences)
    print(f"[extract_kb] {len(sentences)} sentences, {len(vocab)} vocab tokens "
          f"from {args.corpus}")

    if args.relative:
        offsets = sorted(set(j - i for i, j in pairs))
        kbs = extract_relative(sentences, vocab, index, offsets, args.alpha)
        for k, kb in kbs.items():
            path = os.path.join(args.outdir, core.kb_filename("relative", k))
            core.save_kb(path, kb["counts"], kb["probs"], vocab, kb["meta"])
            m = kb["meta"]
            print(f"  offset={k:>2}  sentences_used={m['n_sentences_used']:>4}  "
                  f"pairs={m['n_pairs']:>5}  -> {path}")
    else:
        kbs = extract_absolute(sentences, vocab, index, pairs, args.alpha)
        for (i, j), kb in kbs.items():
            path = os.path.join(args.outdir, core.kb_filename("absolute", i, j))
            core.save_kb(path, kb["counts"], kb["probs"], vocab, kb["meta"])
            m = kb["meta"]
            print(f"  {i}-{j}  sentences_used={m['n_sentences_used']:>4}  "
                  f"-> {path}")


# --------------------------------------------------------------------------- #
#  Self-test
# --------------------------------------------------------------------------- #
def _selftest():
    """Synthetic-corpus correctness check for both absolute and relative extraction."""
    # Fixed-position pattern: "A B C D E F" dominant at position 1-6, with a
    # rarer variant "A B C D E X" so the absolute KB should still strongly
    # prefer F over X (but not with probability 1, exercising smoothing).
    sentences = [["A", "B", "C", "D", "E", "F"]] * 30
    sentences += [["A", "B", "C", "D", "E", "X"]] * 10
    # Position-invariant adjacency pattern: "B" is always immediately
    # followed by "C", regardless of where B falls in the sentence -- an
    # absolute KB fixed at (say) position 2-3 would only see this when B
    # happens to land at position 2, but the relative offset-1 KB should
    # see it everywhere.
    sentences += [["P", "Q", "B", "C"]] * 20      # B at position 3
    sentences += [["B", "C", "R"]] * 20           # B at position 1
    sentences += [["Q", "Q", "Q", "B", "C"]] * 20  # B at position 4

    vocab, index = core.build_vocab(sentences)
    V = len(vocab)
    print(f"[selftest] {len(sentences)} synthetic sentences, {V} vocab tokens")

    # --- absolute: predicting position 6 from position 1 ---
    abs_kbs = extract_absolute(sentences, vocab, index, [(1, 6)], alpha=1.0)
    probs16 = abs_kbs[(1, 6)]["probs"]
    p_f = probs16[index["A"], index["F"]]
    p_x = probs16[index["A"], index["X"]]
    assert p_f > p_x, f"expected P(F|A) > P(X|A), got {p_f} vs {p_x}"
    assert abs(probs16[index["A"]].sum() - 1.0) < 1e-5, "absolute KB row not normalized"
    print(f"[selftest] absolute 1-6: P(F|A)={p_f:.4f} > P(X|A)={p_x:.4f} OK")

    # --- absolute reverse: predicting position 1 from position 6 ---
    # Position 1 is always "A" in this synthetic corpus regardless of what's
    # at position 6, so the reverse conditional should be near-certain -- and
    # it's computed independently (not just a transpose of the forward
    # matrix, which is normalized the other way).
    abs_kbs_rev = extract_absolute(sentences, vocab, index, [(6, 1)], alpha=1.0)
    probs61 = abs_kbs_rev[(6, 1)]["probs"]
    p_a_given_f = probs61[index["F"], index["A"]]
    assert probs61[index["F"]].argmax() == index["A"] and p_a_given_f > 0.7, (
        f"expected P(A|F) (reverse 6-1) to dominate, got {p_a_given_f}")
    assert abs(probs61[index["F"]].sum() - 1.0) < 1e-5, "reverse absolute KB row not normalized"
    print(f"[selftest] absolute reverse 6-1: P(A|F)={p_a_given_f:.4f} OK")

    # --- absolute: a fixed position pair that only rarely sees B->C ---
    abs_kbs_23 = extract_absolute(sentences, vocab, index, [(2, 3)], alpha=1.0)
    probs23 = abs_kbs_23[(2, 3)]["probs"]
    p_c_given_b_fixed = probs23[index["B"], index["C"]]

    # --- relative offset 1: position-invariant B -> C ---
    rel_kbs = extract_relative(sentences, vocab, index, [1], alpha=1.0)
    probs_rel1 = rel_kbs[1]["probs"]
    p_c_given_b_rel = probs_rel1[index["B"], index["C"]]
    assert p_c_given_b_rel > p_c_given_b_fixed, (
        f"relative offset-1 P(C|B)={p_c_given_b_rel} should exceed the "
        f"fixed-position 2-3 P(C|B)={p_c_given_b_fixed} (position-invariant "
        f"pattern should only be fully captured by the relative KB)")
    assert abs(probs_rel1[index["B"]].sum() - 1.0) < 1e-5, "relative KB row not normalized"
    print(f"[selftest] relative offset=1: P(C|B)={p_c_given_b_rel:.4f} > "
          f"fixed 2-3 P(C|B)={p_c_given_b_fixed:.4f} OK")

    # --- relative offset -1 (reverse, sliding-window): C is always
    # immediately preceded by B in this corpus, so P(B|C) via the reverse
    # offset should be near-certain.
    rel_kbs_rev = extract_relative(sentences, vocab, index, [-1], alpha=1.0)
    probs_rel_neg1 = rel_kbs_rev[-1]["probs"]
    p_b_given_c_rel = probs_rel_neg1[index["C"], index["B"]]
    assert probs_rel_neg1[index["C"]].argmax() == index["B"] and p_b_given_c_rel > 0.85, (
        f"expected P(B|C) (offset -1) to dominate, got {p_b_given_c_rel}")
    assert abs(probs_rel_neg1[index["C"]].sum() - 1.0) < 1e-5, "reverse relative KB row not normalized"
    print(f"[selftest] relative offset=-1: P(B|C)={p_b_given_c_rel:.4f} OK")

    # --- pair-to-offset dedup used by --relative CLI mode (incl. reverse) ---
    pairs = parse_pairs(["1-6", "2-6", "3-6", "4-6", "5-6", "6-5"])
    offsets = sorted(set(j - i for i, j in pairs))
    assert offsets == [-1, 1, 2, 3, 4, 5], offsets
    print(f"[selftest] pair->offset dedup for 1-6..5-6 + 6-5 => {offsets} OK")

    # --- parse_pairs validation ---
    parse_pairs(["6-5"])  # reverse pair must be accepted
    for bad in ("3-3", "0-1", "1-0"):
        try:
            parse_pairs([bad])
            raise AssertionError(f"parse_pairs should have rejected {bad!r}")
        except ValueError:
            pass
    print("[selftest] parse_pairs validation OK")

    print("[selftest] extract_knowledge_base OK")


if __name__ == "__main__":
    main()
