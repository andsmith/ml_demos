"""
Shared IO helpers for the NLP core (knowledge-base extraction +
confabulation inference).

Loads the reduced-corpus tokenization exactly as
``named_reduction_explorer.py`` / ``tiny_stories.py`` produced it -- one
sentence per non-delimiter line of ``data/reduced_corpus.txt``, already
whitespace-tokenized into the reduced vocabulary. See
``tiny_stories.render_story``: "one line per (non-empty) sentence" is the
exported format regardless of the ``unit`` config used to build it, so this
module does not need to know how the corpus was reduced -- only how to read
the result.

Also defines the on-disk ``.npz`` knowledge-base format shared by
``extract_knowledge_base.py`` (writer) and ``confabulation_high_level.py``
(reader).
"""

import os

import numpy as np

from tiny_stories import STORY_DELIM

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(HERE, "data", "reduced_corpus.txt")
KB_DIR = os.path.join(HERE, "knowledge_bases")


# --------------------------------------------------------------------------- #
#  Corpus loading
# --------------------------------------------------------------------------- #
def load_sentences(path=CORPUS_PATH):
    """
    Read the reduced corpus into a list of tokenized sentences.

    :param path: path to a ``<|endoftext|>``-delimited, whitespace-tokenized
        reduced corpus (as exported by ``tiny_stories.export_corpus``)
    :return: list of sentences, each a list of word tokens (blank lines and
        the story delimiter are dropped; sentence order is preserved)
    """
    sentences = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line == STORY_DELIM:
                continue
            sentences.append(line.split())
    return sentences


def build_vocab(sentences):
    """
    Build a deterministic vocabulary from tokenized sentences.

    :param sentences: list of list-of-token sentences (as from
        :func:`load_sentences`)
    :return: ``(vocab, index)`` -- ``vocab`` is a list of unique tokens
        sorted by descending frequency (ties broken alphabetically, so the
        result is reproducible run-to-run without relying on dict/set
        iteration order); ``index`` is the ``{token: position in vocab}``
        map
    """
    counts = {}
    for sent in sentences:
        for tok in sent:
            counts[tok] = counts.get(tok, 0) + 1
    vocab = sorted(counts, key=lambda t: (-counts[t], t))
    index = {tok: i for i, tok in enumerate(vocab)}
    return vocab, index


# --------------------------------------------------------------------------- #
#  Knowledge-base file IO
# --------------------------------------------------------------------------- #
def kb_filename(mode, i, j=None):
    """
    Canonical knowledge-base filename for a position pair or offset.

    :param mode: ``"absolute"`` or ``"relative"``
    :param i: for absolute mode, the source (1-indexed) sentence position;
        for relative mode, the offset directly (``j`` is ignored)
    :param j: for absolute mode, the target (1-indexed) sentence position
        (required); unused for relative mode
    :return: bare filename, e.g. ``"kb_abs_1-6.npz"`` / ``"kb_rel_5.npz"``
    """
    if mode == "absolute":
        assert j is not None, "absolute KB filenames need both i and j"
        return f"kb_abs_{i}-{j}.npz"
    if mode == "relative":
        return f"kb_rel_{i}.npz"
    raise ValueError(f"unknown KB mode {mode!r}")


def save_kb(path, counts, probs, vocab, meta):
    """
    Save one knowledge base (co-occurrence counts + smoothed probabilities).

    :param path: output ``.npz`` path
    :param counts: (V, V) int array, raw co-occurrence counts
        ``counts[a, b]`` = number of times source word ``a`` co-occurred
        with target word ``b``
    :param probs: (V, V) float array, row-stochastic ``P(target=b|source=a)``
    :param vocab: list of V tokens; row/column ``k`` refers to ``vocab[k]``
    :param meta: small JSON-serializable dict of provenance (mode,
        position(s)/offset, alpha, n_pairs, ...) -- stored as a numpy 0-d
        object array via ``allow_pickle``
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(
        path,
        counts=np.asarray(counts, dtype=np.int32),
        probs=np.asarray(probs, dtype=np.float32),
        vocab=np.asarray(vocab, dtype=object),
        meta=np.asarray(meta, dtype=object),
    )


def load_kb(path):
    """
    Load one knowledge base saved by :func:`save_kb`.

    :param path: input ``.npz`` path
    :return: dict with keys ``counts`` (V,V int array), ``probs`` (V,V float
        array), ``vocab`` (list of V tokens), ``meta`` (dict)
    """
    with np.load(path, allow_pickle=True) as data:
        return {
            "counts": data["counts"],
            "probs": data["probs"],
            "vocab": list(data["vocab"]),
            "meta": data["meta"].item(),
        }
