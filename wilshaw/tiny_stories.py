"""
TinyStories vocabulary-reduction pipeline (no GUI).

Given the local corpus ``data/TinyStories_valid.txt`` this module:

  1. **loads** the corpus (stories delimited by ``<|endoftext|>``),
  2. **annotates** it once with spaCy -- tokens, lemmas, sentence ids, and
     named entities (characters / places, plus animals & objects detected via
     the ``a <noun> named|called <Name>`` construction) -- and **caches** the
     result to disk keyed by (corpus bytes, model, schema version),
  3. **reduces** each story under an :class:`~named_ontologies.OntologyConfig`
     (entity substitution + light lexical normalization), and
  4. exposes the **vocabulary-vs-coverage tradeoff** used by the explorer, plus
     deterministic **export** (reduced corpus + a reproducible JSON bundle).

The tradeoff is computed from a single monotone curve: for every sample (a
story or a sentence) we record its *threshold rank* -- the frequency rank of
its rarest token.  A sample is covered by a vocabulary budget of the ``k`` most
frequent tokens iff its threshold rank <= ``k``.  Both explorer framings fall
out of the sorted thresholds (see :class:`Reduction`).

Determinism is a hard requirement (see ``tiny_stories_strategy.md``): identical
config + input always produce identical output, and the exported reduced corpus
re-tokenizes (whitespace split) to exactly the exported vocabulary.
"""

from __future__ import annotations

import os

# Force BLAS/BLIS single-threaded before numpy/thinc/spaCy load their native
# libraries.  On this Windows box multi-threaded BLIS otherwise aborts
# ("libblis: Aborting.") during interpreter teardown; single-thread also curbs
# the per-process memory reservation (see memory: willshaw-app-gotchas).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import hashlib
import json
import pickle
import random
import sys
import time
from collections import Counter

import numpy as np

import named_ontologies as onto
from named_ontologies import OntologyConfig

# --------------------------------------------------------------------------- #
#  Constants
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(HERE, "data", "TinyStories_valid.txt")
CACHE_DIR = os.path.join(HERE, ".cache")
STORY_DELIM = "<|endoftext|>"
SPACY_MODEL = "en_core_web_sm"

# Bump when the annotation record schema changes (invalidates the cache).
SCHEMA_VERSION = 4

# The seven vocabulary/sample percentages, drawn in increasing order.
PERCENTS = (10, 20, 50, 75, 90, 95, 100)

# Small, curated synonym-merge table (light lexical reduction).  Applied to
# non-entity word tokens only, after lowercase + lemmatize.
SYNONYMS = {
    # size
    "tiny": "small", "little": "small", "teeny": "small", "wee": "small",
    "small": "small",
    "big": "big", "large": "big", "huge": "big", "giant": "big",
    "enormous": "big", "massive": "big",
    # emotion (coarse)
    "happy": "happy", "glad": "happy", "joyful": "happy", "cheerful": "happy",
    "delighted": "happy", "merry": "happy",
    "sad": "sad", "unhappy": "sad", "gloomy": "sad", "upset": "sad",
    # places / things commonly synonymous
    "woods": "forest", "pond": "lake",
    # kinship / diminutives
    "mommy": "mom", "mummy": "mom", "mama": "mom", "mum": "mom",
    "daddy": "dad", "papa": "dad",
    "kitty": "cat", "kitten": "cat", "puppy": "dog", "doggy": "dog",
    "bunny": "rabbit", "birdie": "bird",
}


# --------------------------------------------------------------------------- #
#  Corpus loading
# --------------------------------------------------------------------------- #
def load_corpus(path=CORPUS_PATH):
    """Return the list of story texts (split on ``<|endoftext|>``, blanks dropped)."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = fh.read()
    stories = [s.strip() for s in raw.split(STORY_DELIM)]
    return [s for s in stories if s]


def _corpus_sha1(path=CORPUS_PATH):
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
#  Annotation (spaCy) + named-object/animal detection
# --------------------------------------------------------------------------- #
def get_nlp():
    """Load the spaCy model tuned for this pipeline.

    We need POS (tagger + attribute_ruler), lemmas (lemmatizer), NER, and
    sentence boundaries -- but *not* the dependency parse.  Dropping the slow
    ``parser`` and using the rule-based ``sentencizer`` roughly halves the
    one-time annotation pass and lowers memory, with no effect on the tokens,
    lemmas, POS, or entities we consume.
    """
    import spacy
    nlp = spacy.load(SPACY_MODEL, exclude=["parser"])
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


def _build_matcher(nlp):
    """Matcher for ``<NOUN> named|called <PROPN>+`` (named animals/objects)."""
    from spacy.matcher import Matcher
    m = Matcher(nlp.vocab)
    m.add("NAMED_THING", [[
        {"POS": {"IN": ["NOUN", "PROPN"]}},
        {"LOWER": {"IN": ["named", "called"]}},
        {"POS": "PROPN", "OP": "+"},
    ]])
    return m


def _entities_for_doc(doc, matcher, context_lower):
    """Extract non-overlapping entities for one doc.

    Returns a list of dicts: ``{start, end, category, leaf, surface}`` where
    start/end are token indices (end exclusive).  Matcher hits (named
    animals/objects/humans) take precedence over raw spaCy NER for the tokens
    they cover.
    """
    claimed = [False] * len(doc)          # token already assigned to an entity
    ents = []

    # 1) "a dog named Spot" pattern -- overrides spaCy's PERSON-biased NER.
    #    Longest matches first so a multi-token name isn't truncated by a
    #    shorter overlapping sub-match.
    matches = sorted(matcher(doc), key=lambda m: -(m[2] - m[1]))
    for _mid, start, end in matches:
        head = doc[start]                 # the common/proper noun
        # the name span is the trailing run of PROPN tokens
        name_start = start + 2            # skip head + "named"/"called"
        while name_start < end and doc[name_start].pos_ != "PROPN":
            name_start += 1
        if name_start >= end:
            continue
        cat, leaf = onto.category_of_head_noun(head.lemma_ or head.text)
        if cat == "character":
            leaf = onto.classify_character(doc[name_start:end].text, context_lower)
        surface = doc[name_start:end].text
        if any(claimed[i] for i in range(name_start, end)):
            continue
        for i in range(name_start, end):
            claimed[i] = True
        ents.append({"start": name_start, "end": end, "category": cat,
                     "leaf": leaf, "surface": surface})

    # 2) spaCy named entities for the categories we care about.
    for ent in doc.ents:
        cat = onto.SPACY_LABEL_CATEGORY.get(ent.label_)
        if cat is None:
            if ent.label_ in ("ORG", "PRODUCT", "WORK_OF_ART", "EVENT"):
                cat = "other"
            else:
                continue
        if any(claimed[i] for i in range(ent.start, ent.end)):
            continue
        if cat == "character":
            leaf = onto.classify_character(ent.text, context_lower)
        elif cat == "place":
            leaf = onto.classify_place(ent.text, context_lower)
        else:
            leaf = onto.CATEGORY_ROOT.get(cat, "THING")
        for i in range(ent.start, ent.end):
            claimed[i] = True
        ents.append({"start": ent.start, "end": ent.end, "category": cat,
                     "leaf": leaf, "surface": ent.text})

    ents.sort(key=lambda e: e["start"])
    return ents


def annotate(stories, progress=None, batch_size=200):
    """Annotate every story; return a list of compact record dicts.

    Each record: ``index``, ``tokens`` (texts), ``lemmas``, ``ws`` (trailing
    whitespace), ``is_word`` (bool), ``is_num`` (bool), ``sent_id`` (per-token
    int), ``ents`` (list, see :func:`_entities_for_doc`).

    :param progress: optional ``callable(done, total)`` for a GUI progress bar.
    """
    nlp = get_nlp()
    matcher = _build_matcher(nlp)
    records = []
    total = len(stories)
    for i, doc in enumerate(nlp.pipe(stories, batch_size=batch_size)):
        # per-token sentence ids
        sent_id = [0] * len(doc)
        for sid, sent in enumerate(doc.sents):
            for t in range(sent.start, sent.end):
                sent_id[t] = sid
        rec = {
            "index": i,
            "tokens": [t.text for t in doc],
            "lemmas": [t.lemma_ for t in doc],
            "ws": [t.whitespace_ for t in doc],
            "is_word": [bool(t.is_alpha) for t in doc],
            "is_num": [bool(t.like_num) for t in doc],
            "sent_id": sent_id,
            "n_sents": (sent_id[-1] + 1) if sent_id else 0,
            "ents": _entities_for_doc(doc, matcher, doc.text.lower()),
        }
        records.append(rec)
        if progress is not None and (i % 200 == 0 or i == total - 1):
            progress(i + 1, total)
    return records


def _cache_path(corpus_path, model, version):
    key = f"{_corpus_sha1(corpus_path)}|{model}|v{version}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"parse_{digest}.pkl")


def load_or_annotate(corpus_path=CORPUS_PATH, progress=None, force=False):
    """Load cached annotations if present, else annotate + cache.

    Returns ``(records, meta)`` where meta records the corpus hash / model /
    spaCy version for reproducibility.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = _cache_path(corpus_path, SPACY_MODEL, SCHEMA_VERSION)
    if os.path.exists(cache) and not force:
        with open(cache, "rb") as fh:
            blob = pickle.load(fh)
        return blob["records"], blob["meta"]

    import spacy
    stories = load_corpus(corpus_path)
    t0 = time.time()
    records = annotate(stories, progress=progress)
    meta = {
        "corpus_path": corpus_path,
        "corpus_sha1": _corpus_sha1(corpus_path),
        "n_stories": len(stories),
        "spacy_model": SPACY_MODEL,
        "spacy_version": spacy.__version__,
        "schema_version": SCHEMA_VERSION,
        "annotate_seconds": round(time.time() - t0, 1),
    }
    with open(cache, "wb") as fh:
        pickle.dump({"records": records, "meta": meta}, fh,
                    protocol=pickle.HIGHEST_PROTOCOL)
    return records, meta


# --------------------------------------------------------------------------- #
#  Reduction
# --------------------------------------------------------------------------- #
def effective_synonyms(config):
    """The merge map actually in force for ``config``.

    ``{}`` when synonym-merge is off; otherwise the config's own table, falling
    back to the built-in :data:`SYNONYMS` when the config carries none (the
    headless / default case).  The explorer always supplies its edited table.
    """
    if not getattr(config, "synonym_merge", False):
        return {}
    return getattr(config, "synonyms", None) or SYNONYMS


def _normalize(text, lemma, is_num, config, synonyms):
    """Normalize one non-entity word token to its counted form."""
    if config.lemmatize:
        base = lemma
    elif config.lowercase:
        base = text.lower()
    else:
        base = text
    if config.lowercase:
        base = base.lower()
    if config.synonym_merge:
        if is_num:
            return "number"
        base = synonyms.get(base, base)
    return base


def _story_substitution(rec, config):
    """The ``{surface: replacement}`` map for a story under ``config``."""
    ents_for_sub = [{"surface": e["surface"], "category": e["category"],
                     "leaf": e["leaf"], "first_pos": e["start"]}
                    for e in rec["ents"]]
    return onto.substitute_story(ents_for_sub, config, rec["index"])


def _emit_tokens(rec, config, synonyms):
    """Yield ``(display_text, ws, word_or_None, sent_id)`` for a reduced story.

    ``word_or_None`` is the vocabulary token (``None`` for punctuation/space,
    which is displayed but not counted).  Entity mentions are replaced two
    ways, so a name is substituted *consistently everywhere* in the story:
    multi-token NER/pattern spans via ``repl_at``/``inside``, and every
    remaining single-token occurrence of a known surface via ``single_sub``
    (covers mentions spaCy's NER missed).
    """
    subs = _story_substitution(rec, config)
    # Register only entities actually replaced; others fall through to ordinary
    # token processing (correct for preserve mode / uncollapsed OTHER).
    repl_at, inside = {}, set()
    for e in rec["ents"]:
        if e["surface"] in subs:
            repl_at[e["start"]] = (subs[e["surface"]], e["end"])
            inside.update(range(e["start"] + 1, e["end"]))
    single_sub = {s: r for s, r in subs.items() if " " not in s}

    tokens, lemmas, ws = rec["tokens"], rec["lemmas"], rec["ws"]
    is_word, is_num, sent_id = rec["is_word"], rec["is_num"], rec["sent_id"]
    i, n = 0, len(tokens)
    while i < n:
        if i in inside:
            i += 1
            continue
        if i in repl_at:
            word, end = repl_at[i]
            yield word, ws[end - 1], word, sent_id[i]
            i = end
            continue
        t = tokens[i]
        if t in single_sub:
            r = single_sub[t]
            yield r, ws[i], r, sent_id[i]
            i += 1
            continue
        w = _normalize(t, lemmas[i], is_num[i], config, synonyms) if is_word[i] else None
        yield t, ws[i], w, sent_id[i]
        i += 1


def reduce_story(rec, config, synonyms):
    """Reduce one story to a list of ``(display_text, ws, word_or_None, sent_id)``."""
    return list(_emit_tokens(rec, config, synonyms))


def _iter_words(rec, config, synonyms):
    """Yield ``(sent_id, word)`` for the counted words of a story (lean path)."""
    for _text, _ws, word, sid in _emit_tokens(rec, config, synonyms):
        if word is not None:
            yield sid, word


def render_sentences(rec, config, synonyms=None):
    """List of reduced sentences, indexed by sentence id (length ``n_sents``).

    Entry ``sid`` is the space-joined words of that sentence (``""`` if the
    sentence reduced to no counted words), so the list index always equals the
    sentence id -- needed to display a specific admitted sentence.
    """
    synonyms = SYNONYMS if synonyms is None else synonyms
    lines = ["" for _ in range(max(rec["n_sents"], 1))]
    buf = {}
    for sid, w in _iter_words(rec, config, synonyms):
        buf.setdefault(sid, []).append(w)
    for sid, words in buf.items():
        if 0 <= sid < len(lines):
            lines[sid] = " ".join(words)
    return lines


def render_story(rec, config, synonyms=None):
    """A readable reduced story: one line per (non-empty) sentence.

    This is exactly the tokenization the exported corpus uses, so a
    whitespace split of the output reproduces the story's vocabulary.
    """
    return "\n".join(s for s in render_sentences(rec, config, synonyms) if s)


# --------------------------------------------------------------------------- #
#  Natural (unconstrained) vocabulary
# --------------------------------------------------------------------------- #
def natural_vocab_size(records):
    """Distinct lowercased alphabetic tokens across the raw corpus (no reduction)."""
    vocab = set()
    for rec in records:
        toks, isw = rec["tokens"], rec["is_word"]
        for i in range(len(toks)):
            if isw[i]:
                vocab.add(toks[i].lower())
    return len(vocab)


def count_sentences(records):
    return int(sum(rec["n_sents"] for rec in records))


# --------------------------------------------------------------------------- #
#  Tradeoff computation
# --------------------------------------------------------------------------- #
class Corpus:
    """Holds the annotated records plus a *normalization cache*.

    The expensive part of a reduction is lexically normalizing every non-entity
    token (lowercase -> lemma -> synonym) across ~5M tokens.  That depends only
    on the three lexical flags, so we cache it per *lexical signature* and, when
    only the entity/name knobs change, re-run just the (cheap) entity
    substitution over the cached base.  Normalized words are interned, so the
    cache is a few million pointers into ~10k unique strings (small).

    Sub-sampling is applied at reduce time: :meth:`reduce` accepts an ``active``
    list of story indices and computes the vocabulary/thresholds over only
    those stories.
    """

    def __init__(self, records):
        self.records = records
        self.n_stories = len(records)
        self._lex_sig = None
        self._base = None            # list[story] -> {sent_id: [word, ...]}
        self._occ = None             # list[story] -> [(sent_id, surface, orig_words)]
        self._unique_ents = None     # list[story] -> [ {surface,category,leaf,first_pos} ]
        self._rulecount = None       # Counter: merge-member base word -> corpus hits
        self._rulecount_sig = None

    # -- entity structure (config-independent, built once) ---------------- #
    def _ensure_entities(self):
        if self._unique_ents is not None:
            return
        uniq = []
        for rec in self.records:
            seen, order = {}, []
            for e in rec["ents"]:
                if e["surface"] not in seen:
                    seen[e["surface"]] = e
                    order.append(e["surface"])
            uniq.append([{"surface": s, "category": seen[s]["category"],
                          "leaf": seen[s]["leaf"], "first_pos": seen[s]["start"]}
                         for s in order])
        self._unique_ents = uniq

    # -- lexical base cache ---------------------------------------------- #
    def _ensure_base(self, config):
        syn = effective_synonyms(config)
        sig = (config.lowercase, config.lemmatize, config.synonym_merge,
               frozenset(syn.items()))
        if sig == self._lex_sig:
            return
        base, occ = [], []
        for rec in self.records:
            b, o = self._build_base_story(rec, config, syn)
            base.append(b)
            occ.append(o)
        self._base, self._occ, self._lex_sig = base, occ, sig

    @staticmethod
    def _build_base_story(rec, config, syn):
        """Split a story into cached non-entity base words + entity occurrences.

        Returns ``(base_by_sent, occurrences)`` where ``base_by_sent`` maps a
        sentence id to its list of normalized non-entity words, and each
        occurrence is ``(sent_id, surface, orig_words)`` -- ``orig_words`` being
        the normalized original words used when the entity is *not* substituted
        (preserve mode / uncollapsed OTHER).
        """
        ents = rec["ents"]
        surfaces = {e["surface"] for e in ents}
        starts = {}
        inside = set()
        for e in ents:
            starts[e["start"]] = (e["end"], e["surface"])
            inside.update(range(e["start"] + 1, e["end"]))

        tokens, lemmas = rec["tokens"], rec["lemmas"]
        is_word, is_num, sent_id = rec["is_word"], rec["is_num"], rec["sent_id"]
        base_by_sent, occ = {}, []
        i, n = 0, len(tokens)
        while i < n:
            if i in inside:
                i += 1
                continue
            if i in starts:
                end, surface = starts[i]
                orig = tuple(sys.intern(_normalize(tokens[j], lemmas[j],
                                                   is_num[j], config, syn))
                             for j in range(i, end) if is_word[j])
                occ.append((sent_id[i], surface, orig))
                i = end
                continue
            t = tokens[i]
            if t in surfaces:
                orig = ((sys.intern(_normalize(t, lemmas[i], is_num[i], config,
                                               syn)),) if is_word[i] else ())
                occ.append((sent_id[i], t, orig))
                i += 1
                continue
            if is_word[i]:
                w = sys.intern(_normalize(t, lemmas[i], is_num[i], config, syn))
                base_by_sent.setdefault(sent_id[i], []).append(w)
            i += 1
        return base_by_sent, occ

    # -- merge-rule statistics (for the lemma editor) --------------------- #
    def rule_member_counts(self, config):
        """Corpus occurrence count of every base word that a merge rule touches.

        Returns a ``Counter`` keyed by the *pre-merge* normalized base form
        (lowercase/lemma per ``config``) for every word that appears as a rule
        key **or** a canonical target.  The lemma editor groups these by
        canonical to show, per rule, how many corpus tokens it collapses.

        Counts are over the **whole corpus** (not the active sub-sample) and are
        independent of the entity/name knobs, so this is cached by
        ``(lowercase, lemmatize, member-set)`` and only recomputed when the
        lexical flags or the rule table actually change.
        """
        syn = getattr(config, "synonyms", None) or SYNONYMS
        members = frozenset(syn) | frozenset(syn.values())
        sig = (config.lowercase, config.lemmatize, members)
        if sig == self._rulecount_sig:
            return self._rulecount
        lower, lemma = config.lowercase, config.lemmatize
        cnt = Counter()
        for rec in self.records:
            tokens, lemmas, is_word = rec["tokens"], rec["lemmas"], rec["is_word"]
            for i, t in enumerate(tokens):
                if not is_word[i]:
                    continue
                b = lemmas[i] if lemma else t
                if lower:
                    b = b.lower()
                if b in members:
                    cnt[b] += 1
        self._rulecount, self._rulecount_sig = cnt, sig
        return cnt

    # -- reduce ----------------------------------------------------------- #
    def reduce(self, config, active=None):
        """Compute a :class:`Reduction` over the ``active`` story set (all if None)."""
        self._ensure_entities()
        self._ensure_base(config)
        active = list(range(self.n_stories)) if active is None else list(active)

        counts = Counter()
        sample_refs, sample_words = [], []
        for i in active:
            subs = onto.substitute_story(self._unique_ents[i], config, i)
            ent_by_sid = {}
            for sid, surface, orig in self._occ[i]:
                if surface in subs:
                    ent_by_sid.setdefault(sid, []).append(sys.intern(subs[surface]))
                elif orig:
                    ent_by_sid.setdefault(sid, []).extend(orig)

            base_i = self._base[i]
            if config.unit == "story":
                words = []
                for ws in base_i.values():
                    words += ws
                for ws in ent_by_sid.values():
                    words += ws
                counts.update(words)
                sample_refs.append(("story", i))
                sample_words.append(words)
            else:
                for sid in sorted(set(base_i) | set(ent_by_sid)):
                    ws = list(base_i.get(sid, ())) + ent_by_sid.get(sid, [])
                    if not ws:
                        continue
                    counts.update(ws)
                    sample_refs.append(("sent", i, sid))
                    sample_words.append(ws)

        return Reduction(config, sample_refs, sample_words, counts,
                         self.records, active)


class Reduction:
    """A computed reduction: vocabulary, per-sample thresholds, and the bars.

    ``sample_refs[i]`` identifies sample ``i``: ``('story', story_index)`` or
    ``('sent', story_index, sent_id)``.  ``thresholds[i]`` is the frequency
    rank (1-based) of that sample's rarest token; a sample is covered by the
    top-``k`` vocabulary iff ``thresholds[i] <= k``.
    """

    def __init__(self, config, sample_refs, sample_words, counts, records,
                 active):
        """Build vocabulary, ranks, and per-sample thresholds.

        :param sample_refs: list of sample identifiers (parallel to
            ``sample_words``).
        :param sample_words: list of per-sample word lists (already reduced).
        :param counts: ``Counter`` of all words over the samples.
        :param records: the full record list (for export/render).
        :param active: list of story indices actually included (the active set).
        """
        self.config = config
        self._records = records
        self.active = active
        self.sample_refs = sample_refs

        ordered = counts.most_common()
        self.vocab = [w for w, _ in ordered]
        self.freqs = [c for _, c in ordered]
        self.rank = {w: i + 1 for i, w in enumerate(self.vocab)}
        self.counts = counts
        self.V0 = len(self.vocab)
        self.N = len(sample_refs)

        rank = self.rank
        thr = np.zeros(self.N, dtype=np.int64)
        for i, ws in enumerate(sample_words):
            if ws:
                thr[i] = max(rank[w] for w in set(ws))
        self.thresholds = thr
        self._sorted_thr = np.sort(thr)

    # -- bars ------------------------------------------------------------- #
    def bars(self, framing, percents=PERCENTS):
        """Return one bar dict per percent for the given framing.

        framing 'sample' (constrain by sample %): height = vocabulary size
        required to keep that fraction of samples.
        framing 'vocab'  (constrain by vocab %):  height = number of samples
        usable when the vocabulary is clamped to that fraction of ``V0``.

        Each dict: ``percent, value, k, admitted, y_max, y_label``.
        """
        out = []
        for p in percents:
            if framing == "sample":
                m = max(1, int(np.ceil(p / 100.0 * self.N)))
                k = int(self._sorted_thr[min(m, self.N) - 1])
                out.append({"percent": p, "value": k, "k": k, "admitted": m,
                            "y_max": self.V0, "y_label": "vocab size"})
            else:
                k = max(1, int(round(p / 100.0 * self.V0)))
                usable = int(np.count_nonzero(self.thresholds <= k))
                out.append({"percent": p, "value": usable, "k": k,
                            "admitted": usable, "y_max": self.N,
                            "y_label": "# samples"})
        return out

    def select(self, k):
        """Vocabulary (top-``k`` tokens) and sample refs admitted by budget ``k``."""
        k = int(max(1, min(k, self.V0)))
        idx = np.nonzero(self.thresholds <= k)[0]
        admitted = [self.sample_refs[i] for i in idx]
        return self.vocab[:k], admitted

    # -- metrics ---------------------------------------------------------- #
    def metrics(self, natural_vocab):
        total_tokens = sum(self.freqs)
        roots = set(onto.CATEGORY_ROOT.values())
        symbols = sum(1 for w in self.vocab
                      if w in roots or (w.isupper() and "_" in w))
        names = self._distinct_names()
        return {
            "natural_vocab": natural_vocab,
            "reduced_vocab": self.V0,
            "reduction_ratio": round(self.V0 / natural_vocab, 4) if natural_vocab else 0,
            "n_samples": self.N,
            "n_stories_active": len(self.active),
            "unit": self.config.unit,
            "unique_ontology_symbols": symbols,
            "canonical_names_introduced": names,
            "avg_tokens_per_sample": round(total_tokens / self.N, 2) if self.N else 0,
            "total_tokens": total_tokens,
        }

    def _distinct_names(self):
        banks = set()
        for cat in ("character", "place", "animal", "object", "other"):
            banks.update(onto._bank_for(cat, ""))
        return sum(1 for w in self.vocab if w in banks)


def compute_reduction(records, config, active=None):
    """Convenience: build a one-off :class:`Corpus` and reduce (no cache reuse)."""
    return Corpus(records).reduce(config, active=active)


def sample_active_set(n_stories, rate, seed):
    """Story indices for a without-replacement subsample at ``rate`` in (0,1].

    Deterministic given ``(n_stories, rate, seed)`` -- ``seed`` bundles the base
    seed and the resample nonce so the Resample button just changes ``seed``.
    Returns a sorted list; the full range when ``rate >= 1``.
    """
    if rate >= 1.0:
        return list(range(n_stories))
    size = max(1, int(round(rate * n_stories)))
    idx = random.Random(seed).sample(range(n_stories), size)
    idx.sort()
    return idx


# --------------------------------------------------------------------------- #
#  Entity-count histograms (status bar)
# --------------------------------------------------------------------------- #
def entity_count_histograms(records, max_bucket=5):
    """Fraction of stories with 1..max_bucket(+) distinct entities.

    Returns ``{'all': [...], 'character': [...], 'place': [...]}`` -- each a
    list of length ``max_bucket`` giving the fraction of stories with exactly
    1,2,...,(max_bucket)+ distinct entity surfaces of that group.
    """
    groups = {"all": None, "character": "character", "place": "place"}
    hist = {g: np.zeros(max_bucket, dtype=np.float64) for g in groups}
    n = max(1, len(records))
    for rec in records:
        surf = {g: set() for g in groups}
        for e in rec["ents"]:
            surf["all"].add(e["surface"])
            if e["category"] == "character":
                surf["character"].add(e["surface"])
            elif e["category"] == "place":
                surf["place"].add(e["surface"])
        for g in groups:
            c = len(surf[g])
            if c >= 1:
                hist[g][min(c, max_bucket) - 1] += 1
    return {g: (hist[g] / n).tolist() for g in groups}


# --------------------------------------------------------------------------- #
#  Export
# --------------------------------------------------------------------------- #
def _admitted_story_indices(reduction, k):
    """Story indices in the active set covered by the top-``k`` vocabulary.

    Restricted to ``reduction.active`` (the sub-sample), so exports never reach
    outside the active set.  With ``k is None`` the whole active set is returned.
    """
    active = set(reduction.active)
    if k is None:
        return active
    thr = reduction.thresholds
    refs = reduction.sample_refs
    if reduction.config.unit == "story":
        covered = {refs[i][1] for i in np.nonzero(thr <= k)[0]}
    else:
        # sentence unit: a story is admitted iff *all* its sentences are covered.
        worst = {}
        for i, ref in enumerate(refs):
            worst[ref[1]] = max(worst.get(ref[1], 0), int(thr[i]))
        covered = {s for s, w in worst.items() if w <= k}
    return active & covered


def export_corpus(path, records, config, reduction=None, k=None):
    """Write the reduced corpus (readable, ``<|endoftext|>``-delimited).

    Each story becomes one-line-per-sentence normalized tokens.  If ``k`` is
    given (a vocabulary budget), only stories fully covered by the top-``k``
    tokens are written ("filtered corpus").  Returns the ``Counter`` of tokens
    actually written (the exported vocabulary).
    """
    syn = effective_synonyms(config)
    keep = _admitted_story_indices(reduction, k) if reduction is not None else None
    written = Counter()
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            if keep is not None and rec["index"] not in keep:
                continue
            text = render_story(rec, config, syn)
            written.update(text.split())
            fh.write(text)
            fh.write(f"\n{STORY_DELIM}\n")
    return written


def export_bundle_json(path, config, reduction, natural_vocab, meta,
                       written_counts=None, selection=None, sampling=None):
    """Write the reproducibility bundle: config + ontology + subs + vocab + metrics."""
    # Per-story substitution maps (only active stories that actually substitute).
    subs = {}
    active = set(reduction.active)
    for rec in reduction._records:
        if rec["index"] not in active:
            continue
        m = _story_substitution(rec, config)
        if m:
            subs[str(rec["index"])] = m

    vocab_counts = written_counts if written_counts is not None else reduction.counts
    vocab_ranked = vocab_counts.most_common()

    bundle = {
        "format": "tinystories-reduced-bundle",
        "format_version": 1,
        "tokenizer": {
            "note": "Reduced corpus is whitespace-tokenizable. Split each line "
                    "on spaces; the multiset of tokens equals this vocabulary.",
            "delimiter": STORY_DELIM,
        },
        "corpus_meta": meta,
        "config": config.to_dict(),
        "sampling": sampling,
        "selection": selection,
        "metrics": reduction.metrics(natural_vocab),
        "ontology": onto.ontology_snapshot(),
        "substitutions": subs,
        "vocabulary": [{"token": t, "freq": c, "rank": i + 1}
                       for i, (t, c) in enumerate(vocab_ranked)],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(bundle, fh, ensure_ascii=False, indent=2)
    return bundle


def verify_reproducible(corpus_path, json_path):
    """Re-tokenize the exported corpus and assert it matches the JSON vocabulary.

    Returns ``(ok, detail)``.
    """
    with open(json_path, "r", encoding="utf-8") as fh:
        bundle = json.load(fh)
    expected = Counter({v["token"]: v["freq"] for v in bundle["vocabulary"]})

    got = Counter()
    with open(corpus_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line == STORY_DELIM:
                continue
            got.update(line.split())

    if got == expected:
        return True, f"OK: {len(expected)} tokens, {sum(expected.values())} occurrences match."
    only_exp = set(expected) - set(got)
    only_got = set(got) - set(expected)
    freq_diff = [t for t in (set(expected) & set(got)) if expected[t] != got[t]]
    return False, (f"MISMATCH: {len(only_exp)} tokens only in JSON, "
                   f"{len(only_got)} only in corpus, {len(freq_diff)} freq diffs.")


# --------------------------------------------------------------------------- #
#  Headless smoke test
# --------------------------------------------------------------------------- #
def _smoke(n_stories=400):
    """Load a slice, annotate, reduce, export, and verify reproducibility."""
    import tempfile
    print(f"[smoke] loading corpus slice ({n_stories} stories)...")
    stories = load_corpus()[:n_stories]

    def prog(done, total):
        if done == total or done % 200 == 0:
            print(f"[smoke]   annotated {done}/{total}")

    records = annotate(stories, progress=prog)
    nat = natural_vocab_size(records)
    print(f"[smoke] stories={len(records)} sentences={count_sentences(records)} "
          f"natural_vocab={nat}")

    corpus = Corpus(records)
    for unit in ("story", "sentence"):
        cfg = OntologyConfig(unit=unit)
        red = corpus.reduce(cfg)
        print(f"[smoke] unit={unit}: reduced_vocab={red.V0} "
              f"samples={red.N} ratio={red.V0 / nat:.3f}")
        print(f"[smoke]   bars(sample)={[b['value'] for b in red.bars('sample')]}")
        print(f"[smoke]   bars(vocab) ={[b['value'] for b in red.bars('vocab')]}")

    # Cache reuse: an entity-only change should NOT rebuild the base cache.
    sig_before = corpus._lex_sig
    red2 = corpus.reduce(OntologyConfig(unit="sentence", name_mode="symbol"))
    assert corpus._lex_sig == sig_before, "entity-only change rebuilt base cache"
    print(f"[smoke] entity-only recompute (symbol mode) vocab={red2.V0} "
          f"(cache reused: {corpus._lex_sig == sig_before})")

    # Sub-sampling: half the stories, deterministic; different seed -> different set.
    a1 = sample_active_set(len(records), 0.5, seed=1)
    a1b = sample_active_set(len(records), 0.5, seed=1)
    a2 = sample_active_set(len(records), 0.5, seed=2)
    assert a1 == a1b and a1 != a2 and len(a1) == len(records) // 2, "bad sampling"
    red_sub = corpus.reduce(OntologyConfig(unit="story"), active=a1)
    print(f"[smoke] subsample 50%: active={len(red_sub.active)} "
          f"samples={red_sub.N} vocab={red_sub.V0}")

    # Export + verify on the story-unit reduction with a mid budget.
    cfg = OntologyConfig(unit="story")
    red = corpus.reduce(cfg)
    bar = red.bars("vocab")[3]           # 75% vocab budget
    tmp = tempfile.mkdtemp(prefix="tinystories_")
    corpus_out = os.path.join(tmp, "reduced_corpus.txt")
    json_out = os.path.join(tmp, "bundle.json")
    written = export_corpus(corpus_out, records, cfg, red, k=bar["k"])
    export_bundle_json(json_out, cfg, red, nat,
                       {"note": "smoke"}, written_counts=written,
                       selection={"framing": "vocab", "percent": 75, "k": bar["k"]})
    ok, detail = verify_reproducible(corpus_out, json_out)
    print(f"[smoke] export k={bar['k']} -> {len(written)} tokens; verify: {detail}")
    assert ok, detail

    # Show a sample reduced story.
    print("[smoke] sample reduced story:")
    print(render_story(records[3], cfg))
    print("[smoke] OK")


if __name__ == "__main__":
    _smoke()
