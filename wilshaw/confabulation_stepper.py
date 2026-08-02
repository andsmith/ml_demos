"""
Pure-logic stepping engine for the confabulation demo app -- no Tk here,
mirrors ``willshaw_core.py``'s role as the numeric/algorithmic core kept
separate from drawing (see ``module_row_art.py`` / ``nlp_demo_tabs.py``).

Decomposes ``confabulation_high_level.py``'s bulk ``infer_next_word`` /
``infer_joint_pair`` into single atomic actions -- one lateral contribution
from one neighbor, or one reduce-by-half attractor step -- so a GUI can
animate them one "Step" click at a time, per ``nl_demo_tab.md``: "we add
excitations to the units doing inference one word at a time... then the
next click of step starts the reduction/willshaw phase (reduce by half
each time if more than one, otherwise take top candidate)."

Reuses ``confabulation_high_level``'s internal combination helpers
directly (``_kb_key``, ``_weighted_row``, ``_finalize_weights``, ``_topn``,
``_EPS``) rather than duplicating them -- pragmatic in-project reuse
across two modules by the same author, not a public library boundary.
"""

import numpy as np

import willshaw_core
from confabulation_high_level import _EPS, _finalize_weights, _kb_key, _topn, _weighted_row

DEFAULT_LIST_N = 16


# --------------------------------------------------------------------------- #
#  Cosmetic word -> sparse-code encoding (never used by the inference math)
# --------------------------------------------------------------------------- #
def build_word_codes(vocab, D, rng):
    """
    Purely cosmetic per-word sparse binary codes, for module-art rendering
    of "locked down" words as sparse activations. Confabulation inference
    never touches these -- it operates entirely on word strings and the
    knowledge-base probability tables (see ``nlp_expansion.md``).

    :param vocab: list of V vocabulary tokens
    :param D: dimensionality
    :param rng: a numpy Generator
    :return: ``(codes, S)`` -- ``codes`` is a ``(V, S)`` int array (row mu
        is ``vocab[mu]``'s sorted active-index code), ``S`` the sparsity used
    """
    S = willshaw_core.optimal_sparsity(D)
    codes = willshaw_core.make_patterns(D, S, len(vocab), rng)
    return codes, S


# --------------------------------------------------------------------------- #
#  One inferring (or locked) unit
# --------------------------------------------------------------------------- #
class UnitState:
    """
    One word module mid-inference: accumulates lateral excitation from its
    neighbors, then performs the reduce/attractor step.

    :param label: identifies this unit for display (e.g. an int sentence
        position for absolute mode, or a short string like "next" for
        relative mode)
    :param neighbors: ordered list of ``(neighbor_label, kb_key, source)``
        -- ``source`` is either a fixed candidate list ``[(word, 1.0)]``
        (a known word) or another ``UnitState`` whose *current*
        ``.candidates`` is read live each time this neighbor is applied
        (mutual coupling between two jointly-unknown units)
    """

    def __init__(self, label, neighbors):
        self.label = label
        self.neighbors = neighbors
        self.candidates = None       # [(word, weight), ...] -- last *committed* (post-reduce) list
        # Live, non-committed preview of the in-progress lateral phase, so a
        # GUI can redraw after every single step (not just at reduce time).
        # Mutual-coupling neighbor lookups (apply_neighbor's `source
        # .candidates`) must never see this -- they read the other unit's
        # last committed round, per the synchronous/Jacobi update -- so it's
        # kept entirely separate from `.candidates`.
        self.display_candidates = None
        self.locked_word = None      # set once len(candidates) == 1
        self.list_n = DEFAULT_LIST_N
        self.log_excitation = None   # (V,) float64, accumulated this round
        self.n_contrib = 0
        self.applied_count = 0       # how many of self.neighbors have been applied so far

    @property
    def has_live_neighbors(self):
        """A "live" neighbor (another UnitState, mutual coupling) can
        change between rounds, so its lateral phase must be redone every
        round. A unit with only fixed (known-word) neighbors never
        changes, so its lateral phase runs once and later rounds just
        reduce. Computed live (not cached) since ``joint_engine`` fills in
        ``.neighbors`` after construction (the two units reference each
        other)."""
        return any(not isinstance(src, list) for _, _, src in self.neighbors)

    @property
    def converged(self):
        return self.locked_word is not None

    def start_round(self, V):
        """Reset the excitation accumulator for a fresh lateral phase."""
        self.log_excitation = np.zeros(V, dtype=np.float64)
        self.n_contrib = 0
        self.applied_count = 0

    def apply_neighbor(self, kb_key, source, kb_set, index, vocab):
        """
        Apply one ``(kb_key, source)`` lateral contribution, then refresh
        ``display_candidates`` so a GUI redraw right after this call shows
        the accumulated-so-far excitation (not just at the next reduce).

        :return: True if it actually added evidence (KB present, source
            resolvable in vocab), False if silently skipped
        """
        self.applied_count += 1
        applied = False
        kb = kb_set.get(kb_key)
        if kb is not None:
            candidates = source if isinstance(source, list) else source.candidates
            if candidates:
                row = _weighted_row(kb, index, candidates)
                if row is not None:
                    self.log_excitation += np.log(row + _EPS)
                    self.n_contrib += 1
                    applied = True
        self._refresh_preview(vocab, kb_set)
        return applied

    def _refresh_preview(self, vocab, kb_set):
        """Non-committing preview of the current partial-round excitation
        (see ``display_candidates``). No-op until at least one neighbor
        has contributed evidence this round."""
        if self.n_contrib == 0:
            return
        weights = _finalize_weights(self.log_excitation, self.n_contrib, kb_set, vocab)
        n = self.list_n if self.candidates is None else len(self.candidates)
        self.display_candidates = _topn(vocab, weights, n)

    def reduce(self, vocab, kb_set):
        """
        Attractor/willshaw step: seed the top-``list_n`` candidates (first
        call after a lateral phase) or halve the current list (subsequent
        calls), locking a single word once the list reaches size 1.
        """
        weights = _finalize_weights(self.log_excitation, self.n_contrib, kb_set, vocab)
        n = self.list_n if self.candidates is None else max(1, len(self.candidates) // 2)
        self.candidates = _topn(vocab, weights, n)
        self.display_candidates = self.candidates
        self.list_n = n
        if n == 1:
            self.locked_word = self.candidates[0][0]
        return self.candidates


# --------------------------------------------------------------------------- #
#  Engine: owns one or more units + the shared lateral/reduce phase pointer
# --------------------------------------------------------------------------- #
class StepEngine:
    """
    Drives one or more :class:`UnitState`\\ s through alternating lateral
    (one neighbor at a time, across all still-active units) and reduce
    (halve every still-active unit's list) phases, until every unit has
    converged to a single locked word.
    """

    def __init__(self, units, vocab, index, kb_set):
        self.units = units
        self.vocab = vocab
        self.index = index
        self.kb_set = kb_set
        self._lateral_queue = []
        self._round_index = 0
        self._begin_round()

    def _active_units(self):
        return [u for u in self.units if not u.converged]

    def _begin_round(self):
        active = self._active_units()
        V = len(self.vocab)
        # A unit with no live (mutual) neighbors only needs its lateral
        # phase on round 0 -- its excitation never changes, so later rounds
        # reuse it and just reduce (see UnitState.has_live_neighbors).
        queue_units = [u for u in active if self._round_index == 0 or u.has_live_neighbors]
        for u in queue_units:
            u.start_round(V)
        self._lateral_queue = [(u, nb_label, kb_key, source)
                                for u in queue_units for (nb_label, kb_key, source) in u.neighbors]
        self._round_index += 1

    @property
    def is_done(self):
        return not self._active_units()

    def next_action(self):
        """
        Describe the effect of the next :meth:`step` call.

        :return: ``(description, highlight)`` -- ``highlight`` is
            ``(unit_label, neighbor_label, kb_key)`` for the next KB arrow
            to draw in neon green during a lateral phase, or ``None``
            during a reduce phase / once done
        """
        if self.is_done:
            return "All units converged.", None
        if self._lateral_queue:
            u, nb_label, kb_key, source = self._lateral_queue[0]
            # nb_label matches the on-screen unit label (for arch drawing);
            # for a known-word neighbor, describe it by the actual word.
            desc = source[0][0] if isinstance(source, list) else f"word {nb_label}"
            return (f"Apply excitation from {desc} to word {u.label}.",
                    (u.label, nb_label, kb_key))
        active = self._active_units()
        sizes = ", ".join(
            f"word {u.label}: {len(u.candidates) if u.candidates else u.list_n}"
            for u in active)
        return f"Reduce candidate list(s) ({sizes}).", None

    def arch_links(self):
        """
        Every KB link relevant to the units still being inferred, for arch
        drawing (see ``module_row_art.draw_kb_arches``).

        :return: list of ``(src_label, dst_label, kb_key, state)`` where
            ``state`` is ``'applied'`` (already used this round, or every
            round for a unit whose neighbors never change), ``'next'``
            (the very next :meth:`step` will apply this one), or
            ``'pending'`` (queued for later this round)
        """
        head = self._lateral_queue[0] if self._lateral_queue else None
        links = []
        for u in self._active_units():
            for i, (nb_label, kb_key, _source) in enumerate(u.neighbors):
                if i < u.applied_count:
                    state = 'applied'
                elif head is not None and head[0] is u and i == u.applied_count:
                    state = 'next'
                else:
                    state = 'pending'
                links.append((nb_label, u.label, kb_key, state))
        return links

    def step(self):
        """Advance exactly one atomic action. Returns False once done."""
        if self.is_done:
            return False
        if self._lateral_queue:
            u, _nb_label, kb_key, source = self._lateral_queue.pop(0)
            u.apply_neighbor(kb_key, source, self.kb_set, self.index, self.vocab)
            return True
        for u in self._active_units():
            u.reduce(self.vocab, self.kb_set)
        if not self.is_done:
            self._begin_round()
        return True


# --------------------------------------------------------------------------- #
#  Constructors for the three demos
# --------------------------------------------------------------------------- #
def _known_label(mode, k, word):
    """
    Display label for the ``k``-th (0-indexed) known word -- must match
    ``nlp_demo_tabs._display_units``'s labeling exactly (1-indexed sentence
    position for absolute mode, the word text itself for relative mode) so
    arch-drawing can match a neighbor link's label to that unit's on-screen
    x-position.
    """
    return (k + 1) if mode == "absolute" else word


def single_engine(known_words, mode, kb_set, vocab, index, target_pos=6,
                   list_n=DEFAULT_LIST_N):
    """Demo 1: infer one unknown word (default: the 6th) from known_words."""
    n_known = len(known_words)
    label = target_pos if mode == "absolute" else "next"
    neighbors = [(_known_label(mode, k, w),
                  _kb_key(mode, k, n_known, target_pos, rel_slot=1), [(w, 1.0)])
                 for k, w in enumerate(known_words)]
    unit = UnitState(label, neighbors)
    unit.list_n = list_n
    return StepEngine([unit], vocab, index, kb_set)


def joint_engine(known_words, mode, kb_set, vocab, index, target_pos=6,
                  list_n=DEFAULT_LIST_N):
    """Demo 2: jointly infer the adjacent pair (default: positions 5, 6)."""
    n_known = len(known_words)
    if mode == "absolute":
        pos_a, pos_b = target_pos - 1, target_pos
        mutual_key_b_from_a, mutual_key_a_from_b = (pos_a, pos_b), (pos_b, pos_a)
    else:
        pos_a, pos_b = "next", "next2"
        mutual_key_b_from_a, mutual_key_a_from_b = 1, -1

    unit_a = UnitState(pos_a, [])
    unit_b = UnitState(pos_b, [])
    unit_a.list_n = unit_b.list_n = list_n
    unit_a.neighbors = [(_known_label(mode, k, w),
                         _kb_key(mode, k, n_known, pos_a, rel_slot=1), [(w, 1.0)])
                         for k, w in enumerate(known_words)] + [(pos_b, mutual_key_a_from_b, unit_b)]
    unit_b.neighbors = [(_known_label(mode, k, w),
                         _kb_key(mode, k, n_known, pos_b, rel_slot=2), [(w, 1.0)])
                         for k, w in enumerate(known_words)] + [(pos_a, mutual_key_b_from_a, unit_a)]
    return StepEngine([unit_a, unit_b], vocab, index, kb_set)


class SlidingRunner:
    """
    Demo 3: repeated Demo-1-style single-unit inference over a sliding
    window, shifting left and generating one new word each time the
    current unit converges, up to ``max_steps`` words (there is no period
    token in the reduced vocabulary to stop on -- see ``nl_demo_tab.md``
    plan notes).
    """

    def __init__(self, seed_words, mode, kb_set, vocab, index, max_steps=12,
                 target_pos=6, list_n=DEFAULT_LIST_N):
        self.mode = mode
        self.kb_set = kb_set
        self.vocab = vocab
        self.index = index
        self.target_pos = target_pos
        self.list_n = list_n
        self.window_size = target_pos - 1
        self.window = list(seed_words[-self.window_size:])
        self.history = list(seed_words)
        self.max_steps = max_steps
        self.n_generated = 0
        self.engine = self._new_engine()

    def _new_engine(self):
        return single_engine(self.window, self.mode, self.kb_set, self.vocab,
                              self.index, target_pos=self.target_pos, list_n=self.list_n)

    @property
    def done(self):
        return self.n_generated >= self.max_steps

    def next_action(self):
        if self.done:
            return "Step cap reached.", None
        return self.engine.next_action()

    def step(self):
        if self.done:
            return False
        changed = self.engine.step()
        if self.engine.is_done:
            new_word = self.engine.units[0].locked_word
            self.history.append(new_word)
            self.n_generated += 1
            self.window = (self.window + [new_word])[-self.window_size:]
            if not self.done:
                self.engine = self._new_engine()
        return changed


# --------------------------------------------------------------------------- #
#  Self-test
# --------------------------------------------------------------------------- #
def _selftest():
    from extract_knowledge_base import extract_absolute, extract_relative
    import nlp_core as core

    # --- Demo 1: single-unit stepping should match infer_next_word's answer ---
    abs_sentences = [["A", "B", "C", "D", "E", "F"]] * 30
    abs_sentences += [["A", "B", "C", "D", "E", "X"]] * 10
    vocab, index = core.build_vocab(abs_sentences)
    abs_kbs = extract_absolute(abs_sentences, vocab, index,
                                [(1, 6), (2, 6), (3, 6), (4, 6), (5, 6)], alpha=1.0)
    abs_kb_set = {k: {**v, "vocab": vocab} for k, v in abs_kbs.items()}

    engine = single_engine(["A", "B", "C", "D", "E"], "absolute", abs_kb_set, vocab, index)
    n_steps = 0
    while not engine.is_done:
        assert engine.step()
        n_steps += 1
    assert engine.units[0].locked_word == "F", engine.units[0].locked_word
    # 5 lateral (one per known word, once) + reduce steps: seed to
    # min(list_n, V) candidates, then halve until 1 (V=7 here: A-F, X).
    n = min(DEFAULT_LIST_N, len(vocab))
    n_reduce = 1
    while n > 1:
        n = max(1, n // 2)
        n_reduce += 1
    assert n_steps == 5 + n_reduce, (n_steps, n_reduce)
    print(f"[selftest] Demo1 engine: converged to 'F' in {n_steps} steps "
          f"(5 lateral + {n_reduce} reduce) OK")

    # step() returns False once done, next_action reports convergence
    assert engine.step() is False
    assert engine.next_action()[0] == "All units converged."
    print("[selftest] Demo1 engine: post-convergence step()/next_action() OK")

    # --- Demo 2: joint stepping should match infer_joint_pair's answer ---
    joint_sentences = [["a", "b", "c", "d", "cat", "meows"]] * 30
    joint_sentences += [["a", "b", "c", "d", "dog", "barks"]] * 25
    joint_sentences += [["a", "b", "c", "d", "cat", "barks"]] * 1
    jvocab, jindex = core.build_vocab(joint_sentences)
    j_pairs = [(1, 5), (2, 5), (3, 5), (4, 5), (1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 5)]
    j_kbs = extract_absolute(joint_sentences, jvocab, jindex, j_pairs, alpha=1.0)
    j_kb_set = {k: {**v, "vocab": jvocab} for k, v in j_kbs.items()}

    jengine = joint_engine(["a", "b", "c", "d"], "absolute", j_kb_set, jvocab, jindex)
    while not jengine.is_done:
        jengine.step()
    words = {u.label: u.locked_word for u in jengine.units}
    assert words == {5: "cat", 6: "meows"}, words
    print(f"[selftest] Demo2 engine: converged to {words} OK")

    # --- Demo 3: sliding window should generate max_steps words ---
    rel_sentences = [["once", "upon", "a", "time", "there", "was", "a", "cat"]] * 40
    rvocab, rindex = core.build_vocab(rel_sentences)
    r_kbs = extract_relative(rel_sentences, rvocab, rindex, [1, 2, 3, 4, 5], alpha=1.0)
    r_kb_set = {k: {**v, "vocab": rvocab} for k, v in r_kbs.items()}

    runner = SlidingRunner(["once", "upon", "a", "time", "there"], "relative",
                            r_kb_set, rvocab, rindex, max_steps=3)
    guard = 0
    while not runner.done and guard < 1000:
        runner.step()
        guard += 1
    assert runner.n_generated == 3, runner.n_generated
    assert runner.history[-3:] == ["was", "a", "cat"], runner.history
    print(f"[selftest] Demo3 runner: generated {runner.history} OK")

    # --- cosmetic word codes: shape/sparsity sanity, no bearing on inference ---
    codes, S = build_word_codes(vocab, 200, __import__("numpy").random.default_rng(0))
    assert codes.shape == (len(vocab), S)
    print(f"[selftest] build_word_codes: D=200 -> S={S}, codes shape {codes.shape} OK")

    print("[selftest] confabulation_stepper OK")


if __name__ == "__main__":
    _selftest()
