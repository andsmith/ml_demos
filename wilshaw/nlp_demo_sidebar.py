"""
Left sidebar for the confabulation demo app: a short title/status frame
plus a scrollable corpus-sentence browser (click a sentence to load its
words into the active demo tab) and a manual word-entry form. Mirrors
``control_panel.py``'s ``Panel``-subclass structure; the scrollable-list
idea follows ``demo_tab.py``'s pattern list, but uses a plain
``tk.Listbox`` since rows are just text (no per-row custom drawing needed).
"""

import tkinter as tk

from gui_base import Panel, tk_color_from_rgb
from layout import FONTS, COLOR_SCHEME as CS
import nlp_core as core
from tiny_stories import STORY_DELIM

DIVIDER = "─" * 34   # horizontal-line glyph, marks a story boundary


def load_sentences_with_breaks(path=core.CORPUS_PATH):
    """
    Like ``nlp_core.load_sentences``, but also reports which sentence
    indices start a new story (for the sidebar's divider lines) --
    display-only bookkeeping ``nlp_core.py`` itself has no need for.

    :return: ``(sentences, story_break_before)`` -- ``story_break_before``
        is the set of sentence indices that begin a new story
    """
    sentences = []
    story_break_before = set()
    new_story = True
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line == STORY_DELIM:
                new_story = True
                continue
            if new_story:
                story_break_before.add(len(sentences))
                new_story = False
            sentences.append(line.split())
    return sentences, story_break_before


class NLPDemoSidebar(Panel):
    """Title/status + corpus browser + manual entry, shared by all 3 demo tabs."""

    def __init__(self, app, bbox_rel, margin_rel=0.0):
        self._sentences, self._story_breaks = load_sentences_with_breaks()
        self._row_to_sentence = []   # parallel to listbox rows; None for divider rows
        super().__init__(app, bbox_rel, margin_rel)

    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #
    def _init_widgets(self):
        bg = self._color_bg
        tk.Label(self._frame, text="NLP Confabulation Demo", bg=bg,
                 fg=self._color_text, font=FONTS['panel_title'],
                 wraplength=200, justify=tk.LEFT).pack(side=tk.TOP, anchor=tk.W,
                                                        padx=6, pady=(6, 2))
        self._status_lbl = tk.Label(self._frame, text="", bg=bg,
                                     fg=tk_color_from_rgb(CS['text_dim']),
                                     font=FONTS['small'], justify=tk.LEFT,
                                     anchor=tk.W, wraplength=200)
        self._status_lbl.pack(side=tk.TOP, anchor=tk.W, padx=6, pady=(0, 6))
        self.set_status("Loading knowledge bases...")

        tk.Label(self._frame, text=f"Corpus sentences ({len(self._sentences)}) "
                                    f"-- click to load:",
                 bg=bg, fg=self._color_text, font=FONTS['section'],
                 wraplength=210, justify=tk.LEFT).pack(side=tk.TOP, anchor=tk.W,
                                                        padx=6, pady=(4, 2))

        list_wrap = tk.Frame(self._frame, bg=bg)
        list_wrap.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=(6, 2))
        sb = tk.Scrollbar(list_wrap, orient=tk.VERTICAL)
        self._list = tk.Listbox(
            list_wrap, bg=tk_color_from_rgb(CS['view_bg']), fg=self._color_text,
            font=FONTS['small'], activestyle='none', highlightthickness=0,
            selectbackground=tk_color_from_rgb(CS['highlight']),
            yscrollcommand=sb.set, exportselection=False)
        sb.config(command=self._list.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._populate_list()
        self._list.bind("<<ListboxSelect>>", self._on_select)

        entry_fr = tk.Frame(self._frame, bg=bg)
        entry_fr.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(6, 8))
        tk.Label(entry_fr, text="Or enter words manually:", bg=bg,
                 fg=self._color_text, font=FONTS['section']).pack(side=tk.TOP, anchor=tk.W)
        self._entry = tk.Entry(entry_fr, font=FONTS['default'])
        self._entry.pack(side=tk.TOP, fill=tk.X, pady=(2, 2))
        self._entry.bind("<Return>", lambda e: self._on_send())
        tk.Button(entry_fr, text="Send", font=FONTS['buttons'],
                  command=self._on_send).pack(side=tk.TOP, fill=tk.X)

    def _populate_list(self):
        self._row_to_sentence = []
        for i, sent in enumerate(self._sentences):
            if i in self._story_breaks and i != 0:
                self._list.insert(tk.END, DIVIDER)
                self._row_to_sentence.append(None)
            self._list.insert(tk.END, " ".join(sent))
            self._row_to_sentence.append(i)

    # ------------------------------------------------------------------ #
    #  Interaction
    # ------------------------------------------------------------------ #
    def _on_select(self, _event):
        sel = self._list.curselection()
        if not sel:
            return
        sent_idx = self._row_to_sentence[sel[0]]
        if sent_idx is None:
            return
        if hasattr(self.app, 'load_words'):
            self.app.load_words(self._sentences[sent_idx])

    def _on_send(self):
        words = self._entry.get().split()
        if words and hasattr(self.app, 'load_words'):
            self.app.load_words(words)

    # ------------------------------------------------------------------ #
    #  Status text (updated by the app once KBs are loaded / mode changes)
    # ------------------------------------------------------------------ #
    def set_status(self, text):
        self._status_lbl.config(text=text)


# --------------------------------------------------------------------------- #
#  Stand-alone test
# --------------------------------------------------------------------------- #
class _FakeApp(object):
    def __init__(self, root):
        self.root = root

    def load_words(self, words):
        print("load_words:", words)


def _test_sidebar():
    root = tk.Tk()
    root.title("Sidebar Test")
    root.geometry("300x900")
    root.configure(bg=tk_color_from_rgb(CS['bg']))
    app = _FakeApp(root)
    sidebar = NLPDemoSidebar(app, {'x_rel': (0.0, 1.0), 'y_rel': (0.0, 1.0)})
    sidebar.set_status(f"corpus: {core.CORPUS_PATH}\n"
                        f"{len(sidebar._sentences)} sentences")
    root.mainloop()


if __name__ == '__main__':
    _test_sidebar()
