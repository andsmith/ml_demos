#!/bin/bash
# Extract the full tiny-demo knowledge-base set for the first 6 word
# modules: all absolute position pairs 1<=i<j<=6, plus the one reverse
# pair 6-5, and the same pairs reinterpreted as relative
# (position-invariant) offsets.
#
# Absolute KBs let module 6 (or any later module) be predicted from a
# fixed earlier position -- useful at the start of a sentence, where
# TinyStories openings are templated. Relative KBs let the *next* word be
# predicted from the last few words regardless of where they fall in the
# sentence -- useful mid-sentence.
#
# The reverse pair 6-5 (and its relative counterpart, offset -1) lets
# units 5 and 6 excite each other in both directions, needed to jointly
# infer them together (confabulation_high_level.py's --joint mode).
set -euo pipefail
cd "$(dirname "$0")"

PY=".venv/Scripts/python.exe"

PAIRS=()
for j in 2 3 4 5 6; do
    for ((i = 1; i < j; i++)); do
        PAIRS+=("$i-$j")
    done
done
PAIRS+=("6-5")

echo "== Absolute KBs (${#PAIRS[@]} pairs): ${PAIRS[*]}"
"$PY" extract_knowledge_base.py "${PAIRS[@]}"

echo "== Relative KBs (same pairs, deduped to offsets 1-5)"
"$PY" extract_knowledge_base.py --relative "${PAIRS[@]}"

echo "== Done. Knowledge bases written to ./knowledge_bases"
