# overview

 This is a spec for a stand-alone app that will demonstrate high-level confabulation. (word prediction)

It is meant to be run by itself (this conversation), or accessed within a planned NLP tab of willshaw_demo.py (future)

# Demos
1 - infer 6th word from start of sentence
2 - infer 5th and 6th word together (simultaneous inference)
3 - run continuously until the period token is encountered (using absolute or relative KBs)


# App layout
All frame dividers are draggable to re-proportion frame sizes/shapes.

## Side-bar
On the left is a column of 2 frames taking up about 25% of the window width, the upper is relatively short: "Title/Status" frame with information about the corpus loaded, KBs loaded, etc. 
The lower is a scrollable window with all the corpus sentences (stories separated by a line).  Clicking one sends it (whichever of its words are relevant) to the main demo window for inference.  There is also a form to enter words manually and a button to send them to the demo

## main frame

The main frame is left of the side-bar is divided into run controls in a thin frame along the top and the "display" frame taking up the rest of the space.

### control bar

It has a big green "Step" button on the left, A slider for the dimensionality (set to 1000 by default), and other controls as described below.  Changing dimensionality resets the app state.

### display frame - tabbed for each demo

This has tabs with large font for the three demos.

Each shows a horizontal row of word modules for consecutive words in a sentence.  Locked-down modules have a single word below them, ones doing inference have a ranked list of candidates (as described below)

# Stepping through the computation

At all times, the KBs that are going to be used the next time "Step" is pressed are colored in neon green (arching arrows described in demo 1).  NExt to the step button is text explaining what clicking it will do

in the lateral phase, we add excitations to the units doing inference one word at a time (if we're inferring 6th word, we first apply influence from first word, then second, ... then fifth.  Then the next click of "step" starts the reduction/willshaw phase (reduce by half each time if more than one otherwise take top candidate.)


## Demo 1 - inferring word 6 given the first 5 words

This shows a row of modules (rendered by module-art) in this simulated region of the cerebral cortex.

For locked-down modules (have a word present) they are shown with their represented word drawn as sparse activations (binary, black for locked-down) corresponding to the vocabulary embedding, with the word in large print below each unit.

The units doing inference will initially be all inactive, then show a simulated "symbol distribution" by determining the ranked lists of words for each unit according to the confabulated probability (top 10) in a 2-column table (%.5f for the probability, then the word) below the unit.  

All knowledge bases in use are drawn as arrows from the "source" to the "destination" unit.  (should be curving arches above the units)

# Demo 2

Similar to demo 1, but using the simulaneous confabulation functionality in confabulation_high_level for words 5 and 6 together. 
# Demo 4)

# Demo 3

Similar to demo 1, but instead of terminating, shifts all words one slot to the left and starts again.
