We are going to re-focus the demo on the mechanistic interpretation of the willshaw attractor circuit.


This is all based on confabulation theory: http://www.scholarpedia.org/article/Confabulation_theory_(computational_intelligence)

Use the reference willshaw_dynamics.md if it helps.

This design will center around a 1-hour long demonstration (defined in the "Demo" section of this document, 3 parts).


# Demo (for background, intended purpose of updates)

The demo for people to learn about this topic will consist of the following sequence:

1.  Why does a willshaw attractor work?  Because it is a "Content-addressable memory" - Demonstration tab (see demonstration-redesign) 

(Set for demos in 1:  D=20, S=5)

(Set |V| = 1)

1.a     show part of the pattern and the rest appears
1.a.i   Show W=V'V is just the outer product of sparse vector
1.a.ii  Show each active neuron of a symbol stimulates all other neurons that are part of the symbol
1.a.iii show damaged V[0] - missing 1 activation and 1 spurious activation

(Set |V| = 2)

1.b    Demonstrate sparsity -> non interference of symbol recall
1.b.i  Show W=V'V Adds weights (entries of W) that don't interfere with each other because of sparsity.
1.b.ii Show damaged V[0] and V[1] recall - same damage pattern as before

(Set |V| = 10)

1.c  More symbols -> starting to show interference in W matrix 
1.c. Show damaged recall of symbols sometimes requires multiple attractor iterations to reach stability.


2.  Capacity experiments - Simulation tab (see simulation-redesign)



3. NLP experiments - (new tab) Next Word Confabulation

# Parameters-redesign

Remove threshold slider, threshold rule selection, optimal theta button.  Since we always know S, we will just use Top-S as the rule for 
thresholding.


# demonstration-redesign

Keep the control frame along the top, but on the left side it has a small frame with three buttons, corresponding to demos 1.a, 1.b and 1.c. 
for each part of the demo. Pushing one resets the app state, changes parameters to what are needed for that part of the demo.

Move the damage controls that are currently there to occupy the rest of the control frame.into a new frame at top of the demonstration main view frame.

Clicking a pattern once will run it through the attractor circuit. Clicking a pattern twice will un-select it, just show the W matrix (un-shaded) and remove side vector, excition/thresholded output /reference patter, and clear the iterations.

Also update the iterations so it just shows the binary activations.  Do not evenly space them out horizontally, draw them piling up on the left with reasonable spacing, and separate the refence pattern by a vertical line.  On the right, add a small frame explaining the number of iterations it took to get to stability (if it did)

Clicking the demo buttons will set params according to the above spec.
New feature - right-click selects a second symbol to activate at 50% (randomly select it's active neurons)
Iteration should end up only with the first symbol active for that demo (1.b)

Above the Patterns scrollable list, put a square module representation (module_art.py for drawing these in TK apps). ("pattern module")


Also ad a "live" representation of the module state, next to the W matrix in the main display. Mousing over the iterations shows the EXCITATION in this representation (shades of gray). unless it is moused over, when it shows the threshold version.



When 2 symbols are selected, color-code their representation in the pattern module.  




# simulation-redesign

(hold off for now)
