# App general description

We are designing a simple TK app to investigate the storage capacity of the Willshaw Feature Attractor, the error-correcting ability, and to demonstrate it's working operation (how it works).

Start here for references:
* http://www.scholarpedia.org/article/Confabulation_theory_(computational_intelligence)
* https://cseweb.ucsd.edu/~atsmith/POL_paper.pdf


It will generate V patterns of exactly S random units active in a D-dimensional setting 1<=S<<D. 

The app has controls on a narrow left-hand frame, Status/title frame at the top, and a "Main view" that is the rest of the window.

The main view has three views (implemented as tabs with labels  in large, noticeable font ): "Demonstration", "Simulation", "Parameter Grid".  The Demo is to visually illustrate the working of a small willshaw attractor.  The Simulation tab runs a monte-carlo simulation of  large willshaw attractors with user-determined parameters.   continuously to accumulate success/failure statistics.  The "parameter Grid"  is to repeatedly run the simulation for different values of the parameters to show asymptotic growth patterns.

## Common controls (to both tabs)

Along the left side are the controls in a relatively narrow panel used for all three modes: 
* Vertical slider for Dimensionality (10-10000 in the simulation, log-scaled, 10 - 100 in Demonstration.
* Vertical slider for Number of Patterns (V), (.1*D - 10*D), log scaled
* Vertical slider for Sparsity S (.01 - 50%)
* Vertical slider for Treshold with a button for theoretical optimal (and a line indicating where that is on the slider)  (See thresholds.md)

Buttons for theoretical optimal values of S given D (to maximize recovery rates , minimize iterations, etc).  I'm not sure how many or what the equations for this are, so look those up.

For the demonstration mode, setting controls resets the main view as described below.
For the simulation mode, changing controls stops the current simulation if its running, changes start/stop button (in the simulation tab) so it can be started again (and activates a "revert & resume" button).  If a new sim starts, old stats are cleared
For the Parameter Grid mode, these controls define the upper limits of the grid search. 


## Demo tab

Two horizontal sliders on top left controls the "damage" that is done to each pattern before the attractor function "recovers" iteratively recovers it.  "Inhibition noise rate" and "Excitation noise rate" (both 0-1, 1% increments), and a checkbox to keep them equal.  Inhibition frac turns off the indicated proportion of each pattern's valid activations.   Excitation Frac turns on random (non-pattern) units.  E.g. if they're equal and at 25%, one quarter of each pattern's active units will be randomly changed to "damage" the signal. 

### Pattern frame / selection

When activated (or when controls change parameters):  Generate the pattern set and the attractor matrix:  Show all patterns in a scrollable frame on the left side of the demo tab.  These are represented as a short (about half text height) lines of light gray (0) and black (1) to represent the sparse vector. (e.g. 010 is Gray-black-gray segments of equal length, etc).   The user can click one to show its "damage and recovery" iteration (The attractor iterations.  Each time the user clicks a symbol, a random 

### Attractor frame 

This shows a black and white heatmap of the attractor matrix W centered in the frame, with the input as a vertical vector on the left and the "output" as two horizontal vectors beneath it (raw excitation, & thresholded activation), and the correct output beneath those two for comparison.

Operation:  When a pattern is clicked by the user, it is "damaged" by the above process, then it shows as a color "vector" left of the attractor matrix (noisy spurious activations are dark red, noisy missing activations light green, correct activations black, aligned by row to the matrix.  This action highlights rows next to active neurons in the pattern in blue, And the sum of those rows appears in greyscale below the W matrix (Excitation). Below that is the thresholded binary output of that iteration and below that is the reference pattern (complete, noise free)

### Iteration frame

Operation When a pattern is selected by the user, The output of each iteration is drawn as vertical color vectors, also including the initial input on the left, and the reference output on the right.  WHenever the user mouses over one of these vectors, it is selected as the iteration number / input to show in the attractor frame.






## Simulation Tab


It will run a monte-carlo simulation continuously when in "run" mode (controlled by a start/stop button) in the top left of its tab.  Also a button for n Trials per pattern set (generate a new pattern set, new network, start experiment again)
1 trial is: For all symbols:
  Pick a partial activation fraction f, choose f * S of that pattern's active neurons to be on (partial activation), count the number of willshaw iterations it takes to get to the complete pattern or up to some maximum (default 10), determine the number of neurons that are correct / incorrect (this is the pattern recovery rate.   The MC simulation is to count the fraction of correct activations for the symbol, determine the number of symbols that can be recovered 100% correct, and in bins with boundaries between 100% and 90%, 80% 70%, 60%, 50%, 25%, 10%, 0%.  For each bin, also compute the average rate of incorrect activations.
 THe "main view" is bar graphs of these statistics (and anything else interesting?)

The simulation should utilize all cores on the machine (via multiprocessing), should bite off chunks to work on of reasonable size to minimize multiprocessing overhead. 



# Parameter grid tab (unimplemented for now, don't even draw it)