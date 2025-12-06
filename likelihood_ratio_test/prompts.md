# ChatGPT prompts for Likelihood Ratio Test Demo

## Initial Prompt

I would like to create a demo for comparing goodness of fit tests using a synthetic dataset and three simple models (underparameterized, "just right", and overparameterized).  This demo will have three plots, one on top of the other, and three horizontal sliders for the controllable parameters, and three lines of text under the sliders.  Updating the sliders should instantly update the plots and text (there is no other interaction).

Specifically:

1. The N_points data points are 1 dimensional (random varaible x), sampled from a 2-gaussian mixture model with unit SD and means separated by mean_dist.  (this is their "true distribution")
2. The three models are:
    "1-comp", a single gaussian fit using sample mean/sd (underparameterized),
    "2-comp", a 2-component gaussian mixture model fit using EM (just right), and 
    "3-comp", a 3-component GMM also fit using EM (overparameterized). 
3. Each plot will show a histogram of the data (using N_bins bins) with the model's fit distribution overlaid as a line, and the subtitle will include the -log(likelihood), perplexity and entropy.  The two and three component models will have dashed lines showing the individual components' distributions.   
4. Underneath the plot are three lines of text showing test statistics comparing:
    - on line 1:  "comp-1" and "comp-2":  (LRT statistic and p-value, AIC score, BIC score.  This should show statistically significant improvement.)
    - on line 2:  The same as line 1, but the null hypothesis will be that "comp-3" represents no improvement over "comp-2".  This should generally be false.
    - on line 3:  The same as lines 1 and 2, but showing the improvement of "comp-3" compared to "comp-1".  This should generally be significant.

Note that the LRT's p-values will be inaccurate because the Wilkes theorem requirements don't apply for GMMs, but we will show this anyway.)

Create a jupyter notebook to show this using the standard python libarries, numpy, scipy, ipywidgets, etc.

### Modification 1
Modify the demo so the plots are on a 2x2 grid (top two and bottom left cells), and the text output lines are in the bottom right cell. Remove the AIC and BIC scores (just focus on LRT). Add hypothetical LRT scores for rejecting null hypothesis with 90% 95% and 99% confidence on each line (given current parameters).

### Modification 2
Modify the demo to resample the synthetic data when a button is pressed, so the sliders are arranged horizontally, and place the "resample" button to the left of the sliders.

### Bug fix 1 (failed)
The most recent update introduced errors, not all plots update when "resample" is pressed (only the 3-component models' best-fit curves, not the histograms and nothing on the other two plots). All plots correctly refresh when the sliders are moved.

[NOTE: named parameter fix suggested in next prompt was for bug introduced by bug fix 1.]

### Bug fix 2 (failed)
Modify code to use kwargs when creating sliders, also fix bug where the mean_dist and n_points sliders only update the bottom plot's curves (should update all parts of all plots).

[NOTE: initial bug was still present, author fixed by changing `resample=true` in `on_slider_change()`, which is overkill, but fast enough.]

[NOTE: The resulting notebook is basically gmm_gof_demo.ipyndb, with minor changes to the text output format and ]