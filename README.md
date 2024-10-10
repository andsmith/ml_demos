# ml_demos
Demos for Machine Learning Reading Group

## EM Demo

Fit a Gaussian Mixture Model using EM

A [*mixture model*](https://en.wikipedia.org/wiki/Mixture_model) parameterizes the distribution of input variables $x$ as the weighted sum of distributions of $N$ mutually-exclusive generative models (the components), $p(x) = \sum_{y}p(x|y) p(y)$, where $p(x|y=i)$ is the *i*-th component distribution and the value of $p(y=i)$ for all $i$ are the mixing probabilities.

This is equivalent to a generative process that samples $x$ by first sampling $y=i$ according to the mixing probabilities, and then sampling the chosen $x\sim p(x|y=i)$.

If each component is a Gaussian distribution, the resulting model is a GMM.  To see a 1-D demo of the EM-algorithm fitting a GMM run: `> python demo_em.py`.  This will generate a random 4-component GMM, and sample 1000 points from it:

![EM_data](/em/assets/EM_data.png)

The sampled points are plotted on the bottom (y-values are random, to better visualize the distribution of x-values) and a histogram above.  Clicking the image fits the model to a 4-component GMM:


![EM_data](/em/assets/EM_algo.png)
The upper plot shows the evolving gaussian components (solid) as well as the mixture distribution (dashed) fit at each iteration.  After convergence or max_iter is reached, the final model is plot along with the model that generated the original data:

![EM_data](/em/assets/EM_final.png)
The upper plot shows the histogram again, with the distribution that generated it and the distribution of the newly fit GMM.  The lower plot shows the individual (weighted) Gaussian distributions within each model.

#### Options:
```
-h, --help     
-r, --n_real   Number of components generating data.
-f, --n_fit    Number of components in the fit model
-i, --iter     Maximum number of iterations to run
-s, --spread   Spread of the true model (SD of dist between comps)
-p, --n_points Number of data points to generate
```            