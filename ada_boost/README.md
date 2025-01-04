# AdaBoost

Boosting algorithms build an accurate ensemble classifer from $N$ "weak learners", i.e. classifiers that do little better than guessing.  Let $X$ be the set of $I$ input points, and $Y$ be the set of their labels,  $y_i \in \{-1, 1\}$.  Let each weak learner, $f_j(x_i)$, output its classification of $x_i$ also as a value in $\{-1, 1\}$. 

Then the goal of [AdaBoost](https://en.wikipedia.org/wiki/AdaBoost) is to learn an ensemble classifier of the form

$$
F(x_i) = \text{Sign}\left(\sum_{ja=1}^N \alpha_j f_j(x_i) \right) \text{   where}
$$

$$
\text{Sign}(v) =  \begin{cases}
-1 & \text{if } v < 0 \\
1 & \text{otherwise.}\\
\end{cases}
$$
I.e. a linear combination of the weak-learner classifiers.  AdaBoost learns each term sequentially: 
* each new weak learner $f_{j+1}(x)$ is trained input that has been re-weighted to focus more on samples that the combined efforts of the previous learners ($1$ through $j$) continue to misclassify.  
* The boosting weight $\alpha_{j+1}$ is then chosen to optimize the output of the ensemble $F(x)$ over the dataset.

### AdaBoost demo

Run `> python ada_demo.py -h` to see the options, most notably `--kind`, determining the dataset used. There are four options:
 * `minimal`: A small dataset with only 6 points to illustrate the algorithm clearly.
 * `bump`: 