# AdaBoost

Boosting algorithms build an accurate ensemble classifer from $N$ "weak learners", i.e. classifiers that do little better than guessing.  Let $X$ be the set of $I$ input points, and $Y$ be the set of their labels,  $y_i \in \\{-1, 1\\}$.  Let each weak learner, $f_j(x_i)$, output its classification of $x_i$ also as a value in $\\{-1, 1\\}$. 

Then the goal of [AdaBoost](https://en.wikipedia.org/wiki/AdaBoost) is to learn an ensemble classifier of the form

$$
F(x_i) = \text{Sign}\left(\sum_{j=1}^N \alpha_j f_j(x_i) \right) \text{where,}
$$


$$
\begin{equation}
\text{Sign}(v)  = 
\begin{cases}
    -1 & \text{if }~~ v < 0 \\
    1 & \text{otherwise.}
\end{cases}
\end{equation}
$$

I.e. a linear combination of the weak-learner classifiers.  AdaBoost learns each term sequentially: 
* each new weak learner $f_{j+1}(x)$ is trained on input that has been re-weighted to focus more on samples that the combined efforts of the previous learners ($1$ through $j$) continue to misclassify.  
* The boosting weight $\alpha_{j+1}$ is then chosen to optimize the output of the ensemble $F(x)$ over the dataset.

### AdaBoost demo

Run `> python ada_demo.py -h` to see the options, most notably `--kind`, determining the dataset used. There are four options:
 * `minimal`: A small dataset with only 6 points to illustrate the algorithm clearly.
 * `bump`: Two classes separable by a line except for a small "bump" in the decision boundary.
 * `checker`:  classic XOR problem
 * `spiral`:  A spiral shaped decision boundary, tough for any classifier using straight lines.
 
![datasets](/ada_boost/assets/datasets.png)

Running `> python ada_demo.py --kind minimal` demonstrates each iteration, using decision stumps as weak learners.  Since these can only form decision boundaries parallel to the X or Y axis, a single stump is unable to separate the two classes.  The first iteration results in a misclassified point.  This point gets a higher weight in the next iteration, whose weak learner correctly classifies this point but now misclassifies another one.  The weights in the final iteration reflect these two points' history of misclassification and the third learner separates them (misclassifying the two points on the bottom).  The final ensemble separates the classes perfectly:
![demo](/ada_boost/assets/demo_minimal.png)

The left-most column shows the weights given to each sample, all equal in iteration 0.  The second column shows the distribution of those weights (sorted values). The third column shows the weak learner added at each iteartion given the weighted data, its misclassified points (outlined in the correct class color), and its weighted loss (y-axis label).  The fourth column shows the decision boundary of the ensemble after adding each weak learner, with misclassified points highlighted.

### compare with Scikit-learn:

Run `> python ada_test.py bump -n 3 -p 15` to run `sklearn.ensemble.AdaBoostClassifier` on the 'bump' dataset:
![scikit_bump](/ada_boost/assets/scikit_adaboost_bump.png)


### Compare decision boundaries of different classifiers
*coming soon...*