## RANSAC:

[Random Sample Consensus (RANSAC)](https://en.wikipedia.org/wiki/Random_sample_consensus) algorithm is a randomized algorithm for fitting simple models from data with outliers:

Let dataset $D = I \cup O$, where $I$ and $O$ are the set of inlier and outlier points respectively.  The points in $I$ are generated by some noisy process that can be modeled by estimating parameters $P$ using at least $n$ samples, but the estimation process will fail if any of those samples are from $O$.

Instead of using this estimator on $D$, known to contain outliers, if we can find a subset that doesn't, *any* subset, then we can use our estimator on that subset and expect reasonable parameters from it.  We can use the rest of the data to check for reasonableness:  if our subset contained no outliers, the model that was estimated from it will fit many other points in the data (the "consensus set") and, conversely, if the subset contained outliers, the resulting model will match very few.  Once a consensus set is found, the original estimator can be used to get the best possible model, i.e. using all possible inliers and no outliers.

The idea behind RANSAC is the smaller this subset is, the easier it will be to guess randomly, so all we have to do is repeat this until the condition is met:
```
define RANSAC(D, Estimator, n, threshold, min_inliers)
    : D - set of data containing outliers
    : Estimator - func. to get model parameters from a subset of D
    : n - minimum number of elements needed for Estimator
    : min_inliers - minimum size of a "valid" consensus set
    : threshold - defines inlier/outlier (w.r.t the model)
    While True:
        samples <- Randomly choose n items from D
        model <- Estimator(samples)
        inliers <- all x in D st. model(x) <= threshold
        if |inliers| >= min_inliers:
            return Estimator(inliers)
```

Or run for as long as we want and return the model estimated from the largest consensus set:
```
define RANSAC(D, n, Estimator, threshold, max_iter)
    best_inlier_set <- {}
    for max_iter iterations:
        samples <- Randomly choose n items from D
        model <- Estimator(samples)
        inliers <- all x in D st. model(x) <= threshold
        if |inliers| > |best_inlier_set|:
            best_inlier_set <- inliers
    return Estimator(best_inlier_set)
```

## Code outline:

```
ransac_sandbox/
    ├── ransac.py            - base classes for RANSAC
    ├── fit_line.py          - subclasses for 2-d line fitting
    ├── match_images.py      - subclasses for affine transformation estimation 
    ├── demo_fit_line.py     - demo for RANSAC line-fitting algorithm.
    ├── demo_match_images.py - demo for (synthetic) image matching
    ├── tune_corner_detection.py - Iterate through parameter space, plot results.
    └── tune_corner_matching.py  - Iterate through params for corner matcher, plot.
```


## Line fitting demo


RANSAC on a line-fitting dataset with noisy inliers and 50% outliers.

Run: `> python demo_fit_line.py` to start the RANSAC line fitter demo.  The demo will:

* Create of a dataset:
  * 100 (noisy) points on a line 
  * 100 random outliers points 

* Plot of each iteration of the RANSAC algorithm as it attempts to find the line:

![ransac_line_demo](/ransac/assets/ransac_line_demo.png)

* After 20 iterations, the results are plotted, showing:
    * the best model found (the one resulting in the most inliers),
    * the final model (estimated from that largest set), and
    * the least-squares fit to the data, obviously not fit the line in a way that is robust to the outliers:

![ransac_line_final](/ransac/assets/ransac_line_final.png)

The output shows the parameters used to generate the line data, and the parameters as estimated by RANSAC, which should be close (up to sign):

```
RANSAC found a solution on iteration 14:
        Inliers: 109 of 200 (54.5 %)

Estimated params of line:
        -0.856*x + 0.388*y + 0.342 = 0

True params of line:
        0.860*x + -0.357*y + -0.365 = 0
```

##### Animation options

The following can be set by changing the value of the `animate_pause_sec` parameter at the bottom of the demo script to any of these values:
* None: no plot / animation,
* 0: pause, wait for user interaction between each iteration, or
* (any number > 0.0): pause this many seconds between each frame.


## Image transformation estimation demo

How do you align two images, given that one is a noisy, warped, and translated copy of the other?   To estimate the transform from one image to the other we need at least three points in each image that correspond, i.e. are of the same spot.  And if we have more, we can get a better estimate.  The perfect setup for RANSAC!

![synth_img](/ransac/assets/synth_img_cropped.png)

###### RANSAC image alignment algorithm outline:

1.  The [Harris Corner Detector](https://en.wikipedia.org/wiki/Harris_corner_detector) is used to locate a set of "feature points" in image 1 and in image 2.
2.  Rotation-invariant descriptors (local color histograms) are extracted around the locations of corners in both images.
3.  A list of candidate correspondences (pairs of corners, one from each image)  is created by comparing the descriptor of every corner in image 1 to the descriptor of every corner in image 2.  The set of all pairs above some threshold is the set D, containing valid correspondences (the correct matches) and spurious ones (the outliers).

4. Recover the transformation parameters used to create image 2 from image 1 using the RANSAC algorithm.  Repeatedly:
    * draw 3 correspondences randomly from D, pairs of corners that have high similarity,
    * calculate the transformation mapping the 3 points in image 1 to their corresponding locations in image 2,  
    * determine which correspondences in $D$ this preserves, this is the inlier set, and
    * if the inlier set is large enough, return the transform estimated from it, else repeat.

###### Running the demo
Run: `> python demo_match_images.py` to start the RANSAC image alignment demo.  The demo will:

* create a synthetic image,
* apply an affine transformation and add noise to create a second, then
* recover the affine transformation using RANSAC.

### Synthetic images

The first image is created in two steps:
  * The "background" consists of random color circles.
  * The "foreground" is a small number of random rectangles.

The second image is created from the first through this process:
  * A random small rotation, translation, and scaling are used to transform the image (preserving width & height)
  * Some fraction of the pixels are randomly switched to a different color:

![alt text](/ransac/assets/noise.png)

(10% of the pixels are changed to a random color)

### Tuning the corner detector:

The expected corners to detect are the corners of the rectangles and intersections of boundaries between shapes.  The color palette is limited to make the corner matching more challenging (i.e. to make them appear more similar).

A set of parameters for the Harris Corner Detector (kSize, blockSize, and k) need to be found that work in the original and transformed image space.

To automatically determine the best settings, we can define a score for the detector and iterate over many random trials.

#### Scoring the detector on the first image:

For a single image, the only criteria are detecting enough corners but not too many false positives.  To ensure the number is in the correct range for this category of images, we use this function to score the detector on the first image:

![detector_func](/ransac/assets/detection_func_single.png)

(run `> python test_detection_tuner.py` to generate)

#### Scoring the detector on the second (and both) images
To score the second image, transform the corners detected from the first image using the same affine transformation that created the second image.  Call these the "true" corners and calculate the accuracy of the detector in the second image (num correct / num detections).  The "combined" score for this set of parameters is the product of the two scores:

![detector_func](/ransac/assets/detection_tuner_score.png)

(The red dots are the detected corners in both image, the blue + symbols are the corners detected in the first image transformed to the second image's coordinates using the same affine matrix that created the second image from the first.)


To iterate through many combinations of paramers use `tune_corner_detector.py`.  This script samples the parameter space and scores each set of parameters on multiple test images, displaying mean value as the color of a pixel on a grid:

![detector_tuning](/ransac/assets/tune_corner_detect.png)

As the image shows, as long as the blocksize is >= 4 and the kSize parameter (size of the Sobel kernel) is sufficiently large to average out noise, there is a good range of values that should work. 



### Tuning the corner matching algorithm:

A good set of parameters for the corner detector can be used to locate the corners in image 1 and image 2.  Then, to determine which corners in image 2 might correspond to a given corner in image 1, a rotation and noise invariant descriptor is extracted from a small neighborhood around the corners and their descriptors are compared.  

The descriptors are simple color histograms, a vector of how many times each distinct color appears in it, with optional smoothing.  Two descriptors can therefore be compared using any histogram similarity metric. (See `TestImage.compare_descriptors()` for a few examples.)

To explore the effects of changing the descriptor parameters and comparison function, run `tune_corner_matching.py`.  This will create a test pair and run the corner matching algorithm with a desired set of parameters, displaying a few corners from image 1 and a few of the best and worst matching corners from image 2:

![matcher tuning](/ransac/assets/corner_matcher.png)


Next to each image patch is a histogram of the amount of each color in it (a visualization of the corner descriptors).   The numerical comperison is in `image_util.compare_descriptors()`.  Note, the first and fourth corners from image 1 (columns) did not match any corners in image 2 with low distance and correspondingly the smallest score is much higher than the best matches in the other examples.

A cutoff of 0.4 is used for the demo.


### Image matching demo:
Run: `> python demo_match_images.py` to start the demo.  This will show the first plot window with two rows:

![demo start](/ransac/assets/match_demo_start.png)

The upper row shows the input images to align (image 2 being a transformation of image 1) and the corners detected in both images.  The lower row shows lines from each corner in image 1 to every corner in image 2 it matches above threshold.

A click/keypress on the plot starts the demo, showing the current and best RANSAC iterations in separate windows (updated as the algorithm runs).

![demo run](/ransac/assets/match_demo_iter.png)

This shows the current iteration window, in the upper row shows the minimum sample, the three possible matching pairs of corners  (a subset of the lower row in the previous plot).   These three are used to estimate the parameters of the affine transformation (2x3 matrix) that might map one image to the other (as well as it's inverse transformation).  

The lower two windows show how the corners line up under this transformation.  Each image shows its own detected corners as green dots and the other image's corners transformed to its image space as circles, blue if they ended up near a detected corner after the transformation, or red otherwise.   In the lower, left plot, an estimated bounding box of image 2 is shown in image 1.

 A separate window shows a similar plot for the best iteration so far.  After 2000 iterations, the final model is displayed, showing all inlier features in the upper row, and the corners & bounding boxes, warped by the transformation estimated from the full inlier set.

 ![demo final](/ransac/assets/match_demo_final.png)

(Demo parameters are mostly in `demo_match_images.py` and `util_affine.py`.)