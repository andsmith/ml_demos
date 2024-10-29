from ransac import solve_ransac
from fit_line import RansacLineData, RansacLine
from util_line import make_line_data, plot_line
import matplotlib.pyplot as plt
import logging

def demo(n_line_pts, n_outliers, **ransac_args):
    """
    Demonstrate RANSAC, finding a line among a set of 2d points.

    :param n_line_pts: number of points in the line
    :param n_outliers: number of outlier points
    :param ransac_args: additional parameters for solve_ransac()
    """
    # run algorithm
    pts, params = make_line_data(n_line_pts, n_outliers)
    data = RansacLineData(pts)
    result = solve_ransac(data, RansacLine, **ransac_args)

    # print results
    print('\n\nRANSAC found the best solution on iteration %i / %i:' %
              (result['best'].iter, result['final'].iter ))
    print('\tInliers: %i of %i (%.1f %%)' % ((result['best'].inlier_mask.sum()),
                                             len(result['best'].inlier_mask),
                                             100*(result['best'].inlier_mask.mean())))
    print("\n\tEstimated params of line:\n\n\t\t%.3f x + %.3f y + %.3f = 0" %
          tuple(result['final'].get_params()))
    print("\n\tTrue params of line:\n\n\t\t%.3f x + %.3f y + %.3f = 0\n\n" % tuple(params))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    demo(n_line_pts=100,
         n_outliers=100,
         max_error=0.03,
         max_iter=30,
         animate_pause_sec=.5  # 0 to pause between iterations, None to disable plotting
         )
