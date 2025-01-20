"""
Kind of clusters user can add to the dataset with the creator.
"""
import numpy as np
from abc import ABC, abstractmethod
from enum import IntEnum
import cv2
from colors import COLORS
from util import bbox_contains, get_ellipse_points, sample_ellipse, sample_gaussian
from layout import APP_CONFIG


class CtrlPt(IntEnum):
    """
    Enum for the control points of the cluster.
    If overlapping, the order of the enum values determines which is drawn on top, 
    selected by a click, etc.
    """
    p0 = 0  # First principal axis, changing its length changes both axes in proportion.
    center = 1  # Place/move cluster by this ctrl point
    p1 = 2  # Second principal axis, changing its length changes only the second axis.


CTRL_ORDER = [CtrlPt.p0, CtrlPt.center, CtrlPt.p1]
CTRL_RAD = 7
DRAW_PT_SIZE = 3


class Cluster(ABC):
    """
    Abstract class for "clusters" that can be added to the dataset.
    Clusters are defined by a shape, position, and relative density (between 0.0 and 1.0).


    A dataset of N points is created with the mixture model generative process, i.e.
    a cluster type is chosen with probability proportional to its relative density, and
    a point is generated from the chosen cluster's distribution.
    """

    def __init__(self, x, y, n, bbox):
        pos = np.array([x, y])
        self._n = n
        self._n_max = APP_CONFIG['max_pts_per_cluster']
        self._bbox = bbox
        self._ctrl = {CtrlPt.center: pos,
                      CtrlPt.p0: pos,
                      CtrlPt.p1: pos,
                      }
        self._rnd_seed = np.random.randint(0, 2**15)
        self._color = (np.random.rand(3)*128).astype(np.uint8).tolist()
        self._ctrl_held = CtrlPt.p1
        self._ctrl_mouse_over = None
        self._colors = {'ctrl_idle': COLORS['cyan'].tolist(),
                        'ctrl_mouse_over': COLORS['red'].tolist(),
                        'ctrl_held': COLORS['neon green'].tolist()}

        # control points are scaled together, but start all at the same point, so the first motion
        # away from that point will use this aspect ratio
        self._default_aspect = 1.0

        self._refresh()

    def get_color(self):
        return self._color

    @abstractmethod
    def _generate(self, n):
        """
        Generate random points.
        For the same N, should be the same points each time.
        :param n: number of points to generate
        :returns: n x 2 array of points
        """
        pass

    def _refresh(self):
        self._points = self._generate(self._n)

    def get_points(self):
        return self._points

    def set_n_pts(self, n):
        self._n = n
        self._refresh()

    def get_n_pts(self):
        return self._n

    def start_adjusting(self):
        if self._ctrl_mouse_over is None:
            raise ValueError("No control point under mouse to adjust")
        self._ctrl_held = self._ctrl_mouse_over
        self._ctrl_mouse_over = None

        return True  # change for single-click actions

    def stop_adjusting(self, bbox):
        self._ctrl_held = None
        # if out of bounds, delete
        if not bbox_contains(bbox, *self._ctrl[CtrlPt.center]):
            return True
        return False

    def update_mouse_pos(self, x, y, tol_px=CTRL_RAD):
        """
        Mouse moved to (x,y), is it over one of our control points?
        if so, set mousover state and return True.
        """
        tol_px_sq = tol_px**2
        for ctrl in CTRL_ORDER:
            pos = self._ctrl[ctrl]
            # print("Q",ctrl)
            # if ctrl==CtrlPt.center:
            # print(pos, x, y, np.sum((np.array([x, y]) - pos)**2), tol_px_sq)
            if np.sum((np.array([x, y]) - pos)**2) < tol_px_sq:
                self._ctrl_mouse_over = ctrl
                # print("Mouse over control point %s" % ctrl)
                return True
        self._ctrl_mouse_over = None
        # print("Mouse not over any control point")
        return False

    def drag_ctrl(self, x, y):
        """
        Drag the control point under the mouse.
        """
        if self._ctrl_held is not None:
            delta = np.array([x, y]) - self._ctrl[CtrlPt.center]

            if self._ctrl_held == CtrlPt.center:
                for ctrl in CTRL_ORDER:
                    self._ctrl[ctrl] += delta
            elif self._ctrl_held == CtrlPt.p0:
                # change p0 and p1 in proportion
                p0_new = np.array([x, y])
                r0 = np.linalg.norm(self._ctrl[CtrlPt.p0] - self._ctrl[CtrlPt.center])
                r1 = np.linalg.norm(self._ctrl[CtrlPt.p1] - self._ctrl[CtrlPt.center])
                r0_new = np.linalg.norm(p0_new - self._ctrl[CtrlPt.center])
                aspect = r1 / r0 if r0 > 5 else self._default_aspect
                r1_new = r0_new * -aspect    # hack to make p1 upwards w/Sierpinski, irrelevant for others
                theta_0_new = np.arctan2(p0_new[1] - self._ctrl[CtrlPt.center][1],
                                         p0_new[0] - self._ctrl[CtrlPt.center][0])
                theta_1_new = theta_0_new + np.pi/2.
                p1_new = self._ctrl[CtrlPt.center] + np.array([r1_new * np.cos(theta_1_new),
                                                               r1_new * np.sin(theta_1_new)])
                self._ctrl[CtrlPt.p0] = p0_new
                self._ctrl[CtrlPt.p1] = p1_new
            elif self._ctrl_held == CtrlPt.p1:
                # Just change length of p1, keep p0 in place
                new_r1 = np.linalg.norm(np.array([x, y]) - self._ctrl[CtrlPt.center])
                old_theta1 = np.arctan2(self._ctrl[CtrlPt.p1][1] - self._ctrl[CtrlPt.center][1],
                                        self._ctrl[CtrlPt.p1][0] - self._ctrl[CtrlPt.center][0])
                self._ctrl[CtrlPt.p1] = self._ctrl[CtrlPt.center] + np.array([new_r1 * np.cos(old_theta1),
                                                                              new_r1 * np.sin(old_theta1)])
            else:
                raise ValueError("Control point held is not valid to drag")
            # self._ctrl[self._ctrl_held] = np.array([x, y])
            # print("Control point %s dragged to (%f, %f)" % (self._ctrl_held, x, y))
        else:
            raise ValueError("No control point held to drag")

        self._refresh()
        
    @staticmethod
    def _draw_line(img, center, p, color):
        # Draw a line on one side of the center point
        cv2.line(img, (int(center[0]), int(center[1])),
                 (int(p[0]), int(p[1]),), color, 1, cv2.LINE_AA)

    @staticmethod
    def _draw_symm_line(img, center, p, color):
        # Draw a line on either side of the center point
        cv2.line(img, (int(center[0]), int(center[1])),
                 (int(p[0]), int(p[1]),), color, 1, cv2.LINE_AA)
        cv2.line(img, (int(center[0]), int(center[1])),
                 (int(2*center[0]-p[0]), int(2*center[1]-p[1])), color, 1, cv2.LINE_AA)

    def _render_control_lines(self, img):
        # draw axes
        Cluster._draw_symm_line(img,self._ctrl[CtrlPt.center], self._ctrl[CtrlPt.p0], self._color)
        Cluster._draw_symm_line(img,self._ctrl[CtrlPt.center], self._ctrl[CtrlPt.p1], self._color)

        # draw ellipse
        pts = get_ellipse_points(self._ctrl[CtrlPt.center], self._ctrl[CtrlPt.p0], self._ctrl[CtrlPt.p1], 100)
        cv2.polylines(img, [pts.astype(int)], isClosed=True, color=self._color, thickness=1, lineType=cv2.LINE_AA)
    
    def _render_control_pts(self, img):
        # draw control points, in this order:
        draw_pts = [CtrlPt.center, CtrlPt.p0, CtrlPt.p1]
        draw_size_multipliers = [1.0, 1.5, 1.00]

        for ctrl, size_mul in zip(draw_pts, draw_size_multipliers):
            pos = self._ctrl[ctrl]
            color = self._colors['ctrl_idle']
            if ctrl == self._ctrl_mouse_over:
                color = self._colors['ctrl_mouse_over']
            elif ctrl == self._ctrl_held:
                color = self._colors['ctrl_held']

            pos_screen = int(pos[0]), int(pos[1])
            cv2.circle(img, pos_screen, int(DRAW_PT_SIZE*size_mul), color, -1, cv2.LINE_AA)

    def render(self, img, show_ctrls, pt_size=3):
        """
        Render the cluster on the image.
        Draw center and points p0, p1, all the appropriate color.
        Draw major/minor axes.
        """

        # print("MOUSEOVER STATE", self._ctrl_mouse_over)
        # print("HOLD STATE", self._ctrl_held)
        # print("\n")

        if show_ctrls:
            self._render_control_lines(img)
            self._render_control_pts(img)

        # draw sample points
        self._points = self._points
        valid = bbox_contains(self._bbox, self._points[:, 0], self._points[:, 1])
        pts = self._points[valid]
        half_size = pt_size // 2
        if pts.shape[0] > 0:
            if pt_size == 1:
                for pt in pts:
                    px, py = int(pt[0]), int(pt[1])
                    img[py, px] = self._color
            else:
                for pt in pts:
                    px, py = int(pt[0]), int(pt[1])
                    img[py-half_size:py+half_size, px-half_size:px+half_size] = self._color


class EllipseCluster(Cluster):
    def __init__(self, x, y, n, bbox):
        super().__init__(x, y, n, bbox)

    def _generate(self, n):
        self._random_state = np.random.RandomState(self._rnd_seed)
        # TODO: Cache these utnil n changes, etc.
        points = sample_ellipse(self._ctrl[CtrlPt.center],
                                self._ctrl[CtrlPt.p0],
                                self._ctrl[CtrlPt.p1], n, self._random_state,)
        return points


class AnnularCluster(EllipseCluster):

    def _generate(self, n):
        self._random_state = np.random.RandomState(self._rnd_seed)
        points = sample_ellipse(self._ctrl[CtrlPt.center],
                                self._ctrl[CtrlPt.p0],
                                self._ctrl[CtrlPt.p1], n, self._random_state,
                                empty_frac=0.5)
        return points


class GaussianCluster(Cluster):
    def __init__(self, x, y, n, bbox):
        super().__init__(x, y, n, bbox)

    def _generate(self, n):
        self._random_state = np.random.RandomState(self._rnd_seed)
        points = sample_gaussian(self._ctrl[CtrlPt.center],
                                 self._ctrl[CtrlPt.p0],
                                 self._ctrl[CtrlPt.p1], n, self._random_state,)
        return points


class SierpinskiCluster(Cluster):
    """
    Generate points in a Sierpinski triangle pattern.

    The points are generated in the rectangle with given center and 
       points p0 and p1 as the height and width:
       
        self._ctrl[CtrlPt.center] is the center of the rectangle.
        self._ctrl[CtrlPt.p0] is the right corner of the base, moving it scales/rotates the triangle
        P1 is the top corner, moving it changes the height of the triangle.       

    """

    def __init__(self, x, y, n, bbox):
        self._init_sierpinski()
        super().__init__(x, y, n, bbox)
        self._default_aspect = 1.56444

    def _init_sierpinski(self, burn_in=100):
        n = APP_CONFIG['max_pts_per_cluster']

        corners = [np.array([-1, -1]), np.array([1, -1]), np.array([0, 1])]
        points = []
        pt = np.random.randn(2)
        for i in range(n + burn_in):
            pt = (pt + corners[np.random.randint(3)]) / 2
            if i >= burn_in:
                points.append(pt)

        self._coords = np.array(points)
        # move up to fit in [-1, 1] x [0, 1] for easier scaling translation
        self._coords[:, 1] = (self._coords[:, 1] + 1) / 2

    def _render_control_lines(self, img):
        SierpinskiCluster._draw_symm_line(img, self._ctrl[CtrlPt.center], self._ctrl[CtrlPt.p0], self._color)
        SierpinskiCluster._draw_line(img, self._ctrl[CtrlPt.center], self._ctrl[CtrlPt.p1], self._color)


    def _generate(self, n):

        # vectors from control points to rectangle side midpoints
        v0 = self._ctrl[CtrlPt.p0] - self._ctrl[CtrlPt.center]
        v1 = self._ctrl[CtrlPt.p1] - self._ctrl[CtrlPt.center]

        angle0 = -np.arctan2(v0[1], v0[0])
        x_scale = np.linalg.norm(v0)
        y_scale = np.linalg.norm(v1)
        # scale
        points = self._coords[:n] * np.array([x_scale, -y_scale])
        # rotate
        r_mat = np.array([[np.cos(angle0), -np.sin(angle0)],
                          [np.sin(angle0), np.cos(angle0)]])
        points = points @ r_mat
        # translate
        points = points + self._ctrl[CtrlPt.center]

        return points   


CLUSTER_TYPES = {'Ellipse': EllipseCluster,
                 'Annulus': AnnularCluster,
                 'Gaussian': GaussianCluster,
                 'Sierpinski': SierpinskiCluster}


def test_sierpinski():
    import matplotlib.pyplot as plt
    sierp = SierpinskiCluster(0, 0, 10000, None)
    pts = sierp._coords
    plt.scatter(pts[:, 0], pts[:, 1], s=1)
    plt.show()


if __name__ == "__main__":
    test_sierpinski()
