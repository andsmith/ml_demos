"""
Kind of clusters user can add to the dataset with the creator.
"""
import numpy as np
from abc import ABC, abstractmethod
from enum import IntEnum
import cv2
from colors import COLORS
from util import bbox_contains
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
CTRL_RAD=5
DRAW_PT_SIZE=3

class Cluster(ABC):
    """
    Abstract class for "clusters" that can be added to the dataset.
    Clusters are defined by a shape, position, and relative density (between 0.0 and 1.0).


    A dataset of N points is created with the mixture model generative process, i.e.
    a cluster type is chosen with probability proportional to its relative density, and
    a point is generated from the chosen cluster's distribution.
    """

    def __init__(self, x, y):
        pos = np.array([x, y])
        self._ctrl = {CtrlPt.center: pos,
                      CtrlPt.p0: pos,
                      CtrlPt.p1: pos,
                      }
        self._rnd_seed = np.random.randint(0, 2**15)
        self._color = (np.random.rand(3)*255).astype(np.uint8)
        self._ctrl_held = CtrlPt.p1
        self._ctrl_mouse_over = None

        self._colors = {'ctrl_idle': COLORS['gray'].tolist(),
                        'ctrl_mouse_over': COLORS['red'].tolist(),
                        'ctrl_held': COLORS['neon green'].tolist()}
        print("Created cluster %s"% (self._ctrl,))

    def start_adjusting(self):
        if self._ctrl_mouse_over is None:
            raise ValueError("No control point under mouse to adjust")
        self._ctrl_held = self._ctrl_mouse_over
        self._ctrl_mouse_over = None

        return True # change for single-click actions

    def stop_adjusting(self, bbox):
        self._ctrl_held = None
        # if out of bounds, delete
        if not bbox_contains(bbox,*self._ctrl[CtrlPt.center]):
            return True
        return False
        

    def update_mouse_pos(self, x, y, tol_px=CTRL_RAD):
        """
        Mouse moved to (x,y), is it over one of our control points?
        if so, set mousover state and return True.
        """
        print("UPDATE MOUSE POS", x, y)
        #import ipdb; ipdb.set_trace()
        tol_px_sq = tol_px**2
        for ctrl in CTRL_ORDER:
            pos = self._ctrl[ctrl]
            #print("Q",ctrl)
            #if ctrl==CtrlPt.center:
                #print(pos, x, y, np.sum((np.array([x, y]) - pos)**2), tol_px_sq)
            if np.sum((np.array([x, y]) - pos)**2) < tol_px_sq:
                self._ctrl_mouse_over = ctrl
                #print("Mouse over control point %s" % ctrl)
                return True
        self._ctrl_mouse_over = None
        #print("Mouse not over any control point")
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
                aspect = r1 / r0 if r0 > 5 else 1.0
                r1_new = r0_new * aspect
                theta_0_new = np.arctan2(p0_new[1] - self._ctrl[CtrlPt.center][1],
                                            p0_new[0] - self._ctrl[CtrlPt.center][0])
                theta_1_new = theta_0_new + np.pi/2.
                p1_new = self._ctrl[CtrlPt.center] + np.array([r1_new * np.cos(theta_1_new),
                                                               r1_new * np.sin(theta_1_new)])                   
                self._ctrl[CtrlPt.p0] =p0_new
                self._ctrl[CtrlPt.p1] = p1_new
            elif self._ctrl_held == CtrlPt.p1:
                # Just change length of p1, keep p0 in place
                new_r1 = np.linalg.norm(np.array([x, y]) - self._ctrl[CtrlPt.center])
                old_theta1 = np.arctan2(self._ctrl[CtrlPt.p1][1] - self._ctrl[CtrlPt.center][1],
                                        self._ctrl[CtrlPt.p1][0] - self._ctrl[CtrlPt.center][0])
                print("OLD THETA",np.rad2deg( old_theta1))
                self._ctrl[CtrlPt.p1] = self._ctrl[CtrlPt.center] + np.array([new_r1 * np.cos(old_theta1),
                                                                            new_r1 * np.sin(old_theta1)])   
            else:
                raise ValueError("Control point held is not valid to drag")



            #self._ctrl[self._ctrl_held] = np.array([x, y])
            #print("Control point %s dragged to (%f, %f)" % (self._ctrl_held, x, y))
        else:
            raise ValueError("No control point held to drag")

    @abstractmethod
    def get_points(self, n):
        """
        Generate a points from the cluster.
        For the same N, should be the same points each time.
        :param n: number of points to generate
        :returns: n x 2 array of points
        """
        pass

    def render(self, img):
        """
        Render the cluster on the image.
        Draw center and points p0, p1, all the appropriate color.
        Draw major/minor axes.
        """
        
        #print("MOUSEOVER STATE", self._ctrl_mouse_over)
        #print("HOLD STATE", self._ctrl_held)
        #print("\n")
        
        # draw control points, in this order:
        draw_pts = [CtrlPt.center, CtrlPt.p0, CtrlPt.p1]

        for ctrl in draw_pts:
            pos = self._ctrl[ctrl]
            color = self._colors['ctrl_idle']
            if ctrl == self._ctrl_mouse_over:
                color = self._colors['ctrl_mouse_over']
            elif ctrl == self._ctrl_held:
                color = self._colors['ctrl_held']
            
            pos_screen = int(pos[0]), int(pos[1])
            cv2.circle(img,pos_screen, DRAW_PT_SIZE, color, -1, cv2.LINE_AA)
        
        # draw major/minor axes
        cv2.line(img, (int(self._ctrl[CtrlPt.center][0]), int(self._ctrl[CtrlPt.center][1])),
                 (int(self._ctrl[CtrlPt.p0][0]), int(self._ctrl[CtrlPt.p0][1])), self._color.tolist(), 1)
        cv2.line(img, (int(self._ctrl[CtrlPt.center][0]), int(self._ctrl[CtrlPt.center][1])),
                    (int(self._ctrl[CtrlPt.p1][0]), int(self._ctrl[CtrlPt.p1][1])), self._color.tolist(), 1)
        


class EllipseCluster(Cluster):
    def __init__(self, x, y):
        super().__init__(x, y)

    def get_points(self, n):
        self._random_state = np.random.RandomState(self._rnd_seed)
        # for now generate N random points in a .1x .1 square about the center
        # then rotate and scale them to the ellipse
        pts = np.random.rand(n, 2) * .1 - .05

        # rotate and scale
        # rotate by angle theta
        theta = np.arctan2(self._ctrl[CtrlPt.p1][1] - self._ctrl[CtrlPt.center][1],
                           self._ctrl[CtrlPt.p1][0] - self._ctrl[CtrlPt.center][0])
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        # scale by lengths of principal axes
        pts = np.dot(pts, R.T)
        pts[:, 0] *= np.linalg.norm(self._ctrl[CtrlPt.p0] - self._ctrl[CtrlPt.center])
        pts[:, 1] *= np.linalg.norm(self._ctrl[CtrlPt.p1] - self._ctrl[CtrlPt.center])
        # translate to center
        pts += self._ctrl[CtrlPt.center]
        return pts


CLUSTER_TYPES = {'Ellipse': EllipseCluster}
