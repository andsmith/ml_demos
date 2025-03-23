import cv2
import numpy as np

def get_annulus_polyline(r_outer, r_inner, n_points=50):
    """
    Return a list of points that form a polyline of an annulus.
    :param r_outer: outer radius
    :param r_inner: inner radius
    :param n_points: number of points
    :return: list of (x, y) points
    """
    # left-right symmetry w/even number of points:
    n_points = n_points +1 if (n_points % 2) == 1 else n_points  

    angles = np.linspace(0, 2*np.pi, n_points+1)
    x_outer = r_outer*np.cos(angles)
    y_outer = r_outer*np.sin(angles)
    x_inner = r_inner*np.cos(angles[::-1])
    y_inner = r_inner*np.sin(angles[::-1])
    
    ring_x = np.concatenate([x_outer, x_inner[::-1]])
    ring_y = np.concatenate([y_outer, y_inner[::-1]])
    return list(zip(ring_x, ring_y))

                
def show_annulus():
    SHIFT = 6
    SHIFT_M = 1 << SHIFT
    points = get_annulus_polyline(20, 15, 100)
    img = np.zeros((50, 50, 3), dtype=np.uint8) + 255
    offset = np.array([25, 25])
    points = np.array((points + offset.T) *SHIFT_M , dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(img, [points], color=(255, 0, 0), lineType=cv2.LINE_AA, shift=SHIFT)

    img_r = img.copy()
    img_r[img[:,:,0] == 255] = [0, 0, 255]

    img_both = (img + img_r[:,::-1,:]) // 2
    cv2.imshow('Annulus w/mirror image.', img_both)
    cv2.waitKey (0)

    cv2.imshow('Annulus', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    show_annulus()
