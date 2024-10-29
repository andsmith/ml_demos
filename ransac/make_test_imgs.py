import numpy as np
import cv2
import pprint
import matplotlib.pyplot as plt
from util_affine import Affine

def draw_img(size, n_rects=10, n_circle_colors=30, n_rect_colors=3):
    """
    Generate an image that will be easy to analyze with a toy feature detector & matcher:
        * Draw 1000 circles of random sizes and colors  (Background)
        * Draw N rectangles of random sizes and colors on top of the circles (Foreground)

    Every pixel will have a color that is either black, one of the circle colors, or one of the rectangle colors.
    This high level of color qantization will make it easy to compare histograms of the color distribution in the images.
    (i.e. to remove the curse of dimensionality in the full 256x256x256 color space)

    :param size: the size of the image (width, height)
    :param n_rects: the number of rectangles to draw on top of the circles
    :param n_circle_colors: the number of colors to use for the circles
    :param n_rect_colors: the number of colors to use for the rectangles

    :returns: WxH - the image, indices into the color palette, and
              Px3 - the color palette
    """
    # image pixel values will be indices into the color palette
    circle_palette = np.random.rand(n_circle_colors, 3).astype(np.float32)
    rect_palette = np.random.rand(n_rect_colors, 3).astype(np.float32)
    full_palette = np.vstack([[[0, 0, 0]], circle_palette, rect_palette]).astype(np.float32)

    img = np.zeros((size[1], size[0]), dtype=np.uint8)

    # Random shape params must be in these ranges
    min_dim = np.min(size)
    rect_size_range = [min_dim//15, min_dim//6]
    circle_rad_range = [min_dim//10, min_dim//5]

    # make image with circles...
    circle_colors = np.random.randint(0, n_circle_colors, 1000)
    for i in range(1000):
        center = np.random.rand(2) * size

        radius = int(np.random.uniform(*circle_rad_range))
        color = int(circle_colors[i])
        cv2.circle(img, tuple(center.astype(int)), radius, color, -1)

    # ...and rectangles
    rect_colors = np.random.randint(0, n_rect_colors, n_rects)
    for i in range(n_rects):
        center = np.random.rand(2) * size
        width, height = np.random.uniform(*rect_size_range, 2)
        color = int(rect_colors[i]) + n_circle_colors
        cv2.rectangle(img, tuple(center.astype(int)), tuple(
            (center + [width, height]).astype(int)), color, -1)
    return img, (full_palette * 255).astype(np.uint8)


def _test_draw_img():
    img, palette = draw_img((256, 256))
    img_rgb = palette[img].astype(np.uint8)
    plt.imshow(img_rgb)
    plt.show()

def transform_img(img, noise_frac=0., max_color_ind=None, rand_transf_params='demo'):
    """
    Transform the image by applying a random affine transformation to the image and palette.

    :param img: WxH - the image
    :param max_color_ind:  largest color value, i.e. palette size
    :param noise_frac:  fraction of pixels to change to a random color
    :param rand_transf_params:  parameters for random transformation generator.

    :returns: WxH - the transformed image, and
              {'M': transformation matrix, 
               'angle': float, rotation angle, 
               'scale': float, 
               'translation': 2x1 )  - the transformation used
    """

    # Make an affine transformation & apply it to create the second image
    size = (img.shape[1], img.shape[0])
    transf = Affine.from_random(rand_transf_params)
    # don't do any interpolation between color values, since they are indices
    #img2 = cv2.warpAffine(img, transf.M[:2,:], size, flags=cv2.INTER_NEAREST, borderValue=int(img[0, 0]))

    img2=transf.warp_image(img)
    # Colors are not aranged in any order so if the transform interpolates pallette indices, artifacts will show.
    c2 = set(img2.reshape(-1))
    c1 = set(img.reshape(-1))
    assert c2.issubset(c1), "Transform added a color not in the original image:  %s" % (c2 - c1,)

    # Add noise by changing some pixels to another color
    if noise_frac > 0:
        max_color_ind = max_color_ind if max_color_ind is not None else np.max(img) + 1
        noise = np.where(np.random.rand(*img.shape) < noise_frac)
        noise_colors = np.random.randint(
            0, max_color_ind, noise[0].size)
        img2[noise] = noise_colors
    return img2, transf


def _test_transform_img():
    img, palette = draw_img((256, 256))
    img2, transform = transform_img(img, noise_frac=0.1, max_color_ind=palette.shape[0])
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(palette[img])
    ax[1].imshow(palette[img2])
    ax[0].set_title('Original')
    ax[1].set_title('Transformed:  \n%s' % (transform,))
    ax[0].axis('off')
    ax[1].axis('off')
    plt.show()


if __name__ == '__main__':
    _test_draw_img()
    _test_transform_img()
