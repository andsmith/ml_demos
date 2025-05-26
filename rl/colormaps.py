import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import cv2


def get_test_colormap_names():
    return ["bwr", "coolwarm", "RdBu", "RdGy", "RdYlGn", "RdYlBu", "seismic", "PuOr", "PRGn", "Spectral"]


def invert_colors(colors):
    return 255 - colors


def get_cmap_colors(name):
    cmap = plt.get_cmap(name)
    colors = cmap(np.linspace(0, 1, 256))[:, :3] * 255

    return colors.astype(int)


_PERMS = [[0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0]]
def permute_channels(colors):
    """
    :param colors:  Nx3 array of RGB colors
    :return:  list of 6 Nx3 arrays of RGB colors, all possible channel permutations
    """
    colors = np.array(colors)
    perms = []
    for perm in _PERMS:
        perms.append(colors[:, tuple(perm)])
    
    return perms


def show_cmap_variants(bkg_color=(246, 238, 227), space=10, height=20, w=256):
    """
    Make each colormap dark in the center, create 6 colormaps by
    permuting the channels, and put them all in the same image, 
    in 2 rows of 3 columns.  Write over each colormap the name its name.
    Write over each block its rgb permutation (G B R, etc.)
    :param bkg_color:  The background color of the image.
    :param space:  The space between the colormaps.
    :param height:  The height of the colormap bands.
    :param w:  The width of the colormap bands.
    :return:  The image with the colormaps.
    """
    n_rows = 2
    n_cols = 4

    maps = {name: invert_colors(get_cmap_colors(name)) for name in get_test_colormap_names()}

    
    map_perms = {name: permute_channels(colors) for name, colors in maps.items()}

    # Add the original at the end of each list of permutations
    for name, colors in map_perms.items():
        map_perms[name].append(invert_colors(colors[0]))
        
    v_pad = space * 2
    h_pad = space * 3
    n_maps = len(maps)
    block_h = n_maps * height + (n_maps + 1) * space
    block_w = w + 2 * space

    header_h = 33

    img_size = (n_cols * block_w + (n_cols + 1) * h_pad, n_rows * (block_h + header_h) + (n_rows + 1) * v_pad)

    img = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
    img[:] = bkg_color

    print("Adding maps to image of size: %i x %i, using %i x %i blocks, with h-padding %i and v-padding %i" %
          (img_size[0], img_size[1], block_w, block_h, h_pad, v_pad))
    for map_ind, (name, color_perms) in enumerate(map_perms.items()):


        for block_ind, colors in enumerate(color_perms):

            row = block_ind // n_cols
            col = block_ind % n_cols

            x = h_pad * (col+1) + col * block_w
            y = v_pad * (row + 1) + row * block_h + header_h* (row+1)

            # Draw rect around block
            cv2.rectangle(img, (x-space, y-space), (x + block_w, y + block_h), (0, 0, 0), 1, cv2.LINE_AA)

            if map_ind == 0:
                y_offset = space - 33
                # write the permutation of the channels
                txt = "Inverted, CMY=%s" %str(_PERMS[block_ind]) if block_ind < len(_PERMS) else "Original 'diverging' RGB"
                cv2.putText(img, txt, (x + 5, y + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 0), 1, cv2.LINE_AA)

            y_offset = map_ind * (height + space) + space

            color_img_256 = np.tile(colors[:, np.newaxis, :], (1, height, 1)).transpose(1, 0, 2)
            color_img = cv2.resize(color_img_256, (w, height), interpolation=cv2.INTER_NEAREST)

            img[y + y_offset:y + y_offset + height, x:x + w, :] = color_img

            # write the name of the colormap
            cv2.putText(img, name, (x + 5, y + y_offset + height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                        cv2.LINE_AA)
    #save   
    cv2.imwrite("colormaps.png", img[:, :, ::-1])

    # show the image
    cv2.imshow("Colormaps", img[:, :, ::-1])
    cv2.waitKey(0)

    cv2.destroyAllWindows()

    return img


if __name__ == "__main__":

    show_cmap_variants()
