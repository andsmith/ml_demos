import numpy as np
import logging
import cv2

from colors import COLOR_LINES, COLOR_BG, COLOR_X, COLOR_O, COLOR_DRAW, COLOR_SELECTED, COLOR_MOUSEOVERED
from game_base import Mark, Result
from tic_tac_toe import get_game_tree_cached, Game
from node_placement import BoxOrganizerPlotter


LAYOUT = {'win_size': (1900, 950)}


SHIFT_BITS = 6
SHIFT = 1 << SHIFT_BITS


class GameGraphApp(object):
    """
    Arrange game states to show the graph structure of the game.

    States will be in layers, all states in a layer will have the same number of marks made so far.
    States with a single parent will be approximately under their parent.
    States with multiple parents will be placed in the middle of their parents.

    Lines will connect states to their successors.

    """

    def __init__(self, max_levels=10, no_plot=False):
        """
        :param size: (width, height) of the app window
        """
        self._tree = get_game_tree_cached(player=Mark.X)
        self._max_levels = max_levels
        if max_levels > 10:
            raise ValueError("Max levels must be <= 10 for Tic-Tac-Toe.")
        self._size = LAYOUT['win_size']
        self._blank_frame = np.zeros((self._size[1], self._size[0], 3), dtype=np.uint8)
        self._blank_frame[:, :] = COLOR_BG
        self._no_plot = no_plot
        self._init_tree()
        self._init_graphics()

        # App state
        self._depth = 1

        # Things to draw
        self._mouseover_state = None
        self._click_state = None
        self._selected_states = [Game.from_strs(["XO ", "   ", "   "])]

        # key by state, value is {'upper': [(x_left, y_left) , (x_right, y_right)],
        #                         'lower': [(x_left, y_left) , (x_right, y_right)]}  (floats, in fractional pixels)
        self._appach_pts = self._find_attach_pts()

        # TODO: update these as states are selected/deselected/mouseovered/etc. rather than every frame.
        self._s_neighbors = []  # states that are neighbors of the selected states list of lists (detph,then state)
        self._n_edges = []  # binary edge matrix, len(s_neighbors[d]) x len(s_neighbors[d+1])
        self._edge_lines = {}
        # list with  {'color': (r,g,b),
        #             'thickness': int,
        #             'coords': [line1, line2, ...]}
        #   where each line is a float numpy array (line strip) of (x1, y1, x2, y2) in pixels.

    def _find_attach_pts(self):
        """
        Attachment points are the upper two and lower two endpoints of the two vertical grid lines.

        Those are found by combining these two:

            * relative points:  self._state_dims[level] has the space size and the attachment points for images of 
              that size: 
                          { 'img_size': grid image's side length
                            'line_t': line width for the 4 grid lines and the bounding box
                            'upper': [(x1, y1), (x2, y2)] attachment points for the upper grid line (floats, fractional pixels)
                            'lower': [(x1, y1), (x2, y2)] attachment points for the lower grid line (same)
                            'space_size': space_size
                            'bar_w': grid line width, int }, and the

            * image offsets:  self._positions[level] is a dict[state] = {'x': (x_min, x_max), 'y': (y_min, y_max)}

        where:

            box_positions, box = BoxOrganizerPlotter.get_layout()


        :returns: dict[state] = {'upper': [(x_left, y_left) , (x_right, y_right)],
                                 'lower': [(x_left, y_left) , (x_right, y_right)]}  (floats, in fractional pixels)
        """
        attach = {}
        for layer, state_list in enumerate(self._states_by_layer):
            print("Getting attachment points for layer %i with %i states" % (layer, len(state_list)))
            dims = self._state_dims[layer]
            upper_pts = np.array(dims['upper'])
            lower_pts = np.array(dims['lower'])

            for state_info in state_list:
                state = state_info['state']
                box_info = self._positions[state]
                x_min = box_info['x'][0]
                y_min = box_info['y'][0]
                offset = np.array([x_min, y_min])
                attach[state] = {'upper': upper_pts + offset,
                                 'lower': lower_pts + offset}
        return attach

    def _init_graphics(self):
        print("Caching images...")
        self._states_frame = self._box_placer.draw(images=self._state_images, dest=self._blank_frame.copy())
        print("\tDone.")

        self._win_name = "Tic-Tac-Toe Game Graph"
        if not self._no_plot:
            cv2.namedWindow(self._win_name, cv2.WINDOW_NORMAL)
            # cv2.setWindowProperty(self._win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.setMouseCallback(self._win_name, self.mouse_callback)

    def _calc_layer_spacing(self, layer_counts):
        """
        for each layer , a dict{'y': (ymin, y_max) the y-extent of the layer in pixels,
                                'n_boxes': number of boxes in the layer,
                                'bar_y': (ymin, y_max) of the bar that will be drawn under the boxes (if not at the bottom)
                                }
        """
        rel = np.sqrt(100 + layer_counts)
        rel = rel/np.sum(rel)
        sizes = (rel * self._size[1]).astype(int)
        y = 0
        bar_w = 5
        layer_spacing = []
        for i, size in enumerate(sizes):
            top = y
            if i < len(sizes) - 1:
                bottom = y + size - bar_w
                bar = (y + size - bar_w, y + size)
            else:
                bottom = self._size[1]
                bar = None
            spacing = {'y': (top, bottom),
                       'n_boxes': layer_counts[i]}
            if bar is not None:
                spacing['bar_y'] = bar
            layer_spacing.append(spacing)
            y += size
        return layer_spacing

    def _init_tree(self):
        """
        Calculate sizes & placement of all states.
            1. Determine number of layers & number of states in each layer
            2. Determine positions of each state
            3. For each state, get lines to draw to it's children states.
        """
        def _get_layer(state):
            return np.sum(state.state != Mark.EMPTY)

        # 1.
        self._term, self._children, self._parents, self._initial = self._tree.get_game_tree(generic=True)
        self._states_by_layer = [[{'id': s,
                                   'state': s,
                                   'color': (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))}
                                  for s in self._term if _get_layer(s) == l] for l in range(self._max_levels)]
        self._states_used = {s['state']: True for state_layer in self._states_by_layer for s in state_layer}
        layer_counts = np.array([len(states) for states in self._states_by_layer])
        logging.info(f"States by layer: {layer_counts}")

        # 2. First get the layer spacing.
        self._layer_spacing = self._calc_layer_spacing(layer_counts)
        self._box_placer = BoxOrganizerPlotter(self._states_by_layer, spacing=self._layer_spacing, size_wh=self._size)

        # 3. get the size of a box in each layer, then make the images
        self._positions, self._box_sizes, _ = self._box_placer.get_layout()
        self._state_images = {}
        self._state_dims = []  # from Game.get_space_size(box_size) for each layer

        logging.info("Generating images...")
        for l_ind, layer_states in enumerate(self._states_by_layer):
            logging.info(f"\tLayer {l_ind} has {len(layer_states)} states")
            box_size = self._box_sizes[l_ind]['box_side_len']
            logging.info(f"\tBox_size: {box_size}")
            space_size = Game.get_space_size(box_size)
            logging.info(f"\tUsing space_size: {space_size}")
            box_dims = Game.get_image_dims(space_size, bar_w_frac=.2)
            logging.info(f"\tUsing tiles of size: {box_dims['img_size']}")

            for state_info in layer_states:

                state = state_info['state']
                # import pprint
                # pprint.pprint(box_dims[l_ind]['])
                self._state_images[state] = state.get_img(box_dims)
            self._state_dims.append(box_dims)

        print("\tmade ", len(self._state_images))

    def _get_state_at(self, pos):
        # Which layer?
        l_ind = self._get_layer_at(pos)
        if l_ind is None:
            return None

        # which box:
        for state_info in self._states_by_layer[l_ind]:
            box_info = self._positions[state_info['state']]
            x_min, x_max = box_info['x']
            y_min, y_max = box_info['y']
            if x_min <= pos[0] <= x_max and y_min <= pos[1] <= y_max:
                return state_info['state']

        return None

    def _get_layer_at(self, pos):
        # Which layer?
        for l_ind, layer in enumerate(self._layer_spacing):

            y_min, y_max = layer['y']
            if y_min <= pos[1] <= y_max:
                return l_ind

    def mouse_callback(self, event, x, y, flags, param):
        pos = (x, y)
        self._mouseover_state = self._get_state_at(pos)
        # print("Mouse over state: ", self._mouseover_state)
        if event == cv2.EVENT_LBUTTONDOWN:
            if self._mouseover_state is not None:
                self._click_state = self._mouseover_state
        elif event == cv2.EVENT_LBUTTONUP:
            if self._mouseover_state is not None:
                # print(f"Clicked on {self._click_state}")
                if self._click_state is not None and self._mouseover_state == self._click_state:
                    if self._click_state in self._selected_states:
                        self._selected_states.remove(self._click_state)
                    else:
                        self._selected_states.append(self._click_state)
                self._click_state = None

    def _draw_edges(self, frame):
        """
        Add edge-lines to the image.
        for depth in self._edge_lines:
            for edge_info in self._edge_lines[depth]:
                color = edge_info['color']
                thickness = edge_info['thickness']
                for line in edge_info['coords']:
                    cv2.polylines(frame, [(line).astype(int)], isClosed=False,
                                  color=color, thickness=thickness, lineType=cv2.LINE_AA)
        """

        # put a red dot on every attachment point
        print("Drawing attachment points for %i states..." % len(self._appach_pts))
        #import ipdb; ipdb.set_trace()
        for state, attach_info in self._appach_pts.items():
            x, y = attach_info['upper'][0]
            x,y=int(x),int(y)
            frame[y:y+3, x:x+3, :] = [255, 0, 0]
            x, y = attach_info['upper'][1]
            x,y=int(x),int(y)
            frame[y:y+3, x:x+3, :] = [0, 255, 0]
            x, y = attach_info['lower'][0]
            x,y=int(x),int(y)
            frame[y:y+3, x:x+3, :] = [0, 0, 255]
            x, y = attach_info['lower'][1]
            x,y=int(x),int(y)
            frame[y:y+3, x:x+3, :] = [255, 128, 0]

    def _scan_neighbors(self):
        """
        Calculate all edges from selected/mouseovered states to their neighbors up to the given depth.
        self._s_neighbors will be a list of lists, s_neighbors[depth]=[state0, state1, ...], length depth+1
        self._n_edges will be a list of binary matrices, n_edges[depth] will be a matrix of size 
            len(s_neighbors[depth]) x len(s_neighbors[depth+1]),  with 1 if there is an edge from state i to state j.
        """
        depth = self._depth
        self._s_neighbors = [[s for s in self._selected_states]]
        self._n_edges = []
        # if self._mouseover_state is not None and self._mouseover_state not in self._s_neighbors[0]:
        #    self._s_neighbors[0].append(self._mouseover_state)

        if depth == 0:
            return
        states_seen = {s for s_list in self._s_neighbors for s in s_list}
        edges_seen = {}  # {(from_state, to_state): None}
        print("Scanning neighbors for %i selected states..." % len(states_seen))
        for d in range(1, depth+1):
            # Scan states to get next set of parents/children
            prev_states = self._s_neighbors[d-1]
            # print("\tScanning depth %i, %i states..."%(d, len(prev_states)))
            new_neighbors = []
            for prev_state in prev_states:
                # print("\t\tState:")
                # print(prev_state.indent(3))
                # print("\t\tChildren:", len(self._children[prev_state]))
                # print("\t\tParents:", len(self._parents[prev_state]))
                children = [child for child in self._children[prev_state] if child in self._states_used]
                parents = [parent for parent in self._parents[prev_state] if parent in self._states_used]
                new_neighbors.extend(children)
                new_neighbors.extend(parents)
            new_neighbors = list(set(new_neighbors))
            new_neighbors = [n for n in new_neighbors if n not in states_seen]
            states_seen.update({n: None for n in new_neighbors})
            self._s_neighbors.append(new_neighbors)
            # print("\t\tUnique new neighbors: ", len(new_neighbors))
            # print("\t\tS  tates in growing tree: ", len(states_seen))

            # Make the edge matrix
            n_prev = len(prev_states)
            n_new = len(new_neighbors)
            edge_matrix = np.zeros((n_prev, n_new), dtype=np.uint8)
            for i, prev_state in enumerate(prev_states):
                for j, new_state in enumerate(new_neighbors):
                    if new_state in self._children[prev_state] or new_state in self._parents[prev_state]:
                        edge_matrix[i, j] = 1

            self._n_edges.append(edge_matrix)

        print("Neighbors: ", [len(n) for n in self._s_neighbors])
        print("Edges: ", [e.shape for e in self._n_edges])

    def _make_frame(self):
        frame = self._states_frame.copy()

        # draw mouseover states
        if self._mouseover_state is not None:
            box_info = self._positions[self._mouseover_state]
            x_span, y_span = box_info['x'], box_info['y']
            cv2.rectangle(frame, (x_span[0], y_span[0]), (x_span[1], y_span[1]), COLOR_MOUSEOVERED, thickness=1)
        # draw selected states
        for s_state in self._selected_states:
            box_info = self._positions[s_state]
            x_span, y_span = box_info['x'], box_info['y']
            cv2.rectangle(frame, (x_span[0], y_span[0]), (x_span[1], y_span[1]), COLOR_SELECTED, thickness=1)

        # draw edges
        # self._scan_neighbors()
        # self._make_edge_list()
        self._draw_edges(frame)

        # draw mouseover state
        # draw lines
        return frame

    def _make_edge_list(self):
        """
        Find the endpoints of each edge in the list of neighbors.
        Draw it the appropriate color.
        TODO: Connect whichever of the 4 connection points is closer (for now just the center)
        """
        self._edge_lines = {}
        for d, edge_matrix in enumerate(self._n_edges):
            depth = d+1
            self._edge_lines[depth] = {COLOR_X: {'thickness': 3,
                                                 'coords': []},
                                       COLOR_O: {'thickness': 3,
                                                 'coords': []}}
            for i in range(edge_matrix.shape[0]):
                from_state = self._s_neighbors[d][i]
                from_box_pos = self._positions[from_state]
                from_center = np.mean(from_box_pos['x']), np.mean(from_box_pos['y'])
                print("from state:\n", from_state)
                print("From point: ", from_center)
                for j in range(edge_matrix.shape[1]):
                    to_state = self._s_neighbors[d+1][j]
                    to_box_pos = self._positions[to_state]
                    to_center = np.mean(to_box_pos['x']), np.mean(to_box_pos['y'])
                    print("to state:\n", to_state)
                    print("To point: ", to_center)
                    move_player = self._children[from_state].get(to_state)[1]
                    if move_player == Mark.X:
                        color = COLOR_X
                    else:
                        color = COLOR_O

                    if edge_matrix[i, j] == 1:
                        edge_info = self._edge_lines[depth][color].get('coords', [])
                        edge_info.append(np.array([from_center[0], from_center[1], to_center[0], to_center[1]]))

        import pprint
        pprint.pprint(self._edge_lines)

    def run(self):
        """
        Run the app
        """
        frame_no = 0
        while True:
            self._out_frame = self._make_frame()
            cv2.imshow(self._win_name, self._out_frame[:, :, ::-1])
            k = cv2.waitKey(1)
            if k == 27 or k == ord('q'):
                break
            elif k == ord(','):
                self._depth = max(0, self._depth - 1)
                print("Depth: ", self._depth)
            elif k == ord('.'):
                self._depth += 1
                print("Depth: ", self._depth)
            frame_no += 1
            if frame_no % 10 == 0:
                logging.info(f"Frame {frame_no}")
        cv2.destroyAllWindows()


def run_app():

    app = GameGraphApp(no_plot=False, max_levels=3)
    # import ipdb; ipdb.set_trace()

    frame = app._make_frame()
    app.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_app()
    logging.info("Exiting.")
