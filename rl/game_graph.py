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
        self._p_depth = 1  # how many grand* parents to show for each selected vertex
        self._c_depth = 1  # how many grand* children to show for each selected vertex

        # Things to draw
        self._mouseover_state = None
        self._click_state = None
        self._selected_states = [Game.from_strs(["X  ", "   ", "   "])]

        # key by state, value is {'upper': [(x_left, y_left) , (x_right, y_right)],
        #                         'lower': [(x_left, y_left) , (x_right, y_right)]}  (floats, in fractional pixels)
        self._appach_pts = self._find_attach_pts()

        # TODO: update these as states are selected/deselected/mouseovered/etc. rather than every frame.
        self._s_neighbors = []  # states that are neighbors of the selected states list of lists (detph,then state)
        self._s_edges = []  # binary edge matrix, len(s_neighbors[d]) x len(s_neighbors[d+1])
        self._edge_lines = {}
        # list with  {'color': (r,g,b),
        #             'thickness': int,
        #             'coords': [line1, line2, ...]}
        #   where each line is a float numpy array (line strip) of (x1, y1, x2, y2) in pixels.
        self._recalc_neighbors()  # if any initial states are selected, find their neighbors.

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
                            'bar_w': grid line width, int }, and 
            * image offsets:  self._positions[level] is a dict[state] = {'x': (x_min, x_max), 'y': (y_min, y_max)}
        where:
            box_positions, box = BoxOrganizerPlotter.get_layout().

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

        # 1.
        self._term, self._children, self._parents, self._initial = self._tree.get_game_tree(generic=True)
        self._states_by_layer = [[{'id': s,
                                   'state': s,
                                   'color': (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))}
                                  for s in self._term if s.n_marked() == l] for l in range(self._max_levels)]
        self._layers_of_states = {s['state']: l for l, layer in enumerate(self._states_by_layer) for s in layer}
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
                self._state_images[state] = state.get_img(box_dims)
            self._state_dims.append(box_dims)

        print("\tmade ", len(self._state_images))

    def _get_state_at(self, pos):
        # Which layer is the x, y position in?
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
                    self._recalc_neighbors()
                self._click_state = None

    def _draw_edges(self, frame):
        """
        Add edge-lines to the image.
        For now:  Attach each edge to wichever attachment point is closest to the center of the parent state.

        # TODO:  Balance left & right attachment points for each state with edges.
        """

        for (color, thickness) in self._edge_lines:
            line_strips = self._edge_lines[(color, thickness)]
            cv2.polylines(frame, line_strips, isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    def _scan_neighbors(self):
        """
        Breadth-first search starting from each highlighted state, up at most self._p_depth, down at most
            self._c_depth.

        Collect all states in those trees, and all edges reaching them.

        By the end,
          - self._s_neighbors will be a dict, keys are selected states, values are dicts{states in the tree : depth from key state}
            rooted at that state.
          - self._s_edges will be a list of l-1 lists (one per layer boundary), each a list of the edges, an
            edge is a dict with:
              'from': state_P, 
              'to': state_C, 
              'player': Mark.X or Mark.O   ( = self._children[state_P][state_C][1])
              'action': action that led from state_P to state_C (= self._children[state_P][state_C][0])
        """

        self._s_neighbors = {}  # state: {neighbor: (depth from state) for all neighbors of state}
        self._s_edges = [[] for _ in range(self._max_levels-1)]  # edges between each layer

        for state in self._selected_states:

            # print("Scanning neighbors for:\n"+state)
            self._s_neighbors[state] = {}  # this state's extended neighbor list, (p,c)-deep in the game tree

            # First, go up:
            next_group = [state]
            for d in range(self._p_depth):
                new_parents = []  # of next_group, to scan next time
                for s in next_group:
                    for p in self._parents[s]:
                        parent_layer_n = self._layers_of_states[p]
                        new_parents.append(p)
                        self._s_neighbors[state][p] = d + 1
                        self._s_edges[parent_layer_n].append({'from': p,  # assign edge to parent layer index
                                                              'to': s,
                                                              'player': self._children[p][s][1],
                                                              'action': self._children[p][s][0]})
                next_group = new_parents
            # print("\tFound %i parents looking up %i levels." % (len(self._s_neighbors[state]), self._p_depth))

            # Then, go down:
            next_group = [state]
            for d in range(self._c_depth):
                new_children = []
                for s in next_group:
                    current_layer_n = self._layers_of_states[s]
                    for c in self._children[s]:
                        if c not in self._states_used:
                            # too low for us
                            continue
                        new_children.append(c)
                        self._s_neighbors[state][c] = d + 1
                        self._s_edges[current_layer_n].append({'from': s,
                                                               'to': c,
                                                               'player': self._children[s][c][1],
                                                               'action': self._children[s][c][0]})
                next_group = new_children
            # print("\tFound %i children looking down %i levels." % (len(self._s_neighbors[state]), self._c_depth))

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
        self._draw_edges(frame)

        return frame
    
    def _recalc_neighbors(self):
        """
        what states/edges need to be drawn differently?
        """

        # Update neighbors of selected states
        self._scan_neighbors()  # Which states are connected?
        self._make_edge_list()  # What are the endpoints of the lines?

    def _make_edge_list(self):
        """
        Find the line coordinates (endpoints, etc) of each edge that needs to be drawn.
        Collect all lines of the same color/thickness together.

        self._edge_lines will be a dict with args for a call to cv2.polylines:
            { ((r,g,b), thickness): [line1, line2, ...] }
        i.e. a dict with (color, thickness) tuple as the key and the list of lines to draw as the value.

        Each line is a int32 numpy array (line strip) of (x1, y1, x2, y2) in pixels.

        Thickness should use the lower state's bar_width, color uses the player who made the move.
        """
        edges_added = {}  # Dict (from, to) -> True, to avoid double-adding edges
        self._edge_lines = {}

        for parent_layer, edge_list in enumerate(self._s_edges):


            for edge in edge_list:
                from_state = edge['from']
                to_state = edge['to']
                player = edge['player']
                # action = edge['action']  # not used yet (could be used to color or label edges)
                child_layer = self._layers_of_states[to_state]
                thickness = self._state_dims[child_layer]['bar_w']

                from_attach, to_attach = self._attach_states(from_state, to_state)

                # find the line
                if player == Mark.X:
                    color = COLOR_X
                elif player == Mark.O:
                    color = COLOR_O

                line = np.array([from_attach,to_attach], dtype=np.int32)

                edge_list = self._edge_lines.get((color, thickness), [])
                edge_list.append(line)
                state_pairs = (from_state, to_state), (to_state, from_state)
                if state_pairs[0] not in edges_added and state_pairs[1] not in edges_added:
                    edges_added[state_pairs[0]] = True
                    edges_added[state_pairs[1]] = True
                    self._edge_lines[(color, thickness)] = edge_list

    def _attach_states(self, from_state, to_state):
        """
        Find the attachment points for the line from from_state to to_state.
        """
        def _box_center(box):
            return np.array([box['x'][0] + box['x'][1], box['y'][0] + box['y'][1]]) / 2

        from_box, to_box = self._positions[from_state], self._positions[to_state]
        from_pos, to_pos = _box_center(from_box), _box_center(to_box)
        # Up or down?  Left or right?
        #import ipdb; ipdb.set_trace()
        if self._layers_of_states[from_state] < self._layers_of_states[to_state]:
            
            if from_pos[0] <= to_pos[0]:
                # from above, moving down and right
                from_attach = self._appach_pts[from_state]['lower'][1]
                to_attach = self._appach_pts[to_state]['upper'][0]

            else:
                # from above, moving down and left
                from_attach = self._appach_pts[from_state]['lower'][0]
                to_attach = self._appach_pts[to_state]['upper'][1]

        else:

            if from_pos[0] < to_pos[0]:
                # from below, moving up and right
                from_attach = self._appach_pts[from_state]['upper'][1]
                to_attach = self._appach_pts[to_state]['lower'][0]

            else:
                # from below, moving up and left
                from_attach = self._appach_pts[from_state]['upper'][0]
                to_attach = self._appach_pts[to_state]['lower'][1]
                
        return from_attach, to_attach 

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
            elif k == ord('['):
                self._p_depth = max(0, self._p_depth - 1)
                print("Highlighting parent states/edges to depth: ", self._p_depth)
                self._recalc_neighbors()
            elif k == ord(']'):
                self._p_depth += 1
                print("Highlighting parent states/edges to depth: ", self._p_depth)
                self._recalc_neighbors()
            elif k == ord(';'):
                self._c_depth = max(0, self._c_depth - 1)
                print("Highlighting parent states/edges to depth: ", self._c_depth)
                self._recalc_neighbors()
            elif k == ord('\''):
                self._c_depth += 1
                print("Highlighting parent states/edges to depth: ", self._c_depth)
                self._recalc_neighbors()
            elif k==ord('c'):
                self._selected_states = []
                self._recalc_neighbors()
            frame_no += 1
            if frame_no % 10 == 0:
                logging.info(f"Frame {frame_no}")
        cv2.destroyAllWindows()


def run_app():

    app = GameGraphApp(no_plot=False, max_levels=10)

    frame = app._make_frame()
    app.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_app()
    logging.info("Exiting.")
