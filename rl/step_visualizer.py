from abc import ABC, abstractmethod
import numpy as np
import logging
import cv2
from tic_tac_toe import Game, Mark, Result
from colors import get_n_colors, shade_color, NEON_GREEN, OFF_WHITE_RGB
from drawing import GameStateArtist
from color_scaler import ColorScaler
from tiny_histogram import MultiHistogram

class PEStep(ABC):
    """
    Some computation just completed, two things should happen:
        1. The permanent images should reflect the new values / changes (_update_iamges, called from constructor).
        2. The display images might be created, if the gui is pausing, or running continuously but in need of a new frame.
             (annotate_imgages, called by the GUI).

    
    Demo will generate these, send them to the GUI, which will apply them as frames are needed and then forget them.

    Include all information needed to update the GUI & methods to alter the state/value/update images & 
    render the step visualization.  GUI will call these when it needs updates.
    """

    def __init__(self, demo, gui):
        self._demo = demo
        self._gui = gui

        self.update_images()  # this one always happens

    @abstractmethod
    def annotate_images(self):
        """
        Calculations ran, now we need to display something with the results.  These images disappear.
        (For speed modes that pause between steps.)
        
        (set self._gui.disp_images)
        """
        pass

    @abstractmethod
    def update_images(self):
        """
        Modify the images so they reflect the current state of the algorithm.  These images store the 
        cumulative results of the algorithm, and are updated as the algorithm runs.

        (set self._gui.base_images)
        """
        pass

    @abstractmethod
    def draw_step_viz(self, img_size):
        """
        Draw the step visualization in the GUI.

        :param img_size:  size of the image to draw.
        :returns:  the image to draw.
        """
        pass

    def post_viz_cleanup(self):
        """
        Called when the step visualization is done.  Clean up any temporary images or data.
        Restore GUI, etc.
        """
        pass

from loop_timing.loop_profiler import LoopPerfTimer as LPT
class StateUpdateStep(PEStep):
    """
    Update for v_new(S). 
    """

    @LPT.time_function
    def __init__(self, demo, gui, state, actions, next_states, rewards, old_values, new_value, bg_color=OFF_WHITE_RGB):
        self._state = state  # the state being updated
        self._actions = actions  # possible actions for the state
        self._next_states = next_states  # distribution of next states for each action
        self._rewards = rewards  # reward for each action
        self._old_value_LUT = old_values  # old value for the state
        self._new_value = new_value  # new value for the state
        self.delta = new_value - old_values[state]
        self._bg_color = bg_color  # background color for the images

        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._title_font_scale = 0.5
        self._font_scale = 0.4
        self._tiny_scale = 0.3
        super().__init__(demo, gui)

        self._init_colors()

    def _init_colors(self):
        n_actions = len(self._actions)
        self._n_next_states = [len(self._next_states[a_ind]) for a_ind in range(n_actions)]
        self._colors = get_n_colors(n_actions)
        self._shades = [shade_color(c, self._n_next_states[c_i]) for c_i, c in enumerate(self._colors)]

    def annotate_images(self):
        """
        1. Draw a thick green box around the state being updated.

        2. For the N possible actions, pick N colors.

        3. For each next state, draw a box around it in a shaded color corresponding to the action leading to it.
           each of the next states for each action.

        Do this for all three images.        
        """
        n_actions = len(self._actions)

        def add_box(state, color, thickness):
            # print("Adding state:\n%s\n" % str(state))
            for img in self._gui.disp_images.values():
                self._gui.box_placer.draw_box(img, state, color=color, thickness=thickness)

        add_box(self._state, NEON_GREEN, 1)
        for a_ind, action in enumerate(self._actions):
            for s_ind, (next_state, prob) in enumerate(self._next_states[a_ind]):
                add_box(next_state, self._shades[a_ind][s_ind], 1)

    @LPT.time_function
    def update_images(self):
        """
        Draw the new color in the updates image.
        Leave the other images untouched.
        """
        new_color = self._gui.color_scalers['updates'].update_and_get_color(self._new_value)
        self._gui.box_placer.draw_box(self._gui.base_images['updates'], self._state, color=new_color, thickness=0)


    
    
    # STEP VIS STUFF BELOW HERE: 
    def _calc_dims(self, w, h, pad=17):
        """
        Calculate where everything goes, e.g. for 3 possible actions using 2 next-state columns:

                +---------------------------------------------+
                |                                             |
                |  Updating state:                            |
                |  [state]                                    |
                |                                             |
                |                Action 1:                    |  
                |                [interm_state 1]             |
                |                                             |
                |      Next state dist:                       |                          
                |      [next_state_1.1]  [next_state_1.2]     |
                |      [next_state_1.3]  [next_state_1.4]     |
                |      [next_state_1.5]                       |
                |                                             |
                |                                             |    
                |                Action 2:  (max)             |                          
                |                [interm_state 2]             |
                |                                             |
                |      Next state dist:                       |                          
                |      [next_state_2.1]                       |
                |                                             |
                |                                             |    
                |                Action 3:                    |                    
                |                [interm_state 3]             |
                |                                             |              
                |      Next state dist:                       |                          
                |      [next_state_3.1] [next_state_3.2]      |   
                |                                             |                                  
                +---------------------------------------------+

        The current state is at the top.
        Intermediate states are centered.
        Under each intermediate state are all the next states possible from it.
        Under each state s is v(s) or the reward for termainal states.
        Under the updating state is the old and new v(s).

        There is a colored box around each action/intermediate state.
        There is a shaded box around each next state.

        :param w:  width of the image to draw.
        :param h:  height of the image to draw.
        :param pad: padding between the boxes & edges. 
        :returns:  dict with keys 'state', 'next_states', and 'leaf_states' with their positions and sizes.
        """
        # import ipdb; ipdb.set_trace()
        # 1. determine how many rows we need to see how big we can make the icons.
        n_action_rows = len(self._actions)
        n_next_state_rows = [np.ceil(len(self._next_states[a_ind])/2) for a_ind in range(n_action_rows)]
        #print("Nnumber of next states per action: %s" % str(self._n_next_states))
        #print("Number of next state rows per action: %s" % str(n_next_state_rows))

        text_h = 18  # height of the text to draw under the states (1 line)
        artists = self._calc_box_sizes(w, h, n_action_rows, n_next_state_rows, text_h, pad)
        box_sizes = {kind: artist.dims['img_size'] for kind, artist in artists.items()}

        title_text_h = cv2.getTextSize("V'(s)=9.9", self._font, self._title_font_scale, 1)[0][1]
        state_h = box_sizes['state']   # height of the state box + text
        dims = {
            'state': {'pos': (pad, pad), 'size': box_sizes['state'],
                      'text1_pos': (pad + 8 + box_sizes['state'],  title_text_h + pad),
                      'text2_pos': (pad + 8 + box_sizes['state'],  title_text_h*2 + pad+10),
                      'artist': artists['state'],
                      'font_scale': self._title_font_scale},
            'inter_states': [],
            'next_states': [],
            'y_lines': []}
        y = state_h
        dims['y_lines'].append(y)
        inter_h = box_sizes['inter_states'] + pad

        inter_x_left = int((w - box_sizes['inter_states']) / 2)

        row_pad = (9-n_action_rows) * 10 + 10

        for a_ind, action in enumerate(self._actions):
            inter_state = {'pos': (inter_x_left, y), 'size': box_sizes['inter_states'],
                           'text1_pos': (inter_x_left + box_sizes['inter_states']+9, y+14),
                           'action': action,
                           'artist': artists['inter_states'],
                           'font_scale': self._font_scale}
            dims['inter_states'].append(inter_state)

            y += inter_h

            next_h = box_sizes['next_states'] + text_h + pad
            dims['next_states'].append([])

            n_next_states = len(self._next_states[a_ind])
            next_cols_centers_x = np.linspace(0, w, n_next_states+2)[1:-1]  # center of the next state boxes
            nex_col_x_lefts = (next_cols_centers_x - box_sizes['next_states'] / 2).astype(int)

            for s_ind, (next_state, prob) in enumerate(self._next_states[a_ind]):

                x = nex_col_x_lefts[s_ind % n_next_states]  # left edge of the next state box
                next_state = {'pos': (x, y), 'size': box_sizes['inter_states'],
                              'text1_pos': (x-8, y+box_sizes['next_states'] + 18),
                              'text2_pos': (x-8, y + box_sizes['next_states'] + 29),
                              'prob': prob,
                              'artist': artists['next_states'],
                              'font_scale': self._tiny_scale}
                dims['next_states'][-1].append(next_state)
                if s_ind % n_next_states == n_next_states-1 and s_ind < len(self._next_states[a_ind]) - 1:
                    y += next_h

            y += next_h  +row_pad # add padding between rows of next states

        return dims

    def _calc_box_sizes(self, w, h, n_action_rows, n_next_state_rows, text_h, pad):
        """
        Keep boxes in proportion, see how big we can make them and still fit.
        Horizontally, each row of next states needs to fit two images.

        Vertically we need to fit:
          1. the state being updated
          2. Each action / intermediate state
          3. Each row of next states

        Solve for the state box size (1), the other two are defined by proportions. 
            - Horizontally constrained:  
                max-width = 2 * interm_box_size + 3 * pad, so solving for interm_box_size:
                interm_box_size = (w - 3 * pad) / 2
            - Vertically constrained:
                max-height = (state_box_size + text_h * 2) + n_actions * (interm_box_size + text_h + pad) + n_next_state_rows (next_box_size + text_h + pad), and
                interm_box_size = state_box_size * interm_ratio
                next_box_size = interm_box_size * next_ratio, so solving for state_box_size gives:
                state_box_size = (h - 2 * pad - text_h * 2) / (n_action_rows + n_next_state_rows + 1)

                state_box_size  = h - text_h * 2 - n_actions * (text_h + pad) - n_next_state_rows * (text_h + pad) 
                                --------------------------------------------------------------------------------
                                        1 + n_actions*interm_ratio + n_next_state_rows * next_ratio
        :returns: artists for each box kind
        """
        box_sizes = {'state': 50, 'inter_states': 40, 'next_states': 30}
        space_sizes = {kind: GameStateArtist.get_space_size(box_size) for kind, box_size in box_sizes.items()}
        artists = {kind: GameStateArtist(space_size=sp_size, bar_w_frac=0) for kind, sp_size in space_sizes.items()}
        return artists

        interm_ratio = 0.8  # to size of state box
        next_ratio = 0.65

        # Get dimensions if horizontally constrained:
        interm_box_size_x = (w - 3 * pad) / 2
        state_box_size_x = interm_box_size_x / interm_ratio
        next_box_size_x = state_box_size_x / next_ratio

        # Get dimensions if vertically constrained:
        numerator = h - 2 * pad - text_h * 2 - n_action_rows * (text_h + pad) - n_next_state_rows * (text_h + pad)
        denom = 1 + n_action_rows * interm_ratio + n_next_state_rows * next_ratio
        state_box_size_y = numerator / denom
        interm_box_size_y = state_box_size_y * interm_ratio
        next_box_size_y = interm_box_size_y * next_ratio

        if state_box_size_x < state_box_size_y:
            return state_box_size_x, interm_box_size_x, next_box_size_x
        else:
            return state_box_size_y, interm_box_size_y, next_box_size_y

    def draw_step_viz(self):
        """
        Draw the state being updated in the top left, the intermediate states resulting from 
        each action distributed in a row under it, and the distribution of next states
        under each intermediate state.

        Under the RL states, print the value function. 
        Under the intermediate states, print the reward values.
        """
        w, h = self._gui.get_step_viz_frame_size()
        dims = self._calc_dims(w, h)
        #import pprint
        #pprint.pprint(dims)
        logging.info("Creating step visualization with size %i x %i" % (w, h))
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = self._bg_color
        #print("BG COLOR:", img[0, 0])
        m = 3

        def add_gamestate(state, info, box_color, text_1=None, text_2=None, thickness=1):
            #import pprint
            #pprint.pprint(info)

            #print("\n")
            x, y = info['pos']
            state_img = info['artist'].get_image(state)
            try:
                img[y:y+state_img.shape[1], x:x+state_img.shape[1]] = state_img
            except ValueError as e:
                logging.warning("Error adding state image: %s" % str(e))
            # Add the text
            if text_1 is not None:
                tx, ty = info['text1_pos']
                font_scale = info['font_scale']
                cv2.putText(img, text_1, (tx, ty), self._font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
            if text_2 is not None:
                tx, ty = info['text2_pos']
                font_scale = info['font_scale']
                cv2.putText(img, text_2, (tx, ty), self._font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

            box_color = int(box_color[0]), int(box_color[1]), int(box_color[2])
            # import ipdb; ipdb.set_trace()
            cv2.rectangle(img, (x-1-m, y-1-m), (x+state_img.shape[1]+m, y+state_img.shape[0]+m), box_color, thickness, cv2.LINE_AA)
            # add the reward value under the intermediate state: (skip for now)
        # import ipdb; ipdb.set_trace()
        # In the top right, draw the updating state in a green box.
        add_gamestate(self._state, dims['state'], NEON_GREEN, text_1="V(s)=%.3f" % self._old_value_LUT[self._state],
                      text_2="V'(s)=%.4f" % self._new_value, thickness =3)

        # show each following state under each action in the appropriate color.
        for a_ind, action in enumerate(self._actions):
            inter_state = self._state.clone_and_move(action, self._gui.player)
            add_gamestate(inter_state, dims['inter_states'][a_ind], self._colors[a_ind],
                          text_1="Reward(a)=%.3f" % self._rewards[a_ind], thickness=2)

            # Show the final states under each intermediate:
            for s_ind, (next_state, prob) in enumerate(self._next_states[a_ind]):

                #if next_state not in self._old_value_LUT:
                #    print("State not in LUT: %s" % str(next_state))

                add_gamestate(next_state, dims['next_states'][a_ind][s_ind], self._shades[a_ind][s_ind],
                              text_1="P: %.2f" % prob,
                              text_2="V: %.2f" % self._old_value_LUT[next_state],
                              thickness=2)

        # for y in dims['y_lines']:
        #    cv2.line(img, (0, y), (w, y), (0, 0, 0), 1)
        return img


class EpochStep(PEStep):
    def __init__(self, demo, gui, v_old, v_new, epoch_ind, states_by_layer):
        self._v_old = v_old  # old value function
        self._v_new = v_new  # new value function
        self._epoch_ind = epoch_ind  # epoch being updated
        #import ipdb; ipdb.set_trace()
        self._delta_v = {state: v_new[state] - v_old[state] for state in v_old.keys()}
        all_deltas = np.array(list(self._delta_v.values()))
        print(np.sum(np.isnan(all_deltas)), np.sum(np.isinf(all_deltas)), np.sum(np.isfinite(all_deltas)))
        print(np.max(all_deltas), np.min(all_deltas), np.mean(all_deltas), np.std(all_deltas))
        self._states_by_layer = states_by_layer  # states by layer

        self._color_scaler = ColorScaler(self._delta_v.values())
        super().__init__(demo, gui)

    def update_images(self):
        """
        Draw the epoch in the base images.
        values:  The update image should now be the values image.
        updates: compute a delta image for the values.
        """
        #import ipdb; ipdb.set_trace()


        delta_img = self._gui.box_placer.draw(colors=self._color_scaler.get_LUT(self._delta_v),dest=self._gui.get_blank('values'))

        self._gui.base_images['values'] = self._gui.base_images['updates']
        self._gui.base_images['updates'] = delta_img

    def annotate_images(self):
        """
        Draw the epoch in the base images.
        values:  The update image should now be the values image.
        updates: compute a delta image for the values.
        """
        # no annotations for epochs, just use the base images.
        self._gui.disp_images['values'] = self._gui.base_images['values']
        self._gui.disp_images['updates'] = self._gui.base_images['updates']
        self._old_gui_disp_titles = self._gui.disp_titles
        self._gui.disp_titles['values'] = "New V(s)" 
        self._gui.disp_titles['updates'] = "Delta V(s)" 
        
    def post_viz_cleanup(self):
        self._gui.disp_titles = self._old_gui_disp_titles   
        

    def draw_step_viz(self):
        """
        Epoch visualization:  For each level, show a small histogram of the changes in value function at that level.
        """
        #import ipdb; ipdb.set_trace()
        blank = self._gui.get_blank('step_vis')
        size = blank.shape[1], blank.shape[0]
        deltas_by_layer = [[self._delta_v[state] for state in self._states_by_layer[lyr]] for lyr in range(6)]
        #import pprint
        #pprint.pprint(deltas_by_layer)
        
        mh = MultiHistogram(img_size=size, value_lists=deltas_by_layer, title = 'delta v(s)')
        #import ipdb; ipdb.set_trace()
        hist_img = mh.draw(blank)
        #cv2.imwrite("epoch_%i_hist.png"% self._epoch_ind, hist_img)
        return hist_img

class PIStep(PEStep):
    def __init__(self, demo, gui, phase, update_info):
        super().__init__(demo, gui)
        self._phase = phase
        self._info = update_info

    def annotate_images(self, images):
        """
        Draw a green box around the state being updated.
        For the N possible actions, pick N colors and draw boxes in those colors around 
        each of the next states for each action.

        Do this for all three images.        
        """
        pass  # TODO:  implement this


class ContinuousStep(PEStep):
    def __init__(self, demo, gui, state_updates):
        super().__init__(demo, gui)
        self._state_updates = state_updates
        self._epoch = demo._epoch  # the epoch being updated

    def annotate_images(self, images):
        """
        Draw a green box around the state being updated.
        For the N possible actions, pick N colors and draw boxes in those colors around 
        each of the next states for each action.

        Do this for all three images.        
        """
        pass  # TODO:  implement this


class FakeGui(object):
    def __init__(self):
        self._size = 431, 980
        self.player = Mark.X
        self.opponent = Mark.O

    def get_step_viz_frame_size(self):
        return self._size


def test_state_update_vis():
    #     def __init__(self, demo, gui, state, actions, next_states, rewards, old_value, new_value, bg_color=(0, 0, 0)):

    state = Game.from_strs(["  ", "   ", "   "])
    actions = state.get_actions()
    print("TEST Actions: %s" % str(actions))
    intermediate_states = [state.clone_and_move(action, Mark.X) for action in actions]
    next_states = [[(inter_state.clone_and_move(act, Mark.O), np.random.rand())
                    for act in inter_state.get_actions()] for inter_state in intermediate_states]
    rewards = [0.0 for _ in actions]
    all_states = [state] + [next_pair[0] for state_dist in next_states for next_pair in state_dist]
    old_values = {state: np.random.rand(1) for state in all_states}
    new_value = 0.5
    step = StateUpdateStep(None, FakeGui(), state, actions, next_states, rewards, old_values, new_value)

    img = step.draw_step_viz()
    cv2.imshow("Step Visualization", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.info("Testing step visualizer")
    test_state_update_vis()
