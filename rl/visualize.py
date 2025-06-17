"""
Adapted from neat/examples/xor/visualize.py
"""

from __future__ import print_function

import copy
import warnings
import logging
import graphviz
import matplotlib.pyplot as plt
import numpy as np
from neat_util import NNetPolicy


class Visualizer(object):
    def __init__(self, axes, config):
        self._config = config
        self._iter = 0
        self._axes = axes
        strongweak_string = "STRONG" if config.strong else "WEAK"
        enc_size_str = NNetPolicy.INPUT_ENC_SIZES[config.genome_config.num_inputs]
        self._title = "NEAT, finding %s solution, pop-size=%i\ninput-encoding: %s,  samp/eval=%i" % (
            strongweak_string, config.pop_size, enc_size_str, config.n_evals)
        
        self._plots = {'curves':{},
                       'speciation':{}}

    def _plot_fitness_stats(self, statistics, ylog=False, view=False):
        """ Plots the population's average and best fitness. """
        ax = self._axes[0]

        generation = range(len(statistics.most_fit_genomes))
        best_fitness = np.array([c.fitness for c in statistics.most_fit_genomes])
        avg_fitness = np.array(statistics.get_fitness_mean())
        stdev_fitness = np.array(statistics.get_fitness_stdev())
        if 'best_fitness' in self._plots['curves']:
            def _update_curve(curve, ydata):
                # safe to assume all generations are here?
                x_data = np.arange(ydata.shape[0])
                curve.set_xdata(x_data)
                curve.set_ydata(ydata)

            _update_curve(self._plots['curves']['best_fitness'], best_fitness) 
            _update_curve(self._plots['curves']['avg_fitness'], avg_fitness)
            _update_curve(self._plots['curves']['+stdev_fitness'], avg_fitness + stdev_fitness) 
            _update_curve(self._plots['curves']['-stdev_fitness'], avg_fitness - stdev_fitness)
            ax.set_xlim(0, len(generation) - 1)
            ax.set_ylim(np.min(avg_fitness - stdev_fitness) - 0.2, np.max(best_fitness) + 0.1)

        else:
            self._plots['curves']['best_fitness'], = ax.plot(generation, best_fitness, 'r-', label="best")
            self._plots['curves']['avg_fitness'], = ax.plot(generation, avg_fitness, 'b-', label="average")
            self._plots['curves']['+stdev_fitness'], = ax.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
            self._plots['curves']['-stdev_fitness'], = ax.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
            
            ax.set_xlabel("Generations")
            ax.set_ylabel("Fitness")
            # add a grid to the axis
            ax.grid(visible=True)
            # add a legend to the axis
            ax.legend(loc='lower right')
            if ylog:
                ax.set_yscale('symlog')

    def _plot_species(self, statistics, view=False ):
        species_sizes = statistics.get_species_sizes()
        num_generations = len(species_sizes)
        curves = np.array(species_sizes).T
        ax = self._axes[1]
        x = np.arange(num_generations)

        species_img = get_species_plot(x, curves, self._config.pop_size)
        import cv2
        cv2.imwrite("species_img.png", species_img[:,:,::-1])

        if 'speciation_img' not in self._plots['speciation']:
            self._plots['speciation']['speciation_img'] = ax.imshow(species_img, aspect='equal')
            # expand margins of the axis
            ax.margins(x=0.01, y=0.01)
            # remove all tics and labels
            ax.set_xticks([])
            ax.set_yticks([])
            # set the aspect ratio to equal
            ax.set_aspect('equal')
            # set the title of the axis
        else:
            # update the image
            self._plots['speciation']['speciation_img'].set_data(species_img)


    def update(self, statistics):
        self._iter+=1
        
        self._plot_fitness_stats(statistics)
        self._plot_species(statistics)
        #set sup title
        plt.suptitle(self._title)
        plt.tight_layout()
        

def get_species_plot(x, curves,pop_size, img_size=(5,3.5), dpi=200):
    """
    Draw the figure to a png.
    :param x: x-axis values
    :param curves: y-axis values for each species
    :param pop_size: population size (int)
    :param img_size: size of the image in inches
    :return: image as a numpy array
    """
    fig, ax = plt.subplots(figsize=img_size, dpi=dpi)
    ax.stackplot(x, *curves)
    ax.set_title("Speciation" + (" (pop_size=%i)" % pop_size if pop_size is not None else ""))
    ax.set_ylabel("Size per Species")
    ax.set_xlabel("Generations")
    # remove box around the plot
    ax.spines['top'].set_visible(False)

    plt.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:,:,0:3]  # remove alpha channel
    plt.close(fig)
    return img  

def test_get_species_plot():
    x = np.arange(10)
    curves = [np.random.randint(0, 10, size=10) for _ in range(5)]
    img = get_species_plot(x, curves, pop_size=100)
    plt.imshow(img)
    plt.show()

def draw_ttt_net(config, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False,
                 node_colors=None, fmt='svg'):
    """ Receives a tic-tac-toe policy genome and draws a neural network with its topology."""
    # Attributes for network nodes.
    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return

    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict

    node_attrs = {
        'shape': 'circle',
        'fontsize': '8',
        'height': '0.1',
        'width': '0.1'}

    # graphviz.Digraph(format=fmt, node_attr=node_attrs)
    dot = graphviz.Graph(engine='neato', node_attr=node_attrs, format=fmt)
    dot.graph_attr['ranksep'] = '1.5'  # Increase vertical spacing between ranks
    dot.graph_attr['nodesep'] = '3'

    inputs = set()
    input_keys = config.genome_config.input_keys

    # Horizontal and vertical spacing between nodes

    def _add_unit_grid(keys, x_offset, y_offset, label_prefix, spacing, x_spread=0):
        """ Helper function to add a grid of input nodes. """

        n_added = 0
        for i in range(3):
            y = -(i * spacing[1]) - y_offset

            for j in range(3):
                x = (i * spacing[0]) + j * x_spread + x_offset

                k = keys[i * 3 + j]
                inputs.add(k)
                name = node_names.get(k, str(k))
                pos = f'{x},{y}!'
                print("Adding input node %s at pos %s" % (name, pos))
                dot.node(name, _attributes={'style': 'filled', 'shape': 'box', 'label': f'{label_prefix}({i},{j})',
                                            'fillcolor': node_colors.get(k, 'lightgray'), 'pos': pos})
                n_added += 1
                if j == 1:
                    center_x = x

        return n_added, center_x
    out_x = 2

    if len(input_keys) == 9:   # (x=1, o=-1, empty=0) for 9 spaces
        # 3x3 grid
        in_out_sep = 3
        n_added, center_x = _add_unit_grid(input_keys, 0, 0, 'X/O/Free', h_spacing=4, spacing=(.5, .5))
        if (n_added != len(input_keys)):
            raise ValueError("Number of input keys does not match number of inputs in genome: %d != %d" %
                             (n_added, len(input_keys)))

    elif len(input_keys) == 18:   # (x=1, o=-1, empty=0) for 9 spaces, then (is_free) for 9 spaces
        in_out_sep = 6
        marked_x = 1
        free_x = 8
        _add_unit_grid(input_keys[:9], marked_x, 0, 'X/O', spacing=(1, .8), x_spread=1)
        _add_unit_grid(input_keys[9:], free_x, 0, 'Free', spacing=(1, .8), x_spread=1)

    elif len(input_keys) == 27:   # (x=1, o=-1, empty=0) for 3 spaces
        in_out_sep = 7
        mark_x = 0.5
        mark_o = 4.75
        mark_empty = 9
        _add_unit_grid(input_keys[:9], mark_x, 0, 'X', spacing=(1, .8), x_spread=1)
        _add_unit_grid(input_keys[9:18], mark_o, 0, 'O', spacing=(1, .8), x_spread=1)
        _add_unit_grid(input_keys[18:], mark_empty, 0, 'Free', spacing=(1, .8), x_spread=1)

    outputs = set()
    output_keys = config.genome_config.output_keys
    n_added = _add_unit_grid(output_keys, out_x, in_out_sep, 'X-Act', spacing=(1.2, .5), x_spread=3.5)[0]
    if (n_added != len(output_keys)):
        raise ValueError("Number of output keys does not match number of outputs in genome: %s != %s" %
                         (n_added, len(output_keys)))
    if prune_unused:
        connections = set()
        for cg in genome.connections.values():
            if cg.enabled or show_disabled:
                connections.add((cg.in_node_id, cg.out_node_id))

        used_nodes = copy.copy(outputs)
        pending = copy.copy(outputs)
        while pending:
            new_pending = set()
            for a, b in connections:
                if b in pending and a not in used_nodes:
                    new_pending.add(a)
                    used_nodes.add(a)
            pending = new_pending
    else:
        used_nodes = set(genome.nodes.keys())

    for n in used_nodes:
        if n in inputs or n in outputs:
            continue
        print("Adding hidden node %s" % n)
        attrs = {'style': 'filled', 'radius': '0.05', 'width': '0.05',
                 'fillcolor': node_colors.get(n, 'white')}
        dot.node(str(n), _attributes=attrs)

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            # if cg.input not in used_nodes or cg.output not in used_nodes:
            #    continue
            input, output = cg.key
            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 2.5))
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.render(filename, view=view)

    return dot

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Test the get_species_plot function
    test_get_species_plot()
