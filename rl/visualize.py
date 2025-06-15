"""
Adapted from neat/examples/xor/visualize.py
"""

from __future__ import print_function

import copy
import warnings

import graphviz
import matplotlib.pyplot as plt
import numpy as np


def plot_stats(statistics, ylog=False, view=False, filename='avg_fitness.svg', pop_size=None):
    """ Plots the population's average and best fitness. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())
    stdev_fitness = np.array(statistics.get_fitness_stdev())
    plt.plot(generation, avg_fitness, 'b-', label="average")
    plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
    plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
    plt.plot(generation, best_fitness, 'r-', label="best")

    plt.title("Population (%i) fitness"% (pop_size,))
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()


def plot_spikes(spikes, view=False, filename=None, title=None):
    """ Plots the trains for a single spiking neuron. """
    t_values = [t for t, I, v, u, f in spikes]
    v_values = [v for t, I, v, u, f in spikes]
    u_values = [u for t, I, v, u, f in spikes]
    I_values = [I for t, I, v, u, f in spikes]
    f_values = [f for t, I, v, u, f in spikes]

    fig = plt.figure()
    plt.subplot(4, 1, 1)
    plt.ylabel("Potential (mv)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, v_values, "g-")

    if title is None:
        plt.title("Izhikevich's spiking neuron model")
    else:
        plt.title("Izhikevich's spiking neuron model ({0!s})".format(title))

    plt.subplot(4, 1, 2)
    plt.ylabel("Fired")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, f_values, "r-")

    plt.subplot(4, 1, 3)
    plt.ylabel("Recovery (u)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, u_values, "r-")

    plt.subplot(4, 1, 4)
    plt.ylabel("Current (I)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, I_values, "r-o")

    if filename is not None:
        plt.savefig(filename)

    if view:
        plt.show()
        plt.close()
        fig = None

    return fig


def plot_species(statistics, view=False, filename='speciation.svg', pop_size=None):
    """ Visualizes speciation throughout evolution. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    species_sizes = statistics.get_species_sizes()
    num_generations = len(species_sizes)
    curves = np.array(species_sizes).T

    fig, ax = plt.subplots()
    ax.stackplot(range(num_generations), *curves)

    plt.title("Speciation" +(" (pop_size=%i)" % pop_size if pop_size is not None else ""))
    plt.ylabel("Size per Species")
    plt.xlabel("Generations")

    plt.savefig(filename)

    if view:
        plt.show()

    plt.close()


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
    # import ipdb; ipdb.set_trace()
    # graphviz.Digraph(format=fmt, node_attr=node_attrs)
    dot = graphviz.Graph(engine='neato', node_attr=node_attrs, format=fmt)
    dot.graph_attr['ranksep'] = '1.5'  # Increase vertical spacing between ranks
    dot.graph_attr['nodesep'] = '3'

    inputs = set()
    input_keys = config.genome_config.input_keys

      # Horizontal and vertical spacing between nodes

    def _add_unit_grid(keys, x_offset, y_offset, label_prefix,spacing,x_spread=0):
        """ Helper function to add a grid of input nodes. """

        n_added = 0
        for i in range(3):
            y= -(i * spacing[1]) - y_offset

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
                if j==1:
                    center_x = x

        return n_added, center_x
    out_x = 2

    if len(input_keys) == 9:   # (x=1, o=-1, empty=0) for 9 spaces
        # 3x3 grid
        in_out_sep = 3
        n_added,center_x=_add_unit_grid(input_keys, 0, 0, 'X/O/Free', h_spacing=4,spacing = (.5, .5))
        if (n_added != len(input_keys)):
            raise ValueError("Number of input keys does not match number of inputs in genome: %d != %d" %
                             (n_added, len(input_keys)))

    elif len(input_keys) == 18:   # (x=1, o=-1, empty=0) for 9 spaces, then (is_free) for 9 spaces
        in_out_sep = 6
        marked_x =1
        free_x = 8
        _add_unit_grid(input_keys[:9], marked_x, 0, 'X/O',spacing =(1, .8), x_spread = 1)
        _add_unit_grid(input_keys[9:], free_x, 0, 'Free', spacing = (1, .8), x_spread =1)
        

    elif len(input_keys) == 27:   # (x=1, o=-1, empty=0) for 3 spaces
        in_out_sep = 7
        mark_x =0.5
        mark_o = 4.75
        mark_empty = 9
        _add_unit_grid(input_keys[:9], mark_x, 0, 'X',spacing = (1, .8), x_spread =1)
        _add_unit_grid(input_keys[9:18], mark_o, 0, 'O', spacing = (1, .8), x_spread =1)
        _add_unit_grid(input_keys[18:], mark_empty, 0, 'Free', spacing = (1, .8), x_spread =1)
        

    outputs = set()
    output_keys = config.genome_config.output_keys
    n_added = _add_unit_grid(output_keys,out_x ,in_out_sep, 'X-Act', spacing=(1.2,.5), x_spread = 3.5)[0]
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
        attrs = {'style': 'filled','radius': '0.05', 'width': '0.05',
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
