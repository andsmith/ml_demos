"""
Generate test/train datasets.
"""
from ripples import Pond, get_drips, get_natural_raindrops
import numpy as np
import matplotlib.pyplot as plt
import json
from interactive import InteractivePond
from esn import EchoStateNetwork as ESN

import logging


def center_test():
    """
    one drop falling in the middle
    """
    n_units = 100
    t_max = 2000
    x_max = 100

    drips = get_drips(t_max, x_max, period=30, amp=20)
    pond = Pond(n_x=n_units, x_max=x_max, decay_factor=.98, wave_scale=3)
    output, input = pond.simulate(drips, iter=int(t_max), t_max=t_max)

    img_out = np.array(output)
    img_in = np.array(input)

    plt.subplot(1, 2, 1)
    plt.imshow(img_out, cmap='hot', interpolation='nearest')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title("Output")

    plt.subplot(1, 2, 2)
    plt.imshow(img_in, cmap='hot', interpolation='nearest')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title("Input")

    plt.show()

def interactive_test():
    """
    Create an interactive pond to find good params for generating a test/train set for the ESN.
    """
    pond_params = get_pond_params()
    pond = InteractivePond(*pond_params)
    pond.simulate_interactive()
    return pond_params


def _train(input, output, pond_params, n_reservoir=100, plot=True):
    """
    Train an ESN on the input/output data.
    """
    n_plot = np.min((600, input.shape[0]))
    extent = (0, n_plot, 0, pond_params['n_x'])

    n_input = input.shape[1]
    n_output = output.shape[1]
    esn = get_esn(n_input, n_reservoir, n_output)

    def _plot(ax):
        ax[0].imshow(input[:n_plot, :].T, cmap='hot', interpolation='nearest', extent=extent)
        ax[0].set_title("input[:%i]    scale = (%.3f - %.3f)" % (n_plot, np.min(input), np.max(input)))
        ax[0].set_ylabel('input state')
        ax[0].set_xticks([])
        ax[1].imshow(output[:n_plot, :].T, cmap='hot', interpolation='nearest', extent=extent)
        ax[1].set_title("output[:%i]   scale = (%.3f - %.3f)" % (n_plot, np.min(output), np.max(output)))
        ax[1].set_ylabel('output state')
        ax[1].set_xticks([])

        predictions = esn.predict(input)
        ax[2].set_title("ESN(%i),   scale = (%.3f - %.3f)" % (esn.n_res,np.min(predictions), np.max(predictions)))
        ax[2].imshow(predictions[2:n_plot, :].T, cmap='hot', interpolation='nearest', extent=extent)
        ax[2].set_ylabel('ESN output')
        ax[2].set_xlabel('time')
        # plt.tight_layout()

    fig, ax = plt.subplots(3, 1, figsize=(12, 4))
    #_plot(ax)
    #plt.suptitle("ESN untrained (close to start training)")
    #plt.show()
    esn.train(input, output, washout=50, batch_size=2000)
    _plot(ax)
    plt.suptitle("ESN trained")
    plt.show()

    return esn


def drip_training(x_var=0):
    n_iter = 15000
    t_max = n_iter/10.
    dt = t_max/n_iter
    pond_params = get_pond_params()
    pond = Pond(**pond_params)
    drops = get_drips(t_max=t_max, x_max=pond_params['x_max'], period=25*dt, amp=20, x_var=x_var)
    output, input = pond.simulate(drops, iter=n_iter, t_max=t_max)
    net = _train(input, output, pond_params, n_reservoir=200)



def get_pond_params():
    return dict(n_x=50, decay_factor=.9, wave_scale=1., speed_factor=3, x_max=100.)

def get_esn(n_input, n_reservoir, n_output):
    return ESN(n_input, n_reservoir, n_output, spectral_radius=0.9996, leak_rate=0.8)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # interactive_test()
    drip_training(x_var=30)
