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



def _train(input, output, pond_params, plot=True):
    """
    Train an ESN on the input/output data.
    """
    n_plot = np.min((600, input.shape[0]))
    extent = (0, n_plot, 0, pond_params['n_x'])

    n_input = input.shape[1]
    n_output = output.shape[1]
    esn = get_esn(n_input, n_output)

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
    esn.train_sequence(input, output,washout=30)
    esn.finish_training()
    _plot(ax)
    plt.suptitle("ESN trained")
    plt.show()

    return esn


def drip_training(n_iter,x_var=0):
    
    t_max = n_iter/10.
    dt = t_max/n_iter
    pond_params = get_pond_params()
    pond = Pond(**pond_params)
    drops = get_drips(t_max=t_max, x_max=pond_params['x_max'], period=10*dt, amp=20, x_var=x_var)
    output, input = pond.simulate(drops, iter=n_iter, t_max=t_max)
    #output = np.clip(output/10, 0, 1)
    #import pickle
    #pickle.dump((input, output), open('drip_data.pkl', 'wb')) 
    net = _train(input, output, pond_params)


def rain_training(n_iter,n_drops):
    t_max = n_iter/10.
    pond_params = get_pond_params()
    pond = Pond(**pond_params)
    drops = get_natural_raindrops(n_drops, t_max, pond_params['x_max'], amp_mean=10)
    output, input = pond.simulate(drops, iter=n_iter, t_max=t_max)
    net = _train(input, output, pond_params)

def _thread_proc():
    pond_params = get_pond_params()
    pond = InteractivePond(**pond_params)
    pond.simulate_interactive(realtime=True)
    return pond_params

def interactive_test():
    """
    Create an interactive pond to find good params for generating a test/train set for the ESN.
    """
    from multiprocessing import Process
    t = Process(target=_thread_proc)
    t.start() 
    return t

def get_pond_params():
    pp= dict(n_x=25, decay_factor=1, wave_scale=1.0, speed_factor=3, x_max=100., max_wave_age=0)
    return pp

def get_esn(n_input, n_output):
    return ESN(n_input, n_reservoir=200, n_output=n_output, spectral_radius=0.99, leak_rate=0.0, linear_out=True, input_scale=1, feedback_scale=0.0)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    #entertainment = interactive_test()
    #rain_training(n_iter = 40000, n_drops = 2000)
    #drip_training(n_iter=100000, x_var=20)
    #entertainment.join()

    _thread_proc()
