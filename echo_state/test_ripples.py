from ripples import Wave, Pond, get_natural_raindrops, get_drips
import numpy as np
import matplotlib.pyplot as plt
import logging
import matplotlib.animation as animation

def test_wave():
    n_steps = 3000
    n_pixels = 500  # discretize [0, x_max] into this many pixels
    x_max = 100
    dt = 0.1
    t_max = n_steps*dt
    
    wave = Wave(x_0=35, a=20.,  x_right=x_max, speed_factor=.1, decay_factor=.995, amp_thresh=1, reflect=(True, True),scale=.5)

    amps = []
    for iter in range(n_steps):
        amps.append(wave.get_amplitudes(n_pixels))
        if not wave.tick(dt=dt):
            break
    img = np.array(amps)
    plt.subplot(2, 1, 1)
    plt.imshow(img.T, cmap='hot', interpolation='nearest')
    plt.ylabel('x')
    plt.xlabel('t')
    plt.yticks([0, n_pixels], [0, x_max])
    plt.xticks([0, n_steps], [0, t_max])
    plt.title("Decayed to %.6f after %i iterations" % (np.max(img[-1, :]), iter+1))
    plt.colorbar()
    plt.subplot(2,1, 2)
    plt.plot(np.linspace(0,x_max, n_pixels), img[0,:], label='wave at t=0')
    plt.ylim(-5, 100)
    plt.show()


def plot_raindrop_size_dist():

    n = 500000
    raindrops = get_natural_raindrops(n, 100, 100, 10)
    masses = [drop['mass'] for drop in raindrops]
    counts, bins = np.histogram(masses, bins=100, density=False)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    plt.plot(bin_centers, counts, 'o-')
    plt.xlabel('Raindrop Mass')
    plt.ylabel('Counts')
    plt.title('Natural raindrop size distribution')
    plt.show()


def test_pond(n_drops=100):
    n_iter = 1000
    decay = 0.995
    x_max = 500
    speed_factor = 0.1
    mean_amp = 20
    t_max =1000.

    pond = Pond(n_x=300, x_max=x_max, decay_factor=decay, speed_factor=speed_factor, )
    pond_lo_res = Pond(n_x=100, x_max=x_max, decay_factor=decay, speed_factor=speed_factor)
    drops = get_natural_raindrops(n_drops, n_iter, x_max, amp_mean=mean_amp)
    waves,_ = pond.simulate(drops,t_max=t_max)
    waves_2,_ = pond_lo_res.simulate(drops,t_max=t_max)
    waves_3,_ = pond_lo_res.simulate(drops, iter=500,t_max=t_max)
    img1 = np.array(waves)
    img2 = np.array(waves_2)
    img3 = np.array(waves_3)
    imgs = [img1, img2, img3]
    titles = ["x_units = %i, t_steps = %i, xmax=%.0f" % (img.shape[1], img.shape[0] ,x_max) for img in imgs]

    a_0 = imgs[0].shape[1]/imgs[0].shape[0]


    def _plt(img, i, title):
        plt.subplot(1, len(imgs), i+1)
        plt.imshow(img, cmap='hot', interpolation='nearest')

        if i > 0:
            aspect = imgs[i].shape[1]/imgs[i].shape[0]
            scale = aspect / a_0
            plt.gca().set_aspect(scale)

        # plt.axis('square')
        plt.xlabel('x')
        plt.ylabel('t')
        
        plt.title(title)

    plt.figure(figsize=(12, 6))
    for i in range(len(imgs)):
        _plt(imgs[i], i, titles[i])
    plt.show()
    fig, ax = plt.subplots()
    n_frames = np.max([img.shape[0] for img in imgs])
    lines = [ax.plot(np.linspace(0, x_max, imgs[i].shape[1]), imgs[i][0, :], label="%i" % i)[0] for i in range(len(imgs))]
    #for iter in range(n_frames):
    def animate(iter):
        ind_frac = (iter/n_frames)
        # plot current pond state
        for i in range(len(imgs)):
            row = int(imgs[i].shape[0] * ind_frac)
            lines[i].set_ydata(imgs[i][row, :])
        ax.set_title("t = %.2f" % (iter*t_max/n_frames))
        ax.set_xlabel('x')
        ax.set_ylabel('height')
        ax.legend()
        plt.axis('equal')
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, 100)
        return lines

    
    ani = animation.FuncAnimation(fig, animate, frames=n_frames, interval=10, blit=False, repeat=False)
    plt.show()

def test_esn_test_train(n_drops=5,xmax=100.,t_max=500.):
    pond = Pond(n_x=50, x_max=xmax, decay_factor=.98,wave_scale=1)
    #drops = get_natural_raindrops(n_drops, t_max, xmax, amp_mean=10)
    drops = get_drips(t_max, xmax,period=150, amp=20)
    output, input = pond.simulate(drops,iter=int(t_max), t_max=t_max)
    img_out = np.array(output)
    img_in = np.array(input)
    extent = [0, xmax, 0, t_max]
    plt.subplot(1, 2,1)
    plt.imshow(img_out, cmap='hot', interpolation='nearest',extent=extent)
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title("Output")

    plt.subplot(1, 2, 2)
    plt.imshow(img_in, cmap='hot', interpolation='nearest',extent=extent)   
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title("Input")

    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) 
    #plot_raindrop_size_dist()
    #test_wave()
    #test_pond()
    test_esn_test_train()
    #plt.waitforbuttonpress()

