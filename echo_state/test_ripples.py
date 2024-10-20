from ripples import Wave, Pond, get_natural_raindrops
import numpy as np
import matplotlib.pyplot as plt

def test_wave():
    n_steps = 300
    n_pixels = 100 # discretize [0, x_max] into this many pixels
    x_max = 100

    wave = Wave(x_0=35, a=20.,  x_max=x_max, speed_factor=.1, decay_factor=.95, amp_thresh=1,reflect=(True, True) )

    amps=[]
    x = np.linspace(0, x_max, n_pixels)   
    for iter in range(n_steps):
        amps.append(wave.get_amplitudes(x))
        print(wave)
        if not wave.tick(dt=1.0):
            break
    img = np.array(amps)
    plt.imshow(img, aspect='equal', cmap='hot', interpolation='nearest')
    plt.xlabel('x');plt.ylabel('t')
    plt.title("Decayed to %.6f after %i iterations" % (np.min(img[-1,:]), iter))
    plt.axis('equal')
    plt.colorbar()
    plt.show()



def plot_raindrop_size_dist():
    
    n= 100000
    raindrops = get_natural_raindrops(n, 100, 100, 10)
    counts, bins = np.histogram(raindrops['mass'], bins=100, density=False)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    plt.plot(bin_centers, counts,'o-')
    plt.xlabel('Raindrop Mass')
    plt.ylabel('Counts')
    plt.title('Natural raindrop size distribution')
    plt.show()

def test_pond(n_drops=100):
    n_iter = 1000
    decay = 0.995
    x_max=500
    pond = Pond(n_x=600, x_max=x_max, decay_factor=decay)
    pond_lo_res = Pond(n_x=100, x_max=x_max, decay_factor=decay)
    drops = get_natural_raindrops(n_drops, n_iter, x_max, amp_mean=10)
    waves = pond.simulate(drops)
    waves_2 = pond_lo_res.simulate(drops)
    waves_3 = pond_lo_res.simulate(drops, iter=500)
    img1 = np.array(waves)
    img2 = np.array(waves_2)
    img3 = np.array(waves_3)
    imgs = [img1, img2, img3]
    titles = ["x_units = %i, t_steps = %i" % (img.shape[1], img.shape[0]) for img in imgs]

    a_0 = imgs[0].shape[1]/imgs[0].shape[0]
    print("Aspect1: ", a_0)

    def _plt(img, i, title):
        plt.subplot(1,len(imgs),i+1)
        plt.imshow(img, cmap='hot', interpolation='nearest')

        if i>0:
            aspect = imgs[i].shape[1]/imgs[i].shape[0]
            scale = aspect / a_0
            print("Aspect2: ", plt.gca().get_aspect())
            plt.gca().set_aspect(scale)\

        #plt.axis('square')
        plt.xlabel('x');plt.ylabel('t')
        plt.title(title)
            
    plt.figure(figsize=(12,6))
    for i in range(len(imgs)):
        _plt(imgs[i], i,titles[i])
        
    plt.show()


if __name__ == "__main__":
    test_pond()
