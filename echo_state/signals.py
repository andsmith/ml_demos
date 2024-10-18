import numpy as np



def make_square(period, n_samples):
    """
    Generate a square wave.
    """
    return np.array([1 if i % period < period / 2 else -1 for i in range(n_samples)],dtype=np.float32)


def make_sawtooth(period, n_samples):
    """
    Generate a sawtooth wave.
    """
    return np.array([2 * (i % period) / period - 1 for i in range(n_samples)],dtype=np.float32)


def make_triangle(period, n_samples):
    """
    Generate a triangle wave.
    """
    return np.array([1 - 4 * abs((i % period) - period / 2) / period for i in range(n_samples)],dtype=np.float32)


def make_sine(period, n_samples):
    """
    Generate a sine wave.
    """
    return np.array([np.sin(2 * np.pi * i / period) for i in range(n_samples)],dtype=np.float32)


def make_complex(period_f, n_samples, n_harmonics):
    """
    Generate a complex wave.
    """
    overtone_coeffs = np.random.rand(n_harmonics) * 1/ np.arange(1, n_harmonics + 1)
    freq = 2 * np.pi / period_f
    t = np.arange(n_samples,dtype=np.float32)
    wave = np.zeros(n_samples,dtype=np.float32)
    for i in range(n_harmonics):
        overtone = np.sin(freq * (i + 1) * t)
        overtone *= overtone_coeffs[i]
        if i == 0:
            wave = overtone
        else:
            wave += overtone
    return wave

def test_signals():
    """
    Test the signal generation functions.
    """
    import matplotlib.pyplot as plt
    n_samples = 1000
    period = 50
    for i, wave in enumerate([make_square, make_sawtooth, make_triangle, make_sine]):
        plt.subplot(2, 3, i + 1)
        plt.plot(wave(period, n_samples))
    

    n_harmonics = 1000
    period_f = 50
    plt.subplot(2, 3, 5)
    plt.plot(make_complex(period_f, n_samples, n_harmonics))
    plt.show()  


if __name__ == "__main__":
    test_signals()