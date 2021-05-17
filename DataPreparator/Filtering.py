import math
import numpy as np
import scipy.stats as stats
from scipy import signal as scising
from tqdm import tqdm


def filtering(signals):

    new_signal = np.zeros(signals.shape)

    for i in range(0, signals.shape[1]):
        tmp = own_NRMAS(signals[:, i], 3)
        tmp = own_NRMAS(tmp, 7)
        new_signal[:, i] = tmp

    return new_signal

def own_NRMAS_index(vector, window, index):

    smoothed_value = 0
    max_size = (window - 1) / 2
    smoothing_window = np.arange(-max_size, max_size + 1, 1)

    vector_window = []
    for j in range(window):

        if (0 <= (index + smoothing_window[j]) < len(vector)):
            vector_window.append(vector[int(index + smoothing_window[j])])

    return (vector_window[0] + vector_window[-1]) / 2

def own_NRMAS(vector, window):

    smoothed_vector = np.zeros(len(vector))

    if (window % 2) == 0:
        print("Error window size even")
        return

    for i in range(len(vector)):
        smoothed_vector[i] = own_NRMAS_index(vector, window, i)

    return smoothed_vector

def freq_normalizer(signal, original_freq=360, new_freq=250):
    """
    Use fourrier to adapt the frequency to 360 hz
    """
    # Get parameters
    new_len = math.floor((signal.shape[1] / original_freq) * new_freq)
    # A new matrix to store
    #new_signal = np.zeros((signal.shape[0], new_len))
    #print('signal resampling...')
    new_signal = scising.resample(signal, new_len, axis=1)
    #for i in tqdm(range(0, signal.shape[0])):
    #    new_signal[i, :] = scising.resample(signal[i, :], new_len)

    return new_signal