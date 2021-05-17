import math
import numpy as np
import scipy.stats as stats
from scipy import signal as scising
from tqdm import tqdm
from numba import njit, jit
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, f1_score, classification_report
import matplotlib.pyplot as plt


def filtering(signals, freq_adapter=True):

    new_signal = np.zeros(signals.shape)

    for i in range(0, signals.shape[0]):
        tmp = own_NRMAS(signals[i, :], 3)
        tmp = own_NRMAS(tmp, 7)
        new_signal[i, :] = tmp

    # Adapt frequency
    if freq_adapter:
        sng_lst = []
        for i in range(new_signal.shape[0]):
            tmp = freq_normalizer(new_signal[i, :])
            sng_lst.append(tmp)
        new_signal = np.zeros((len(sng_lst), len(sng_lst[0])))
        for i in range(len(sng_lst)):
            new_signal[i, :] = sng_lst[i]

    return new_signal

@njit
def own_NRMAS_index(vector, window, index):

    smoothed_value = 0
    max_size = (window - 1) / 2
    smoothing_window = np.arange(-max_size, max_size + 1, 1)

    vector_window = []
    for j in range(window):

        if (0 <= (index + smoothing_window[j]) < len(vector)):
            vector_window.append(vector[int(index + smoothing_window[j])])

    return (vector_window[0] + vector_window[-1]) / 2

@njit
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
    new_len = math.floor((signal.shape[0] / original_freq) * new_freq)
    # A new matrix to store
    #new_signal = np.zeros((signal.shape[0], new_len))
    #print('signal resampling...')
    new_signal = scising.resample(signal, new_len, axis=0)
    #for i in tqdm(range(0, signal.shape[0])):
    #    new_signal[i, :] = scising.resample(signal[i, :], new_len)

    return new_signal

def RocCurves(target, prediction, classes, name, epoch=0):

    cmap = plt.cm.Blues
    # Compute for each target classes:
    for symbol in classes.keys():
        # Compute the rec curve
        class_idx = classes[symbol][1][0]
        y = np.zeros(target.shape[0])
        for i in range(target.shape[0]):
            if target[i] == class_idx:
                y[i] = 1
        print(y)
        prds = prediction[:, class_idx].flatten()
        fpr, tpr, thresholds = roc_curve(y, prds)

        # Compute teh AUC
        #auc = roc_auc_score(y, prediction[:, class_idx])

        # Plot it
        plt.plot(fpr, tpr, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        try:
            plt.savefig('Plot/RocCurves/{}-label-{}-epoch{}.png'.format(name, symbol, epoch))
        except:
            plt.savefig('Plot/RocCurves/{}-label-{}-epoch{}.png'.format(name, 's{}'.format(class_idx), epoch))
        plt.close()


