from RDetector import RDetector
import Utils
import torch
import numpy as np
import pandas as pd
import os
import wfdb
from scipy.signal import find_peaks


MIT_PATH = 'D:/ECG/mit'
DEVICE = 'cuda:0'
MODEL_NAME = 'Bittium_I'

def prepare_data():

    # Get the MIT files list
    tmp_rec_idx_lst = pd.read_csv('{}/RECORDS'.format(MIT_PATH))['number'].to_numpy()

    # Remove already done
    tmp_r_done_lst = os.listdir('Data/Test_Data/Pred')
    r_done_lst = []
    for itm in tmp_r_done_lst:
        r_done_lst.append(int(itm.replace('.npy', '')))

    rec_idx_lst = []
    for itm in tmp_rec_idx_lst:
        already = False
        for i in range(len(r_done_lst)):
            if itm == r_done_lst[i]:
                already = True
                break
        if not already:
            rec_idx_lst.append(itm)

    # Restore a model
    model = RDetector(name=MODEL_NAME).to(DEVICE)
    model.restore(epoch=3)

    # For each files
    for i in range(len(rec_idx_lst)):
        print('File {} / {}'.format(i, len(rec_idx_lst)))
        # Read the signal
        signal = wfdb.rdrecord('{}/{}'.format(MIT_PATH, rec_idx_lst[i]))
        signal = signal.p_signal.T
        signal *= 1000
        signal = Utils.filtering(signal, freq_adapter=True)

        # Annot the signal
        annot_sign, r = model.annot_peaks(signal, device=DEVICE, pre_trait=False)

        # Save data
        np.save('Data/Test_Data/Signal/{}.npy'.format(rec_idx_lst[i]), signal)
        np.save('Data/Test_Data/Pred/{}.npy'.format(rec_idx_lst[i]), annot_sign)

def score_data_treshold(signal, annot_signal, target_idx, threshold_lst, distance=75):

    # Get peaks index
    peaks, _ = find_peaks(annot_signal, distance=distance)
    np.diff(peaks)

    # Accept as true if distance from the target lower than
    acceptation_th = 50

    total_peaks = []

    for th in threshold_lst:

        tmp_array = []
        for itm in peaks:
            if annot_signal[itm] > th:
                tmp_array.append(itm)
        total_peaks.append(tmp_array)

    tot_true_pos = []
    tot_false_pos = []
    tot_false_neg = []
    tot_target = []
    tot_annot = []

    # Computing fake positive/negative and true positive/negative
    for a in range(len(total_peaks)):
        print('Iter {} / {}'.format(a, len(total_peaks)))
        arr = total_peaks[a]

        # Store distance between target and closest prediction
        target_dist = np.ones(len(target_idx)) * (1000)
        target_assos_idx = np.ones(len(target_idx)) * -1
        predict_asso_idx = np.ones(len(arr)) * -1
        for i in range(len(target_dist)):
            for j in range(len(arr)):
                if abs(arr[j] - target_idx[i]) < target_dist[i] and abs(arr[j] - target_idx[i]) < acceptation_th:
                    target_dist[i] = abs(arr[j] - target_idx[i])
                    target_assos_idx[i] = j
                    predict_asso_idx[j] = i

        true_pos = 0
        false_pos = 0
        false_neg = 0
        for i in range(len(target_assos_idx)):
            if target_assos_idx[i] == -1:
                false_neg += 1
            else:
                true_pos += 1
        # Get false pos
        for i in range(len(predict_asso_idx)):
            if predict_asso_idx[i] == -1:
                false_pos += 1

        total_annot = len(arr)
        total_target = len(target_idx)

        tot_true_pos.append(true_pos)
        tot_false_pos.append(false_pos)
        tot_false_neg.append(false_neg)
        tot_annot.append(total_annot)
        tot_target.append(total_target)

    return tot_true_pos, tot_false_pos, tot_false_neg, tot_annot, tot_target





if __name__ == '__main__':

    prepare_data()