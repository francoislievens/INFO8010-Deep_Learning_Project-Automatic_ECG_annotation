"""
This code preapre a dataset from original bittium
recording from the CHU.
.edf files are first segmented in 30 min section.

"""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pyedflib
import math
import scipy.stats as stats
import torch
import random
import json
from Filtering import *

# Change user
ANDREAS = False

FRANCOIS = True

# Paths to use
if FRANCOIS:
    ORIGINAL_FILE_PATH = 'D:/ECG/ECG_DATABASE_A'
    SEGMENTED_FILE_PATH = 'D:/ECG/ECG_DATABASE_A/Segmented'
    SEG_BEAT_TENSOR_PATH = 'D:/ECG/ECG_DATABASE_A/Seg_beat_tensor'

elif ANDREAS:

    with open('C:/Users/dadou/Documents/Github/Deep-ECG-Project/andreas_param.json', 'r') as myfile:
        ANDREAS_json = myfile.read()
    ANDREAS_param = json.loads(ANDREAS_json)

    ORIGINAL_FILE_PATH = ANDREAS_param["DATASET_PATH"]
    SEGMENTED_FILE_PATH = ANDREAS_param["SEGMENTED_FILE"]
    SEG_BEAT_TENSOR_PATH = ANDREAS_param["SEGMENTED_BEAT_TENSOR_PATH"]



# Action to perform
SEGMENTE = True
BUILD_TENSORS = True

# Others parameters
SEG_DURATION = 30       # Duration of an edf segment in min
S_RATE = 250            # Sampling frequency
BEAT_SIZE = 350         # Size of a beat signal
PROP_NOISE = 0.3        # Proportion of noisy samples to keep




def segmenter(files, destination):
    """
    This method segmente original files in segment of
    the same length
    :param files: the list of file names. .edf and .csv have to be present in directory
    :param destination: the path of the destination directory
    """
    # Get already segmented files list
    already_done = os.listdir(destination)
    print(already_done)


    for i in range(len(files_list)):
        print('Segmentation: file {} / {}'.format(i, len(files_list)))
        file_name = files_list[i]

        # Check if the file is not already done
        already = False
        for itm in already_done:
            if file_name in itm:
                already = True
                break
        if already:
            continue

        # Load annotations
        annot = pd.read_csv('{}/{}.csv'.format(ORIGINAL_FILE_PATH, file_name), sep=';')
        # Check if extended annotations
        if annot.shape[1] < 20:
            print('Wrong annotations format')
            continue
        # Rename the first column
        annot = annot.rename(columns={annot.columns[0]: 'Start'})

        # Load the edf file
        signal, signal_headers, header = pyedflib.highlevel.read_edf('{}/{}.edf'.format(ORIGINAL_FILE_PATH, file_name))

        # Get the number of 60 min file (drop incompleted, 250hz files)
        nb_files = math.ceil(signal.shape[1] / (S_RATE * 60 * SEG_DURATION))
        print('signal shape: ', signal.shape)
        print('nb files: ', nb_files)
        # Change annot ms to seconds
        annot['Start'] = (annot['Start'] / 1000)
        annot['RR'] = (annot['RR'] / 1000)

        # Get the number of samples in sub files
        size = SEG_DURATION * 60 * S_RATE
        # Write all sub-files
        print('Writing files...')
        start_time_idx = 0
        end_time_idx = 0
        for j in tqdm(range(nb_files)):
            start_idx = j * size
            end_idx = (j + 1) * size
            start_time = j * SEG_DURATION * 60
            end_time = (j + 1) * SEG_DURATION * 60

            # Get time idx
            start_test = False
            end_test = False
            for l in range(start_time_idx, annot.shape[0]):

                if annot.iloc[l]['Start'] >= start_time and not start_test:
                    start_time_idx = l
                    start_test = True
                if annot.iloc[l]['Start'] >= end_time and not end_test:
                    end_time_idx = l
                    end_test = True
                    break

            # If no annotations
            if end_time_idx - start_time_idx < 1:
                continue
            # Get sub-annotation file
            sub_annot = annot.iloc[start_time_idx:end_time_idx].copy()
            # Adapt index:
            sub_annot['Start'] -= start_time

            # Get sub recording
            sub_signal = signal[:, start_idx:end_idx]
            # Save the signal
            pyedflib.highlevel.write_edf('{}/{}_{}.edf'.format(SEGMENTED_FILE_PATH,
                                                               file_name, j),
                                         sub_signal, signal_headers, header)
            # Save the dataframe
            sub_annot.to_csv('{}/{}_{}.csv'.format(SEGMENTED_FILE_PATH, file_name, j),
                             sep=';', header=True, index=False)

            # Change index
            start_time_idx = end_time_idx


def tensor_builder():
    """
    This method use 30 min segemented file to build a tensor
    for each signal that can be use by the dataset builder
    """
    # File already performed:
    a_done = os.listdir(SEG_BEAT_TENSOR_PATH)
    # Make the list of files
    tmp_file_lst = os.listdir(SEGMENTED_FILE_PATH)
    file_lst = []
    for itm in tmp_file_lst:
        if '.edf' in itm:
            tmp = itm.replace('.edf', '')
            # Check if already done
            done = False
            for fait in a_done:
                if tmp in fait:
                    done = True
                    break
            if not done:
                file_lst.append(tmp)

    # Get a normal distribution for peaks annotataion
    tmp_x = np.arange(0, 21)
    r_box = stats.norm.pdf(tmp_x, 10, math.sqrt(8))
    r_box *= (1 / np.max(r_box))

    print('Data encoding...')
    for i in range(0, len(file_lst)):
        print('Encoding file {} / {} name: {}'.format(i, len(file_lst), file_lst[i]))

        # Get the file name
        file_name = file_lst[i]

        # Load the file in memory
        signal, signal_headers, header = pyedflib.highlevel.read_edf('{}/{}.edf'.format(SEGMENTED_FILE_PATH,
                                                                                        file_name))
        # Load annotations
        annot = pd.read_csv('{}/{}.csv'.format(SEGMENTED_FILE_PATH, file_name), sep=';')

        # Change the scale to index
        annot[annot.columns[0]] *= S_RATE
        sym_lst = annot['Type'].tolist()
        idx_lst = annot[annot.columns[0]].tolist()

        # Build a signal for R wave annotation
        r_signal = np.zeros((1, signal.shape[1]))
        print('Annotation signal building...')
        for j in tqdm(range(0, len(idx_lst))):

            if idx_lst[j] < 10:
                idx_lst[j] = 10
            elif idx_lst[j] >= signal.shape[1] - 10:
                idx_lst[j] = signal.shape[1] - 10
            if idx_lst[j] + len(r_box) >= r_signal.shape[1]:
                break
            # Put the normal distribution at the good place
            r_signal[0, int(idx_lst[j]):(int(idx_lst[j])+len(r_box))] = r_box

        # Build a tensor of beats
        print('Write tensor pickles...')
        # Beat list for the threes signals
        beats_lst = []
        for j in range(0, signal.shape[0]):
            beats_lst.append([])
        start_idx = 0
        for j in range(0, len(idx_lst)-1):
            # Get annotations
            note = annot.iloc[j]
            # Get beat idx
            end_idx = int((idx_lst[j] + idx_lst[j+1]) / 2)
            beat_duration = end_idx - start_idx
            # Limit beat size:
            final_end_idx = end_idx
            if beat_duration > BEAT_SIZE:
                final_end_idx = start_idx + BEAT_SIZE
                beat_duration = BEAT_SIZE
            # Get R annotation signal
            tmp_r = torch.zeros((1, BEAT_SIZE))
            bt = torch.Tensor(r_signal[0, start_idx:final_end_idx])
            if bt.size(0) != beat_duration:
                start_idx = end_idx
                continue
            tmp_r[0, 0:beat_duration] = bt
            # For each dirivation
            for d in range(1, signal.shape[0]):
                # Check if interpretable
                if not math.isnan(note['{}#QRS'.format(d)]):
                    beat = torch.zeros((2, BEAT_SIZE))
                    beat[1, :] = tmp_r
                    beat[0, 0:beat_duration] = torch.Tensor(own_NRMAS(signal[d-1, start_idx:final_end_idx], 3))
                    beats_lst[d-1].append(beat)
                # Keep very noisy elements with no annotations
                rdn = random.uniform(0, 1)
                #if note['NOISE'] > 30:
                #    beat = torch.zeros((2, BEAT_SIZE))
                #    beat[0, 0:beat_duration] = torch.Tensor(own_NRMAS(signal[d-1, start_idx:final_end_idx], 3))
                #    beats_lst[d-1].append(beat)
            start_idx = end_idx
            # Save tensors
        for t in range(len(beats_lst)):
            torch.save(beats_lst[t], '{}/{}_{}.pt'.format(SEG_BEAT_TENSOR_PATH, file_name, t))




if __name__ == '__main__':

    # Get original file list
    or_files_tot = os.listdir(ORIGINAL_FILE_PATH)

    # Store names who have data and annotations
    files_list = []
    for itm in or_files_tot:
        if '.edf' in itm:
            name = itm.replace('.edf', '')
            # Check if annotation file present
            for i in range(len(or_files_tot)):
                if '{}.csv'.format(name) == or_files_tot[i]:
                    # Save the file name
                    files_list.append(name)

    if SEGMENTE:
        segmenter(files_list, destination=SEGMENTED_FILE_PATH)

    if BUILD_TENSORS:
        tensor_builder()



