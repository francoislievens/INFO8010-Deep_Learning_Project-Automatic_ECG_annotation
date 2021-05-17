import torch
import numpy as np
import pandas as pd
import tqdm
import wfdb
from Utils import filtering
from tqdm import tqdm
import pickle
import json
import random

ANDREAS = False
FRANCOIS = True

if FRANCOIS:
    MIT_PATH = 'D:/ECG/mit'

elif ANDREAS:
    with open('C:/Users/dadou/Documents/Github/Deep-ECG-Project/andreas_param.json', 'r') as myfile:
        ANDREAS_json = myfile.read()
    ANDREAS_param = json.loads(ANDREAS_json)
    MIT_PATH = ANDREAS_param['DATASET_PATH']

BLACK_LIST = ['|', 'Q', 'x', 'j', 'a', 'J', '[', ']', 'E', 'S', 'e', '+', '~', '\"']

class MIT_Dataset(torch.utils.data.Dataset):

    def __init__(self, device='cpu',
                 train=True,
                 train_prop=0.8,  # Proportion to take as test set
                 seed=1,  # Seed to shuffle and split the dataset
                 copy_from=None,  # Copy from another dataset
                 debog=False,
                 ):
        super(MIT_Dataset, self).__init__()

        # Dictionary who contain each class symbol and occurences
        self.classes = {}

        self.device = device
        self.debog = debog
        self.train_prop = train_prop
        self.train = train

        # Store beats data
        self.beats = []
        self.target = []

        # Restore data
        if copy_from is not None:
            self.dataset_copier(copy_from)
        else:
            try:
                print('Serialized dataset loading...')
                self.restore()
                print('... Done')
            # If not already serialzed:
            except:
                print('No serialized dataset found, process data...')
                self.read_raw_data()
                self.tensorize()
                self.serialize()
                print('... Done.')

        # Build index tensor
        idx_arr = np.arange(0, len(self.beats))
        np.random.seed(seed)
        np.random.shuffle(idx_arr)
        # Keep train or test elements
        treshold = int(train_prop * len(self.beats))

        self.index = None
        if train:
            self.index = idx_arr[0:treshold]
        else:
            self.index = idx_arr[treshold:]

    def read_raw_data(self, segmentation_type="center_r"):
        print('\033[93m(Creating new dataset)\033[0m' + f" Processing raw data using: {segmentation_type}")

        # Recover all files numbers
        rec_idx_lst = pd.read_csv(f"{MIT_PATH}/RECORDS")['number'].to_numpy()

        # Read all files
        beat_idx = 0
        class_idx = 0
        end = len(rec_idx_lst)

        # If debog we use only 2 files
        if self.debog:
            end = 2

        for i in range(0, end):
            print(f"Loading file {i} / {len(rec_idx_lst)} - {rec_idx_lst[i]}")
            tmp_rec_w = wfdb.rdrecord(f"{MIT_PATH}/{rec_idx_lst[i]}")
            tmp_annot_w = wfdb.rdann(f"{MIT_PATH}/{rec_idx_lst[i]}", 'atr')

            # Get signals, annotations idx and symbols
            signal = tmp_rec_w.p_signal.T
            signal = signal * 1000
            annot_idx = tmp_annot_w.sample
            # Convert annot idx to new sample rate
            annot_idx = annot_idx / 360
            annot_idx = annot_idx * 250
            annot_idx = annot_idx.astype(int)
            annot_symbol = tmp_annot_w.symbol

            # Adapt the signal
            signal = filtering(signal, freq_adapter=True)

            if segmentation_type == "r_to_r":
                with tqdm(total=len(annot_idx), position=0, leave=True) as pbar:
                    for j in range(len(annot_idx)):

                        try:
                            start_idx = annot_idx[j]
                            end_idx = annot_idx[j + 1]
                        except:
                            print(f"Segmented for {j}")
                            break

                        tmp_symbol = annot_symbol[j]

                        # For each derivations
                        for d in range(signal.shape[0]):
                            tmp_beat = torch.zeros(350).to(self.device)
                            tmp = torch.Tensor(signal[d, start_idx:end_idx]).to(self.device)
                            if tmp.size(0) <= tmp_beat.size(0):
                                tmp_beat[0:tmp.size(0)] = tmp
                            else:
                                tmp_beat = tmp[0:350]
                            self.beats.append(tmp_beat)
                            try:
                                self.classes[tmp_symbol][0].append(beat_idx)
                            # If the class does not already exist
                            except:
                                self.classes[tmp_symbol] = [[beat_idx], [class_idx]]
                                class_idx += 1
                            # Add class index
                            cls_idx = self.classes[tmp_symbol][1][0]
                            self.target.append(cls_idx)
                            beat_idx += 1

                        start_idx = end_idx
                        pbar.update(1)

            if segmentation_type == "center_r":
                with tqdm(total=len(annot_idx), position=0, leave=True) as pbar:
                    for j in range(len(annot_idx)):

                        start_idx = annot_idx[j] - 175
                        end_idx = annot_idx[j] + 175
                        if start_idx < 0:
                            start_idx = 0
                        if end_idx > signal.shape[1]:
                            end_idx = signal.shape[1]

                        tmp_symbol = annot_symbol[j]

                        # For each derivations
                        for d in range(signal.shape[0]):
                            tmp_beat = torch.zeros(350).to(self.device)
                            tmp = torch.Tensor(signal[d, start_idx:end_idx]).to(self.device)
                            tmp_beat[350 - tmp.size(0):350] = tmp
                            try:
                                self.classes[tmp_symbol][0].append(beat_idx)
                            # If the class does not already exist
                            except:
                                # Check if not blacklisted:
                                blaked = False
                                for itm in BLACK_LIST:
                                    if itm in tmp_symbol:
                                        blaked = True
                                        break
                                if blaked:
                                    break
                                else:
                                    # Add the new class to the dict
                                    self.classes[tmp_symbol] = [[beat_idx], [class_idx]]
                                    class_idx += 1

                            # Add class index
                            cls_idx = self.classes[tmp_symbol][1][0]
                            self.target.append(cls_idx)
                            # And the beat signal
                            self.beats.append(tmp_beat)
                            beat_idx += 1

                        start_idx = end_idx
                        pbar.update(1)

    def tensorize(self):

        beat_tensor = torch.zeros((len(self.beats), 350), device=self.device)
        i = 0
        print('Tensor transformation...')
        with tqdm(total=len(self.beats), position=0, leave=True) as pbar:
            for itm in self.beats:

                end_idx = len(itm)
                if end_idx > 350:
                    end_idx = 350
                beat_tensor[i, 0:end_idx] = itm[0:end_idx]
                pbar.update(1)
                i += 1

    def serialize(self):

        deb = ''
        if self.debog:
            deb = '_debog'
        torch.save(self.beats, 'Data/inputs_data{}.pt'.format(deb))
        torch.save(self.target, 'Data/target_data{}.pt'.format(deb))
        with open('Data/classes_dict{}.pickle'.format(deb), 'wb') as file:
            pickle.dump(self.classes, file, protocol=pickle.HIGHEST_PROTOCOL)

    def restore(self):

        deb = ''
        if self.debog:
            deb = '_debog'
        self.beats = torch.load('Data/inputs_data{}.pt'.format(deb))
        self.target = torch.load('Data/target_data{}.pt'.format(deb))
        with open('Data/classes_dict.pickle{}'.format(deb), 'rb') as file:
            self.classes = pickle.load(file)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):

        return self.beats[self.index[index]], self.target[self.index[index]], self.index[index]

    def dataset_copier(self, dataset):

        self.beats = dataset.beats
        self.target = dataset.target
        self.classes = dataset.classes

    def balance_data(self):

        # for i in range(24):
        #     print(f" nb of class {i}: {self.target.count(i)}")

        index_ones = []
        for i in range(len(self.beats)):
            if self.target[i] == 1:
                index_ones.append(i)

        random.seed(231)
        to_delete = []
        for i in range(len(index_ones) - 25000):
            to_delete.append(random.choice(index_ones))

        # Reverse and sort to start deleting by the end sinon ça fait foirer les indices (il est tard)
        to_delete = sorted(to_delete, reverse=True)

        index = self.index.tolist()
        for i in tqdm(range(len(to_delete) - 1)):
            try:
                del self.target[to_delete[i]]
                del self.beats[to_delete[i]]
            except:
                print(f"Len self.target: {len(self.target)}")
                print(f"Len self.beats: {len(self.beats)}")
                print(f"to_delete[i]: {to_delete[i]}")
                raise Exception("y'a un truc qui cloche")
            try:
                index.remove(to_delete[i])
            except:
                continue

        self.index = np.array(index)


if __name__ == '__main__':
    test = MIT_Dataset(debog=False)

    test_2 = MIT_Dataset(debog=False, copy_from=test, train=False)
    print(len(test))
    print(len(test_2))