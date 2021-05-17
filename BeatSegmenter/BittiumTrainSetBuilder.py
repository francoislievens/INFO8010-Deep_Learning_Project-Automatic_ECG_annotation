import torch
import numpy as np
import tqdm
from tqdm import tqdm
import os

BITTIUM_FILE_PATH = 'D:/ECG/ECG_DATABASE_A/Seg_beat_tensor'


class BittiumTrainSetBuilder(torch.utils.data.Dataset):

    def __init__(self,
                 name='Bittium_A',
                 memory_size=50,       # Number of recording initially in memory
                 upd_size=50,           # The number of recording to change at each epoch
                 train=True,            # To build a training set
                 train_prop=0.8,        # Proportion of data keeped for training purposes
                 ):

        super().__init__()

        self.name = name

        # Get the file list
        self.file_lst = os.listdir(BITTIUM_FILE_PATH)
        treshold = int(train_prop*len(self.file_lst))
        if not train:
            self.file_lst = self.file_lst[treshold:]
        else:
            self.file_lst = self.file_lst[:treshold]


        # Number of records to load
        self.memory_size = memory_size
        # Number of file to update each epcoh
        self.upd_size = upd_size

        # Store data
        self.data = []
        self.fill_memory()

        # index to use
        self.index = np.arange(len(self.data))
        # Shuffle it
        np.random.shuffle(self.index)



    def fill_memory(self):
        """
        Generate a random dataset of the selected memory size
        """
        print('Initial dataset loading...')
        # Get random files
        idx_lst = np.arange(len(self.file_lst))
        # Shuffle it
        np.random.shuffle(idx_lst)
        idx_lst = idx_lst[0:self.memory_size]

        # Load tensors
        tmp_data = []
        for idx in tqdm(idx_lst):
            name = self.file_lst[idx]
            # Load the file and concatenate
            new_data = torch.load('{}/{}'.format(BITTIUM_FILE_PATH, name))
            tmp_data = tmp_data + new_data

        self.data = tmp_data

    def update_memory(self):

        # First we load the wantend number of tensors
        idx_lst = np.arange(len(self.file_lst))
        np.random.shuffle(idx_lst)
        idx_lst = idx_lst[0:self.upd_size]
        tmp_tensors = []
        for i in range(self.upd_size):
            name = self.file_lst[idx_lst[i]]
            tmp_tensors = tmp_tensors + torch.load('{}/{}'.format(BITTIUM_FILE_PATH, name))

        new_size = len(tmp_tensors)
        # Delete the number of new  element at the begin (oldest)
        self.data = self.data[new_size:]
        self.data = self.data + tmp_tensors

        # Shuffle the index array
        np.random.shuffle(self.index)



    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        # Return input - target

        return self.data[self.index[index]][0], self.data[self.index[index]][1]





