"""
This class implement a Transformer based deep
learning model who detect R waves in order to
segmente ECG's signals beats by beats.

"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import math
import numpy as np
from tqdm import tqdm
from Utils import *

class RDetector(nn.Module):

    def __init__(self, name='Weights_Bittium_I', device='cuda:0'):

        super(RDetector, self).__init__()

        # ======================================= #
        # Hyper parameters
        # ======================================= #

        # The size of word embedding: have to be a multiple of the nb of heads
        self.name = name
        self.embed_size = 20
        self.kernel_size = 50
        self.nb_heads = 5
        self.forward_expansion = 6
        self.forward_expansion_decoder = 2
        self.max_len = 350     # Max len of a sequence (ok for a beat)
        self.max_total_len = 70000       # Max sequence len for annotations
        self.dropout_val = 0.3
        self.normalize = False       # Normalize signal in input
        self.device = device

        # Peaks detection treshold: proportion of max value
        self.peaks_treshold = 0.1

        # Path to save weights
        self.weights_path = 'Model/Weights_{}.pt'.format(self.name)
        # Path to save loss tracking
        self.loss_track_path = 'Model/loss_track_{}.csv'.format(self.name)

        # ======================================= #
        # Input adaptation
        # ======================================= #

        # Convolution for signal embedding
        self.embed_conv = nn.Conv1d(in_channels=1,
                                    out_channels=self.embed_size,
                                    kernel_size=self.kernel_size,
                                    stride=1,
                                    padding=25,
                                    bias=False)
        # Instance normalization
        self.istance_norm = nn.InstanceNorm2d(num_features=self.embed_size, affine=True)
        # The positional encoding:
        self.pos_encoding = PositionalEncoding(self.embed_size, self.max_total_len)

        # Dropout on embedding and positional encoding
        self.dropout = nn.Dropout(self.dropout_val)

        # ======================================= #
        # The Encoder part:
        # We are using 3 transformers blocks
        # ======================================= #

        self.bk1 = TransformerBlock(self.embed_size, self.nb_heads,
                                    self.forward_expansion,
                                    dropout=self.dropout_val)
        self.bk2 = TransformerBlock(self.embed_size, self.nb_heads,
                                    self.forward_expansion,
                                    dropout=self.dropout_val)
        self.bk3 = TransformerBlock(self.embed_size, self.nb_heads,
                                    self.forward_expansion,
                                    dropout=self.dropout_val)
        self.bk4 = TransformerBlock(self.embed_size, self.nb_heads,
                                    self.forward_expansion,
                                    dropout=self.dropout_val)

        # ======================================= #
        # The Decoder part:
        # Simply use fc layers
        # ======================================= #

        self.dec_fc1 = nn.Linear(self.embed_size, self.embed_size * self.forward_expansion_decoder)
        self.re1 = nn.ReLU()

        self.dec_fc2 = nn.Linear(self.embed_size * self.forward_expansion_decoder, self.embed_size * self.forward_expansion_decoder)
        self.re2 = nn.ReLU()

        self.dec_fc3 = nn.Linear(self.embed_size * self.forward_expansion_decoder, self.embed_size)
        self.re3 = nn.ReLU()

        self.dec_fc4 = nn.Linear(self.embed_size, 1)

        # Softmax for the end
        self.sm = nn.Softmax(dim=2)




    def forward(self, x):

        # Normalization
        if self.normalize:
            x = nn.functional.normalize(x, dim=1)

        beat_size = x.size(1)
        # Adapt the shape for conv1D
        x = torch.reshape(x, (x.size(0), 1, x.size(1)))
        # Build embedding form of signals using conv layer
        out = self.embed_conv(x)
        out = out[:, :, 0:beat_size]

        # Add the positional encoding
        out = self.pos_encoding(out)

        # Transpose last two dim to have embedding on the last dim
        out.transpose_(1, 2)

        # First encoders
        out = self.bk1(out, out, out, mask=None)
        out = self.bk2(out, out, out, mask=None)
        out = self.bk3(out, out, out, mask=None)
        out = self.bk4(out, out, out, mask=None)

        # Decoder part
        out = self.re1(self.dec_fc1(out))
        out = self.re2(self.dec_fc2(out))
        out = self.re3(self.dec_fc3(out))
        out = torch.sigmoid(self.dec_fc4(out))

        return out

    def save(self, epoch=None):

        if epoch is not None:
            torch.save(self.state_dict(), 'Model/Weights_{}_epoch_{}.pt'.format(self.name, epoch))
        else:
            torch.save(self.state_dict(), self.weights_path)

    def restore(self, epoch=0):

        if epoch != 0:
            print('restore at epoch {}'.format(epoch))
            self.load_state_dict(torch.load('{}_epoch_{}.pt'.format(self.weights_path.replace('.pt', ''), epoch)))
        else:
            self.load_state_dict(torch.load(self.weights_path))


    def annot_signal(self, signal, device='cuda:0'):

        self.eval()

        # Max size of signal to perform at the same time
        reading_max_len = 5000
        # Store result signal
        results = []

        # Read all the signal step by step
        idx = 0
        total_size = 0
        # Use a progress bar:
        pbar = tqdm(total=len(signal))
        while idx < len(signal):
            resize=False
            end = idx + reading_max_len
            if end > len(signal):
                end = len(signal)
                resize=True
            # Update the progress bar
            pbar.update(end - idx)
            # Get the tensor
            if self.normalize:
                tmp_tensor = nn.functional.normalize(torch.Tensor(signal[idx:end]).view(1, end - idx).to(device), dim=1)
            else:
                tmp_tensor = torch.Tensor(signal[idx:end]).view(1, end - idx).to(device)

            if resize:
                tmp = torch.zeros(1, reading_max_len).to(device)
                tmp[0, 0:end-idx] = tmp_tensor
                tmp_tensor = tmp

            # Make a prediction
            self.eval()
            with torch.no_grad():
                tmp_prd = self(tmp_tensor)
            # Ad to results
            results.append(tmp_prd[0, :, 0].cpu().detach().numpy())
            total_size += len(results[-1])
            idx += reading_max_len

        # Store in one array
        opt = np.zeros((total_size,))
        idx = 0
        for itm in results:
            end = idx + len(itm)
            opt[idx:end] = itm
            idx = end

        return opt

    def annot_peaks(self, signals, device=None, pre_trait=True):
        """
        R peaks annotation
        :param signals: 2D array: first dim = signal index
        :return: R confidence signal, r wave index's
        """
        if device is None:
            device = self.device
        # First get annotation signal values
        annot_sign = self.annot_multi_signal(signals, device=device, pre_trait=pre_trait)

        # Find peaks
        peaks, _ = find_peaks(annot_sign, distance=75)
        np.diff(peaks)

        # Store R index annotation
        r = []
        # Delete value under a certain treshold
        for itm in peaks:
            if annot_sign[itm] > self.peaks_treshold:
                r.append(itm)
        return annot_sign, r

    def annot_multi_signal(self, signals, device='cuda:0', pre_trait=True):

        # If pre traitement of the signa: apply filters
        if pre_trait:
            print('Filtering...')
            signals = filtering(signals)
            print('... Done')
        # Store predictiosn for each signals
        preds = np.zeros(signals.shape)
        for i in range(signals.shape[0]):
            # Predict for the signal
            preds[i, :] = self.annot_signal(signals[i, :], device=device)[0:preds.shape[1]]

        # Do the max prediction:
        sm = np.max(preds, axis=0)

        return sm

class SelfAttention(nn.Module):
    """
    The self attetion module
    """
    def __init__(self, embed_size=50, nb_heads=5):
        """
        :param embed_size: This size is the kernel size of the enbedding
        convolutional layer.
        :param nb_heads: The number of heads in the self attention process
        """
        super(SelfAttention, self).__init__()

        self.embed_size = embed_size
        # WARNING: the embed_size have to be a multiple of the number of heads
        self.nb_heads = nb_heads
        self.heads_dim = int(embed_size / nb_heads)

        # Layer to generate the values matrix
        self.fc_values = nn.Linear(self.heads_dim, self.heads_dim, bias=False)
        # Layer to generate keys
        self.fc_keys = nn.Linear(self.heads_dim, self.heads_dim, bias=False)
        # Layer for Queries
        self.fc_queries = nn.Linear(self.heads_dim, self.heads_dim, bias=False)

        # A fully connected layer to concatenate results
        self.fc_concat = nn.Linear(self.nb_heads * self.heads_dim, embed_size)

        # The softmax step
        self.sm = nn.Softmax(dim=3)


    def forward(self, values, keys, query, mask=None):

        # Get the number of training samples
        n = query.shape[0]
        # Get original shapes
        v_len = values.size()[1]
        k_len = keys.size()[1]
        q_len = keys.size()[1]

        # Split embedded inputs into the number of heads
        v = values.view(n, v_len, self.nb_heads, self.heads_dim)
        k = keys.view(n, k_len, self.nb_heads, self.heads_dim)
        q = query.view(n, q_len, self.nb_heads, self.heads_dim)

        # Feed it in appropriate layer
        v = self.fc_values(v)
        k = self.fc_keys(k)
        q = self.fc_queries(q)

        # Matrix dot product between qureries and keys
        prdc = torch.einsum('nqhd,nkhd->nhqk', [q, k])

        # Apply mask if present
        if mask is not None:
            prdc = prdc.masked_fill(mask == 0, float('-1e20'))  # don't use zero

        # The softmax step
        #attention = self.sm(prdc / (self.embed_size ** (1/2)))
        attention = torch.softmax(prdc / (self.embed_size ** (1 / 2)), dim=3)

        # Product with values
        # Output shape: (n, query len, heads, head_dim
        out = torch.einsum('nhql,nlhd->nqhd', [attention, v])

        # Concatenate heads results (n x query len x embed_size)
        out = torch.reshape(out, (n, q_len, self.nb_heads * self.heads_dim))

        # Feed the last layer
        return self.fc_concat(out)


class TransformerBlock(nn.Module):

    def __init__(self, embed_size=50, nb_heads=5, forward_expansion=4, dropout=0.1):
        super(TransformerBlock, self).__init__()

        # The self attention element
        self.attention = SelfAttention(embed_size, nb_heads)

        # The first normalization after attention block
        self.norm_A = nn.LayerNorm(embed_size)

        # The feed forward part
        self.feed = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        # The second normalization
        self.norm_B = nn.LayerNorm(embed_size)

        # A batch normalization instead of the classical dropout
        #self.bn = nn.BatchNorm1d(embed_size)

        # Or a classical dropout to avoid overfitting
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, v, k, q, mask=None):

        # The attention process
        out = self.attention(v, k, q, mask)

        # The first normalization after attention + skip connection
        out = self.norm_A(out + q)

        # A batch normalization instead of the classical dropout
        #out = self.bn(out)

        # The feed forward part
        fw = self.feed(out)

        # The second normalization + skip connection
        out = self.norm_B(fw + out)

        # An other batch normalization
        #out = self.bn(out)

        # Dropout
        out = self.dropout(out)

        return out




class PositionalEncoding(nn.Module):

    def __init__(self, embed_size=50, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.embed_size = embed_size
        self.max_len = max_len

        # Store a matrix with all possible positions
        pe = torch.zeros(embed_size, max_len)
        for pos in range(0, max_len):
            for i in range(0, embed_size, 2):
                pe[i, pos] = math.sin(pos / (10000 ** ((2 * i) / embed_size)))
                pe[i+1, pos] = math.cos(pos / (10000 ** ((2 * (i+1)) / embed_size)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):

        # Get seq size
        seq_len = x.size(2)

        # If size is greater that pos embedding saved in memory:
        if seq_len > self.max_len:
            self.adapt_len(seq_len)
        # Add positional embedding
        x = x[:, 0:self.embed_size, 0:seq_len] + self.pe[0, :, :seq_len].to('cuda:0')
        return x

    def adapt_len(self, new_len):
        """
        Normally not use any more
        """
        self.max_len = new_len

        # Store a matrix with all possible positions
        pe = torch.zeros(self.embed_size, self.max_len)
        for pos in range(0, self.max_len):
            for i in range(0, self.embed_size, 2):
                pe[i, pos] = math.sin(pos / (10000 ** ((2 * i) / self.embed_size)))
                pe[i+1, pos] = math.cos(pos / (10000 ** ((2 * (i+1)) / self.embed_size)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)












