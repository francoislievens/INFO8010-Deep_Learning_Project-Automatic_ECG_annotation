"""
This class implement a Transformer based deep
learning model who detect R waves in order to
segmente ECG's signals beats by beats.

"""
import torch
import torch.nn as nn
import math
import numpy as np
from tqdm import tqdm

class BeatClassifier(nn.Module):

    def __init__(self, name='A', device='cuda:0'):

        super(BeatClassifier, self).__init__()

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
        self.dropout_val = 0.1
        #self.nb_classes = 23
        self.nb_classes = 9
        self.init_weights = True
        self.device = device

        # Class dict
        self.classes = ['N', 'A', 'V', '/', 'f', 'F', 'L', 'R', '!']
        self.classes_dict = {'N': 0,
                             'A': 1,
                             'V': 2,
                             '/': 3,
                             'f': 4,
                             'F': 5,
                             'L': 6,
                             'R': 7,
                             '!': 8}

        # Peaks detection treshold: proportion of max value
        self.peaks_treshold = 0.0

        # Path to save weights
        self.weights_path = 'Model/Weights_{}.pt'.format(self.name)
        # Path to save loss tracking
        self.loss_track_path = 'Model/loss_track_{}.csv'.format(self.name)

        # ======================================= #
        # Input adaptation
        # ======================================= #

        # Convolution for word embedding
        self.embed_conv = nn.Conv1d(in_channels=1,
                                    out_channels=self.embed_size,
                                    kernel_size=self.kernel_size,
                                    stride=1,
                                    padding=25,
                                    bias=False)
        # The positional encoding:
        self.pos_encoding = PositionalEncoding(self.embed_size, self.max_len)

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

        self.dec_fc1 = nn.Linear(self.embed_size * self.max_len, self.embed_size * self.forward_expansion_decoder)
        self.re1 = nn.ReLU()

        self.dec_fc2 = nn.Linear(self.embed_size * self.forward_expansion_decoder, self.embed_size * self.forward_expansion_decoder)
        self.re2 = nn.ReLU()

        self.dec_fc3 = nn.Linear(self.embed_size * self.forward_expansion_decoder, self.embed_size)
        self.re3 = nn.ReLU()

        self.dec_fc4 = nn.Linear(self.embed_size, self.nb_classes)
        self.re4 = nn.Sigmoid()

        # Softmax for the end
        self.sm = nn.Softmax(dim=1)

        # =================================== #
        #   Init weights
        # =================================== #

        if self.init_weights:
            self.initialize_weights()

    def forward(self, x):

        beat_size = x.size(1)
        n = x.size(0)
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

        # Average outputs of encoder
        out = out.view(n, self.embed_size * self.max_len)

        # Decoder part
        out = self.re1(self.dec_fc1(out))
        out = self.re2(self.dec_fc2(out))
        out = self.re3(self.dec_fc3(out))
        out = self.re4(self.dec_fc4(out))

        # Reshape to match original
        #out = self.sm(out)

        return out

    def save(self, epoch=None):

        if epoch is not None:
            torch.save(self.state_dict(), 'Model/Weights_{}_epoch_{}.pt'.format(self.name, epoch))
        torch.save(self.state_dict(), self.weights_path)

    def restore(self, epoch=None):

        if epoch is not None:
            self.load_state_dict(torch.load('Model/Weights_{}_epoch_{}.pt'.format(self.name, epoch)))
            print('Successfully restore model {} at epoch {}'.format(self.name, epoch))
        else:
            self.load_state_dict(torch.load(self.weights_path))

    def initialize_weights(self):
        """
        Initialize all the weights in the nn.Modules
        :return:
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_uniform_(module.weight)

                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def annot(self, signal, beat_idx, batch_size=200):
        """
        This method annot each bet to an Arhytmia class
        :param signal: a matrix with the number of leads on the first dimension
        :param beat_idx: A list of beats index
        :return: a list of symbol associated to the corresponding class
        """
        #avg_prediction = torch.zeros((len(beat_idx), self.nb_classes)).to(self.device)
        avg_prediction = None
        drop_first = False
        drop_last = False
        tot_pred = []
        for l in range(signal.shape[0]):
            # Tensorize the lead
            beats_lst = []
            tmp_signal = torch.Tensor(signal[l, :])
            for idx in beat_idx:
                start_idx = idx - 175
                end_idx = idx + 175
                # Drop first and last beat if needed
                if start_idx < 0:
                    drop_first = True
                    continue
                if end_idx >= tmp_signal.size(0):
                    drop_last = True
                    continue
                else:
                    beats_lst.append(torch.Tensor(tmp_signal[start_idx:end_idx]))
            # Make predictions
            self.eval()
            tmp_prds = []
            with torch.no_grad():
                start = 0
                while start < len(beats_lst):
                    end = start + batch_size
                    if end > len(beats_lst):
                        end = len(beats_lst)
                    beat_tensor = torch.zeros((end-start, 350))
                    l = 0
                    for t in range(start, end):
                        beat_tensor[l, :] = beats_lst[t]
                        l += 1
                    #staker = torch.cat(beats_lst[start:end], dim=0).to(self.device)
                    prds = self.forward(beat_tensor.to(self.device))
                    tmp_prds.append(prds)
                    start = end
            prds = torch.cat(tmp_prds, dim=0)

            tot_pred.append(prds)

        sm_pred = torch.zeros(len(tot_pred[0]), 9).to(self.device)
        for j in range(len(tot_pred[0])):
            for k in range(0, 9):
                for l in range(0, len(tot_pred)):
                    sm_pred[j, k] += tot_pred[l][j, k]

        # Make a softmax on predictions
        sm_pred = self.sm(sm_pred)

        # Get max index
        max_class = torch.argmax(sm_pred, dim=1).detach().cpu().numpy()
        sm_pred = sm_pred.detach().cpu().numpy()

        # Build annotations symbols array
        annot = []
        if drop_first:
            annot.append('N')
        for i in range(max_class.shape[0]):
            annot.append(self.classes[max_class[i]])
        if drop_last:
            annot.append('N')

        return annot, sm_pred




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

    def __init__(self, embed_size=50, max_len=400):
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
        x = x[:, 0:self.embed_size, 0:seq_len] + self.pe[:, :seq_len].to('cuda:0')
        return x

    def adapt_len(self, new_len):

        self.max_len = new_len

        # Store a matrix with all possible positions
        pe = torch.zeros(self.embed_size, self.max_len)
        for pos in range(0, self.max_len):
            for i in range(0, self.embed_size, 2):
                pe[i, pos] = math.sin(pos / (10000 ** ((2 * i) / self.embed_size)))
                pe[i+1, pos] = math.cos(pos / (10000 ** ((2 * (i+1)) / self.embed_size)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)












