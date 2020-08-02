import numpy as np
import json, sys, os
from torch import nn
import torch
from torch.distributions import kl_divergence, Normal
from torch.optim.lr_scheduler import ExponentialLR

def load_dataset_path(fn='model_config.json'):
    with open(fn) as f:
        paths = json.load(f)['dataset_path']

    train_val_path = paths['hpc_data_path']
    return train_val_path

def load_params_dict(key, fn='model_config.json'):
    with open(fn) as f:
        dict = json.load(f)[key]
    return dict


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

def standard_normal(shape):
    N = Normal(torch.zeros(shape), torch.ones(shape))
    if torch.cuda.is_available():
        N.loc = N.loc.cuda()
        N.scale = N.scale.cuda()
    return N


def loss_function(recon_pitch, pitch, recon_dur, dur,
                  dist, pitch_criterion, dur_criterion, normal,
                  weights=(1, .5, .1)):
    # bs = dist.mean.size(0)
    pitch_loss = pitch_criterion(recon_pitch, pitch)
    recon_dur = recon_dur.view(-1, 5, 2)
    dur = dur.view(-1, 5)
    dur0 = dur_criterion(recon_dur[:, 0, :], dur[:, 0])
    dur1 = dur_criterion(recon_dur[:, 1, :], dur[:, 1])
    dur2 = dur_criterion(recon_dur[:, 2, :], dur[:, 2])
    dur3 = dur_criterion(recon_dur[:, 3, :], dur[:, 3])
    dur4 = dur_criterion(recon_dur[:, 4, :], dur[:, 4])

    # dur_loss = dur_criterion(recon_dur, dur)  # (bs * 32 * 15 * 5)
    # dur_loss = dur_loss.view(-1, 5)  # some hard code here
    w = torch.tensor([1, 0.6, 0.4, 0.3, 0.3],
                      dtype=float,
                      device=recon_dur.device)
    # dur_loss = (dur_loss * w).mean()
    dur_loss = w[0] * dur0 + w[1] * dur1 + w[2] * dur2 + w[3] * dur3 + \
               w[4] * dur4
    kl_div = kl_divergence(dist, normal).mean()
    loss = weights[0] * pitch_loss + weights[1] * dur_loss + \
        weights[2] * kl_div
    return loss, pitch_loss, dur_loss, kl_div


# Useful function for how long epochs take
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


class MinExponentialLR(ExponentialLR):
    def __init__(self, optimizer, gamma, minimum, last_epoch=-1):
        self.min = minimum
        super(MinExponentialLR, self).__init__(optimizer, gamma, last_epoch=-1)

    def get_lr(self):
        return [
            max(base_lr * self.gamma ** self.last_epoch, self.min)
            for base_lr in self.base_lrs
        ]


def scheduled_sampling(i, high=0.7, low=0.05):
    x = 10 * (i - 0.5)
    z = 1 / (1 + np.exp(x))
    y = (high - low) * z + low
    return y



def piano_roll_to_target(pr):
    #  pr: (32, 128, 3), dtype=bool

    # Assume that "not (first_layer or second layer) = third_layer"
    pr[:, :, 1] = np.logical_not(np.logical_or(pr[:, :, 0], pr[:, :, 2]))
    # To int dtype can make addition work
    pr = pr.astype(int)
    # Initialize a matrix to store the duration of a note on the (32, 128) grid
    pr_matrix = np.zeros((32, 128))

    for i in range(31, -1, -1):
        # At each iteration
        # 1. Assure that the second layer accumulates the note duration
        # 2. collect the onset notes in time step i, and mark it on the matrix.

        # collect
        onset_idx = np.where(pr[i, :, 0] == 1)[0]
        pr_matrix[i, onset_idx] = pr[i, onset_idx, 1] + 1
        if i == 0:
            break
        # Accumulate
        # pr[i - 1, :, 1] += pr[i, :, 1]
        # pr[i - 1, onset_idx, 1] = 0  # the onset note should be set 0.
        pr[i, onset_idx, 1] = 0  # the onset note should be set 0.
        pr[i - 1, :, 1] += pr[i, :, 1]

    return pr_matrix


def target_to_3dtarget(pr_mat, max_note_count=11, max_pitch=107, min_pitch=22,
                       pitch_pad_ind=88, dur_pad_ind=2,
                       pitch_sos_ind=86, pitch_eos_ind=87):
    """
    :param pr_mat: (32, 128) matrix. pr_mat[t, p] indicates a note of pitch p,
    started at time step t, has a duration of pr_mat[t, p] time steps.
    :param max_note_count: the maximum number of notes in a time step,
    including <sos> and <eos> tokens.
    :param max_pitch: the highest pitch in the dataset.
    :param min_pitch: the lowest pitch in the dataset.
    :param pitch_pad_ind: see return value.
    :param dur_pad_ind: see return value.
    :param pitch_sos_ind: sos token.
    :param pitch_eos_ind: eos token.
    :return: pr_mat3d is a (32, max_note_count, 6) matrix. In the last dim,
    the 0th column is for pitch, 1: 6 is for duration in binary repr. Output is
    padded with <sos> and <eos> tokens in the pitch column, but with pad token
    for dur columns.
    """
    pitch_range = max_pitch - min_pitch + 1  # including pad
    pr_mat3d = np.ones((32, max_note_count, 6), dtype=int) * dur_pad_ind
    pr_mat3d[:, :, 0] = pitch_pad_ind
    pr_mat3d[:, 0, 0] = pitch_sos_ind
    cur_idx = np.ones(32, dtype=int)
    for t, p in zip(*np.where(pr_mat != 0)):
        pr_mat3d[t, cur_idx[t], 0] = p - min_pitch
        binary = np.binary_repr(int(pr_mat[t, p]) - 1, width=5)
        pr_mat3d[t, cur_idx[t], 1: 6] = \
            np.fromstring(' '.join(list(binary)), dtype=int, sep=' ')
        cur_idx[t] += 1
    pr_mat3d[np.arange(0, 32), cur_idx, 0] = pitch_eos_ind
    return pr_mat3d


def get_low_high_dur_count(pr_mat):
    # pr_mat (32, 128)
    # return the maximum duration
    # return the pitch range
    # return the number of notes at each column

    pitch_range = np.where(pr_mat != 0)[1]
    low_pitch = pitch_range.min()
    high_pitch = pitch_range.max()
    pitch_dur = pr_mat.max()
    num_notes = np.count_nonzero(pr_mat, axis=-1)
    return low_pitch, high_pitch, pitch_dur, num_notes
