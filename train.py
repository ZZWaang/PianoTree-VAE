import torch
from torch import nn
import os
import sys
import json
import numpy as np
import utils
from torch import optim
from utils import MinExponentialLR, loss_function, scheduled_sampling
from model import VAE
from tensorboardX import SummaryWriter
from dataset import PolyphonicDataset
import time
from torch.utils.data import DataLoader

###############################################################################
# Load config
###############################################################################
config_fn = './model_config.json'
train_hyperparams = utils.load_params_dict('train_hyperparams', config_fn)
model_params = utils.load_params_dict('model_params', config_fn)
data_repr_params = utils.load_params_dict('data_repr', config_fn)
project_params = utils.load_params_dict('project', config_fn)
dataset_path = utils.load_dataset_path(config_fn)

BATCH_SIZE = train_hyperparams['batch_size']
LEARNING_RATE = train_hyperparams['learning_rate']
DECAY = train_hyperparams['decay']
PARALLEL = train_hyperparams['parallel']
if sys.platform in ['win32', 'darwin']:
    PARALLEL = False
N_EPOCH = train_hyperparams['n_epoch']
CLIP = train_hyperparams['clip']
UP_AUG = train_hyperparams['up_aug']
DOWN_AUG = train_hyperparams['down_aug']
INIT_WEIGHT = train_hyperparams['init_weight']
WEIGHTS = tuple(train_hyperparams['weights'])
TFR1 = tuple(train_hyperparams['teacher_forcing1'])
TFR2 = tuple(train_hyperparams['teacher_forcing2'])

###############################################################################
# Initialize project
###############################################################################
PROJECT_NAME = project_params['project_name']
MODEL_PATH = project_params['model_path']
LOG_PATH = project_params['log_path']

if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)
else:
    ans = input('Model path is not empty. '
                'Are you sure you want to continue?[y/n]')
    if ans == 'y':
        pass
    else:
        quit()

if not os.path.exists(LOG_PATH):
    os.mkdir(LOG_PATH)
else:
    ans = input('Log path is not empty. '
                'Are you sure you want to continue?[y/n]')
    if ans == 'y':
        pass
    else:
        quit()

loss_writer = SummaryWriter(os.path.join(LOG_PATH, 'loss'))
kl_writer = SummaryWriter(os.path.join(LOG_PATH, 'kl_div'))
pitch_writer = SummaryWriter(os.path.join(LOG_PATH, 'pitch_loss'))
dur_writer = SummaryWriter(os.path.join(LOG_PATH, 'dur_loss'))
print('Project initialized.', flush=True)

###############################################################################
# load data
###############################################################################
train_path = os.path.join(dataset_path, 'pop909+mlpv_t32_train_fix1.npy')
val_path = os.path.join(dataset_path, 'pop909+mlpv_t32_val_fix1.npy')
train_dataset = PolyphonicDataset(train_path, DOWN_AUG, UP_AUG)
val_dataset = PolyphonicDataset(val_path, 0, 0)
train_loader = DataLoader(train_dataset, BATCH_SIZE, True)
val_loader = DataLoader(val_dataset, BATCH_SIZE, True)

print(len(train_dataset), len(val_dataset), flush=True)
print('Dataset loaded!', flush=True)

###############################################################################
# model parameter
###############################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VAE(max_simu_note=data_repr_params['max_simu_note'],
            max_pitch=data_repr_params['max_pitch'],
            min_pitch=data_repr_params['min_pitch'],
            pitch_sos=data_repr_params['pitch_sos'],
            pitch_eos=data_repr_params['pitch_eos'],
            pitch_pad=data_repr_params['pitch_pad'],
            dur_pad=data_repr_params['dur_pad'],
            dur_width=data_repr_params['dur_width'],
            num_step=data_repr_params['num_time_step'],
            note_emb_size=model_params['note_emb_size'],
            enc_notes_hid_size=model_params['enc_notes_hid_size'],
            enc_time_hid_size=model_params['enc_time_hid_size'],
            z_size=model_params['z_size'],
            dec_emb_hid_size=model_params['dec_emb_hid_size'],
            dec_time_hid_size=model_params['dec_time_hid_size'],
            dec_notes_hid_size=model_params['dec_notes_hid_size'],
            dec_z_in_size=model_params['dec_z_in_size'],
            dec_dur_hid_size=model_params['dec_dur_hid_size'],
            device=device
            )

if INIT_WEIGHT:
    model.apply(utils.init_weights)
    print('The parameters in the model are initialized!')
print(f'The model has {utils.count_parameters(model):,} trainable parameters')

if PARALLEL:
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model = model.to(device)
    model = model.module
else:
    model = model.to(device)
print('Model loaded!')

###############################################################################
# Optimizer and Criterion
###############################################################################
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
pitch_loss_func = nn.CrossEntropyLoss(ignore_index=model.pitch_pad)
dur_loss_func = nn.CrossEntropyLoss(ignore_index=model.dur_pad)
normal = utils.standard_normal(model.z_size)
if DECAY:
    scheduler = MinExponentialLR(optimizer, gamma=0.9999, minimum=1e-5)


###############################################################################
# Main
###############################################################################
def train(model, train_loader,
          optimizer, pitch_criterion, dur_criterion, normal, weights,
          decay, clip, epoch):
    model.train()
    epoch_loss = 0.
    epoch_pitch_loss = epoch_dur_loss = epoch_kl_loss = 0.
    num_batch = len(train_loader)

    for i, batch in enumerate(train_loader):
        if torch.cuda.is_available():
            batch = batch.cuda()
        optimizer.zero_grad()
        tfr1 = scheduled_sampling(((epoch + i / num_batch) / N_EPOCH),
                                  TFR1[0], TFR1[1])
        tfr2 = scheduled_sampling(((epoch + i / num_batch) / N_EPOCH),
                                  TFR2[0], TFR2[1])
        recon_pitch, recon_dur, dist = model(batch, False, True, tfr1, tfr2)
        recon_pitch = recon_pitch.view(-1, recon_pitch.size(-1))
        recon_dur = recon_dur.view(-1, recon_dur.size(-1))
        loss, pitch_loss, dur_loss, kl_loss = \
            loss_function(recon_pitch,
                          batch[:, :, 1:, 0].contiguous().view(-1),
                          recon_dur,
                          batch[:, :, 1:, 1:].contiguous().view(-1),
                          dist, pitch_criterion, dur_criterion, normal,
                          weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        epoch_pitch_loss += pitch_loss.item()
        epoch_dur_loss += dur_loss.item()
        epoch_kl_loss += kl_loss.item()

        loss_writer.add_scalar('batch_train_loss %d' % num_batch, loss.item(),
                               epoch * num_batch + i)
        kl_writer.add_scalar('batch_train_loss %d' % num_batch, kl_loss.item(),
                             epoch * num_batch + i)
        pitch_writer.add_scalar('batch_train_loss %d' % num_batch,
                                pitch_loss.item(),
                                epoch * num_batch + i)
        dur_writer.add_scalar('batch_train_loss %d' % num_batch,
                              dur_loss.item(),
                              epoch * num_batch + i)
        if decay:
            scheduler.step()
    return (epoch_loss / num_batch, epoch_pitch_loss / num_batch,
            epoch_dur_loss / num_batch, epoch_kl_loss / num_batch)


def evaluate(model, val_loader,
             pitch_criterion, dur_criterion, normal, weights, epoch):
    model.eval()
    epoch_loss = 0.
    epoch_pitch_loss = epoch_dur_loss = epoch_kl_loss = 0.
    num_batch = len(val_loader)
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if torch.cuda.is_available():
                batch = batch.cuda()
            tfr1 = scheduled_sampling(((epoch + i / num_batch) / N_EPOCH),
                                      TFR1[0], TFR1[1])
            tfr2 = scheduled_sampling(((epoch + i / num_batch) / N_EPOCH),
                                      TFR2[0], TFR2[1])
            recon_pitch, recon_dur, dist = model(batch, False, True,
                                                 tfr1, tfr2)
            recon_pitch = recon_pitch.view(-1, recon_pitch.size(-1))
            recon_dur = recon_dur.view(-1, recon_dur.size(-1))
            loss, pitch_loss, dur_loss, kl_loss = \
                loss_function(recon_pitch,
                              batch[:, :, 1:, 0].contiguous().view(-1),
                              recon_dur,
                              batch[:, :, 1:, 1:].contiguous().view(-1),
                              dist, pitch_criterion, dur_criterion, normal,
                              weights)
            epoch_loss += loss.item()
            epoch_pitch_loss += pitch_loss.item()
            epoch_dur_loss += dur_loss.item()
            epoch_kl_loss += kl_loss.item()
            loss_writer.add_scalar('batch_eval_loss %d' % num_batch,
                                   loss.item(),
                                   epoch * num_batch + i)
            kl_writer.add_scalar('batch_eval_loss %d' % num_batch,
                                 kl_loss.item(),
                                 epoch * num_batch + i)
            pitch_writer.add_scalar('batch_eval_loss %d' % num_batch,
                                    pitch_loss.item(),
                                    epoch * num_batch + i)
            dur_writer.add_scalar('batch_eval_loss %d' % num_batch,
                                  dur_loss.item(),
                                  epoch * num_batch + i)
    return (epoch_loss / num_batch, epoch_pitch_loss / num_batch,
            epoch_dur_loss / num_batch, epoch_kl_loss / num_batch)


best_valid_loss = float('inf')
for epoch in range(N_EPOCH):
    print(f'Start Epoch: {epoch + 1:02}', flush=True)

    start_time = time.time()
    train_loss, train_pl, train_dl, train_kl = \
        train(model, train_loader,
              optimizer, pitch_loss_func, dur_loss_func, normal,
              WEIGHTS, DECAY, CLIP, epoch)

    valid_loss, valid_pl, valid_dl, valid_kl = \
        evaluate(model, val_loader,
                 pitch_loss_func, dur_loss_func, normal,
                 WEIGHTS, epoch)
    end_time = time.time()

    epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)

    torch.save(model.state_dict(), os.path.join(MODEL_PATH,
                                                'pgrid-epoch-model.pt'))

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(),
                   os.path.join(MODEL_PATH, 'pgrid-best-valid-model.pt'))

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s',
          flush=True)
    print(
        f'\tTrain Loss: {train_loss:.3f}', flush=True)
    print(
        f'\t Val. Loss: {valid_loss:.3f}', flush=True)
    loss_writer.add_scalar('epoch_train_loss', train_loss, epoch)
    loss_writer.add_scalar('epoch_eval_loss', valid_loss, epoch)

    kl_writer.add_scalar('epoch_train_loss', train_kl, epoch)
    pitch_writer.add_scalar('epoch_train_loss', train_pl, epoch)
    dur_writer.add_scalar('epoch_train_loss', train_dl, epoch)

    kl_writer.add_scalar('epoch_eval_loss', valid_kl, epoch)
    pitch_writer.add_scalar('epoch_eval_loss', valid_pl, epoch)
    dur_writer.add_scalar('epoch_eval_loss', valid_dl, epoch)

torch.save(model.state_dict(), os.path.join(MODEL_PATH, 'pgrid-last-model.pt'))
print('Model Saved!', flush=True)


if __name__ == '__main__':
    pass
