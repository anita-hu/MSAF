#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Amirreza Shaban (from MMTM github https://github.com/haamoon/mmtm)
# Copyright (c) 2020 Anita Hu and Kevin Su
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys

sys.path.append('mfas')
from models.central.ntu import Visual, Skeleton
from models.auxiliary.I3D import I3D
from main_found_ntu import get_dataloaders

import torch
import argparse
import os
import glob
import copy
import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from msaf_ntu import MSAFNet


# %% Parse inputs
def parse_args():
    parser = argparse.ArgumentParser(description='Modality optimization.')
    parser.add_argument('--rgb_net', type=str, help='name of rgb model that is loaded', default='i3d',
                        choices=['resnet', 'i3d'])
    parser.add_argument('--checkpointdir', type=str, help='output base dir', default='checkpoints')
    parser.add_argument('--datadir', type=str, help='data directory', default='dataset')
    parser.add_argument('--ske_cp', type=str, help='Skeleton net checkpoint (assuming is contained in checkpointdir)',
                        default='skeleton_32frames_85.24.checkpoint')
    parser.add_argument('--rgb_cp', type=str, help='RGB net checkpoint (assuming is contained in checkpointdir)',
                        default='i3d_32frames_85.63.checkpoint')
    parser.add_argument('--test_cp', type=str, help='Full net checkpoint (assuming is contained in checkpointdir)',
                        default='')
    parser.add_argument('--num_outputs', type=int, help='output dimension', default=60)
    parser.add_argument('--batchsize', type=int, help='batch size', default=4)
    parser.add_argument('--epochs', type=int, help='training epochs', default=20)
    parser.add_argument('--use_dataparallel', help='Use several GPUs', action='store_true', dest='use_dataparallel',
                        default=False)
    parser.add_argument('--j', dest='num_workers', type=int, help='Dataloader CPUS', default=16)
    parser.add_argument('--modality', type=str, help='', default='both')
    parser.add_argument('--no-verbose', help='verbose', action='store_false', dest='verbose', default=True)
    parser.add_argument('--no-multitask', dest='multitask', help='Multitask loss', action='store_false', default=True)

    parser.add_argument("--vid_len", action="store", default=(32, 32), dest="vid_len", type=int, nargs='+',
                        help="length of video, as a tuple of two lengths, (rgb len, skel len)")
    parser.add_argument("--drpt", action="store", default=0.4, dest="drpt", type=float, help="dropout")

    parser.add_argument('--no_bad_skel', action="store_true",
                        help='Remove the 300 bad samples, espec. useful to evaluate', default=False)
    parser.add_argument("--no_norm", action="store_true", default=False, dest="no_norm",
                        help="Not normalizing the skeleton")

    parser.add_argument("--fc_final_preds", default=False, type=bool, help="Add fc layer to fuse modalities at the end")

    parser.add_argument('--train', action='store_true', default=False, help='training')
    return parser.parse_args()


def update_lr(optimizer, multiplier=.1):
    state_dict = optimizer.state_dict()
    for param_group in state_dict['param_groups']:
        param_group['lr'] = param_group['lr'] * multiplier
    optimizer.load_state_dict(state_dict)


def step(branch, input_data, optimizers, criteria, is_training):
    rgb, ske, label = input_data
    optimizer = optimizers[branch]
    # Track history only in training
    with torch.set_grad_enabled(is_training):
        output = model((rgb, ske))
        # Predict
        _, preds1 = torch.max(output[0], 1)
        _, preds2 = torch.max(output[1], 1)
        _, preds = torch.max(output[0] + output[1], 1)
        # Backward
        optimizer.zero_grad()
        loss = criteria(output[branch], label)
        # Backward into the branch
        if is_training:
            loss.backward()
            optimizer.step()
    return loss, preds1, preds2, preds


def train_mmtm_track_acc(model, criteria, optimizers, dataloaders, device=None, num_epochs=200):
    torch.autograd.set_detect_anomaly(True)
    best_model_sd = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    # setup tensorboard
    log_dir = args.checkpointdir + '/logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'dev']:
            print('Epoch {}, Phase {}'.format(epoch + 1, phase))
            is_training = (phase == 'train')
            model.train(is_training)

            # Learning rate schedule
            if is_training and (epoch == 5 or epoch == 20):
                update_lr(optimizers[0], multiplier=.1)
                update_lr(optimizers[1], multiplier=.1)

            running_loss1, running_loss2 = 0.0, 0.0
            running_corrects, running_corrects1, running_corrects2 = 0, 0, 0
            ndata = 0

            # Iterate over data
            for data in tqdm(dataloaders[phase]):
                input_data = [data[n].to(device) for n in ['rgb', 'ske', 'label']]
                # print(input_data[0].shape)
                if input_data[0].shape[2] == 0:
                    continue

                # Update Visual Branch
                loss1, preds1, _, preds = step(0, input_data, optimizers, criteria, is_training)
                # Update Skeleton Branch
                loss2, _, preds2, _ = step(1, input_data, optimizers, criteria, is_training)

                # Update statistics
                batch_size = input_data[0].size(0)
                running_loss1 += loss1.item() * batch_size
                running_corrects1 += torch.sum(preds1 == input_data[2].data)
                running_loss2 += loss2.item() * batch_size
                running_corrects2 += torch.sum(preds2 == input_data[2].data)
                running_corrects += torch.sum(preds == input_data[2].data)
                ndata = ndata + batch_size

            avg_loss = (running_loss1 + running_loss2) / ndata / 2
            epoch_loss = [avg_loss,
                          running_loss1 / ndata,
                          running_loss2 / ndata]

            epoch_acc = [running_corrects.double() / ndata,
                         running_corrects1.double() / ndata,
                         running_corrects2.double() / ndata]

            print('Acc Multimodal: {:.4f}, Acc Visual: {:.4f}, Acc Skeleton: {:.4f}'.format(*epoch_acc))
            print('Loss Avg: {:.6f}, Loss Visual: {:.6f}, Loss Skeleton: {:.6f}'.format(*epoch_loss))

            # tensorboard update
            writer.add_scalar('Acc_{}/Multimodal'.format(phase), epoch_acc[0], epoch)
            writer.add_scalar('Acc_{}/Visual'.format(phase), epoch_acc[1], epoch)
            writer.add_scalar('Acc_{}/Skeleton'.format(phase), epoch_acc[2], epoch)
            writer.add_scalar('Loss_{}/Avg'.format(phase), epoch_loss[0], epoch)
            writer.add_scalar('Loss_{}/Visual'.format(phase), epoch_loss[1], epoch)
            writer.add_scalar('Loss_{}/Skeleton'.format(phase), epoch_loss[2], epoch)
            writer.flush()

            # Keep the best model
            if not is_training:  # and (avg_loss < best_loss or epoch % 5 == 0):
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_model_sd = copy.deepcopy(model.state_dict())
                filename = (args.checkpointdir +
                            '/msaf_ntu_epoch{}_val_loss{:.4f}.checkpoint'.format(epoch + 1, avg_loss))
                torch.save(model.state_dict(), filename)
                print('Saving ' + filename)

    model.load_state_dict(best_model_sd)
    model.train(False)
    return best_loss


def test_mmtm_track_acc(model, dataloaders, device=None):
    model.train(False)
    phase = 'test'

    running_corrects, running_corrects1, running_corrects2 = 0, 0, 0

    # Iterate over data
    ndata = 0
    for data in tqdm(dataloaders[phase]):
        # Get the inputs
        rgb, ske, label = [data[n].to(device) for n in ['rgb', 'ske', 'label']]

        # Forward
        output = model((rgb, ske))

        # Predict
        preds1 = torch.argmax(output[0], dim=-1)
        preds2 = torch.argmax(output[1], dim=-1)
        preds = torch.argmax(output[0] + output[1], dim=-1)

        # Update statistics
        running_corrects += torch.sum(preds == label.data)
        running_corrects1 += torch.sum(preds1 == label.data)
        running_corrects2 += torch.sum(preds2 == label.data)
        ndata += rgb.size(0)

    acc = running_corrects.double() / ndata
    acc_vis = running_corrects1.double() / ndata
    acc_ske = running_corrects2.double() / ndata
    return acc, acc_vis, acc_ske


def test_model(model, dataloaders, args, device):
    filename = os.path.join(args.checkpointdir, args.test_cp)
    checkpoint = torch.load(filename, map_location=device) if device.type == 'cpu' else torch.load(filename)
    model.load_state_dict(checkpoint)
    print('Loading ' + filename)

    # hardware tuning
    if torch.cuda.device_count() > 1 and args.use_dataparallel:
        model = torch.nn.DataParallel(model)
    model.to(device)
    test_model_acc = test_mmtm_track_acc(model, dataloaders, device=device)
    return test_model_acc


def train_model(model, dataloaders, args, device):
    criteria = torch.nn.CrossEntropyLoss()

    # loading pretrained weights
    skemodel_filename = os.path.join(args.checkpointdir, args.ske_cp)
    rgbmodel_filename = os.path.join(args.checkpointdir, args.rgb_cp)
    model.skeleton.load_state_dict(torch.load(skemodel_filename))
    model.visual.load_state_dict(torch.load(rgbmodel_filename))

    # optimizers
    optimizers = [torch.optim.Adam(model.get_visual_params(), lr=.0001, weight_decay=1e-4),
                  torch.optim.Adam(model.get_skeleton_params(), lr=.0001, weight_decay=1e-4)]

    # hardware tuning
    if torch.cuda.device_count() > 1 and args.use_dataparallel:
        model = torch.nn.DataParallel(model)
    model.to(device)

    val_model_acc = train_mmtm_track_acc(model, criteria, optimizers, dataloaders, device=device,
                                         num_epochs=args.epochs)
    return val_model_acc


if __name__ == "__main__":
    args = parse_args()
    print("The configuration of this run is:")
    print(args, end='\n\n')

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")

    dataloaders = get_dataloaders(args)

    model = MSAFNet(args)
    model.set_return_both(True)

    visual = Visual(args) if args.rgb_net == 'resnet' else I3D(num_classes=60, dropout_drop_prob=0.5,
                                                               input_channel=3, spatial_squeeze=True)
    skeleton = Skeleton(args)
    model.set_visual_skeleton_nets(visual, skeleton)

    if args.train:
        print("Training MSAF network")
        train_model(model, dataloaders, args, device)
    else:
        if args.test_cp:
            print("Evaluating single model...")
            test_acc = test_model(model, dataloaders, args, device)
            print('Acc Multimodal: {:.4f}, Acc Visual: {:.4f}, Acc Skeleton: {:.4f}'.format(*test_acc))
        else:
            print("Evaluating all fusion models in checkpoint dir...")
            checkpoint_dir = os.path.join(args.checkpointdir, "msaf_ntu*")
            for each_w in glob.glob(checkpoint_dir):
                each_w = os.path.basename(each_w)
                args.test_cp = each_w
                test_acc = test_model(model, dataloaders, args, device)
                print('Acc Multimodal: {:.4f}, Acc Visual: {:.4f}, Acc Skeleton: {:.4f}'.format(*test_acc))
