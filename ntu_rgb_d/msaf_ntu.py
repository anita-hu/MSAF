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

import torch
import torch.nn as nn
import torch.nn.functional as F
from mfas.models.auxiliary.resnet.resnet import transform_input

import sys

sys.path.append('..')
from MSAF import MSAF


class MSAFNet(nn.Module):
    def __init__(self, args):
        super(MSAFNet, self).__init__()
        self.visual = None
        self.rgb_net_name = args.rgb_net
        self.skeleton = None
        self.final_pred = None

        if args.rgb_net == 'resnet':
            self.msaf1 = MSAF(in_channels=[1024, 128, 128], block_channel=64, block_dropout=0, lowest_atten=0.5,
                              reduction_factor=4)
            self.msaf2 = MSAF(in_channels=[2048, 512], block_channel=256, block_dropout=0, lowest_atten=0.5,
                              reduction_factor=4)
        elif args.rgb_net == 'i3d':
            self.msaf1 = MSAF(in_channels=[832, 128, 128], block_channel=64, block_dropout=0, lowest_atten=0.5,
                              reduction_factor=4)
            self.msaf2 = MSAF(in_channels=[1024, 512], block_channel=256, block_dropout=0, lowest_atten=0.5,
                              reduction_factor=4)
        else:
            print("RGB net not resnet or i3d")
            raise NotImplementedError

        self.return_interm_feas = False
        self.return_both = False
        if hasattr(args, 'fc_final_preds') and args.fc_final_preds:
            print("Using fc final prediction")
            self.final_pred = nn.Linear(60 * 2, 60)

    def get_msaf_params(self):
        parameters = []
        if hasattr(self, "msaf1"):
            parameters.append({'params': self.msaf1.parameters()})
        if hasattr(self, "msaf2"):
            parameters.append({'params': self.msaf2.parameters()})
        return parameters

    def get_visual_params(self):
        parameters = [{'params': self.visual.parameters()}]
        if hasattr(self, "msaf1"):
            parameters.append({'params': self.msaf1.parameters()})
        if hasattr(self, "msaf2"):
            parameters.append({'params': self.msaf2.parameters()})
        return parameters

    def get_skeleton_params(self):
        parameters = [{'params': self.skeleton.parameters()}]
        if hasattr(self, "msaf1"):
            parameters.append({'params': self.msaf1.parameters()})
        if hasattr(self, "msaf2"):
            parameters.append({'params': self.msaf2.parameters()})
        return parameters

    def set_visual_skeleton_nets(self, visual, skeleton, return_interm_feas=False):
        self.visual = visual
        self.skeleton = skeleton
        self.return_interm_feas = return_interm_feas

    def set_return_both(self, p):
        self.return_both = p

    def forward(self, tensor_tuple):
        frames, skeleton = tensor_tuple[:2]

        ############################################## SKELETON INIT BLOCK
        N, C, T, V, M = skeleton.size()  # N0, C1, T2, V3, M4
        motion = skeleton[:, :, 1::, :, :] - skeleton[:, :, 0:-1, :, :]
        motion = motion.permute(0, 1, 4, 2, 3).contiguous().view(N, C * M, T - 1, V)
        motion = F.interpolate(motion, size=(T, V), mode='bilinear',
                               align_corners=False).contiguous().view(N, C, M, T, V).permute(0, 1, 3, 4, 2)

        # sk_logits = []
        sk_hidden = []
        for i in range(self.skeleton.num_person):
            # position
            # N0,C1,T2,V3 point-level
            out1 = self.skeleton.conv1(skeleton[:, :, :, :, i])
            out2 = self.skeleton.conv2(out1)
            # N0,V1,T2,C3, global level
            out2 = out2.permute(0, 3, 2, 1).contiguous()
            out3 = self.skeleton.conv3(out2)
            out_p = self.skeleton.conv4(out3)

            # motion
            # N0,T1,V2,C3 point-level
            out1m = self.skeleton.conv1m(motion[:, :, :, :, i])
            out2m = self.skeleton.conv2m(out1m)
            # N0,V1,T2,C3, global level
            out2m = out2m.permute(0, 3, 2, 1).contiguous()
            out3m = self.skeleton.conv3m(out2m)
            out_m = self.skeleton.conv4m(out3m)

            # concat
            out4 = torch.cat((out_p, out_m), dim=1)
            sk_hidden.append([out1, out2, out3, out4])

        # clean hidden representations
        new_sk_hidden = []
        for h1, h2 in zip(sk_hidden[0], sk_hidden[1]):
            new_sk_hidden.append(torch.max(h1, h2))

        out4_p0 = sk_hidden[0][-1]
        out4_p1 = sk_hidden[1][-1]

        out5_p0 = self.skeleton.conv5(out4_p0)
        sk_hidden[0].append(out5_p0)
        out5_p1 = self.skeleton.conv5(out4_p1)
        sk_hidden[1].append(out5_p1)

        out5_max = torch.max(out5_p0, out5_p1)

        ################################################ VISUAL INIT BLOCK
        # Changing temporal and channel dim to fit the inflated resnet input requirements
        B, T, W, H, C = frames.size()
        frames = frames.view(B, 1, T, W, H, C)
        frames = frames.transpose(1, -1)
        frames = frames.view(B, C, T, W, H)
        frames = frames.contiguous()

        if self.rgb_net_name == 'resnet':
            rgb_resnet = self.visual.cnn

            # 5D -> 4D if 2D conv at the beginning
            frames = transform_input(frames, rgb_resnet.input_dim, T=T)

            # 1st conv
            frames = rgb_resnet.conv1(frames)
            frames = rgb_resnet.bn1(frames)
            frames = rgb_resnet.relu(frames)
            frames = rgb_resnet.maxpool(frames)

            # 1st residual block
            frames = transform_input(frames, rgb_resnet.layer1[0].input_dim, T=T)
            frames = rgb_resnet.layer1(frames)
            fm1 = frames

            # 2nd residual block
            frames = transform_input(frames, rgb_resnet.layer2[0].input_dim, T=T)
            frames = rgb_resnet.layer2(frames)
            fm2 = frames

            # 3rd residual block
            frames = transform_input(frames, rgb_resnet.layer3[0].input_dim, T=T)
            frames = rgb_resnet.layer3(frames)
            fm3 = frames
        else:
            fm2 = self.visual.features[:13](frames)
            fm3 = self.visual.features[13:15](fm2)

        ###################################### FIRST msaf
        # fm3, out5_p0 (first person), out5_p1 (second person) => fm3, out5_p0, out5_p1
        fm3, out5_p0, out5_p1 = self.msaf1([fm3, out5_p0, out5_p1])
        ######################################

        # skeleton
        out6_p0 = self.skeleton.conv6(out5_p0)
        sk_hidden[0].append(out6_p0)
        out6_p1 = self.skeleton.conv6(out5_p1)
        sk_hidden[1].append(out6_p0)
        out6_max = torch.max(out6_p0, out6_p1)
        out7 = out6_max

        # max out logits
        out7 = out7.view(out7.size(0), -1)
        out8 = self.skeleton.fc7(out7)

        # visual
        if self.rgb_net_name == 'resnet':
            # 4th residual block
            frames = transform_input(frames, rgb_resnet.layer4[0].input_dim, T=T)
            frames = rgb_resnet.layer4(frames)
            final_fm = transform_input(frames, rgb_resnet.out_dim, T=T)
        else:
            final_fm = self.visual.features[15](fm3)

        ########################################## SECOND msaf
        # final_fm, out8 => final_fm, out8
        final_fm, out8 = self.msaf2([final_fm, out8])
        ##########################################

        # skeleton
        outf = self.skeleton.fc8(out8)

        new_sk_hidden.append(out5_max)
        new_sk_hidden.append(out6_max)
        new_sk_hidden.append(out7)
        new_sk_hidden.append(out8)

        t = outf
        assert not (torch.isnan(t).any())  # find out nan in tensor
        skeleton_features = [new_sk_hidden, outf]

        # visual
        if self.rgb_net_name == 'resnet':
            # Temporal pooling
            vis_out5 = self.visual.temporal_pooling(final_fm)
            vis_out6 = self.visual.classifier(vis_out5)
            visual_features = [fm1, fm2, fm3, final_fm, vis_out5, vis_out6]
        else:
            vis_out5 = self.visual.features[16:](final_fm)
            if self.visual.spatial_squeeze:
                vis_out5 = vis_out5.squeeze(3)
                vis_out5 = vis_out5.squeeze(3)

            vis_out6 = torch.mean(vis_out5, 2)
            visual_features = [fm2, fm3, final_fm, vis_out5, vis_out6]

        if self.return_interm_feas:
            return visual_features, skeleton_features

        ### LATE FUSION
        vis_pred = vis_out6
        skeleton_pred = outf
        if self.final_pred is None:
            pred = (skeleton_pred + vis_pred) / 2
        else:
            pred = self.final_pred(torch.cat([skeleton_pred, vis_pred], dim=-1))

        if self.return_both:
            return vis_pred, skeleton_pred

        return pred
