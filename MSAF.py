# Copyright (c) 2020 Anita Hu and Kevin Su
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
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
import math


# The probability of dropping a block
class BlockDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super(BlockDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, " "but got {}".format(p)
            )
        self.p: float = p

    def forward(self, X):
        if self.training:
            blocks_per_mod = [x.shape[1] for x in X]
            mask_size = torch.Size([X[0].shape[0], sum(blocks_per_mod)])
            binomial = torch.distributions.binomial.Binomial(probs=1 - self.p)
            mask = binomial.sample(mask_size).to(X[0].device) * (1.0 / (1 - self.p))
            mask_shapes = [list(x.shape[:2]) + [1] * (x.dim() - 2) for x in X]
            grouped_masks = torch.split(mask, blocks_per_mod, dim=1)
            grouped_masks = [m.reshape(s) for m, s in zip(grouped_masks, mask_shapes)]
            X = [x * m for x, m in zip(X, grouped_masks)]
            return X, grouped_masks
        return X, None


# squeeze dim default 1: i.e. channel in (bs, channel, height, width, ...)
# Parameters:
# in_channels: a list of channel numbers for modalities
# block_channel: the channel number of each equal-sized block
# reduction_factor: c' = c / reduction_factor, where c is block_channel
# lowest_atten: float number between 0 and 1. Attention value will be mapped to:
#   lowest_atten + attention_value * (1 - lowest_atten)
class MSAFBlock(nn.Module):
    def __init__(self, in_channels, block_channel, block_dropout=0., lowest_atten=0., reduction_factor=4):
        super(MSAFBlock, self).__init__()
        self.block_channel = block_channel
        self.in_channels = in_channels
        self.lowest_atten = lowest_atten
        self.num_modality = len(in_channels)
        self.reduced_channel = self.block_channel // reduction_factor
        self.block_dropout = BlockDropout(p=block_dropout) if 0 < block_dropout < 1 else None
        self.joint_features = nn.Sequential(
            nn.Linear(self.block_channel, self.reduced_channel),
            nn.BatchNorm1d(self.reduced_channel),
            nn.ReLU(inplace=True)
        )
        self.num_blocks = [math.ceil(ic / self.block_channel) for ic in
                           in_channels]  # number of blocks for each modality
        self.last_block_padding = [ic % self.block_channel for ic in in_channels]
        self.dense_group = nn.ModuleList([nn.Linear(self.reduced_channel, self.block_channel)
                                          for i in range(sum(self.num_blocks))])
        self.soft_attention = nn.Softmax(dim=0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # X: a list of features from different modalities
    def forward(self, X):
        bs_ch = [x.size()[:2] for x in X]
        for bc, ic in zip(bs_ch, self.in_channels):
            assert bc[1] == ic, "X shape and in_channels are different. X channel {} but got {}".format(str(bc[1]),
                                                                                                        str(ic))
        # pad channel if block_channel non divisible
        padded_X = [F.pad(x, (0, pad_size)) for pad_size, x in zip(self.last_block_padding, X)]

        # reshape each feature map: [batch size, N channels, ...] -> [batch size, N blocks, block channel, ...]
        desired_shape = [[x.shape[0], nb, self.block_channel] + list(x.shape[2:]) for x, nb in zip(padded_X, self.num_blocks)]
        reshaped_X = [torch.reshape(x, ds) for x, ds in zip(padded_X, desired_shape)]

        if self.block_dropout:
            reshaped_X, masks = self.block_dropout(reshaped_X)

        # element wise sum of blocks then global ave pooling on channel
        elem_sum_X = [torch.sum(x, dim=1) for x in reshaped_X]
        gap = [F.adaptive_avg_pool1d(sx.view(list(sx.size()[:2]) + [-1]), 1) for sx in elem_sum_X]

        # combine GAP over modalities and generate attention values
        gap = torch.stack(gap).sum(dim=0)  # / (self.num_modality - 1)
        gap = torch.squeeze(gap, -1)
        gap = self.joint_features(gap)
        atten = self.soft_attention(torch.stack([dg(gap) for dg in self.dense_group])).permute(1, 0, 2)
        atten = self.lowest_atten + atten * (1 - self.lowest_atten)

        # apply attention values to features
        atten_shapes = [list(x.shape[:3]) + [1] * (x.dim() - 3) for x in reshaped_X]
        grouped_atten = torch.split(atten, self.num_blocks, dim=1)
        grouped_atten = [a.reshape(s) for a, s in zip(grouped_atten, atten_shapes)]
        if self.block_dropout and self.training:
            reshaped_X = [x * m * a for x, m, a in zip(reshaped_X, masks, grouped_atten)]
        else:
            reshaped_X = [x * a for x, a in zip(reshaped_X, grouped_atten)]
        X = [x.reshape(org_x.shape) for x, org_x in zip(reshaped_X, X)]

        return X


class MSAF(nn.Module):
    def __init__(self, in_channels, block_channel, block_dropout, lowest_atten=0., reduction_factor=4,
                 split_block=1):
        super(MSAF, self).__init__()
        self.num_modality = len(in_channels)
        self.split_block = split_block
        self.blocks = nn.ModuleList([MSAFBlock(in_channels, block_channel, block_dropout, lowest_atten,
                                               reduction_factor) for i in range(split_block)])

    # X: a list of features from different modalities
    def forward(self, X):
        if self.split_block == 1:
            ret = self.blocks[0](X)  # only 1 MSAF block
        else:
            # split into multiple time segments, assumes at dim=2
            segment_shapes = [[x.shape[2] // self.split_block] * self.split_block for x in X]
            for x, seg_shape in zip(X, segment_shapes):
                seg_shape[-1] += x.shape[2] % self.split_block
            segmented_x = [torch.split(x, seg_shape, dim=2) for x, seg_shape in zip(X, segment_shapes)]
    
            # process segments using MSAF blocks
            ret_segments = [self.blocks[i]([x[i] for x in segmented_x]) for i in range(self.split_block)]

            # put segments back together
            ret = [torch.cat([r[m] for r in ret_segments], dim=2) for m in range(self.num_modality)]

        return ret


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m1 = torch.rand(4, 32, 64, 64, 50).to(device)
    m2 = torch.rand(4, 16, 32, 32, 53).to(device)
    x = [m1, m2]
    net = MSAF([32, 16], 8, block_dropout=0.2, reduction_factor=4, split_block=5).to(device)
    y = net(x)
    print(y[0].shape, y[1].shape)
