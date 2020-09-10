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


# The probability of dropping a chunk
class BlockDropout(nn.Module):
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super(BlockDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, " "but got {}".format(p)
            )
        self.p: float = p
        self.inplace: bool = inplace

    def forward(self, x):
        if self.training:
            total_blocks = sum([len(sx) for sx in x])
            mask_size = torch.Size([total_blocks])
            binomial = torch.distributions.binomial.Binomial(probs=1 - self.p)
            mask = binomial.sample(mask_size) * (1.0 / (1 - self.p))
            mask_id = 0
            for mod in x:
                for x_mod in mod:
                    x_mod *= mask[mask_id]
                    mask_id += 1
            return x, mask
        return x, None


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
        self.block_dropout = BlockDropout(p=block_dropout, inplace=True) if 0 < block_dropout < 1 else None
        self.joint_features = nn.Sequential(
            nn.Linear(self.block_channel, self.reduced_channel),
            nn.BatchNorm1d(self.reduced_channel),
            nn.ReLU(inplace=True)
        )
        self.num_blocks = [math.ceil(ic / self.block_channel) for ic in
                           in_channels]  # number of blocks for each modality
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
        # split each modality into chunks
        spliced_x = [list(torch.split(x, self.block_channel, dim=1)) for x, bc in zip(X, bs_ch)]
        # pad channel if block_channel non divisible
        for sx in spliced_x:
            padding_shape = list(sx[-1].size())
            padding_shape[1] = self.block_channel - sx[-1].shape[1]
            sx[-1] = torch.cat([sx[-1], torch.zeros(torch.Size(padding_shape)).to(self.device)], dim=1)

        # apply BlockDropout
        if self.block_dropout:
            spliced_x, mask = self.block_dropout(spliced_x)
        # element wise sum
        spliced_x_sum = [torch.stack(sx).sum(dim=0) for sx in spliced_x]
        # global ave pooling on channel
        gap = [F.adaptive_avg_pool1d(sx.view(list(sx.size()[:2]) + [-1]), 1) for sx in spliced_x_sum]
        # combine GAP over modalities
        gap = torch.stack(gap).sum(dim=0)  # / (self.num_modality - 1)
        gap = torch.squeeze(gap, -1)
        gap = self.joint_features(gap)
        # pass into attention
        atten = self.soft_attention(torch.stack([dg(gap) for dg in self.dense_group]))
        # apply attention to each group
        atten_id = 0
        for sx_mod in spliced_x:
            for sx_chunk in sx_mod:
                att = self.lowest_atten + atten[atten_id] * (1 - self.lowest_atten)
                ns = len(sx_chunk.size()) - len(att.size())
                if self.block_dropout and self.training:
                    sx_chunk *= (mask[atten_id] * att.view(list(att.size()) + ns * [1]))
                else:
                    sx_chunk *= (att.view(list(att.size()) + ns * [1]))
                atten_id += 1
        # concat channel wise
        ret = [torch.cat(sx_mod, dim=1)[:, :ic] for sx_mod, ic in zip(spliced_x, self.in_channels)]
        return ret


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
            ret = self.blocks[0](X)
            return ret

        # split into multiple time segments
        segmented_x = [list(torch.split(x, x.shape[2] // self.split_block, dim=2)) for x in X]
        for x in segmented_x:
            if len(x) > self.split_block:
                x[-2] = torch.cat(x[-2:], dim=2)

        ret_segments = []
        for i in range(self.split_block):
            ret_segments.append(self.blocks[i]([x[i] for x in segmented_x]))

        ret = [torch.cat([r[m] for r in ret_segments], dim=2) for m in range(self.num_modality)]

        return ret


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m1 = torch.rand(4, 64, 50).to(device)
    m2 = torch.rand(4, 32, 53).to(device)
    x = [m1, m2]
    net = MSAF([64, 32], 16, block_dropout=0.2, reduction_factor=4, split_block=5).to(device)
    y = net(x)
    print(y[0].shape, y[1].shape)
