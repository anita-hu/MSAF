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

import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class MFCCNet(nn.Module):
    def __init__(self, features_only=False):
        super(MFCCNet, self).__init__()
        self.features_only = features_only
        self.conv1 = nn.Conv1d(in_channels=13, out_channels=32, kernel_size=7, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=2)
        self.maxpool = nn.MaxPool1d(kernel_size=8)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(p=0.2)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(64)
        self.flatten = Flatten()
        self.dropout2 = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1664, 8)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.bn2(x)
        x = self.dropout1(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.flatten(x)
        x = self.dropout2(x)
        if self.features_only:
            return x
        x = self.fc1(x)

        return x
