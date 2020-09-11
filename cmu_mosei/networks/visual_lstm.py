# Copyright (c) 2020 Anita Hu
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

'''
Input output shapes
visual (FACET): (20, 35)
label: (1) -> [sentiment]
'''


class FACETVisualLSTMNet(nn.Module):
    def __init__(self, features_only=False):
        super(FACETVisualLSTMNet, self).__init__()
        self.features_only = features_only
        self.lstm1 = nn.LSTM(input_size=35, hidden_size=32, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=32, hidden_size=32, num_layers=1, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        last = x[:, -1, :]

        return self.classifier(last)
