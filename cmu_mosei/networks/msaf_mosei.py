# Copyright (c) 2020 Anita Hu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the 'Software'), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn

import sys
sys.path.append('..')
from MSAF import MSAF


class MSAFLSTMNet(nn.Module):
    def __init__(self, model_param):
        super(MSAFLSTMNet, self).__init__()
        # Locations: the features from the previous layer index will be passed to a msaf module
        self.msaf_locations = [1]
        # MSAF blocks
        self.msaf = nn.ModuleList([
            MSAF(in_channels=[32, 64, 128], block_channel=16, block_dropout=0.2, lowest_atten=0., reduction_factor=2,
                 split_block=5),
        ])
        self.max_feature_layers = 2  # number of layers in unimodal models before classifier

        if "visual" in model_param:
            visual_model = model_param["visual"]["model"]
            # visual model layers
            self.visual_model = nn.ModuleList([
                visual_model.lstm1,
                visual_model.lstm2
            ])
            self.visual_id = model_param["visual"]["id"]
            print("########## Visual ##########")
            print(visual_model)

        if "audio" in model_param:
            audio_model = model_param["audio"]["model"]
            # audio model layers
            self.audio_model = nn.ModuleList([
                audio_model.lstm1,
                audio_model.lstm2
            ])
            self.audio_id = model_param["audio"]["id"]
            print("########## Audio ##########")
            print(audio_model)

        if "bert" in model_param:
            text_model = model_param["bert"]["model"]
            # text model layers
            self.text_model = nn.ModuleList([
                text_model.lstm1,
                text_model.lstm2
            ])
            self.text_id = model_param["bert"]["id"]
            print("########## Bert ##########")
            print(text_model)

        self.multimodal_classifier = nn.Sequential(
            nn.Linear(224, 1),
        )

    def forward(self, x):
        msaf_block_idx = 0
        for i in range(self.max_feature_layers):
            if i in self.msaf_locations:
                x = [item.permute(0, 2, 1) for item in x]  # msaf requires channels at dim=1
                x = self.msaf[msaf_block_idx](x)
                x = [item.permute(0, 2, 1) for item in x]
                msaf_block_idx += 1
            if hasattr(self, "visual_id"):
                x[self.visual_id], _ = self.visual_model[i](x[self.visual_id])
            if hasattr(self, "audio_id"):
                x[self.audio_id], _ = self.audio_model[i](x[self.audio_id])
            if hasattr(self, "text_id"):
                x[self.text_id], _ = self.text_model[i](x[self.text_id])

        x = [item[:, -1, :] for item in x]

        return self.multimodal_classifier(torch.cat(x, -1))
