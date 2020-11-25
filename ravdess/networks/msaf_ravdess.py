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

import sys
sys.path.append('..')
from MSAF import MSAF


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class MSAFNet(nn.Module):
    def __init__(self, model_param):
        super(MSAFNet, self).__init__()
        # The inputs to these layers will be passed through msaf before being passed into the layer
        self.msaf_locations = {
            "video": [6, 7],
            "audio": [5, 11],
        }
        # MSAF blocks
        self.msaf = nn.ModuleList([
            MSAF(in_channels=[1024, 32], block_channel=16, block_dropout=0.2, reduction_factor=4),
            MSAF(in_channels=[2048, 64], block_channel=32, block_dropout=0.2, reduction_factor=4)
        ])
        self.num_msaf = len(self.msaf)

        if "video" in model_param:
            video_model = model_param["video"]["model"]
            # video model layers
            video_model = nn.Sequential(
                video_model.conv1,  # 0
                video_model.bn1,  # 1
                video_model.maxpool,  # 2
                video_model.layer1,  # 3
                video_model.layer2,  # 4
                video_model.layer3,  # 5
                video_model.layer4,  # 6
                video_model.avgpool,  # 7
                Flatten(),  # 8
                video_model.fc  # 9
            )
            self.video_model_blocks = self.make_blocks(video_model, self.msaf_locations["video"])
            self.video_id = model_param["video"]["id"]
            print("########## Video ##########")
            for vb in self.video_model_blocks:
                print(vb)

        if "audio" in model_param:
            audio_model = model_param["audio"]["model"]
            # audio model layers
            audio_model = nn.Sequential(
                audio_model.conv1,  # 0
                nn.ReLU(inplace=True),  # 1
                audio_model.bn1,  # 2
                audio_model.conv2,  # 3
                nn.ReLU(inplace=True),  # 4
                audio_model.maxpool,  # 5
                audio_model.bn2,  # 6
                audio_model.dropout1,  # 7
                audio_model.conv3,  # 8
                nn.ReLU(inplace=True),  # 9
                audio_model.bn3,  # 10
                audio_model.flatten,  # 11
                audio_model.dropout2,  # 12
                audio_model.fc1  # 13
            )
            self.audio_model_blocks = self.make_blocks(audio_model, self.msaf_locations["audio"])
            self.audio_id = model_param["audio"]["id"]
            print("########## Audio ##########")
            for ab in self.audio_model_blocks:
                print(ab)

    def forward(self, x):
        for i in range(self.num_msaf + 1):
            if hasattr(self, "video_id"):
                x[self.video_id] = self.video_model_blocks[i](x[self.video_id])
            if hasattr(self, "audio_id"):
                x[self.audio_id] = self.audio_model_blocks[i](x[self.audio_id])
            if i < self.num_msaf:
                x = self.msaf[i](x)
                
        return sum(x)

    # split model into blocks for msafs. Model in Sequential. recipe in list
    def make_blocks(self, model, recipe):
        blocks = [nn.Sequential(*(list(model.children())[i:j])) for i, j in zip([0] + recipe, recipe + [None])]
        return nn.ModuleList(blocks)
