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

import glob
import os
import cv2
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset

"""
Video label in filename:
Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
Vocal channel (01 = speech, 02 = song).
Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
Repetition (01 = 1st repetition, 02 = 2nd repetition).
Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
"""
video_dict = {
    "modality": [1, 2, 3],
    "vocal_channel": [1, 2],
    "emotion": [1, 2, 3, 4, 5, 6, 7, 8],
    "emotional_intensity": [1, 2],
    "statement": [1, 2],
    "repetition": [1, 2],
    "actor": list(range(1, 25))
}


def parse_gt(video_name):
    video_keys = video_dict.keys()
    gt = {}
    y = video_name.strip(".mp4")
    y = y.split("-")
    for k, v in zip(video_keys, y):
        gt[k] = int(v)
    return gt


class RAVDESSDataset(Dataset):
    """RAVDESS dataset."""

    def __init__(self, actor_folds, modality=('video', 'audio'), transform=None):
        """
        :param actor_folds: a list of actor folder paths
        :param modality: list of modalities, default ('video', 'audio')
        :param transform: data transform or a dict with
        {
            "image_transform": transform function or None
            "audio_transform": transform function or None
        }
        """
        self.video_list = []
        self.transform = transform
        self.modality = modality

        actor_counter = 0
        for each_actor in actor_folds:
            for each_video in glob.glob(os.path.join(each_actor, "*.mp4")):
                self.video_list.append(each_video)
            actor_counter += 1

        print("{} videos for {} actors found.".format(str(len(self.video_list)), str(actor_counter)))

    def __len__(self):
        return len(self.video_list)

    def _read_images(self, idx, transform):
        X = []
        video_path = self.video_list[idx]
        frames_path = os.path.join(video_path, "frames")
        frames_lst = sorted(glob.glob(os.path.join(frames_path, "*.jpg")), key=lambda x: x[6:])

        for frame in frames_lst:
            img = cv2.imread(frame)
            if transform:
                img = transform(img)
                img = img.squeeze_(0)  # remove fake batch dimension
            else:
                img = torch.tensor(img)
            X.append(img)
        X = torch.stack(X, dim=0)

        return {'images': X.float()}

    def _read_audio(self, idx, transform):
        video_path = self.video_list[idx]
        audio_path = os.path.join(video_path, "audios/featuresMFCC.npy")

        if transform:
            wav_file = os.path.join(video_path, "audios/audio.wav")
            X, sample_rate = librosa.load(wav_file, duration=2.45, sr=22050 * 2, offset=0.5)
            X = transform(X)
            sample_rate = np.array(sample_rate)
            features = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)
        else:
            features = np.load(audio_path)

        return {'mfcc': torch.tensor(features)}

    def __getitem__(self, idx):
        gt = parse_gt(os.path.basename(self.video_list[idx]))
        y = torch.LongTensor([gt["emotion"] - 1])  # id starts in 1

        sample = {'emotion': y}

        if "video" in self.modality:
            if isinstance(self.transform, dict) and "image_transform" in self.transform:
                image_transform = self.transform["image_transform"]
            else:
                image_transform = self.transform
            image_feature = self._read_images(idx, image_transform)
            sample.update(image_feature)
        if "audio" in self.modality:
            if isinstance(self.transform, dict) and "audio_transform" in self.transform:
                audio_transform = self.transform["audio_transform"]
            else:
                audio_transform = self.transform
            audio_features = self._read_audio(idx, audio_transform)
            sample.update(audio_features)

        return sample
