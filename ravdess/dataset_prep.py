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

import numpy as np
import glob
import os
import argparse
from tqdm import tqdm
import librosa
from video_utils import VideoProcessor


def preprocess_video(dataset_folder):
    output_folder = os.path.join(dataset_folder, "preprocessed")
    landmark_folder = os.path.join(dataset_folder, "landmarks")

    for each_actor in glob.glob(os.path.join(dataset_folder, "Actor*")):
        actor_name = os.path.basename(each_actor)
        output_actor_folder = os.path.join(output_folder, actor_name)

        if not os.path.exists(output_actor_folder):
            os.makedirs(output_actor_folder)

        for each_video in glob.glob(os.path.join(each_actor, "*.mp4")):
            name = each_video.split("/")[-1].split("-")[0]
            if int(name) == 2:
                continue
            video_name = os.path.basename(each_video)
            landmark_path = os.path.join(landmark_folder, video_name[:-4] + '.csv')
            frames_folder = os.path.join(output_actor_folder, video_name)
            if not os.path.exists(frames_folder):
                os.mkdir(frames_folder)
            video_processor = VideoProcessor(video_path=each_video, landmark_path=landmark_path,
                                             output_folder=frames_folder, extract_audio=True)
            video_processor.preprocess(seq_len=30, target_resolution=(224, 224))


def get_audio_paths(path):
    audio_files = []
    actors = os.listdir(path)
    for a in actors:
        path_to_folders = os.path.join(path, a)
        folders = os.listdir(path_to_folders)
        audio_files += [os.path.join(path_to_folders, p) for p in folders]
    return audio_files


def mfcc_features(dataset_folder):
    path = os.path.join(dataset_folder, "preprocessed")
    files = get_audio_paths(path)
    for f in tqdm(files):
        X, sample_rate = librosa.load(os.path.join(f, 'audios/audio.wav'),
                                      duration=2.45, sr=22050 * 2, offset=0.5)
        sample_rate = np.array(sample_rate)
        mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)
        np.save(os.path.join(f, 'audios/featuresMFCC.npy'), mfccs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, help='dataset directory', default='RAVDESS')
    args = parser.parse_args()

    print("Processing videos...")
    preprocess_video(args.datadir)

    print("Generating MFCC features...")
    mfcc_features(args.datadir)

    print("Preprocessed dataset located in ", os.path.join(args.datadir, 'preprocessed'))
