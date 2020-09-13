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

import os
import cv2
import numpy as np
from moviepy.editor import VideoFileClip


class VideoProcessor(object):
    def __init__(self, video_path, landmark_path, output_folder, extract_audio):
        # path of the video file
        self.video_path = video_path
        self.landmarks_path = landmark_path
        self.extract_audio = extract_audio
        self.frames_folder = os.path.join(output_folder, "frames")
        self.audios_folder = os.path.join(output_folder, "audios")

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        if not os.path.exists(self.audios_folder):
            os.mkdir(self.audios_folder)
        if not os.path.exists(self.frames_folder):
            os.mkdir(self.frames_folder)

    def preprocess(self, seq_len=30, target_resolution=(224, 224)):
        """
        extract frames and audio from the video,
        store the cropped frames and audio file in the output folders
        seq_len: how many frames will be extracted from the video.
                  Considering all videos from this dataset have similar duration
                  video_duration = seq_len / fps
        target_resolution: (desired_height, desired_width) of the facial frame extracted
        """
        video = VideoFileClip(self.video_path, audio=self.extract_audio, target_resolution=target_resolution)
        if self.extract_audio:
            video.audio.write_audiofile(os.path.join(self.audios_folder, "audio.wav"))

        times = list(np.arange(0, video.duration, video.duration/seq_len))
        if len(times) < seq_len:
            times.append(video.duration)
        times = times[:seq_len]

        # extract 2D points from csv
        data = np.genfromtxt(self.landmarks_path, delimiter=',')[1:]
        lm_times = [int(np.ceil(t)) for t in list(np.arange(0, len(data), len(data) / seq_len))]
        if len(lm_times) < seq_len:
            lm_times.append(len(data) - 1)
        lm_times = lm_times[:seq_len]
        index_x = (298, 366)
        index_y = (366, 433)
        landmarks_2d_x = [data[t, index_x[0] - 1:index_x[1] - 1] * (1 / 1280) for t in lm_times]
        landmarks_2d_y = [data[t, index_y[0] - 1:index_y[1]] * (1 / 720) for t in lm_times]

        for i, t in enumerate(times):
            img = cv2.cvtColor(video.get_frame(t), cv2.COLOR_BGR2RGB)
            # extract roi from landmarks and crop
            xs, ys = landmarks_2d_x[i], landmarks_2d_y[i]
            bottom = int(max(ys) * img.shape[0])
            right = int(max(xs) * img.shape[1])
            top = int(min(ys) * img.shape[0])
            left = int(min(xs) * img.shape[1])

            cropped = cv2.resize(img[top:bottom, left:right, :], target_resolution)
            cv2.imwrite(os.path.join(self.frames_folder, "frame_{0:.2f}.jpg".format(t)), cropped)

        print("Video duration {} seconds. Extracted {} frames".format(video.duration, len(times)))
