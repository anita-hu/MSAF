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
import face_recognition
from moviepy.editor import VideoFileClip


class VideoProcessor(object):
    def __init__(self, video_path, output_folder, extract_audio):
        # path of the video file
        self.video_path = video_path
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
        if len(times) < 30:
            times.append(video.duration)
        times = times[:30]

        for t in times:
            img = cv2.cvtColor(video.get_frame(t), cv2.COLOR_BGR2RGB)
            # extract landmarks and crop
            landmark = face_recognition.face_landmarks(img)
            points = []
            for each_key in landmark:
                for each_point in landmark[each_key]:
                    points.append(each_point)

            up = int(max(points, key=lambda p: p[1])[1])
            right = int(max(points, key=lambda p: p[0])[0])
            bottom = int(min(points, key=lambda p: p[1])[1])
            left = int(min(points, key=lambda p: p[0])[0])
            cropped = cv2.resize(img[bottom:up, left:right, :], target_resolution)

            cv2.imwrite(os.path.join(self.frames_folder, "frame_{0:.2f}.jpg".format(t)), cropped)

        print("Video duration {} seconds. Extracted {} frames".format(str(video.duration), str(len(times))))
