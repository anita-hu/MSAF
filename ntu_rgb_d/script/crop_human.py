import sys
import os
import tqdm
import glob
sys.path.append('pytorch-YOLOv4')
from models import Yolov4
from tool.utils import post_processing, load_class_names, plot_boxes_cv2
from tool.torch_utils import do_detect

import torch
import numpy as np
import cv2
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Human cropping.')
    parser.add_argument('--yolo_weight_path', type=str, help='Yolov4 weight', default='/home/lang/Downloads/yolov4.pth')
    parser.add_argument('--video_dir', type=str, help='NTU dataset folder', default='/home/lang/Data/project/msaf/NTU/')
    parser.add_argument('--conf_thresh', type=float, help='Yolov4 confidence threshold', default=0.7)
    parser.add_argument('--vid_len', type=int, help='video sequence length', default=24)
    parser.add_argument('--step', type=int, help='step. Should be divisible by vid_len', default=8)
    return parser.parse_args()

def load_video(path, vid_len=24):
    cap = cv2.VideoCapture(path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    heigth = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Init the numpy array
    video = np.zeros((vid_len, width, heigth, 3)).astype(np.float32)
    taken = np.linspace(0, num_frames, vid_len).astype(int)

    np_idx = 0
    for fr_idx in range(num_frames):
        ret, frame = cap.read()

        if cap.isOpened() and fr_idx in taken:
            video[np_idx, :, :, :] = frame.astype(np.float32)
            np_idx += 1
    cap.release()
    return video


# for the boxes detected for each image
# we filter out only human's
# generate a minimum square normalized bbox (x1 y1 x2 y2)  that contains
# all humans in the image, or non-square if one side goes beyond
# (so that we don't have to resize which distorts images)
# if no human detected in the image, return original image (i.e. [0 1 0 1])
def process_box(boxes):
    human_boxes = [b for b in boxes if b[-1] == 0]
    if len(human_boxes) == 0:  # if no human detected
        return [0, 0, 1, 1]
    min_x1 = min([b[0] for b in human_boxes])
    max_x2 = max([b[2] for b in human_boxes])
    min_y1 = min([b[1] for b in human_boxes])
    max_y2 = max([b[3] for b in human_boxes])
    side_diff_x = max_x2 - min_x1
    side_diff_y = max_y2 - min_y1
    ctr_x = min_x1 + side_diff_x / 2
    ctr_y = min_y1 + side_diff_y / 2
    if side_diff_y > side_diff_x:
        max_x2 = min(ctr_x + side_diff_y / 2, 1)
        min_x1 = max(0, ctr_x - side_diff_y / 2)
    else:
        max_y2 = min(ctr_y + side_diff_x / 2, 1)
        min_y1 = max(0, ctr_y - side_diff_x / 2)

    return [min_x1, min_y1, max_x2, max_y2]

# process a batch of boxes
def process_boxes(boxes):
    return [process_box(b) for b in boxes]

# e.g.
# boxes = [[0.51835126, 0.26324576, 0.5738782, 0.69262147, 0.98613703, 0.98613703, 0],
#         [0.41210455, 0.25013545, 0.47323123, 0.71688616, 0.964836, 0.964836, 0]]
# process_boxes(boxes) -->
# [0.25961601999999995, 0.25013545, 0.7263667300000001, 0.71688616]

# visualizing bbox for a video
def visualization():
    # first test path is two person, second is one person
    # test_path = "/home/lang/Data/project/msaf/NTU/nturgbd_rgb/avi_256x256_30/S001C001P001R001A055_rgb.avi"
    # test_path = "/home/lang/Data/project/msaf/NTU/nturgbd_rgb/avi_256x256_30/S001C001P001R001A001_rgb.avi"
    test_video = "/home/lang/Data/project/msaf/NTU/nturgbd_rgb/avi_256x256_30/S014C002P025R001A011_rgb.avi"
    test_output = "/home/lang/Data/project/msaf/NTU/nturgbd_rgb/human_crop/S014C002P025R001A011_rgb.npy"
    class_names = load_class_names("pytorch-YOLOv4/data/coco.names")
    video = load_video(test_video, 24)
    boxes = np.load(test_output)
    for b, img in zip(boxes, video):
        print(b)
        vis_img = plot_boxes_cv2(img, [list(b)+[1,1,0]], 'predictions.jpg', class_names)
        cv2.imshow("img", vis_img/255)
        cv2.waitKey()


# visualization()


if __name__ == "__main__":
    args = parse_args()
    yolo_weight_path = args.yolo_weight_path
    video_dir = args.video_dir
    conf_thresh = args.conf_thresh
    vid_len = args.vid_len
    step = args.step
    assert vid_len % step == 0

    video_dir = os.path.join(video_dir, "nturgbd_rgb")
    output_dir = os.path.join(video_dir, "human_crop")
    if not os.path.exists(output_dir):
        print("Creating folder to store crop")
        os.mkdir(output_dir)

    # yolov4
    # init yolov4
    print("Initializing yolov4...")
    model = Yolov4(yolov4conv137weight=None, n_classes=80, inference=True)
    pretrained_dict = torch.load(yolo_weight_path, map_location=torch.device('cuda'))
    model.load_state_dict(pretrained_dict)
    use_cuda = True
    if use_cuda:
        model.cuda()

    for each_video in tqdm.tqdm(glob.glob(os.path.join(video_dir, "avi_256x256_30/*.avi"))):
        video = load_video(each_video, vid_len)
        video_name = os.path.basename(each_video).strip(".avi")

        sized = np.array([cv2.resize(img, (416, 416)) for img in video])
        sized = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in video])
        video_boxes = []
        for i in range(len(sized) // step):
            img = sized[i*step:(i+1)*step]
            boxes = do_detect(model, img, conf_thresh, 0.6, use_cuda)
            fixed_boxes = process_boxes(boxes)
            video_boxes += fixed_boxes
            # this is for visualizing
            # boxes[0] = [fixed_boxes + [1, 1, 0]]
            # class_names = load_class_names("pytorch-YOLOv4/data/coco.names")
            # plot_boxes_cv2(img, boxes[0], 'predictions.jpg', class_names)
        #np.save(os.path.join(output_dir, video_name+".npy"), np.array(video_boxes))
