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
import datetime
import argparse
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data
import torchvision.transforms as transforms
from networks import *
from ravdess import RAVDESSDataset
from main_utils import train, validation


# Parameters
modalities = ["video", "audio"]

# setting seed
seed = 1234
torch.autograd.set_detect_anomaly(True)
np.random.seed(seed)
torch.manual_seed(seed)  # cpu
torch.cuda.manual_seed_all(seed)  # gpu
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# define model input
def get_X(device, sample):
    ret = []
    if "video" in modalities:
        images = sample["images"].to(device)
        images = images.permute(0, 2, 1, 3, 4)  # swap to be (N, C, D, H, W)
        ret.append(images)
    if "audio" in modalities:
        mfcc = sample["mfcc"].to(device)
        ret.append(mfcc)
    n = ret[0].size(0)
    return ret, n


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, help='dataset directory', default='RAVDESS/preprocessed')
    parser.add_argument('--k_fold', type=int, help='k for k fold cross validation', default=6)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.001)
    parser.add_argument('--batch_size', type=int, help='batch size', default=4)
    parser.add_argument('--num_workers', type=int, help='num workers', default=4)
    parser.add_argument('--epochs', type=int, help='train epochs', default=70)
    parser.add_argument('--checkpointdir', type=str, help='directory to save/read weights', default='checkpoints')
    parser.add_argument('--no_verbose', action='store_true', default=False, help='turn off verbose for training')
    parser.add_argument('--log_interval', type=int, help='interval for displaying training info if verbose', default=10)
    parser.add_argument('--no_save', action='store_true', default=False, help='set to not save model weights')
    parser.add_argument('--train', action='store_true', default=False, help='training')

    args = parser.parse_args()

    print("The configuration of this run is:")
    print(args, end='\n\n')

    # Detect devices
    use_cuda = torch.cuda.is_available()  # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU

    # Data loading parameters
    params = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': args.num_workers, 'pin_memory': True} \
        if use_cuda else {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': args.num_workers}

    # define data transform
    train_transform = {
        "image_transform": transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.4)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        "audio_transform": None
    }

    val_transform = {
        "image_transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        "audio_transform": None
    }

    # loss function
    loss_func = torch.nn.CrossEntropyLoss()

    # top k categorical accuracy: k
    training_topk = (1,)
    val_topk = (1, 2, 4,)

    all_folder = sorted(list(glob.glob(os.path.join(args.datadir, "Actor*"))))

    # train mode or eval mode
    if args.train:
        # kfold training
        s = int(len(all_folder) / args.k_fold)
        top_scores = []
        for i in range(args.k_fold):
            val_fold = all_folder[i * s: i * s + s]
            train_fold = all_folder[:i * s] + all_folder[i * s + s:]
            training_set = RAVDESSDataset(train_fold, modality=modalities, transform=train_transform)
            training_loader = data.DataLoader(training_set, **params)
            val_set = RAVDESSDataset(val_fold, modality=modalities, transform=val_transform)
            val_loader = data.DataLoader(val_set, **params)

            print("Fold " + str(i + 1))
            print("Train fold: ")
            print([os.path.basename(act) for act in train_fold])
            print("val fold: ")
            print([os.path.basename(act) for act in val_fold])

            # record training process
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = os.path.join(args.checkpointdir, 'logs/fold{}_{}'.format(i + 1, current_time))
            writer = SummaryWriter(log_dir=train_log_dir)

            # define model
            model_param = {}
            if "video" in modalities:
                video_model = resnet50(
                    num_classes=8,
                    shortcut_type='B',
                    cardinality=32,
                    sample_size=224,
                    sample_duration=30)
                video_model_path = os.path.join(args.checkpointdir, "resnext50/fold_{}_resnext50_best.pth".format(i+1))
                video_model_checkpoint = torch.load(video_model_path) if use_cuda else \
                    torch.load(video_model_path, map_location=torch.device('cpu'))
                video_model.load_state_dict(video_model_checkpoint)
                model_param.update(
                    {"video": {
                        "model": video_model,
                        "id": modalities.index("video")
                    }})

            if "audio" in modalities:
                audio_model = MFCCNet()
                audio_model_path = os.path.join(args.checkpointdir, "mfccNet/fold_{}_mfccNet_best.pth".format(i + 1))
                audio_model_checkpoint = torch.load(audio_model_path) if use_cuda else \
                    torch.load(audio_model_path, map_location=torch.device('cpu'))
                audio_model.load_state_dict(audio_model_checkpoint)
                model_param.update(
                    {"audio": {
                        "model": audio_model,
                        "id": modalities.index("audio")
                    }})

            multimodal_model = MSAFNet(model_param)
            multimodal_model.to(device)

            # Adam parameters
            num_parameters = multimodal_model.parameters()
            optimizer = torch.optim.Adam(num_parameters, lr=args.lr)

            # keep track of epoch test scores
            test = []
            best_acc_1 = 0
            for epoch in range(args.epochs):
                # train, test model
                train_losses, train_scores = train(get_X, args.log_interval, multimodal_model, device, training_loader,
                                                   optimizer, loss_func, training_topk, epoch)
                epoch_test_loss, epoch_test_score = validation(get_X, multimodal_model, device, loss_func, val_loader,
                                                               val_topk)

                if not args.no_save and epoch_test_score[0] > best_acc_1:
                    best_acc_1 = epoch_test_score[0]
                    torch.save(multimodal_model.state_dict(),
                               os.path.join(args.checkpointdir, 'fold_{}_msaf_ravdess_best.pth'.format(i + 1)))
                    print("Epoch {} model saved!".format(epoch + 1))

                # save results
                writer.add_scalar('Loss/train', np.mean(train_losses), epoch)
                writer.add_scalar('Loss/test', epoch_test_loss, epoch)
                for each_k, k_score in zip(training_topk, train_scores):
                    writer.add_scalar('Scores_top{}/train'.format(each_k), np.mean(k_score), epoch)
                for each_k, k_score in zip(val_topk, epoch_test_score):
                    writer.add_scalar('Scores_top{}/test'.format(each_k), np.mean(k_score), epoch)
                test.append(epoch_test_score)
                writer.flush()
            test = np.array(test)
            for j, each_k in enumerate(val_topk):
                max_idx = np.argmax(test[:, j])
                print('Best top {} test score {:.2f}% at epoch {}'.format(each_k, test[:, j][max_idx], max_idx + 1))
            top_scores.append(test[:, 0].max())
        print("Scores for each fold: ")
        print(top_scores)
        print("Averaged score for {} fold training: {:.2f}%".format(args.k_fold, sum(top_scores) / len(top_scores)))

    else:
        # kfold eval
        s = int(len(all_folder) / args.k_fold)
        top_scores = []
        for i in range(args.k_fold):
            val_fold = all_folder[i * s: i * s + s]
            train_fold = all_folder[:i * s] + all_folder[i * s + s:]
            val_set = RAVDESSDataset(val_fold, modality=modalities, transform=val_transform)
            val_loader = data.DataLoader(val_set, **params)

            print("Fold " + str(i + 1))
            print("val fold: ")
            print([os.path.basename(act) for act in val_fold])

            # define model
            model_param = {}
            if "video" in modalities:
                video_model = resnet50(
                    num_classes=8,
                    shortcut_type='B',
                    cardinality=32,
                    sample_size=224,
                    sample_duration=30)
                model_param.update(
                    {"video": {
                        "model": video_model,
                        "id": modalities.index("video")
                    }})

            if "audio" in modalities:
                audio_model = MFCCNet()
                model_param.update(
                    {"audio": {
                        "model": audio_model,
                        "id": modalities.index("audio")
                    }})

            model_path = os.path.join(args.checkpointdir, 'fold_{}_msaf_ravdess_best.pth'.format(i + 1))
            model = MSAFNet(model_param)
            checkpoint = torch.load(model_path) if use_cuda else torch.load(model_path,
                                                                            map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint)
            model.to(device)
            epoch_test_loss, epoch_test_score = validation(get_X, model, device, loss_func, val_loader, val_topk)
            top_scores.append(epoch_test_score[0])

        print("Scores for each fold: ")
        print(top_scores)
        print("Averaged score for {} fold: {:.2f}%".format(args.k_fold, sum(top_scores) / len(top_scores)))
