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

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from scipy.stats import pearsonr


def convert2class(out):
    if out < 0:
        return -1
    elif out > 0:
        return 1
    else:
        return 0


def convert7class(out):
    return np.clip(np.rint(out), -3, 3)


def acc_2_f_score(y_pred, y_true):
    pred_cls = []
    true_cls = []
    for y_p, y_t in zip(y_pred, y_true):
        if y_t == 0:
            continue
        pred_cls.append(convert2class(y_p))
        true_cls.append(convert2class(y_t))

    cm = confusion_matrix(true_cls, pred_cls, labels=[-1, 1])
    acc = np.sum(np.diag(cm)) / np.sum(cm)
    f_score = f1_score(true_cls, pred_cls, labels=[-1, 1], average='weighted')

    return f_score, acc, cm


def acc_7(y_pred, y_true):
    pred_cls = []
    true_cls = []
    for y_p, y_t in zip(y_pred, y_true):
        pred_cls.append(convert7class(y_p))
        true_cls.append(convert7class(y_t))

    cm = confusion_matrix(true_cls, pred_cls, labels=[-3, -2, -1, 0, 1, 2, 3])
    acc = np.sum(np.diag(cm)) / np.sum(cm)

    return acc, cm


def train(get_X, log_interval, model, device, train_loader, optimizer, loss_func, epoch, verbose=True):
    # set model as training mode
    model.train()

    losses = []
    all_y_true = []
    all_y_pred = []

    N_count = 0  # counting total trained sample in one epoch

    for batch_idx, sample in enumerate(train_loader):
        # distribute data to device
        X, n = get_X(device, sample)
        y_true = sample['label'].to(device)  # .view(-1, )
        output = model(X)

        N_count += n
        optimizer.zero_grad()

        loss = loss_func(output, y_true)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

        all_y_true.append(y_true.cpu().numpy())
        all_y_pred.append(output.detach().cpu().numpy())

        # show information
        if (batch_idx + 1) % log_interval == 0 and verbose:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item()))

    all_y_pred = np.concatenate(all_y_pred)
    all_y_true = np.concatenate(all_y_true)
    acc7, _ = acc_7(all_y_pred, all_y_true)
    f1, acc2, _ = acc_2_f_score(all_y_pred, all_y_true)
    corr, _ = pearsonr(all_y_pred.squeeze(), all_y_true.squeeze())

    scores = [acc7*100, acc2*100, f1*100, corr]
    print('Train Epoch: {} Acc 7: {:.2f}%; Acc 2: {:.2f}%; F1 score: {:.2f}%; Corr: {:.4f}\n'.format(epoch + 1, *scores))

    return np.mean(losses), scores


def validation(get_X, model, device, loss_func, val_loader, print_cm=False):
    # set model as testing mode
    model.eval()

    test_loss = []
    all_y_true = []
    all_y_pred = []
    samples = 0

    with torch.no_grad():
        for sample in val_loader:
            # distribute data to device
            X, _ = get_X(device, sample)
            y_true = sample['label'].to(device)
            output = model(X)

            loss = loss_func(output, y_true)
            test_loss.append(loss.item())  # sum up batch loss
            all_y_true.append(y_true.cpu().numpy())
            all_y_pred.append(output.cpu().numpy())

            samples += len(y_true)

    all_y_pred = np.concatenate(all_y_pred)
    all_y_true = np.concatenate(all_y_true)
    acc7, cm7 = acc_7(all_y_pred, all_y_true)
    if print_cm:
        print("7 class sentiment confusion matrix")
        print(cm7)
    corr, _ = pearsonr(all_y_pred.squeeze(), all_y_true.squeeze())
    f1, acc2, cm2 = acc_2_f_score(all_y_pred, all_y_true)

    test_score = [acc7*100, acc2*100, f1*100, corr]
    test_loss = np.mean(test_loss)

    # print info
    print('\nTest set ({:d} samples): Average MAE loss: {:.4f}'.format(samples, test_loss))
    print('Acc 7: {:.2f}%; Acc 2: {:.2f}%; F1 score: {:.2f}%; Corr: {:.4f}\n'.format(*test_score))

    return test_loss, test_score
