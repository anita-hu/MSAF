import torch
import numpy as np
from sklearn.metrics import confusion_matrix


# calculate the accuracy that the true label is in y_true's top k rank
# k: list of int. each <= num_cls
# y_pred: np array of probablities. (batch_size * cls_num) (output of softmax)
# y_true: batch_size * 1.
# return: list of acc given list of k
def accuracy_topk(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train(get_X, log_interval, model, device, train_loader, optimizer, loss_func, metric_topk, epoch):
    # set model as training mode
    model.train()

    losses = []
    scores = [[]] * len(metric_topk)
    N_count = 0  # counting total trained sample in one epoch

    for batch_idx, sample in enumerate(train_loader):
        # distribute data to device
        X, n = get_X(device, sample)
        y = sample["emotion"].to(device)  # dim = (batch, 1)
        with torch.autograd.detect_anomaly():
            output = model(X)

            N_count += n
            optimizer.zero_grad()

            loss = loss_func(output, y)
            losses.append(loss.item())

            step_score = accuracy_topk(output, y, topk=metric_topk)
            for i, ss in enumerate(step_score):
                scores[i].append(int(ss))

            loss.backward()
            optimizer.step()

        # show information
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item()))
            for i, each_k in enumerate(metric_topk):
                print("Top {} accuracy: {:.2f}%".format(each_k, float(step_score[i])))

    return losses, scores


def validation(get_X, model, device, loss_func, val_loader, metric_topk, show_cm=False):
    # set model as testing mode
    model.eval()

    test_loss = []
    all_y = []
    all_y_pred = []

    with torch.no_grad():
        for sample in val_loader:
            # distribute data to device
            X, _ = get_X(device, sample)
            y = sample["emotion"].to(device)
            output = model(X)

            loss = loss_func(output, y)
            test_loss.append(loss.item())  # sum up batch loss

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(output)

    test_loss = np.mean(test_loss)

    # compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    test_score = [float(t_acc) for t_acc in accuracy_topk(all_y_pred, all_y, topk=metric_topk)]

    # show information
    print('\nTest set ({:d} samples): Average loss: {:.4f}'.format(len(all_y), test_loss))
    for i, each_k in enumerate(metric_topk):
        print("Top {} accuracy: {:.2f}%".format(each_k, test_score[i]))
    print("\n")

    if show_cm:
        cm = confusion_matrix(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())
        print("Confusion matrix")
        print(cm)

    return test_loss, test_score
