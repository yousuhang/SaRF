from collections import namedtuple

import h5py
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np

Metrics = namedtuple("Metrics", ["AUC", "ACC", "F1"])


class Evaluator:
    def __init__(self, split, data_path, test_vol = False, slice_number = 7):
        self.data_path = data_path
        self.split = split
        if self.split == 'val':
            self.label_name = 'Valid_Labels'
        elif self.split == 'test':
            self.label_name = 'Test_Labels'
        else:
            raise ValueError
        self.test_vol = test_vol
        self.slice_number = slice_number
        with h5py.File(self.data_path, 'r') as hf:
            self.labels = np.array(hf[self.label_name])
            if self.test_vol:
                self.labels = np.array(hf[self.label_name])[::self.slice_number]
        self.labels_onehot = np.zeros((self.labels.size, self.labels.max() + 1))
        self.labels_onehot[np.arange(self.labels.size), self.labels] = 1

    def evaluate(self, y_score, task = 'multi-label, binary-class', selected_labels=None):
        assert y_score.shape[0] == self.labels.shape[0]


        auc = getAUC(self.labels_onehot, y_score, task, selected_labels=selected_labels)
        acc = getACC(self.labels_onehot, y_score, task, selected_labels=selected_labels)
        f1 = getF1(self.labels_onehot, y_score, task, selected_labels=selected_labels)
        metrics = Metrics(auc, acc, f1)

        return metrics


def getAUC(y_true, y_score, task, selected_labels = None):
    '''AUC metric.
    :param y_true: the ground truth labels, shape: (n_samples, n_labels) or (n_samples,) if n_labels==1
    :param y_score: the predicted score of each class,
    shape: (n_samples, n_labels) or (n_samples, n_classes) or (n_samples,) if n_labels==1 or n_classes==1
    :param task: the task of current dataset
    '''
    y_true = y_true.squeeze()
    y_score = y_score.squeeze()

    if task == 'multi-label, binary-class':
        auc = 0
        for i in range(y_score.shape[1]):
            label_auc = roc_auc_score(y_true[:, i], y_score[:, i])
            auc += label_auc
        ret = auc / y_score.shape[1]
    elif task == 'multi-label, binary-class, selected classes':
        auc = 0
        for i in selected_labels:
            label_auc = roc_auc_score(y_true[:, i], y_score[:, i])
            auc += label_auc
        ret = auc / len(selected_labels)
    elif task == 'binary-class':
        if y_score.ndim == 2:
            y_score = y_score[:, -1]
        else:
            assert y_score.ndim == 1
        ret = roc_auc_score(y_true, y_score)
    else:
        auc = 0
        for i in range(y_score.shape[1]):
            y_true_binary = (y_true == i).astype(float)
            y_score_binary = y_score[:, i]
            auc += roc_auc_score(y_true_binary, y_score_binary)
        ret = auc / y_score.shape[1]

    return ret


def getACC(y_true, y_score, task, threshold=0.5, selected_labels = None):
    '''Accuracy metric.
    :param y_true: the ground truth labels, shape: (n_samples, n_labels) or (n_samples,) if n_labels==1
    :param y_score: the predicted score of each class,
    shape: (n_samples, n_labels) or (n_samples, n_classes) or (n_samples,) if n_labels==1 or n_classes==1
    :param task: the task of current dataset
    :param threshold: the threshold for multilabel and binary-class tasks
    '''
    y_true = y_true.squeeze()
    y_score = y_score.squeeze()

    if task == 'multi-label, binary-class':
        y_pre = y_score > threshold
        acc = 0
        for label in range(y_true.shape[1]):
            label_acc = accuracy_score(y_true[:, label], y_pre[:, label])
            acc += label_acc
        ret = acc / y_true.shape[1]
    elif task == 'multi-label, binary-class, selected classes':
        y_pre = y_score > threshold
        acc = 0
        for label in selected_labels:
            label_acc = accuracy_score(y_true[:, label], y_pre[:, label])
            acc += label_acc
        ret = acc / len(selected_labels)
    elif task == 'binary-class':
        if y_score.ndim == 2:
            y_score = y_score[:, -1]
        else:
            assert y_score.ndim == 1
        ret = accuracy_score(y_true, y_score > threshold)
    else:
        ret = accuracy_score(y_true, np.argmax(y_score, axis=-1))

    return ret

def getF1(y_true, y_score, task, threshold = 0.5, selected_labels = None):
    '''F1 score metric.
    :param y_true: the ground truth labels, shape: (n_samples, n_labels) or (n_samples,) if n_labels==1
    :param y_score: the predicted score of each class,
    shape: (n_samples, n_labels) or (n_samples, n_classes) or (n_samples,) if n_labels==1 or n_classes==1
    :param task: the task of current dataset
    # :param threshold: the threshold for multilabel and binary-class tasks
    '''
    y_true = y_true.squeeze()
    y_score = y_score.squeeze()

    if task == 'multi-label, binary-class':
        y_pre = y_score > threshold
        f1 = 0
        for label in range(y_true.shape[1]):
            label_f1 = f1_score(y_true[:, label], y_pre[:, label])
            f1 += label_f1
        ret = f1 / y_true.shape[1]
    elif task == 'multi-label, binary-class, selected classes':
        y_pre = y_score > threshold
        f1 = 0
        for label in selected_labels:
            label_f1 = accuracy_score(y_true[:, label], y_pre[:, label])
            f1 += label_f1
        ret = f1 / len(selected_labels)
    elif task == 'binary-class':
        pass
    else:
        ret = accuracy_score(y_true, np.argmax(y_score, axis=-1))

    return ret