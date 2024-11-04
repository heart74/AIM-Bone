"""metrics.py - Metrics for model evaluation 
"""

import numpy as np
from sklearn import metrics


def find_best_threshold(fpr, tpr):
    diff = np.array(tpr) - np.array(1.-fpr)
    diff = diff**2
    index = np.argmin(diff)
    return index


def find_nearest_fpr(fpr, fpr_target):
    diff = np.array(fpr) - fpr_target
    diff = diff**2
    index = np.argmin(diff)
    return index

def evaluate_ROC(y_label, y_score, pos_label=1):
    fpr, tpr, threshold = metrics.roc_curve(y_label, y_score, pos_label=pos_label)
    auc = metrics.auc(fpr, tpr)
    index = find_best_threshold(fpr, tpr)
    acc_0 = tpr[index]
    acc_1 = 1. - fpr[index]
    ap = metrics.average_precision_score(y_label, y_score, pos_label=1)
    # import pdb; pdb.set_trace()
    # print(fpr)
    # print(tpr)
    # print(threshold)
    # print(index)
    # print(auc)
    # print(acc_0)
    # print(acc_1)

    return acc_0, acc_1, threshold[index], auc, ap


def evaluate_at_fpr(y_label, y_score, pos_label=1, fpr_target=0.1):
    fpr, tpr, threshold = metrics.roc_curve(y_label, y_score, pos_label=pos_label)
    auc = metrics.auc(fpr, tpr)
    index = find_nearest_fpr(fpr, fpr_target)
    tpr_target = tpr[index]
    fpr_nearest = fpr[index]
    
    # print(fpr)
    # print(tpr)
    # print(threshold)
    # print(index)
    # print(auc)
    # print(acc_0)
    # print(acc_1)

    return tpr_target, fpr_nearest, threshold[index], auc


def get_confusion_matrix(y_true, y_pred):
    return metrics.confusion_matrix(y_true, y_pred, labels=[0,1])


def score_thre_confusionmatrix(label, score, thre):
    """
    from score and threshold, return confusion matrix
    :param label: array, shape = [n_samples]
    :param score: array, shape = [n_samples]
    :param thre: scalar
    :return: array, shape = [4, 2]
    """
    label = np.array(label)
    score = np.array(score)
    preds = score > thre
    confusion_matrix = get_confusion_matrix(label, preds)
    return confusion_matrix



if __name__ == '__main__':
    y_label = [0, 0, 1, 1]
    y_score = [0.1, 0.4, 0.35, 0.8]
    evaluate_ROC(y_label, y_score)