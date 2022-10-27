"""
Anh Viet Pham
Doc1: 20 Popular Machine Learning Metrics. Part 1: Classification & Regression Evaluation Metrics:
==> https://towardsdatascience.com/20-popular-machine-learning-metrics-part-1-classification-regression-evaluation-metrics-1ca3e282a2ce
Doc2: Confusion Matrix, Accuracy, Precision, Recall, F1 Score
==> https://medium.com/analytics-vidhya/confusion-matrix-accuracy-precision-recall-f1-score-ade299cf63cd
"""

import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle

y_true = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
y_pred = np.array([0, 1, 0, 2, 1, 1, 0, 2, 1, 2])


def acc(y_true, y_pre):
    correct = np.sum(y_true == y_pred)
    print(f'Correct: {correct}')
    return float(correct / y_true.shape[0])


def confusion_matrix(y_true, y_pre):
    N = np.unique(y_true).shape[0]
    print(np.unique(y_true))
    cm = np.zeros((N, N))
    print(f'N {N}')
    print(f'cm {cm}')
    for i in range(y_true.shape[0]):
        cm[y_true[i], y_pre[i]] += 1
    return cm


def accuracy_avp():
    print(f'y_true: {y_true}, shape: {y_true.shape}')
    print(f'y_pre: {y_pred}, shape: {y_pred.shape}')
    print(f'Accuracy: {acc(y_true, y_pred)}')

    print("=" * 50)
    cnf = confusion_matrix(y_true, y_pred)
    print(cnf)
    print(f'Accuracy Confusion Matrix: {np.diagonal(cnf).sum() / cnf.sum()}')

    print(f'Axis = 1: {cnf.sum(axis=1, keepdims=True)}')
    print(f'Axis = 0: {cnf.sum(axis=0, keepdims=True)}')
    print(f'Axis = 1: {cnf.sum(axis=1)}')
    print(f'Axis = 0: {cnf.sum(axis=0)}')

    print("=" * 50)
    print("Confusion Matrix Normalization")
    normalization_cnf = cnf / cnf.sum(axis=1, keepdims=True)
    print(normalization_cnf)


if __name__ == '__main__':
    print("=" * 50)
    print("True/ False Positive/ Negative")
    n0, n1 = 20, 30
    score0 = np.random.rand(n0) / 2
    print(f'Score 0: {score0}')
    label0 = np.zeros(n0, dtype=int)
    print(f'Label 0: {label0}')
    score1 = np.random.rand(n1) / 2 + .2
    print(f'Score 1: {score1}')
    label1 = np.ones(n1, dtype=int)
    print(f'Label 1: {label1}')
    scores = np.concatenate((score0, score1))
    y_true = np.concatenate((label0, label1))

    print('True Label:')
    print(y_true)
    print('Scores:')
    print(scores)

    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)
    print('Thresholds:')
    print(thresholds)
    print("=" * 50)
    print("False Positive Rate")
    print(fpr)
    print("=" * 50)
    print("True Positive Rate")
    print(tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
