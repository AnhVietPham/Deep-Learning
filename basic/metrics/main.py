"""
Anh Viet Pham
Doc1: 20 Popular Machine Learning Metrics. Part 1: Classification & Regression Evaluation Metrics:
==> https://towardsdatascience.com/20-popular-machine-learning-metrics-part-1-classification-regression-evaluation-metrics-1ca3e282a2ce
Doc2: Confusion Matrix, Accuracy, Precision, Recall, F1 Score
==> https://medium.com/analytics-vidhya/confusion-matrix-accuracy-precision-recall-f1-score-ade299cf63cd
"""

import numpy as np


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


if __name__ == '__main__':
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
    y_pred = np.array([0, 1, 0, 2, 1, 1, 0, 2, 1, 2])
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

    print("=" * 50)
    print("True/ False Positive/ Negative")
