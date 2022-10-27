"""
Anh Viet Pham
https://towardsdatascience.com/roc-curve-and-auc-from-scratch-in-numpy-visualized-2612bb9459ab
"""

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from celluloid import Camera
import matplotlib.pyplot as plt
import seaborn as sns

X, y = make_classification(n_samples=1000, n_informative=10, n_features=20, flip_y=0.2)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

prob_vector = model.predict_proba(X_test)[:, 1]


def true_false_positive(threshold_vector, y_test):
    true_positive = np.equal(threshold_vector, 1) & np.equal(y_test, 1)
    true_negative = np.equal(threshold_vector, 0) & np.equal(y_test, 0)
    false_positive = np.equal(threshold_vector, 1) & np.equal(y_test, 0)
    false_negative = np.equal(threshold_vector, 0) & np.equal(y_test, 1)

    tpr = true_positive.sum() / (true_positive.sum() + false_negative.sum())
    fpr = false_positive.sum() / (false_positive.sum() + true_negative.sum())
    return tpr, fpr


def roc_from_scratch(probabilities, y_test, partitions=100):
    roc = np.array([])
    for i in range(partitions + 1):
        threshold_vector = np.greater_equal(probabilities, i / partitions).astype(int)
        tpr, fpr = true_false_positive(threshold_vector, y_test)
        print(f'tpr: {tpr}')
        print(f'fpr: {fpr}')
        roc = np.append(roc, [fpr, tpr])
    return roc.reshape(-1, 2)


def roc_test():
    print(prob_vector)
    print(prob_vector.shape)
    sns.set()
    plt.figure(figsize=(15, 7))

    ROC = roc_from_scratch(prob_vector, y_test, partitions=10)
    print(ROC)
    plt.scatter(ROC[:, 0], ROC[:, 1], color='#0F9D58', s=100)
    plt.title('ROC Curve', fontsize=20)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.show()


def roc_animation():
    camera = Camera(plt.figure(figsize=(17, 9)))
    for i in range(30):
        ROC = roc_from_scratch(prob_vector, y_test, partitions=(i + 1) * 5)
        plt.scatter(ROC[:, 0], ROC[:, 1], color='#0F9D58', s=100)
        plt.title('ROC Curve', fontsize=20)
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
        camera.snap()
    anim = camera.animate(blit=True, interval=300)
    plt.show()


if __name__ == '__main__':
    # roc_test()
    roc_animation()
