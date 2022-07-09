import random
import numpy as np


def svm_loss_naive(W, X, y, reg):
    d, C = W.shape
    _, N = X.shape

    loss = 0
    dW = np.zeros_like(W)
    for n in range(N):
        xn = X[:, n]
        score = W.T.dot(xn)
        for j in range(C):
            if j == y[n]:
                continue
            margin = 1 - score[y[n]] + score[j]
            if margin > 0:
                loss += margin
                dW[:, j] += xn
                dW[:, y[n]] -= xn

    loss /= N
    loss += 0.5 * reg * np.sum(W * W)
    dW /= N
    dW += reg * W
    return loss, dW


if __name__ == '__main__':
    # N, C, d = 10, 3, 5
    reg = .1
    W = np.array([[1, 2, 3], [1, 2, 3]], dtype='float')
    X = np.array([[1, 2, 3, 4], [1, 2, 3, 4]], dtype='float')
    y = np.random.randint(3, size=4)
    print(W.shape)
    print(X.shape)
    print(y.shape)
    # W = np.random.randn(d, C)
    # X = np.random.randn(d, N)
    # y = np.random.randint(C, size=N)
    print(f'loss without regularization: {svm_loss_naive(W, X, y, 0)[0]}')
    print(f'loss with regularization:{svm_loss_naive(W, X, y, .1)[0]}')
