import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = pd.read_csv('data_linear.csv').values
    print(data.shape)
    N = data.shape[0]
    x = data[:, 0].reshape(-1, 1)
    y = data[:, 1].reshape(-1, 1)
    plt.scatter(x, y)
    plt.xlabel("Mét vuông")
    plt.ylabel("Giá")

    x = np.hstack((np.ones((N, 1)), x))
    w = np.array([0., 1.]).reshape(-1, 1)
    # predict = np.dot(x, w)
    # plt.plot((x[0][1], x[N - 1][1]), (predict[0], predict[N - 1]), 'r')
    # plt.show()
    numOfIteration = 100
    cost = np.zeros((numOfIteration, 1))
    lr = 0.00001
    for i in range(0, numOfIteration):
        r = np.dot(x, w) - y
        cost[i] = 0.5 * np.sum(r * r)
        w[0] -= lr * np.sum(r)
        w[1] -= lr * np.sum(np.multiply(r, x[:, 1].reshape(-1, 1)))
        print(cost[i])

    print(f'Predict w[0]: {w[0]}')
    print(f'Predict w[1]: {w[1]}')
    predict = np.dot(x, w)
    plt.plot((x[0][1], x[N - 1][1]), (predict[0], predict[N - 1]), 'r')
    plt.show()

    x1 = 50
    y1 = w[0] + w[1] * 50
    print('Giá nhà cho 50m^2 là : ', y1)
