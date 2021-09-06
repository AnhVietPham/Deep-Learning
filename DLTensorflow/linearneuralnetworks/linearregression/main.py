import math
import numpy as np
import tensorflow as tf
from DLTensorflow.linearneuralnetworks.utils.timer import Timer
from d2l import tensorflow as d2l


def speed_test(a, b, n):
    c = tf.Variable(tf.zeros(n))
    timer = Timer()
    for i in range(n):
        c[i].assign(a[i] + b[i])
    print(f'{timer.stop():.5f} sec')

    timer.start()
    d = a + b
    print(f'{timer.stop():.5f} sec')


def normal_distribution(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma ** 2)
    return p * np.exp((-1 / (2 * sigma ** 2)) * (x - mu) ** 2)


if __name__ == '__main__':
    n = 10000
    a = tf.ones(n)
    b = tf.ones(n)
    speed_test(a, b, n)
    x = np.arange(-9, 9, 0.01)
    params = [(0, 1), (0, 2), (3, 1)]
    d2l.plot(x, [normal_distribution(x, mu, sigma) for mu, sigma in params], xlabel='x', ylabel='p(x)',
             figsize=(4.5, 2.5),
             legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
