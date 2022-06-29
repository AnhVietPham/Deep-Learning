import numpy as np
import tensorflow as tf
from mpl_toolkits import mplot3d
from d2l import tensorflow as d2l


def f(x):
    return x ** 2


def f_grad(x):
    return 2 * x


def gd(eta, f_grad):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x)
        results.append(float(x))
    print(f'Epoch 10, x: {x:f}')
    return results


def show_trace(results, f):
    n = max(abs(min(results)), abs(max(results)))
    f_line = tf.range(-n, n, 0.01)
    d2l.set_figsize()
    d2l.plot([f_line, results], [[f(x) for x in f_line], [
        f(x) for x in results]], 'x', 'f(x)', fmts=['-', '-o'])


def train_2d(trainer, steps=20):
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(steps):
        x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    return results


def show_trace_2d(f, results):
    """Show the trace of 2D variables during optimization."""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = np.meshgrid(np.arange(-5.5, 1.0, 0.1),
                         np.arange(-3.0, 1.0, 0.1))
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')


f = lambda x1, x2: x1 ** 2 + 2 * x2 ** 2
gradf = lambda x1, x2: (2 * x1, 4 * x2)


def gd(x1, x2, s1, s2):
    (g1, g2) = gradf(x1, x2)
    return (x1 - eta * g1, x2 - eta * g2, 0, 0)


if __name__ == "__main__":
    # results = gd(0.3, f_grad)
    # show_trace(results, f)
    eta = 0.1
    results = train_2d(gd)
    print(results)
