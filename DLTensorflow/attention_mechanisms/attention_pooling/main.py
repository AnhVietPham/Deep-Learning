import tensorflow as tf
from d2l import tensorflow as d2l

tf.random.set_seed(seed=1322)

n_train = 50
x_train = tf.sort(tf.random.uniform(shape=(n_train,), maxval=5))


def f(x):
    return 2 * tf.sin(x) + x ** 0.8


y_train = f(x_train) + tf.random.normal((n_train,), 0.0, 0.5)
x_test = tf.range(0, 5, 0.1)
y_truth = f(x_test)
n_test = len(x_test)

result = tf.reduce_mean(y_train)


def plot_kernel_reg(y_hat):
    d2l.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'],
             xlim=[0, 5], ylim=[-1, 5])
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5)


class NWKernelRegression(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = tf.Variable(initial_value=tf.random.uniform(shape=(1,)))

    def call(self, queries, keys, values, **kwargs):
        queries = tf.repeat(tf.expand_dims(queries, axis=1), repeats=keys.shape[1], axis=1)
        self.attention_weights = tf.nn.softmax(-((queries - keys) * self.w) ** 2 / 2, axis=1)
        return tf.squeeze(tf.matmul(tf.expand_dims(self.attention_weights, axis=1), tf.expand_dims(values, axis=-1)))


if __name__ == '__main__':
    print(n_test)
    # y_hat = tf.repeat(tf.reduce_mean(y_train), repeats=n_test)
    X_repeat = tf.repeat(tf.expand_dims(x_train, axis=0), repeats=n_train, axis=0)
    attention_weights = tf.nn.softmax(-(X_repeat - tf.expand_dims(x_train, axis=1)) ** 2 / 2, axis=1)
    y_hat = tf.matmul(attention_weights, tf.expand_dims(y_train, axis=1))
    X = tf.ones((2, 1, 4))
    Y = tf.ones((2, 4, 6))
    weights = tf.ones((2, 10)) * 0.1
    # values = tf.reshape(tf.range(20.0), shape=(2, 10))
    X_tile = tf.repeat(tf.expand_dims(x_train, axis=0), repeats=n_train, axis=0)
    Y_tile = tf.repeat(tf.expand_dims(y_train, axis=0), repeats=n_train, axis=0)
    keys = tf.reshape(X_tile[tf.cast(1 - tf.eye(n_train), dtype=tf.bool)], shape=(n_train, -1))
    values = tf.reshape(Y_tile[tf.cast(1 - tf.eye(n_train), dtype=tf.bool)], shape=(n_train, -1))
    net = NWKernelRegression()
    loss_object = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.5)

    for epoch in range(5):
        with tf.GradientTape() as t:
            loss = loss_object(y_train, net(x_train, keys, values)) * len(y_train)
        grads = t.gradient(loss, net.trainable_variables)
        optimizer.apply_gradients(zip(grads, net.trainable_variables))
