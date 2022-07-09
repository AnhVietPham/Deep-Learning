import tensorflow as tf
from d2l import tensorflow as d2l


def masked_softmax(X, valid_lens):
    if valid_lens is None:
        return tf.nn.softmax(X, axis=-1)
    else:
        shape = X.shape
        if len(valid_lens.shape) == 1:
            valid_lens = tf.repeat(valid_lens, repeats=shape[1])
        else:
            valid_lens = tf.reshape(valid_lens, shape=-1)

        X = d2l.sequence_mask(tf.reshape(X, shape=(-1, shape[-1])), valid_lens, value=-1e6)
        return tf.nn.softmax(tf.reshape(X, shape=shape), axis=-1)


if __name__ == '__main__':
    print(masked_softmax(tf.random.uniform(shape=(2, 2, 4)), tf.constant([2, 3])))
