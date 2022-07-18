import numpy as np
import tensorflow as tf

if __name__ == '__main__':
    x = np.array([[1, 0, 1, 0, 1],
                  [0, 0, 0, 0, 0],
                  [1, 0, 1, 0, 1],
                  [0, 0, 0, 0, 0],
                  [1, 0, 1, 0, 1]], dtype=np.float32)
    x_pad = np.pad(x, pad_width=2)
    x_in = x_pad.reshape((1, 9, 9, 1))
    kernel = np.ones(shape=(3, 3, 1, 1), dtype=np.float32)
    output = tf.nn.atrous_conv2d(x_in, kernel, rate=2, padding='VALID')
    print(output)
