import tensorflow as tf
from d2l import tensorflow as d2l

batch_size, num_steps = 1, 5
train_iter, vocab_iter = d2l.load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10)


def get_params(vocab_size, num_hiddens):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return tf.random.normal(shape=shape, stddev=0.01, mean=0, dtype=tf.float32)

    def three():
        return (tf.Variable(normal((num_inputs, num_hiddens)), dtype=tf.float32),
                tf.Variable(normal((num_hiddens, num_hiddens)), dtype=tf.float32),
                tf.Variable(tf.zeros(num_hiddens), dtype=tf.float32))

    W_xz, W_hz, b_z = three()
    W_xr, W_hr, b_r = three()
    W_xh, W_hh, b_h = three()

    W_hq = tf.Variable(normal((num_hiddens, num_outputs)), dtype=tf.float32)
    b_q = tf.Variable(tf.zeros(num_outputs), dtype=tf.float32)
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    return params


def init_gru_state(batch_size, num_hiddens):
    return (tf.zeros((batch_size, num_hiddens)),)


def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        X = tf.reshape(X, [-1, W_xh.shape[0]])
        Z = tf.sigmoid(tf.matmul(X, W_xz) + tf.matmul(H, W_hz) + b_z)
        R = tf.sigmoid(tf.matmul(X, W_xr) + tf.matmul(H, W_hr) + b_r)
        H_tilda = tf.tanh(tf.matmul(X, W_xh) + tf.matmul(R * H, W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = tf.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return tf.concat(outputs, axis=0), (H,)


if __name__ == '__main__':
    vocab_size, num_hiddens, device_name = len(vocab_iter), 1, d2l.try_gpu()._device_name
    strategy = tf.distribute.OneDeviceStrategy(device_name)
    X = tf.reshape(tf.range(5), (1, 5))
    print("Corpus:")
    print(train_iter.corpus)
    print(vocab_iter.idx_to_token)
    print("Vocab length:")
    print(len(vocab_iter))
    print("Id Token:")
    print(vocab_iter.token_freqs)
    print("Token Freqs:")
    str_idx = ""
    str_tokens = ""
    print(vocab_iter.token_to_idx)
    for idx in train_iter.corpus:
        str_idx += ' ' + str(idx)
    print(str_idx)
    for idx in train_iter.corpus:
        str_tokens += ' ' + vocab_iter.to_tokens(idx)
    print(str_tokens)

    with strategy.scope():
        model = d2l.RNNModelScratch(len(vocab_iter), num_hiddens, init_gru_state, gru, get_params)
    # state = model.begin_state(X.shape[0])
    # Y, new_state = model(X, state)
    # print(Y.shape)
    # print(len(new_state))
    # print(new_state[0].shape)
    print(d2l.predict_ch8('time traveller ', 5, model, vocab_iter))
