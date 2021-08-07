from sklearn.datasets import make_circles
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.initializers import RandomUniform
from matplotlib import pyplot
from keras_visualizer import visualizer
from numpy import where
from keras.callbacks import TensorBoard


def plot_scatter_data(X, y):
    for i in range(2):
        sample_ix = where(y == i)
        pyplot.scatter(X[sample_ix, 0], X[sample_ix, 1], label=str(i))
    pyplot.legend()
    pyplot.show()


def plot_history(model):
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    # prepare callback
    tb = TensorBoard(histogram_freq=1, write_grads=True)
    # fit model
    history = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=500, verbose=0, callbacks=[tb])
    _, train_acc = model.evaluate(trainX, trainY, verbose=0)
    _, test_acc = model.evaluate(testX, testY, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    pyplot.plot(history.history['accuracy'], label='train')
    pyplot.plot(history.history['val_accuracy'], label='test')
    pyplot.legend()
    pyplot.show()


def build_model_1():
    model = Sequential()
    init = RandomUniform(minval=0.0, maxval=1.0)
    model.add(Dense(5, input_dim=2, activation='tanh', kernel_initializer=init))
    model.add(Dense(1, activation='sigmoid', kernel_initializer=init))
    plot_history(model)


def build_model_2():
    model = Sequential()
    init = RandomUniform(minval=0.0, maxval=1.0)
    model.add(Dense(5, input_dim=2, activation='tanh', kernel_initializer=init))
    model.add(Dense(5, activation='tanh', kernel_initializer=init))
    model.add(Dense(5, activation='tanh', kernel_initializer=init))
    model.add(Dense(5, activation='tanh', kernel_initializer=init))
    model.add(Dense(5, activation='tanh', kernel_initializer=init))
    model.add(Dense(1, activation='sigmoid', kernel_initializer=init))
    plot_history(model)


def build_model_with_relu():
    model = Sequential()
    model.add(Dense(5, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    plot_history(model)


def build_model_with_relu_26_layer():
    model = Sequential()
    model.add(Dense(5, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    plot_history(model)


if __name__ == "__main__":
    X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = scaler.fit_transform(X)
    n_train = 500
    trainX, testX = X[:n_train, :], X[n_train:, :]
    trainY, testY = y[:n_train], y[n_train:]
    # plot_scatter_data(X, y)
    build_model_1()
    # build_model_2()
    # build_model_with_relu()
    # build_model_with_relu_26_layer()
