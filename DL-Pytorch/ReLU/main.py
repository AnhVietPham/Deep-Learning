from sklearn.datasets import make_circles
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.initializers import RandomUniform
from matplotlib import pyplot

if __name__ == "__main__":
    X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = scaler.fit_transform(X)
    n_train = 500
    trainX, testX = X[:n_train, :], X[n_train:, :]
    trainY, testY = y[:n_train], y[n_train:]

    model = Sequential()
    init = RandomUniform(minval=0.0, maxval=1.0)
    model.add(Dense(5, input_dim=2, activation='tanh', kernel_initializer=init))
    model.add(Dense(1, activation='sigmoid', kernel_initializer=init))
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    history = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=500, verbose=0)
    _, train_acc = model.evaluate(trainX, trainY, verbose=0)
    _, test_acc = model.evaluate(testX, testY, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    pyplot.plot(history.history['accuracy'], label='train')
    pyplot.plot(history.history['val_accuracy'], label='test')
    pyplot.legend()
    pyplot.show()
    # for i in range(2):
    #     sample_ix = where(y == i)
    #     pyplot.scatter(X[sample_ix, 0], X[sample_ix, 1], label=str(i))
    # pyplot.legend()
    # pyplot.show()
