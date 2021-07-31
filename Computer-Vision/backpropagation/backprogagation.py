import numpy as np


# https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        self.W = []
        self.layers = layers
        self.alpha = alpha

        for i in np.arange(0, len(layers) - 2):
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            self.W.append(w / np.sqrt(layers[i]))

        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

    def __repr__(self):
        return "NeuralNetwork: {}".format("-".join(str(l) for l in self.layers))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        return x * (1 - x)

    def fit(self, X, y, epochs=1000, displayUpdate=100):
        X = np.c_[X, np.ones((X.shape[0]))]

        for epoch in np.arange(0, epochs):
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)
            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                loss = self.calculate_loss(X, y)
                print("epoch={}, loss={:.7f}".format(epoch + 1, loss))

    def fit_partial(self, x, y):
        A = [np.atleast_2d(x)]
        for layer in np.arange(0, len(self.W)):
            net = A[layer].dot(self.W[layer])
            out = self.sigmoid(net)
            A.append(out)

        error = A[-1] - y
        D = [error * self.sigmoid_deriv(A[-1])]

        # Backward
        for layer in np.arange(len(A) - 2, 0, -1):
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            D.append(delta)

        D = D[::-1]
        for layer in np.arange(0, len(self.W)):
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

    def predict(self, X, bias=True):
        p = np.atleast_2d(X)
        if bias:
            p = np.c_[p, np.ones((p.shape[0]))]

        for layer in np.arange(0, len(self.W)):
            p = self.sigmoid(np.dot(p, self.W[layer]))
        return p

    def calculate_loss(self, X, target):
        target = np.atleast_2d(target)
        predictions = self.predict(X, bias=False)
        loss = 0.5 % np.sum((predictions - target) ** 2)
        return loss


if __name__ == "__main__":
    # construct the XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    nn = NeuralNetwork([2, 2, 1], alpha=0.5)
    print("Before")
    print(nn.W)
    nn.fit(X, y, epochs=20000)
    print("After")
    print(nn.W)

    # now that our network is trained, loop over the XOR data points
    for (x, target) in zip(X, y):
        # make a prediction on the data point and display the result
        # to our console
        pred = nn.predict(x)[0][0]
        step = 1 if pred > 0.5 else 0
        print("[INFO] data={}, ground-truth={}, pred={:.4f}, step={}".format(
            x, target[0], pred, step))
