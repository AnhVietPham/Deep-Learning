import numpy as np

# And, OR, XOR datasets

class Perceptron:
    def __init__(self, N, alpha=0.1):
        self.W = np.random.rand(N + 1) / np.sqrt(N)
        self.alpha = alpha

    def step(self, x):
        return 1 if x > 0 else 0

    def fit(self, X, y, epochs=10):
        X = np.c_[X, np.ones((X.shape[0]))]
        for epoch in np.arange(0, epochs):
            for (x, target) in zip(X, y):
                p = self.step(np.dot(x, self.W))
                if p != target:
                    error = p - target
                    self.W += -self.alpha * error * x

    def predict(self, X, addBias=True):
        X = np.atleast_2d(X)
        if addBias:
            X = np.c_[X, np.ones((X.shape[0]))]

        return self.step(np.dot(X, self.W))


if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [0], [0], [1]])
    print(X.shape)
    print("Training perceptron...")
    p = Perceptron(X.shape[1], alpha=0.1)
    p.fit(X, y, epochs=20)
    print("Testing perceptron...")
    for (x, target) in zip(X, y):
        pre = p.predict(x)
        print("Data={}, Ground-Truth={}, pred={}".format(x, target[0], pre))
