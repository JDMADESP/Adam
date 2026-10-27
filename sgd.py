import numpy as np

class SGD:
    def __init__(self, theta, lr=0.001):
        self.theta = np.array(theta, dtype=float, copy=True)
        self.lr = lr

    def step(self, gradient):
        gradient = np.asarray(gradient, dtype=float)

        if gradient.shape != self.theta.shape:
            raise ValueError("Gradient shape must match theta shape")

        self.theta -= self.lr * gradient

        return self.theta.copy()
