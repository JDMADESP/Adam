import numpy as np

class Adam:
    def __init__(self, theta, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        '''
            lr - learning rate
            beta1 - first moment decay rate
            beta2 - second moment decay rate
            eps - small positive constant to prevent division by zero
        '''
        # holds the weights being optimized
        self.theta = np.array(theta, dtype=float, copy=True)

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.t = 0
        self.m = np.zeros_like(self.theta)
        self.v = np.zeros_like(self.theta)

    def step(self, gradient):
        # double check that it is numpy array of floating point numbers
        gradient = np.asarray(gradient, dtype=float)

        if gradient.shape != self.theta.shape:
            raise ValueError("Gradient shape must match theta shape")
        
        self.t += 1

        self.m = (self.beta1 * self.m) + ((1 - self.beta1) * gradient)
        self.v = (self.beta2 * self.v) + ((1 - self.beta2) * (gradient ** 2))
        m_t_hat = self.m / (1 - (self.beta1**(self.t)))
        v_t_hat = self.v / (1 - self.beta2**(self.t))

        self.theta = self.theta - (self.lr * (m_t_hat / (np.sqrt(v_t_hat) + self.eps)))

        return self.theta.copy()

