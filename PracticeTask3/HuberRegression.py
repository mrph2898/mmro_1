from sklearn.base import BaseEstimator
import numpy as np


class HuberReg(BaseEstimator):
    def __init__(self, delta=1.0, gd_type='stochastic',
                 tolerance=1e-4, max_iter=1000, w0=None, alpha=1e-3, eta=1e-2):
        """
        gd_type: 'full' or 'stochastic'
        tolerance: for stopping gradient descent
        max_iter: maximum number of steps in gradient descent
        w0: np.array of shape (d) - init weights
        eta: learning rate
        alpha: momentum coefficient
        """
        self.delta = delta
        self.gd_type = gd_type
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.w0 = w0
        self.alpha = alpha
        self.w = None
        self.eta = eta
        # list of loss function values at each training iteration
        self.loss_history = None

    def fit(self, X, y):
        """
        X: np.array of shape (l, d)
        y: np.array of shape (l)
        ---
        output: self
        """
        self.loss_history = []
        self.w0 = np.zeros(X.shape[1])
        cur_w = self.w0
        previous_step_size = 1
        iter_counter = 0
        w_change = self.w0
        while previous_step_size > self.tolerance and\
                iter_counter < self.max_iter:
            self.w = np.copy(cur_w)
            cur_w -= self.eta * self.calc_gradient(X, y) -\
                self.alpha * w_change
            w_change = cur_w - self.w
            previous_step_size = np.linalg.norm(w_change)
            self.loss_history = np.append(self.loss_history,
                                          self.calc_loss(X, y))
            iter_counter += 1
        self.w = np.copy(cur_w)
        return self

    def predict(self, X):
        if self.w is None:
            raise Exception('Not trained yet')
        return np.matmul(X, self.w)

    def score(self, X, y):
        return 1 - ((((y - self.predict(X)) ** 2).sum()) /
                    (((y - self.predict(X).mean()) ** 2).sum()))

    def huber_derivate(self, z):
        return np.where(np.abs(z) <= self.delta, z, self.delta * np.sign(z))

    def calc_gradient(self, X, y):
        """
        X: np.array of shape (l, d) (l can be equal to 1 if stochastic)
        y: np.array of shape (l)
        ---
        output: np.array of shape (d)
        """
        if self.w is None:
            raise Exception('Not trained yet')
        if self.gd_type == "full":
            z = y - np.matmul(X, self.w.T)
            N = z.shape[0]
            return -np.matmul(self.huber_derivate(z), X) / N
        else:
            i_k = np.random.randint(X.shape[0])
            return (-X[i_k] *
                    self.huber_derivate(y[i_k] -
                                        np.dot(X[i_k], self.w)))

    def huber_loss(self, z):
        N = z.shape[0]
        loss = np.where(np.abs(z) <= self.delta, 0.5 * ((z) ** 2),
                        self.delta * np.abs(z) - 0.5 * (self.delta ** 2))
        return np.sum(loss) / N

    def calc_loss(self, X, y):
        """
        X: np.array of shape (l, d)
        y: np.array of shape (l)
        ---
        output: float
        """
        if self.w is None:
            raise Exception('Not trained yet')
        return float(self.huber_loss(y - np.matmul(X, self.w.T)))
