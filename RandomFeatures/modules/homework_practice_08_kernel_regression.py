import numpy as np
from sklearn.base import RegressorMixin
from sklearn.gaussian_process.kernels import RBF


class KernelRidgeRegression(RegressorMixin):
    """
    Kernel Ridge regression class
    """

    def __init__(
        self,
        lr=0.01,
        regularization=1.0,
        tolerance=1e-2,
        max_iter=1000,
        batch_size=64,
        kernel_scale=1.0,
    ):
        """
        :param lr: learning rate
        :param regularization: regularization coefficient
        :param tolerance: stopping criterion for square of euclidean norm of weight difference
        :param max_iter: stopping criterion for iterations
        :param batch_size: size of the batches used in gradient descent steps
        :parame kernel_scale: length scale in RBF kernel formula
        """

        self.lr: float = lr
        self.regularization: float = regularization
        self.w: np.ndarray | None = None

        self.tolerance: float = tolerance
        self.max_iter: int = max_iter
        self.batch_size: int = batch_size
        self.loss_history: list[float] = []
        self.kernel = RBF(kernel_scale)

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculating loss for x and y dataset
        :param x: features array
        :param y: targets array
        """
        # raise NotImplementedError
        kernel_calc = self.kernel(x)
        loss = 1/2*(np.linalg.norm(kernel_calc @ self.w - y))**2 + self.regularization/2 * (self.w.T @ kernel_calc @ self.w)
        return loss

    def calc_grad(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculating gradient for x and y dataset
        :param x: features array
        :param y: targets array
        """
        # raise NotImplementedError
        kernel_calc = self.kernel(x)
        ids = self.indexes
        grad = kernel_calc @ (kernel_calc @ self.w[ids] - y) + self.regularization * (kernel_calc @ self.w[ids])
        return grad

    def fit(self, x: np.ndarray, y: np.ndarray) -> "KernelRidgeRegression":
        """
        Получение параметров с помощью градиентного спуска
        :param x: features array
        :param y: targets array
        :return: self
        """
        # raise NotImplementedError
        self.X_train = x
        n_samples = x.shape[0]
        self.w = np.zeros(n_samples)
        for i in range(self.max_iter):
            self.indexes = np.random.choice(n_samples, self.batch_size, replace=False)
            x_batch = x[self.indexes]
            y_batch = y[self.indexes]

            grad = self.calc_grad(x_batch, y_batch)
            self.w[self.indexes] -= self.lr*grad

            loss = self.calc_loss(x, y)
            self.loss_history.append(loss)

            if np.linalg.norm(grad) < self.tolerance:
                break

        return self.loss_history

    def fit_closed_form(self, x: np.ndarray, y: np.ndarray) -> "KernelRidgeRegression":
        """
        Получение параметров через аналитическое решение
        :param x: features array
        :param y: targets array
        :return: self
        """
        # raise NotImplementedError
        kernel_calc = self.kernel(x)
        self.X_train = x
        # w = (ФФ^T + lambda)^-1 @ y
        lambdas = self.regularization * np.eye(kernel_calc.shape[0])
        self.w = np.linalg.inv(kernel_calc + lambdas) @ y
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicting targets for x dataset
        :param x: features array
        :return: prediction: np.ndarray
        """
        # raise NotImplementedError
        kernel_calc = self.kernel(x, self.X_train)
        return kernel_calc @ self.w
