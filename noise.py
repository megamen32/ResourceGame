import numpy as np


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mean, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mean = mean
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = (
                self.x_prev
                + self.theta * (self.mean - self.x_prev) * self.dt
                + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        self.x_prev = x

        # Преобразовываем значения в диапазоне [-1, 1]
        x = np.tanh(x)

        # Конвертируем в дискретные действия
        discrete_x = np.where(x < -0.33, 1, np.where(x < 0.33, 2, 0))

        return discrete_x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mean)
