"""
Gaussian Process Model for the Drone framework.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


class DroneGaussianProcess:
    def __init__(self,
                 length_scale=1.0,
                 length_scale_bounds=(1e-5, 1e5),
                 nu=1.5,  # Using Matern kernel with v=3/2
                 alpha=1e-6,  # Noise level
                 normalize_y=True,
                 n_restarts_optimizer=10,
                 sliding_window_size=30):
        # Initialize the kernel
        self.kernel = Matern(length_scale=length_scale,
                             length_scale_bounds=length_scale_bounds,
                             nu=nu)

        # Initialize the Gaussian Process model
        self.model = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=alpha,
            normalize_y=normalize_y,
            n_restarts_optimizer=n_restarts_optimizer
        )

        # Data storage
        self.X = None  # Action-context pairs
        self.y = None  # Rewards

        # Sliding window for managing computational complexity
        self.sliding_window_size = sliding_window_size

    def update(self, X, y):
        if self.X is None:
            self.X = X
            self.y = y
        else:
            self.X = np.vstack((self.X, X))
            self.y = np.append(self.y, y)

        # Apply sliding window to limit computational complexity
        if len(self.y) > self.sliding_window_size:
            self.X = self.X[-self.sliding_window_size:]
            self.y = self.y[-self.sliding_window_size:]

        # Fit the GP model
        self.model.fit(self.X, self.y)

    def predict(self, X):
        if self.X is None or len(self.X) == 0:
            # If no data is available yet, return prior
            return np.zeros(X.shape[0]), np.ones(X.shape[0]) * np.sqrt(self.kernel.k(X, X).diagonal())

        # Use the GP model to predict mean and std
        mean, std = self.model.predict(X, return_std=True)
        return mean, std

    def get_data(self):
        return self.X.copy() if self.X is not None else None, self.y.copy() if self.y is not None else None

    def reset(self):
        self.X = None
        self.y = None

    def __str__(self):
        return f"DroneGaussianProcess(kernel={self.kernel}, sliding_window_size={self.sliding_window_size})"
