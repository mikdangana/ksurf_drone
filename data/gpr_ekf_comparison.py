import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from filterpy.kalman import ExtendedKalmanFilter
from sklearn.metrics import mean_squared_error
from math import sqrt

# --- Load or generate data ---
def load_or_generate_data(filename=None, n=200):
    if filename:
        df = pd.read_csv(filename)
        return df['value'].values
    else:
        x = np.linspace(0, 10, n)
        y = np.sin(x) + np.random.normal(0, 0.2, size=x.shape)
        return y

# --- Define EKF model ---
def hx(x):
    return np.array([x[0]])

def fx(x, dt):
    return np.array([x[0] + x[1] * dt, x[1]])

def H_jacobian(x):
    return np.array([[1., 0.]])

def F_jacobian(x, dt):
    return np.array([[1., dt],
                     [0., 1.]])

def run_ekf(data, dt=1.0):
    ekf = ExtendedKalmanFilter(dim_x=2, dim_z=1)
    ekf.x = np.array([data[0], 0.0])
    #ekf.F = lambda x,*args: F_jacobian(x, dt)
    #ekf.H = lambda x,*args: H_jacobian(x)
    ekf.R = np.array([[0.04]])  # Measurement noise
    ekf.Q = np.eye(2) * 1e-4    # Process noise
    ekf.P *= 10

    estimates = []
    for z in data:
        ekf.predict_update(np.array([z])) #, hx, fx, args=(dt,))
        estimates.append(ekf.x[0])
    return np.array(estimates)

# --- GPR predictor ---
def run_gpr(data):
    x = np.arange(len(data)).reshape(-1, 1)
    kernel = RBF(length_scale=5.0) + WhiteKernel(noise_level=0.1)
    gpr = GaussianProcessRegressor(kernel=kernel)
    gpr.fit(x, data)
    y_pred, _ = gpr.predict(x, return_std=True)
    return y_pred

# --- Main comparison ---
def compare(filename=None):
    data = load_or_generate_data(filename)
    ekf_pred = run_ekf(data)
    gpr_pred = run_gpr(data)

    rmse_ekf = sqrt(mean_squared_error(data, ekf_pred))
    rmse_gpr = sqrt(mean_squared_error(data, gpr_pred))

    print(f"RMSE EKF: {rmse_ekf:.4f}")
    print(f"RMSE GPR: {rmse_gpr:.4f}")

    plt.figure(figsize=(12, 6))
    plt.plot(data, label="True", color='black', linewidth=1)
    plt.plot(ekf_pred, label="EKF", linestyle='--')
    plt.plot(gpr_pred, label="GPR", linestyle=':')
    plt.legend()
    plt.title("Time Series Prediction: GPR vs EKF")
    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compare()  # or compare("your_file.csv")

