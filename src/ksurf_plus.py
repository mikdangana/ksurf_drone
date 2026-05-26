import numpy as np
import pandas as pd
import os, sys, torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from filterpy.kalman import ExtendedKalmanFilter as EKF
from filterpy.kalman import KalmanFilter
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset


TRAIN_SPLIT = 0.7


# Load CSV file
def load_data(csv_file, column_name):
    """Load CSV file and extract a specified column as a NumPy array."""
    df = pd.read_csv(csv_file)
    return df[column_name].values

# Define Kalman Filter model
def initialize_kalman_filter(Q, R):
    """Initialize a simple Kalman Filter model."""
    kf = KalmanFilter(dim_x=2, dim_z=1)  # 2 state variables, 1 measurement variable

    # State transition matrix
    kf.F = np.array([[1, 1], [0, 1]])  # Constant velocity model

    # Measurement function
    kf.H = np.array([[1, 0]])  # We only measure the first state (position)

    # Process noise covariance (Q)
    kf.Q = np.array([[Q, 0], [0, Q]])

    # Measurement noise covariance (R)
    kf.R = np.array([[R]])

    # Initial state estimate
    kf.x = np.array([[0], [0]])  # Initial position and velocity

    # Initial uncertainty
    kf.P = np.eye(2) * 1.0

    return kf


# Apply Kalman Filter for inference
def apply_kalman_filter(data, Q, R):
    """Run the Kalman filter on the full dataset using learned Q and R."""
    kf = initialize_kalman_filter(Q, R)
    estimates = []

    for z in data:
        kf.predict()
        kf.update(z)
        estimates.append(ekf.x[0, 0])  # Store filtered position estimate

    return np.array(estimates)



# Apply Kalman Filter for inference
def apply_ekf_filter(data, Q, R, dt=1):
    """Run the Kalman filter on the full dataset using learned Q and R."""
    #kf = initialize_kalman_filter(Q, R)
    ekf = initialize_ekf(Q, R, dt)  # Initialize with learned parameters
    estimates = []

    for z in data:
        #ekf.predict()
        #ekf.update(z, HJacobian=H_jacobian)
        ekf.F = F_jacobian(ekf.x, dt)
        ekf.H = H_jacobian(ekf.x)
        ekf.predict_update(z, H_jacobian, h_x)  # EKF update
        estimates.append(ekf.x[0, 0])  # Store filtered position estimate

    return np.array(estimates)

# Define LSTM model
class LSTMKalmanTuner(nn.Module):
    """LSTM model to predict Q and R parameters for the Kalman Filter."""
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(LSTMKalmanTuner, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)  # Output Q and R

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])  # Take the last output of the LSTM
        Q, R = torch.exp(output[:, 0]), torch.exp(output[:, 1])  # Ensure positivity
        return Q, R


# Create sequences for training
def create_sequences(data, seq_length=10):
    """Convert time series data into sequences for training the LSTM."""
    sequences, targets = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        targets.append(data[i+seq_length])  # Predict next step
    return np.array(sequences), np.array(targets)


def kalman_loss(Q_pred, R_pred, sequences, targets):
    """Compute the Kalman filter loss using predicted Q and R."""
    loss = torch.zeros(1, requires_grad=True, dtype=torch.float32)  # ✅ Ensure loss tracks gradients

    batch_size = sequences.shape[0]

    for i in range(batch_size):
        kf = initialize_kalman_filter(Q_pred[i].item(), R_pred[i].item())  # Initialize with learned parameters

        for t in range(sequences.shape[1]):  # Run through the sequence
            kf.predict()
            kf.update(sequences[i, t])

        # Compute squared error between Kalman Filter output and target
        kf.predict()  # One more prediction step
        error = (kf.x[0, 0] - targets[i]) ** 2  # Squared error
        loss = loss + error  # ✅ Accumulate loss while maintaining gradients

    return loss / batch_size  # ✅ Keep it inside computation graph


def train_lstm1(data, seq_length=10, epochs=100, batch_size=16, lr=0.001):
    """Train an LSTM model to learn Q and R parameters from time-series data."""

    # Prepare data
    sequences, targets = create_sequences(data, seq_length)
    sequences = torch.tensor(sequences, dtype=torch.float32).unsqueeze(-1)  # Add feature dimension
    targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)
    train_loader = DataLoader(TensorDataset(sequences, targets), batch_size=batch_size, shuffle=True)

    # Initialize model
    model = LSTMKalmanTuner()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for seq_batch, target_batch in train_loader:
            optimizer.zero_grad()
            Q_pred, R_pred = model(seq_batch)  # Get learned Q and R
            loss = kalman_loss(Q_pred, R_pred, seq_batch.squeeze(-1), target_batch.squeeze(-1))  # Compute loss
            loss.backward()  # ✅ Now it works because loss tracks gradients
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    return model



# Nonlinear state transition function
def f_x(x, dt):
    """Nonlinear state transition function (Example: Quadratic Acceleration)."""
    return np.array([[x[0, 0] + x[1, 0] * dt + 0.5 * x[2, 0] * dt**2],  # Position
                     [x[1, 0] + x[2, 0] * dt],  # Velocity
                     [x[2, 0]]])  # Acceleration (constant)

# Jacobian of f_x (State transition function)
def F_jacobian(x, dt):
    """Jacobian matrix for the nonlinear state transition function."""
    return np.array([[1, dt, 0.5 * dt**2],
                     [0, 1, dt],
                     [0, 0, 1]])

# Measurement function (Linear in this case)
def h_x(x):
    """Measurement function: Extracts position from the state."""
    return np.array([[x[0, 0]]])  # We only measure position

# Jacobian of h_x (Measurement function)
def H_jacobian(x):
    """Jacobian matrix for the measurement function."""
    return np.array([[1, 0, 0]])

# Define Extended Kalman Filter (EKF)
def initialize_ekf(Q, R, dt=1):
    """Initialize an Extended Kalman Filter (EKF)."""
    ekf = EKF(dim_x=3, dim_z=1)  # 3 state variables (position, velocity, acceleration), 1 measurement

    # Initial state estimate
    ekf.x = np.array([[0], [0], [0]])  # Initial position, velocity, acceleration

    # Process noise covariance
    ekf.Q = np.eye(3) * Q  # Learned from LSTM

    # Measurement noise covariance
    ekf.R = np.array([[R]])  # Learned from LSTM

    # Initial state transition matrix (updated dynamically)
    ekf.F = F_jacobian(ekf.x, dt)

    # Initial measurement function (updated dynamically)
    ekf.H = H_jacobian(ekf.x)

    return ekf

# Compute EKF loss (Prediction error)
def ekf_loss(Q_pred, R_pred, sequences, targets, dt=1):
    """Compute the EKF loss using predicted Q and R."""
    loss = torch.zeros(1, requires_grad=True, dtype=torch.float32)  # ✅ Ensure loss tracks gradients

    batch_size = sequences.shape[0]

    for i in range(batch_size):
        ekf = initialize_ekf(Q_pred[i].item(), R_pred[i].item(), dt)  # Initialize with learned parameters

        for t in range(sequences.shape[1]):  # Run through the sequence
            ekf.F = F_jacobian(ekf.x, dt)
            ekf.H = H_jacobian(ekf.x)
            ekf.predict_update(sequences[i, t], H_jacobian, h_x)  # EKF update

        # Compute squared error between EKF output and target
        ekf.predict()  # One more prediction step
        error = (ekf.x[0, 0] - targets[i]) ** 2  # Squared error
        loss = loss + error  # ✅ Accumulate loss while maintaining gradients

    return loss / batch_size  # ✅ Keep it inside computation graph

# Train LSTM model
def train_lstm(data, seq_length=10, epochs=100, batch_size=16, lr=0.001):
    """Train an LSTM model to learn Q and R parameters from time-series data."""

    # Prepare data
    sequences, targets = create_sequences(data, seq_length)
    sequences = torch.tensor(sequences, dtype=torch.float32).unsqueeze(-1)  # Add feature dimension
    targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)
    train_loader = DataLoader(TensorDataset(sequences, targets), batch_size=batch_size, shuffle=True)

    # Initialize model
    model = LSTMKalmanTuner()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for seq_batch, target_batch in train_loader:
            optimizer.zero_grad()
            Q_pred, R_pred = model(seq_batch)  # Get learned Q and R
            loss = ekf_loss(Q_pred, R_pred, seq_batch.squeeze(-1), target_batch.squeeze(-1))  # Compute loss
            loss.backward()  # ✅ Now it works because loss tracks gradients
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    return model



# Apply trained LSTM to predict Q and R
def apply_lstm_kalman_filter(model, data, seq_length=10, skip_lstm=False):
    """Use trained LSTM model to predict Q and R for each time step and apply the Kalman Filter."""
    if skip_lstm:
        filtered_data = apply_ekf_filter(data[seq_length:], 1, 1) 
        return filtered_data, 1, 1
    sequences, _ = create_sequences(data, seq_length)
    sequences = torch.tensor(sequences, dtype=torch.float32).unsqueeze(-1)

    model.eval()
    with torch.no_grad():
        Q_pred, R_pred = model(sequences)

    # Convert to NumPy
    Q_values, R_values = Q_pred.numpy(), R_pred.numpy()

    print(f"Q = {Q_values.mean()}, R = {R_values.mean()}")
    # Apply Kalman filter with learned parameters
    filtered_data = apply_ekf_filter(data[seq_length:], Q_values.mean(), R_values.mean())
    
    return filtered_data, Q_values, R_values


def summary_stats(data, filtered_data, tag=""):
    err = np.subtract(data[-len(filtered_data)+1:], filtered_data[:-1])
    mae = np.square(err).mean() #np.abs(err).mean()
    var = np.square(err-mae).std()
    print(f"mae = {mae}, std = {var} {tag}")
    return mae, var


def test_ksurf_plus():
    global TRAIN_SPLIT
    f = os.path.join(sys.path[0],'..','data','twitter_trace.csv')
    f = sys.argv[sys.argv.index("-f")+1] if "-f" in sys.argv else f
    xcol = sys.argv[sys.argv.index("-x")+1] if "-x" in sys.argv else 'Tweets 09-May-2023'
    ycol = sys.argv[sys.argv.index("-y")+1] if "-y" in sys.argv else 'Tweets 09-May-2023'
 
    # Load CSV file
    csv_file = f  # Change this to your actual file
    column_name = xcol  # Change this to your actual column name
    
    data = load_data(csv_file, column_name)

    # Train LSTM to learn Q and R using Kalman loss
    print("Training LSTM to learn Kalman Filter parameters...")
    print(f"data.len = {len(data)}, split = {int(TRAIN_SPLIT*len(data))}")
    lstm_model = train_lstm(data[0:int(TRAIN_SPLIT*len(data))], seq_length=10, epochs=100, batch_size=16, lr=0.001)

    # Apply LSTM-trained Kalman Filter
    print("Applying LSTM-based Kalman Filter...")
    filtered_data, Q_values, R_values = apply_lstm_kalman_filter(lstm_model, data[int(TRAIN_SPLIT*len(data)):], seq_length=10)
    maes = [[], []]
    for i in range(10):
        TRAIN_SPLIT = 0.1*(i+1)
        print(f"computing maes = {maes}, split = {TRAIN_SPLIT}")
        lstm_model = train_lstm(data[0:int(TRAIN_SPLIT*len(data))], seq_length=10, epochs=100, batch_size=16, lr=0.001)
        maes[0].append(summary_stats(data, apply_lstm_kalman_filter(lstm_model, data, seq_length=10, skip_lstm=False)[0], tag="(LSTM)")[0])
        maes[1].append(summary_stats(data, apply_lstm_kalman_filter(lstm_model, data, seq_length=10, skip_lstm=True)[0], tag="(No LSTM)")[0])
    # Plot results
    mae_fig = plt.figure(figsize=(10, 5))
    plt.plot(np.subtract(np.array(maes[0]), np.array(maes[1])), label="LSTM MAE  - None-LSTM MAE")
    plt.title("LSTM MAE - Non-LSTM MAE")
    plt.xlabel("Training data split")
    plt.ylabel("CPU MAE Delta")
    mae_fig.savefig("ksurf_plus_lstm_vs_non_lstm.png")
    print("Saved figure to ksurf_plus_lstm_vs_non_lstm.png")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(data[int(TRAIN_SPLIT*len(data)):], label="Raw Data", alpha=0.5)
    plt.plot(np.arange(10+1, len(filtered_data)+10+1), filtered_data, label="LSTM-Kalman Output", linewidth=2)
    plt.xlabel("Time Step")
    plt.ylabel(column_name)
    plt.legend()
    plt.title("LSTM-Kalman Filter Performance on CPU Usage")
    plt.show()



# Main function
if __name__ == "__main__":
    test_ksurf_plus()
