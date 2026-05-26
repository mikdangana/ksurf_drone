import numpy as np
import pandas as pd
import os, sys, torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from filterpy.kalman import ExtendedKalmanFilter# as EKF
from filterpy.kalman import KalmanFilter
from math import ceil
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
    ekf = ExtendedKalmanFilter(dim_x=3, dim_z=1)  # 3 state variables (position, velocity, acceleration), 1 measurement

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
def train_lstm(data, seq_length=10, epochs=300, batch_size=16, lr=0.001):
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
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss}")

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



class ClearTrack:
    def __init__(self, seq_length=10, epochs=100, batch_size=16, lr=0.001, sliding_window_size=30):
        self.seq_length = seq_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model = None
        self.trained = False
        self.X = None
        self.y = None
        self.sliding_window_size = sliding_window_size
        self.filtered_data = None
        self.Q_values = None
        self.R_values = None

    def update(self, X, y):
        """Update the training data and re-train the LSTM filter."""
        if self.X is None:
            self.X = X
            self.y = y
        else:
            self.X = np.vstack((self.X, X))
            self.y = np.append(self.y, y)

        if len(self.y) > self.seq_length + 1:
            # Apply sliding window
            self.X = self.X[-self.sliding_window_size:]
            self.y = self.y[-self.sliding_window_size:]

            self.model = train_lstm(
                np.array(self.y),
                seq_length=self.seq_length,
                epochs=self.epochs,
                batch_size=self.batch_size,
                lr=self.lr
            )
            self.trained = True

    def predict(self, X):
        """Predict using the trained LSTM-based filter."""
        if not self.trained or self.y is None or len(self.y) <= self.seq_length:
            return np.zeros(X.shape[0]), np.ones(X.shape[0])  # mimic prior mean, std

        # Filter predictions for each row in X using the full sequence
        filtered, Qs, Rs = apply_lstm_kalman_filter(
            self.model,
            np.array(self.y),
            seq_length=self.seq_length
        )
        self.filtered_data = filtered
        self.Q_values = Qs
        self.R_values = Rs

        mean = np.array([filtered[-1]] * X.shape[0])
        std = np.array([np.sqrt(Qs[-1])] * X.shape[0]) if Qs else np.ones(X.shape[0])
        return mean, std

    def reset(self):
        self.model = None
        self.trained = False
        self.X = None
        self.y = None
        self.filtered_data = None
        self.Q_values = None
        self.R_values = None




def test_clear_track():
    global TRAIN_SPLIT
    f = os.path.join(sys.path[0],'..','data','twitter_trace.csv')
    f = sys.argv[sys.argv.index("-f")+1] if "-f" in sys.argv else f
    xcol = sys.argv[sys.argv.index("-x")+1] if "-x" in sys.argv else 'Tweets 09-May-2023'
    ycol = sys.argv[sys.argv.index("-y")+1] if "-y" in sys.argv else 'Tweets 09-May-2023'
    L = float(sys.argv[sys.argv.index("-l")+1]) if "-l" in sys.argv else 0.0001
    E = int(sys.argv[sys.argv.index("-e")+1]) if "-e" in sys.argv else 10
    print("test_clear_track().f, x, y, L, E =", f, xcol, ycol, L, E)
 
    # Load CSV file
    csv_file = f  # Change this to your actual file
    column_name = xcol  # Change this to your actual column name
    
    data = load_data(csv_file, column_name)
    TRAIN_SPLIT = 0.4

    # Train LSTM to learn Q and R using Kalman loss
    print("Training LSTM to learn Kalman Filter parameters...")
    print(f"data.len = {len(data)}, split = {int(TRAIN_SPLIT*len(data))}")
    lstm_model = train_lstm(data[0:int(TRAIN_SPLIT*len(data))], seq_length=10, epochs=E, batch_size=16, lr=L)

    # Apply LSTM-trained Kalman Filter
    print("Applying LSTM-based Kalman Filter...")
    filtered_data, Q_values, R_values = apply_lstm_kalman_filter(lstm_model, data[int(TRAIN_SPLIT*len(data)):], seq_length=10)
    filtered_ekf, _, _ = apply_lstm_kalman_filter(lstm_model, data, seq_length=10, skip_lstm=True)
    n = min(len(data), len(filtered_data), len(filtered_ekf))

    df_out = pd.DataFrame({
        "raw": np.asarray(data[:n], dtype=float),
        "lstm_kf": np.asarray(filtered_data[:n], dtype=float),
        "ekf": np.asarray(filtered_ekf[:n], dtype=float),
    })

    df_out.to_csv("clear_track_data.csv", index=False)
    #lstm_model = train_lstm(data[0:int(TRAIN_SPLIT*len(data))], seq_length=10, epochs=100, batch_size=16, lr=0.001)
    maes, L = [[], []], ceil(len(data)/10)
    for i in range(10):
        print(f"computing maes = {maes}, split = {TRAIN_SPLIT}")
        datai = data[i*L:(i+1)*L]
        maes[0].append(summary_stats(datai, apply_lstm_kalman_filter(lstm_model, datai, seq_length=10, skip_lstm=False)[0], tag="(LSTM)")[0])
        maes[1].append(summary_stats(datai, apply_lstm_kalman_filter(lstm_model, datai, seq_length=10, skip_lstm=True)[0], tag="(No LSTM)")[0])
    print("mse.delta = ", np.array(maes[0])-np.array(maes[1]))
    print("mse.delta,l = ", np.array(maes[0]).mean()-np.array(maes[1]).mean(),L)
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
    plt.plot(np.arange(10+1, len(filtered_ekf)+10+1), filtered_ekf, label="Kalman Output", linewidth=2)
    plt.xlabel("Time Step")
    plt.ylabel(column_name)
    plt.legend()
    plt.title("LSTM-Kalman Filter Performance on CPU Usage")
    plt.show()


import os, sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil
# assumes train_lstm, apply_lstm_kalman_filter, load_data already exist in this file

# -------------------- Caching helpers --------------------

def _results_cache_path() -> str:
    return "ksurf_epoch_results.json"

def save_epoch_results_json(path: str, results: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

def load_epoch_results_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# -------------------- Decile / metric helpers --------------------

def _decile_segments(arr: np.ndarray, k: int = 10):
    """Yield k segments (last one possibly shorter) for decile-wise eval."""
    L = ceil(len(arr) / k)
    for i in range(k):
        seg = arr[i*L : (i+1)*L]
        yield seg

def _align_for_metrics(raw_seg: np.ndarray, filt_seg: np.ndarray, seq_len: int):
    """
    Align raw and filtered for MAE/MSE:
      filtered starts after seq_len lookback, so compare raw_seg[seq_len:] with filtered.
    """
    if len(raw_seg) <= seq_len or len(filt_seg) == 0:
        return np.array([]), np.array([])
    raw = raw_seg[seq_len:]
    m = min(len(raw), len(filt_seg))
    return raw[-m:], filt_seg[-m:]

def _decile_mae_mse_for_method(arr: np.ndarray, model, seq_len: int, use_lstm: bool):
    """
    For each decile segment, compute (MAE, MSE) for either:
      - LSTM+EKF (use_lstm=True)
      - EKF baseline (use_lstm=False, skip_lstm=True)
    """
    maes, mses = [], []
    for seg in _decile_segments(arr, k=10):
        if len(seg) <= seq_len + 1:
            maes.append(np.nan); mses.append(np.nan); continue
        filt, *_ = apply_lstm_kalman_filter(model, seg, seq_length=seq_len, skip_lstm=(not use_lstm))
        raw_aln, filt_aln = _align_for_metrics(seg, filt, seq_len)
        if len(raw_aln) == 0:
            maes.append(np.nan); mses.append(np.nan); continue
        err = raw_aln - filt_aln
        maes.append(float(np.nanmean(np.abs(err))))
        mses.append(float(np.nanmean(err*err)))
    return np.array(maes, dtype=float), np.array(mses, dtype=float)


def _decile_delta_mae_ci(arr: np.ndarray, model, seq_len: int):
    """
    For each decile segment, compute the mean of paired differences:
        d_i = |raw_i - lstm_i| - |raw_i - ekf_i|
    and its 95% CI half-width (1.96 * std(d)/sqrt(n)), using aligned samples.
    Returns:
        delta_means: np.array shape (10,)
        delta_cis:   np.array shape (10,)  # half-widths for 95% CI
    """
    delta_means, delta_cis = [], []
    for seg in _decile_segments(arr, k=10):
        if len(seg) <= seq_len + 1:
            delta_means.append(np.nan); delta_cis.append(np.nan); continue
        # Filter both ways on the SAME segment
        lstm_filt, *_ = apply_lstm_kalman_filter(model, seg, seq_length=seq_len, skip_lstm=False)
        ekf_filt,  *_ = apply_lstm_kalman_filter(model, seg, seq_length=seq_len, skip_lstm=True)

        raw_lstm, lstm_aln = _align_for_metrics(seg, lstm_filt, seq_len)
        raw_ekf,  ekf_aln  = _align_for_metrics(seg, ekf_filt,  seq_len)
        m = min(len(raw_lstm), len(raw_ekf), len(lstm_aln), len(ekf_aln))
        if m == 0:
            delta_means.append(np.nan); delta_cis.append(np.nan); continue

        raw  = raw_lstm[-m:]        # same aligned raw for both
        lstm = lstm_aln[-m:]
        ekf  = ekf_aln[-m:]

        d = np.abs(raw - lstm) - np.abs(raw - ekf)  # paired differences
        mu = float(np.nanmean(d))
        if m >= 2:
            se = float(np.nanstd(d, ddof=1) / np.sqrt(m))
            ci = 1.96 * se
        else:
            ci = np.nan
        delta_means.append(mu)
        delta_cis.append(ci)

    return np.array(delta_means, dtype=float), np.array(delta_cis, dtype=float)


# -------------------- Plotting-only from cache --------------------

def plot_from_cache(cache_path: str = None):
    """Load cached epoch results and produce the two plots (no training)."""
    cache_path = cache_path or _results_cache_path()
    fontsize = 18
    print("plot_from_cache().cache_path,exists =", cache_path, os.path.exists(cache_path))

    if not os.path.exists(cache_path):
        print("[cache] file not found:", os.path.abspath(cache_path))
        return
    D = load_epoch_results_json(cache_path)

    # Extract baseline + epochs
    baseline_mae_dec = np.array(D["baseline"]["mae_dec"], dtype=float)
    baseline_mse_dec = np.array(D["baseline"]["mse_dec"], dtype=float)
    baseline_mae_mean = float(D["baseline"]["mae_mean"])
    baseline_mse_mean = float(D["baseline"]["mse_mean"])
    epochs_list = [r["epochs"] for r in D["epochs"]]

    # Bar chart
    labels = ["EKF"] + [f"L-EKF $\epsilon$={E}" for E in epochs_list]
    bar_vals = [baseline_mae_mean] + [float(r["mae_mean"]) for r in D["epochs"]]
    mse_vals = [baseline_mse_mean] + [float(r["mse_mean"]) for r in D["epochs"]]
    # 1) Use a categorical colormap (tab10/tab20 are great for distinct colors)
    cmap = plt.get_cmap("tab10")              # 10 distinct colors
    colors = cmap(np.arange(len(labels)) % 10)
    print("plot_from_cache().colors =", colors)

    plt.figure(figsize=(10, 5))
    bars = plt.bar(range(len(labels)), bar_vals, color=colors,edgecolor="black")
    plt.xticks(range(len(labels)), labels, rotation=0)
    plt.ylabel("Mean MAE",fontsize=fontsize)
    plt.title("Mean L-EKF MAE by Epoch",fontsize=fontsize)
    for rect, mse in zip(bars, mse_vals):
        y = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2.0, y,
                 f"MSE={mse:.4g}", ha="center", va="bottom", fontsize=fontsize)
    plt.tight_layout()
    plt.savefig("ksurf_epoch_bar_mae_mse.png")
    print("Saved bar chart -> ksurf_epoch_bar_mae_mse.png")
    plt.show()

    print("print_from_cache().epochs,epochs[0],baseline =", len(D["epochs"]), len(D["epochs"][0]["mae_dec"]), len(baseline_mae_dec))

    # Delta lines
    plt.figure(figsize=(11, 5))
    x_dec = np.arange(1, 11)
    for r in D["epochs"]:
        if "delta_mae_dec" in r:
            delta = np.array(r["delta_mae_dec"], dtype=float)
            yerr  = np.array(r.get("delta_mae_ci", [np.nan]*len(delta)), dtype=float)
        else:
            # Backward compatibility: compute delta from cached MAE arrays (no CI)
            #delta = np.array(r["mae_dec"], dtype=float) - np.array(D["baseline"]["mae_dec"], dtype=float)
            delta = np.array(r["mae_dec"], dtype=float) - baseline_mae_dec
            yerr  = None
        e, mean, std = r['epochs'], np.nanmean(delta), np.nanstd(delta)
        label = f"e={e} ($\Delta$ MAE $\\rho$={mean:.3g}, $\sigma$={std:.3g})"
        print(label)
        #plt.plot(x_dec, delta, marker="o", label=f"e={r['epochs']} ($\delta$MAE $\rho$={np.nanmean(delta):.3g})")
        if yerr is None or np.all(np.isnan(yerr)):
            plt.plot(x_dec, delta, marker="o", label=label)
           # plt.plot(x_dec, delta, marker="o", label=label)
        else:
            plt.errorbar(x_dec, delta, yerr=yerr, marker="o", capsize=4, label=label)

    plt.axhline(0.0, color="gray", linewidth=1, linestyle="--")
    plt.xlabel("Decile",fontsize=fontsize)
    plt.ylabel("L-EKF MAE - EKF MAE",fontsize=fontsize)
    plt.title("L-EKF MAE vs EKF MAE per Decile",fontsize=fontsize)
    plt.xticks(x_dec)
    plt.legend(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig("ksurf_epoch_delta_lines.png")
    print("Saved delta lines -> ksurf_epoch_delta_lines.png")
    plt.show()

# -------------------- Main multi-epoch test (compute + cache + plot) --------------------

def test_clear_track_multi_epoch():
    """
    Train/evaluate for epochs in {1,10,20,30}, cache results to JSON,
    and plot (1) mean MAE bars with MSE annotations, (2) delta lines vs EKF.
    On subsequent runs, if cache exists and matches settings, we reuse it.
    """
    # -------- CLI / defaults --------
    f = os.path.join(sys.path[0], '..', 'data', 'twitter_trace.csv')
    f = sys.argv[sys.argv.index("-f")+1] if "-f" in sys.argv else f
    xcol = sys.argv[sys.argv.index("-x")+1] if "-x" in sys.argv else 'Tweets 09-May-2023'
    lr   = float(sys.argv[sys.argv.index("-l")+1]) if "-l" in sys.argv else 1e-4
    seq_len = 10
    epochs_list = [1, 10, 20, 30]
    train_split = 0.4
    cache_path = _results_cache_path()
    force = ("--recompute" in sys.argv)  # optional flag to force recompute

    print("test_clear_track_multi_epoch().file, x, lr, cache =", f, xcol, lr, cache_path)

    # -------- Load data --------
    data = load_data(f, xcol).astype(float)
    train_n = int(train_split * len(data))
    train_data = data[:train_n]

    # -------- Try cache --------
    use_cache = False
    if os.path.exists(cache_path) and not force:
        try:
            D = load_epoch_results_json(cache_path)
            meta = D.get("meta", {})
            if (meta.get("file") == os.path.abspath(f) and
                meta.get("column") == xcol and
                #int(meta.get("len", -1)) == int(len(data)) and
                #int(meta.get("train_n", -1)) == int(train_n) and
                #int(meta.get("seq_len", -1)) == int(seq_len) and
                #float(meta.get("lr", -1.0)) == float(lr) and
                meta.get("epochs_list", []) == epochs_list):
                print("[cache] using cached results:", os.path.abspath(cache_path))
                use_cache = True
                # go straight to plotting
                plot_from_cache(cache_path)
                return
            else:
                print("[cache] cache exists but metadata mismatch; recomputing.")
        except Exception as e:
            print("[cache] failed to read cache; recomputing. err=", e)

    # -------- Compute baseline EKF --------
    print("[EKF] computing baseline decile metrics...")
    baseline_mae_dec, baseline_mse_dec = _decile_mae_mse_for_method(data, None, seq_len, use_lstm=False)
    baseline_mae_mean = float(np.nanmean(baseline_mae_dec))
    baseline_mse_mean = float(np.nanmean(baseline_mse_dec))
    print(f"[EKF] mean MAE={baseline_mae_mean:.6g}, mean MSE={baseline_mse_mean:.6g}")

    # -------- Train & evaluate each epoch spec --------
    results_epochs = []
    for E in epochs_list:
        print(f"[train] epochs={E}")
        model = train_lstm(train_data, seq_length=seq_len, epochs=E, batch_size=16, lr=lr)
        mae_dec, mse_dec = _decile_mae_mse_for_method(data, model, seq_len, use_lstm=True)
        mae_mean = float(np.nanmean(mae_dec))
        mse_mean = float(np.nanmean(mse_dec))
        # NEW: per-decile delta MAE vs EKF with 95% CI half-widths
        delta_means, delta_cis = _decile_delta_mae_ci(data, model, seq_len)
        print(f"[eval] epochs={E}: mean MAE={mae_mean:.6g}, mean MSE={mse_mean:.6g}")
        results_epochs.append({
            "epochs": E,
            "mae_dec": mae_dec.tolist(),
            "mse_dec": mse_dec.tolist(),
            "mae_mean": mae_mean,
            "mse_mean": mse_mean,
            "delta_mae_dec": delta_means.tolist(),
            "delta_mae_ci":  delta_cis.tolist(),
        })

    # -------- Save cache --------
    results_payload = {
        "meta": {
            "file": os.path.abspath(f),
            "column": xcol,
            "len": int(len(data)),
            "train_n": int(train_n),
            "seq_len": int(seq_len),
            "lr": float(lr),
            "epochs_list": epochs_list,
            "timestamp": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "baseline": {
            "mae_dec": baseline_mae_dec.tolist(),
            "mse_dec": baseline_mse_dec.tolist(),
            "mae_mean": baseline_mae_mean,
            "mse_mean": baseline_mse_mean,
        },
        "epochs": results_epochs,
    }
    save_epoch_results_json(cache_path, results_payload)
    print("Saved plot data ->", os.path.abspath(cache_path))

    # -------- Plot from the just-saved cache --------
    plot_from_cache(cache_path)


# --- NEW IMPORTS (safe to add near the top) ---
import json, time, math, datetime as dt
from typing import Dict, List, Optional, Tuple

try:
    import requests
except Exception:  # fallback if requests is unavailable
    requests = None
    import urllib.request
    import urllib.parse

# ----------------------------------------------------------------------
# NEW: Low-latency Prometheus HTTP client (query_range)
# ----------------------------------------------------------------------
def prometheus_query_range(
    base_url: str,
    promql: str,
    start: dt.datetime,
    end: dt.datetime,
    step: str = "5s",
    timeout: int = 20,
    bearer_token: Optional[str] = None,
    verify_tls: bool = True,
    extra_headers: Optional[Dict[str, str]] = None,
) -> List[Tuple[float, float]]:
    """
    Fetch a time series via Prometheus HTTP API /api/v1/query_range.

    Returns: list of (timestamp_seconds, value_float).
    """
    # Normalize URL
    base = base_url.rstrip("/")
    url = f"{base}/api/v1/query_range"

    params = dict(
        query=promql,
        start=str(int(start.timestamp())),
        end=str(int(end.timestamp())),
        step=step,
    )
    headers = {"Accept": "application/json"}
    if extra_headers:
        headers.update(extra_headers)
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"

    def _parse_json(raw: bytes) -> List[Tuple[float, float]]:
        import json as _json
        payload = _json.loads(raw.decode("utf-8"))
        if payload.get("status") != "success":
            raise RuntimeError(f"Prometheus error: {payload.get('error', 'unknown')}")
        result = payload.get("data", {}).get("result", [])
        if not result:
            return []
        # Use first series (you can aggregate in PromQL to ensure single series)
        series = result[0]["values"]  # [ [ts, "value"], ... ]
        out = []
        for t, v in series:
            try:
                out.append((float(t), float(v)))
            except Exception:
                # skip NaN/Inf or parse errors
                pass
        return out

    if requests is not None:
        with requests.Session() as s:
            s.headers.update(headers)
            resp = s.get(url, params=params, timeout=timeout, verify=verify_tls)
            resp.raise_for_status()
            return _parse_json(resp.content)
    else:
        # urllib fallback
        qs = urllib.parse.urlencode(params)
        req = urllib.request.Request(f"{url}?{qs}", headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return _parse_json(r.read())


# ----------------------------------------------------------------------
# NEW: Build a dataframe from multiple queries and align on timestamps.
# ----------------------------------------------------------------------
def _align_series(
    series_map: Dict[str, List[Tuple[float, float]]]
) -> "pd.DataFrame":
    """
    series_map: name -> list[(ts, val)]  (ts in seconds)
    Returns a DataFrame with columns: ['ts', <names...>] aligned on ts.
    """
    frames = []
    for name, seq in series_map.items():
        if not seq:
            continue
        df = pd.DataFrame(seq, columns=["ts", name])
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["ts"])

    df = frames[0]
    for f in frames[1:]:
        df = pd.merge(df, f, on="ts", how="outer")
    df = df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    return df


# ----------------------------------------------------------------------
# NEW: Quick Q,R estimation from filtered signal (residual-based)
# ----------------------------------------------------------------------
def _estimate_qr_from_series(
    y: np.ndarray, filtered: np.ndarray
) -> Tuple[float, float]:
    """
    Heuristic estimators:
      - R_hat: variance of measurement residuals (y - filtered) on alignment
      - Q_hat: variance of increments of filtered state (diff)
    Safeguards non-negativity.
    """
    if len(filtered) == 0 or len(y) < 2:
        return 1.0, 1.0
    # Align: filtered is typically produced with some lookback; use tail match
    m = min(len(y), len(filtered))
    y_tail = y[-m:]
    f_tail = filtered[-m:]
    res = y_tail - f_tail
    R_hat = float(np.nan_to_num(np.var(res), nan=1.0, posinf=1.0, neginf=1.0))
    inc = np.diff(f_tail)
    Q_hat = float(np.nan_to_num(np.var(inc), nan=1.0, posinf=1.0, neginf=1.0))
    # Avoid zero
    R_hat = max(R_hat, 1e-9)
    Q_hat = max(Q_hat, 1e-9)
    return Q_hat, R_hat


# ----------------------------------------------------------------------
# NEW: Main ingestion + tuning entry point (standalone)
# ----------------------------------------------------------------------
def collect_prometheus_and_tune(
    base_url: str,
    x_queries: List[str],
    y_query: str,
    start: Optional[dt.datetime] = None,
    end: Optional[dt.datetime] = None,
    step: str = "5s",
    csv_path: str = "ksurf_metrics.csv",
    qr_path: str = "ksurf_qr.json",
    bearer_token: Optional[str] = None,
    verify_tls: bool = True,
    min_points: int = 300,
    seq_length: int = 10,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pull metrics from Prometheus, write/append CSV, and when enough data exists:
    train the LSTM to get Q,R; also compute residual-based Q,R; store matrices.

    Returns:
        X (shape [n, k]) and y (shape [n,]) ready for Ksurf.update(X, y).
    """
    # Time window defaults: last 1 hour
    now = dt.datetime.utcnow() if end is None else end
    if start is None:
        start = now - dt.timedelta(hours=1)

    # Fetch y
    y_series = prometheus_query_range(
        base_url, y_query, start, now, step,
        bearer_token=bearer_token, verify_tls=verify_tls
    )
    # Fetch X features
    series_map = {"y": y_series}
    for i, q in enumerate(x_queries):
        seq = prometheus_query_range(
            base_url, q, start, now, step,
            bearer_token=bearer_token, verify_tls=verify_tls
        )
        series_map[f"x{i}"] = seq

    # Align and clean
    df = _align_series(series_map)
    if df.empty:
        print("[prom] no data returned for the given window")
        return np.zeros((0, len(x_queries))), np.zeros((0,))

    # Drop rows with missing y, and forward-fill X if needed
    df = df.dropna(subset=["y"]).reset_index(drop=True)
    x_cols = [c for c in df.columns if c.startswith("x")]
    if x_cols:
        df[x_cols] = df[x_cols].fillna(method="ffill").fillna(method="bfill")

    # Convert from epoch seconds to pandas datetime for the CSV
    df["ts_iso"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    # Order columns for CSV and append
    csv_cols = ["ts", "ts_iso"] + x_cols + ["y"]
    if not os.path.exists(csv_path):
        df[csv_cols].to_csv(csv_path, index=False)
    else:
        df[csv_cols].to_csv(csv_path, index=False, mode="a", header=False)

    # Build X, y arrays for Ksurf API
    X = df[x_cols].values.astype(np.float32) if x_cols else np.zeros((len(df), 0), dtype=np.float32)
    y = df["y"].values.astype(np.float32)

    # If enough points, train the LSTM (uses functions already in this file)
    if len(y) >= max(min_points, seq_length + 2):
        # Train on all y for stability; you can also use a split if desired
        model = train_lstm(
            data=y, seq_length=seq_length,
            epochs=epochs, batch_size=batch_size, lr=lr
        )
        # Apply to get filtered and (Q,R) predictions
        filtered, Q_vals, R_vals = apply_lstm_kalman_filter(model, y, seq_length=seq_length)
        # LSTM-derived scalars (geomean/mean are both reasonable; use mean)
        Q_lstm = float(np.asarray(Q_vals).mean()) if np.size(Q_vals) else 1.0
        R_lstm = float(np.asarray(R_vals).mean()) if np.size(R_vals) else 1.0
        # Residual-based fallback/confirmation
        Q_resid, R_resid = _estimate_qr_from_series(y, filtered)

        # Matrix forms for EKF (3x3 for your initialize_ekf, 1x1 for R)
        Q_mat = (np.eye(3) * Q_lstm).tolist()
        R_mat = np.array([[R_lstm]], dtype=float).tolist()

        payload = dict(
            timestamp=dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            n_samples=int(len(y)),
            seq_length=int(seq_length),
            epochs=int(epochs),
            step=str(step),
            Q_scalar_lstm=Q_lstm,
            R_scalar_lstm=R_lstm,
            Q_scalar_residual=Q_resid,
            R_scalar_residual=R_resid,
            Q_matrix_ekf=Q_mat,
            R_matrix_ekf=R_mat,
            csv_path=os.path.abspath(csv_path),
        )
        with open(qr_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[prom] wrote Q/R to {os.path.abspath(qr_path)}")
    else:
        print(f"[prom] collected {len(y)} points (<{min_points}); skipping LSTM for now.")

    return X, y

# --- add somewhere near the top (helpers) ---
def _default_mean_latency_query(namespace: str) -> str:
    # mean = rate(sum) / rate(count) over 5m window
    return (
        f'sum(rate(http_request_duration_seconds_sum{{namespace="{namespace}"}}[5m]))'
        ' / '
        f'sum(rate(http_request_duration_seconds_count{{namespace="{namespace}"}}[5m]))'
    )


def _fallback_y_candidates() -> str:
    """Mean latency from Prometheus' own HTTP server."""
    return (
        "sum(rate(prometheus_http_request_duration_seconds_sum[5m]))"
        " / "
        "sum(rate(prometheus_http_request_duration_seconds_count[5m]))"
    )


def _default_x_queries(namespace: str) -> List[str]:
    ns = namespace
    return [
        f"sum(rate(http_requests_total{{namespace='{ns}'}}[5m]))",
        "avg(node_cpu_utilization)",
        "avg(node_memory_utilization)",
        "avg(node_network_transmit_bytes_total + node_network_receive_bytes_total)"
        # spot_price handled as constant column after alignment
    ]

def _fallback_x_queries() -> List[str]:
    return [
            "sum(rate(prometheus_http_requests_total[5m]))",
            # CPU-seconds per second used by this process (≈ cores used)
            "rate(process_cpu_seconds_total[5m])",
            # Absolute RSS as last resort
            "process_resident_memory_bytes",
            # Outbound bytes/s served by Prometheus HTTP (response size sum rate)
            "sum(rate(prometheus_http_response_size_bytes_sum[5m]))"
    ]


# ----------------------------------------------------------------------
# OPTIONAL: Example CLI usage (keeps your existing main intact)
#   python clear_track.py --prom http://localhost:9090 \
#       --x 'rate(container_cpu_usage_seconds_total[30s])' \
#       --y 'container_memory_working_set_bytes' \
#       --window 3600 --step 5s
# ----------------------------------------------------------------------
if __name__ == "__main__" and "--prom" in sys.argv:
    # Parse minimal CLI args for quick testing
    def _arg(flag, dflt=None):
        return sys.argv[sys.argv.index(flag)+1] if flag in sys.argv else dflt

    base = _arg("--prom", "http://localhost:9090")
    yq = _arg("--y", None)
    if yq is None:
      yq = _fallback_y_candidates() #_default_mean_latency_query("default")
    xqs = [_arg("--x")] if "--x" in sys.argv else None
    if xqs is None:
      xqs = _fallback_x_queries() #_default_x_queries("default")
    window = int(_arg("--window", "3600"))
    step = _arg("--step", "5s")
    csv_out = _arg("--csv", "ksurf_metrics.csv")
    qr_out = _arg("--qr", "ksurf_qr.json")

    end = dt.datetime.utcnow()
    start = end - dt.timedelta(seconds=window)

    X, y = collect_prometheus_and_tune(
        base_url=base,
        x_queries=[q for q in xqs if q],
        y_query=yq,
        start=start,
        end=end,
        step=step,
        csv_path=csv_out,
        qr_path=qr_out,
        min_points=300,
        seq_length=10,
        epochs=200,
        batch_size=32,
        lr=1e-3,
    )
    print(f"[prom] collected shapes: X={X.shape}, y={y.shape}")



# Main function
elif __name__ == "__main__":
    test_clear_track_multi_epoch()
