from ekf import *
from clear_track import apply_lstm_kalman_filter, train_lstm

import os
import json
import numpy as np
from typing import Optional, Any


def _as_array(x: Any) -> np.ndarray:
    """Convert JSON scalar/matrix to numpy array."""
    if isinstance(x, (int, float)):
        return np.array([[float(x)]], dtype=float)
    arr = np.array(x, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    return arr


def _coerce_to_shape(mat_or_scalar: Any, target_shape: tuple) -> np.ndarray:
    """
    Coerce a given scalar/matrix to the target square shape.
    Rules:
      - If scalar => scalar * I(target_dim)
      - If matrix with same shape => use as-is
      - If matrix larger => take top-left submatrix
      - If matrix smaller => pad to target with zeros (keep on diagonal if square 1x1)
    """
    tgt_r, tgt_c = target_shape
    if tgt_r != tgt_c:
        # We only handle square Q/R here; if your EKF uses non-square R/H,
        # adapt accordingly.
        pass

    tgt_n = tgt_r
    M = _as_array(mat_or_scalar)

    # scalar case (1x1) -> expand to identity * scalar
    if M.shape == (1, 1) and tgt_n > 1:
        return np.eye(tgt_n, dtype=float) * float(M[0, 0])

    # exact match
    if M.shape == (tgt_n, tgt_n):
        return M

    # larger matrix -> crop
    if M.shape[0] >= tgt_n and M.shape[1] >= tgt_n:
        return M[:tgt_n, :tgt_n]

    # smaller -> pad
    out = np.zeros((tgt_n, tgt_n), dtype=float)
    r = min(tgt_n, M.shape[0])
    c = min(tgt_n, M.shape[1])
    out[:r, :c] = M[:r, :c]
    # If it was a scalar (or effectively diagonal), keep scalar on diagonal
    if M.shape == (1, 1):
        np.fill_diagonal(out, float(M[0, 0]))
    return out


class Ksurf:
    def __init__(
        self,
        nmsmt: int = 2,                # measurement dimension
        dx: int = 2,                   # state dimension
        n_components: int = 10,
        att_fname: Optional[str] = None,
        att_col: Optional[str] = None,
        sliding_window_size: Optional[int] = None,
        qr_path: str = "/home/ubuntu/kfpca/src/ksurf_qr.json",  # <— default path to learned Q/R
    ):
        # Create the PCA+EKF wrapper
        self.kf = PCAKalmanFilter(
            nmsmt=nmsmt,
            dx=dx,
            n_components=n_components,
            normalize=True,
            att_fname=att_fname,
            att_col=att_col
        )
        self.X = []
        self.y = []

        # Try to load learned Q/R (if present), then adapt to EKF shapes
        self._try_load_qr_and_apply(qr_path)

    def _try_load_qr_and_apply(self, qr_path: str) -> None:
        """Load Q/R from json if present and apply to ekf with shape adaptation."""
        if not hasattr(self.kf, "ekf"):
            return  # nothing to do

        try:
            Q_cur = np.array(self.kf.ekf.Q, dtype=float)
            R_cur = np.array(self.kf.ekf.R, dtype=float)
            q_shape = Q_cur.shape
            r_shape = R_cur.shape
        except Exception:
            # If EKF doesn’t expose Q/R yet, bail out gracefully
            return

        if not os.path.isfile(qr_path):
            print(f"[ksurf] Q/R file not found: {qr_path} (keeping defaults)")
            return

        try:
            with open(qr_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[ksurf] Failed to read {qr_path}: {e} (keeping defaults)")
            return

        # Prefer matrix fields; else fall back to scalar (lstm or residual)
        Q_src = None
        R_src = None

        if "Q_matrix_ekf" in data:
            Q_src = _as_array(data["Q_matrix_ekf"])
        elif "Q_scalar_lstm" in data:
            Q_src = _as_array([[float(data["Q_scalar_lstm"])]])

        if "R_matrix_ekf" in data:
            R_src = _as_array(data["R_matrix_ekf"])
        elif "R_scalar_lstm" in data:
            R_src = _as_array([[float(data["R_scalar_lstm"])]])

        # As a last resort, try residual scalars if matrices/scalars above missing
        if Q_src is None and "Q_scalar_residual" in data:
            Q_src = _as_array([[float(data["Q_scalar_residual"])]])
        if R_src is None and "R_scalar_residual" in data:
            R_src = _as_array([[float(data["R_scalar_residual"])]])

        if Q_src is None and R_src is None:
            print(f"[ksurf] No usable Q/R fields in {qr_path} (keeping defaults)")
            return

        # Coerce to EKF shapes
        try:
            if Q_src is not None:
                Q_new = _coerce_to_shape(Q_src, q_shape)
                self.kf.ekf.Q = Q_new
            if R_src is not None:
                R_new = _coerce_to_shape(R_src, r_shape)
                self.kf.ekf.R = R_new
            print(f"[ksurf] Loaded Q/R from {qr_path} → "
                  f"Q.shape={self.kf.ekf.Q.shape}, R.shape={self.kf.ekf.R.shape}")
        except Exception as e:
            print(f"[ksurf] Failed to apply Q/R from {qr_path}: {e} (keeping defaults)")

    def update(self, X, y):
        """Store action-context X and target y, and update EKF with correctly shaped vectors."""
        print("update().X, y =", X, y)
        self.X.extend(X)
        self.y.extend(y)

        # Ensure y is flattened to 1D list of scalars
        y_flat = np.array(self.y).flatten().tolist()

        # Require at least 2 measurements to proceed
        if len(y_flat) < 2:
            return

        priors = None  # prevent NameError on final print

        for i in range(1, len(y_flat)):
            print("ksurf.update().y_flat =", y_flat)
            msmts = [y_flat[i - 1], y_flat[i]]

            # Normalize & set H with past X
            msmts = self.kf.pca_normalize(msmts, is_scalar=False)
            x_window = self.X[i - 1:i + 1]
            self.kf.to_H([[x, x] for x in x_window], [msmts])

            # Reshape to match EKF's expected shape: (2, 1)
            msmts_reshaped = np.reshape(msmts[-2:], (2, 1))
            self.kf.update([msmts_reshaped])
        print("update() done, priors =", priors)

    def predict(self, X):
        """Use PCA attention to generate predictions for each X row."""
        print("predict().X,y =", X, self.y)
        if len(self.y) < 2:
            return np.zeros(len(X)), np.ones(len(X))  # no prior data

        # Apply PCA attention
        attn_input = np.array(self.y[-self.kf.n_components:])
        values = self.kf.predict(X)

        mean = np.array([values[-1][-1]] * len(X))
        std = np.array([1.0] * len(X))  # placeholder
        print("predict().mean,std,values =", mean, std, values)
        return mean, std

    def reset(self):
        print("reset()")
        # If you want to preserve learned Q/R, pass the same qr_path again on re-init
        # or pull self.kf.ekf.Q/R into local vars and restore after re-init.
        pass


if __name__ == "__main__":
    None

