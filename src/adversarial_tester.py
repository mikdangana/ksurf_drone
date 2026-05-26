"""adversarial_tester.py

Generate *adversarial / unseen* versions of a CPU‑usage time‑series CSV,
run `test_pca()` from **ekf.py** on every generated file, collect the RMSE
(or whatever error `test_pca()` returns), and plot how the error grows with
severity.

Key change (2025‑08‑04, rev 2)
-----------------------------
`test_pca()` now receives **the mutated metric as -x** and **the baseline
metric as -y** so we measure how much the adversarial corruption degrades the
model’s ability to predict the *clean* series.

Scenario generators implemented
-------------------------------
1. **Gaussian noise** – add N(0, σ) to every sample; σ in {1 %, 2 %, … 20 % of
   the original series std}.
2. **Spike noise** – insert `k` random spikes of height `spike_mag × σ`;
   k∈{1 %, 5 %, 10 % of len(series)}.
3. **Linear drift** – add a ramp that reaches `drift_mag × σ` at the end.
4. **Delayed monitoring** – shift the series by `k = severity × len(series)`
   samples to emulate metrics that arrive late.
5. **Missing data** – randomly drop `k = severity × len(series)` samples and
   interpolate to keep the time series length constant.

Outputs
-------
* `data_adversarial/SCENARIO_#.csv` – mutated data sets containing both the
  original column *CPU* and the adversarial column *CPU_adv*.
* `results.csv`            – table: scenario, severity, error
* `results.png`            – matplotlib plot of error vs. severity.

Run:
-----
```
python adversarial_tester.py  \
       -f data/cpu_trace.csv \
       -x CPU  # baseline column (will be saved as -y)
```
"""

import argparse, sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Scenario definitions
# -----------------------------------------------------------------------------

def gaussian(series: np.ndarray, severity: float) -> np.ndarray:
    """Add iid Gaussian noise. *severity* is a σ‑multiplier (e.g. 0.05 → 5 %)."""
    sigma = severity * np.std(series)
    return series + np.random.normal(0, sigma, size=series.shape)


def spikes(series: np.ndarray, severity: float) -> np.ndarray:
    """Inject random spikes. *severity* is the fraction of samples to spike."""
    out = series.copy()
    k = max(1, int(len(series) * severity))
    idx = np.random.choice(len(series), k, replace=False)
    spike_sigma = 4 * np.std(series)
    out[idx] += np.random.normal(0, spike_sigma, size=k)
    return out


def drift(series: np.ndarray, severity: float) -> np.ndarray:
    """Add linear drift reaching ±severity·σ at the end."""
    sigma = np.std(series)
    ramp = np.linspace(0, severity * sigma, len(series))
    return series + ramp


def delayed(series: np.ndarray, severity: float) -> np.ndarray:
    """Simulate monitoring delay by shifting the series to the right."""
    k = int(len(series) * severity)
    if k == 0:
        return series.copy()
    out = np.empty_like(series)
    out[:k] = series[0]
    out[k:] = series[:-k]
    return out


def missing(series: np.ndarray, severity: float) -> np.ndarray:
    """Randomly remove samples (set to NaN) and interpolate."""
    out = series.astype(float).copy()
    k = max(1, int(len(series) * severity))
    idx = np.random.choice(len(series), k, replace=False)
    out[idx] = np.nan
    return pd.Series(out).interpolate(limit_direction="both").values


N=13

SCENARIOS0 = [
    ("gaussian", gaussian, [0.2+0.0001*2**i for i in range(N)]),
    ("drift",    drift,    [0.2+0.0001*2**i for i in range(N)]), 
    ("missing",  missing,  [0.2+0.0001*2**i for i in range(N)]) 
]
SCENARIOS1 = [
    ("spikes",   spikes,   [0.2+0.0001*2**i for i in range(N)]), 
    ("delay",    delayed,  [0.2+0.0001*2**i for i in range(N)]) 
]


# -----------------------------------------------------------------------------
# Harness to run ekf.test_pca on a generated file (with caching)
# -----------------------------------------------------------------------------


def run_test_pca_cached(
    csv_path: Path,
    x_col: str,
    y_col: str,
    std_flag: bool,
    cache_df: pd.DataFrame,
) -> tuple[float, pd.DataFrame]:
    """Return error, populating *cache_df* if call is new."""
    # Key used for cache lookup
    scenario = csv_path.stem.split("_")[0]
    severity = float(csv_path.stem.split("_")[1])
    mask = (
        (cache_df.scenario == scenario)
        & (cache_df.severity == severity)
        & (cache_df.std_flag == std_flag)
    )
    if mask.any():
        return float(cache_df.loc[mask, "error"].iloc[0]), cache_df

    # --- cache miss → compute ---
    import ekf  # local import to avoid dependency at import‑time

    save_argv = sys.argv[:]
    sys.argv = [
        "ekf.py",
        "--testpcacsv",
        "-f",
        str(csv_path),
        "-x",
        x_col,
        "-y",
        y_col,
    ]
    try:
        err_raw = ekf.test_pca()
    finally:
        sys.argv = save_argv

    if isinstance(err_raw, (list, tuple)):
        err = float(err_raw[1] if std_flag else err_raw[0])

    # Append to cache
    cache_df = pd.concat(
        [
            cache_df,
            pd.DataFrame(
                [
                    {
                        "scenario": scenario,
                        "severity": severity,
                        "std_flag": std_flag,
                        "error": err,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    return err, cache_df


# -----------------------------------------------------------------------------
# Harness to run ekf.test_pca on a generated file
# -----------------------------------------------------------------------------

def run_test_pca(csv_path: Path, x_col: str, y_col: str, std_flag: bool) -> float:
    """Invoke `ekf.test_pca()` programmatically and return its error value."""
    import ekf  # local import

    save_argv = sys.argv[:]
    sys.argv = [
        "ekf.py",
        "--testpcacsv",
        "-f",
        str(csv_path),
        "-x",
        x_col,  # mutated
        "-y",
        y_col,  # baseline
    ]
    try:
        err = ekf.test_pca()
    finally:
        sys.argv = save_argv

    if isinstance(err, (list, tuple)):
        return float(err[1] if std_flag else err[0])
    return float(err)


# -----------------------------------------------------------------------------
# Evaluation helper (runs one GAUSS×STD combination)
# -----------------------------------------------------------------------------


def evaluate(
    scenarios: list[tuple[str, callable, list[float]]],
    baseline: np.ndarray,
    df_template: pd.DataFrame,
    out_dir: Path,
    std_flag: bool,
    cache_df: pd.DataFrame,
    x_col_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results: list[tuple[str, float, float]] = []

    for name, func, severities in scenarios:
        for sev in severities:
            mutated = func(baseline, sev)
            mdf = df_template.copy()
            adv_col = f"{x_col_name}_adv"
            mdf[adv_col] = mutated
            fname = out_dir / f"{name}_{sev:.3f}.csv"
            mdf.to_csv(fname, index=False)
            err, cache_df = run_test_pca_cached(fname, adv_col, x_col_name, std_flag, cache_df)
            results.append((name, sev, err))
    res_df = pd.DataFrame(results, columns=["scenario", "severity", "error"])
    return res_df, cache_df


# -----------------------------------------------------------------------------
# Matplotlib defaults – larger, clearer fonts
# -----------------------------------------------------------------------------

plt.rcParams.update(
    {
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 18,
    }
)


# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", default="data/cpu_analysis.csv", help="Source CSV with baseline CPU metric")
    parser.add_argument("-x", default="CPU", help="Baseline metric column (default: CPU)")
    parser.add_argument("--outdir", default="data_adversarial", help="Directory for outputs & cache")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--gauss", dest="gauss", action="store_true", help="Highlight GAUSS quadrant (default)")
    group.add_argument("--nogauss", dest="gauss", action="store_false", help="Highlight NOGAUSS quadrant")
    parser.set_defaults(gauss=True)
    parser.add_argument("--std", action="store_true", help="Plot error std instead of mean (highlighted quadrant)")
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(exist_ok=True)
    cache_file = out_dir / "cache.csv"
    cache_df = pd.read_csv(cache_file) if cache_file.exists() else pd.DataFrame(columns=["scenario", "severity", "std_flag", "error"])

    df = pd.read_csv(args.f)
    if args.x not in df.columns:
        raise ValueError(f"Column {args.x} not found in {args.f}")

    baseline = df[args.x].to_numpy(dtype=float)

    # Run all four combinations (gauss TRUE/FALSE × std TRUE/FALSE)
    combinations = [
        (True, False),
        (True, True),
        (False, False),
        (False, True),
    ]
    subplot_titles = [
        "Gaussian, Drift & Missing Data Error", "Gaussian, Drift, Missing Data Standard Deviation", "Spikes, Delay Error", "Spikes, Delay Standard Deviation"
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for (gauss_flag, std_flag), ax, title in zip(combinations, axes, subplot_titles):
        scenarios = SCENARIOS0 if gauss_flag else SCENARIOS1
        res_df, cache_df = evaluate(scenarios, baseline, df, out_dir, std_flag, cache_df, args.x)

        for name in res_df.scenario.unique():
            sub = res_df[res_df.scenario == name]
            ax.plot(sub.severity, sub.error, marker="o", label=name)

        mean_err = res_df.error.mean()
        std_err = res_df.error.std(ddof=0)
        ax.axhline(mean_err + 2 * std_err, ls="--", lw=1, label="2σ")
        ax.axhline(mean_err + 3 * std_err, ls=":", lw=1, label="3σ")

        ax.set_title(title) 
        ax.set_xlabel("Severity (\u03BB)")
        ax.set_ylabel("Error std" if std_flag else "Error")
        ax.grid(True, ls="--", alpha=0.4)
        if len(res_df.scenario.unique()) <= 5:  # keep legend compact
            ax.legend() 

        # If this quadrant matches CLI flags, also save a single‑plot version
        if gauss_flag == args.gauss and std_flag == args.std:
            plt.figure(figsize=(6, 4))
            for name in res_df.scenario.unique():
                sub = res_df[res_df.scenario == name]
                plt.plot(sub.severity, sub.error, marker="o", label=name)
            plt.axhline(mean_err + 2 * std_err, ls="--", lw=1, label="2σ")
            plt.axhline(mean_err + 3 * std_err, ls=":", lw=1, label="3σ")
            plt.xlabel("Severity (\u03BB or σ‑multiplier)")
            plt.ylabel("Error std" if std_flag else "Error")
            plt.title(f"Highlighted quadrant: {title}")
            plt.legend()
            plt.grid(True, ls="--", alpha=0.4)
            plt.tight_layout()
            plt.savefig(out_dir / "results.png", dpi=180)
            plt.close()

    # Persist updated cache after all runs
    cache_df.to_csv(cache_file, index=False)

    fig.tight_layout()
    fig.suptitle("EKF PCA error under adversarial perturbations (4‑quadrant view)", y=1.02, fontsize=14)
    fig.savefig(out_dir / "results_multi.png", dpi=180, bbox_inches="tight")
    plt.show()


# -----------------------------------------------------------------------------
# Main experiment driver
# -----------------------------------------------------------------------------

def main1():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", default="data/cpu_analysis.csv", help="Source CSV with baseline CPU metric")
    ap.add_argument("-x", default="CPU", help="Baseline metric column (default: CPU)")
    ap.add_argument("--outdir", default="data_adversarial", help="Directory for mutated CSVs")
    ap.add_argument("--std", action="store_true", help="Log/plot error std instead of mean when available")
    ap.add_argument("--gauss", action="store_true", help="Log/plot error gauss/delay/drift instead of spikes/missing when available")
    args = ap.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(exist_ok=True)
    cache_file = out_dir / "cache.csv"
    cache_df = pd.read_csv(cache_file) if cache_file.exists() else pd.DataFrame(columns=["scenario", "severity", "std_flag", "error"])


    df = pd.read_csv(args.f)
    if args.x not in df.columns:
        raise ValueError(f"Column {args.x} not found in {args.f}")

    baseline = df[args.x].values.astype(float)
    results: List[Tuple[str, float, float]] = []

    for name, func, severities in SCENARIOS0 if args.gauss else SCENARIOS1:
        for sev in severities:
            mutated = func(baseline, sev)
            mdf = df.copy()
            adv_col = f"{args.x}_adv"
            mdf[adv_col] = mutated
            fname = out_dir / f"{name}_{sev:.3f}.csv"
            mdf.to_csv(fname, index=False)
            # mutated → -x, baseline → -y
            err, cache_df = run_test_pca_cached(fname, adv_col, args.x, args.std, cache_df)
            results.append((name, sev, err))
            status = "cached" if (cache_df[(cache_df.scenario == name) & (cache_df.severity == sev) & (cache_df.std_flag == args.std)].shape[0] > 1) else "computed"
            print(f"{name:8s} sev={sev:<5} -> err={err:.4f}  [{status}]")

    # Persist cache
    cache_df.to_csv(cache_file, index=False)

    res_df = pd.DataFrame(results, columns=["scenario", "severity", "error"])
    res_df.to_csv(out_dir / "results.csv", index=False)

    plt.figure(figsize=(8, 5))
    for name in res_df["scenario"].unique():
        sub = res_df[res_df.scenario == name]
        plt.plot(sub["severity"], sub["error"], marker="o", label=name)

    # 2σ and 3σ reference lines
    mean_err = res_df.error.mean()
    std_err = res_df.error.std(ddof=0)
    plt.axhline(mean_err + 2 * std_err, ls="--", linewidth=1.2, label="2σ threshold")
    plt.axhline(mean_err + 3 * std_err, ls=":", linewidth=1.2, label="3σ threshold")
    plt.xlabel("σ‑multiplier")
    plt.ylabel("Error Standard Deviation" if args.std else "Error")
    plt.title("EKF PCA error under adversarial perturbations (p=32)")
    plt.legend()
    plt.grid(True, ls="--", alpha=.4)
    plt.tight_layout()
    plt.savefig(out_dir / "results.png", dpi=180)
    plt.show()


if __name__ == "__main__":
    main()

