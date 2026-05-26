# analyze_mubench_p90_99_conf.py
# Aggregates multiple iterations per (algorithm, threshold), plots 95% CIs,
# and shows μ, σ (and optionally variance) in legends. Also computes pairwise
# significance stats (Welch's t, Hedges' g) per threshold.

import os
import re
import glob
import math
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- add below the existing label helpers ---

def percent_to_milli(threshold_percent: float) -> float:
    """
    Map percent threshold to millipods:
      1% -> 2 m, 0.5% -> 1 m, 0.25% -> 0.5 m, etc.
    """
    return threshold_percent * 2.0

def labels_milli(index_vals):
    """Format tick labels in millipods."""
    return [f"{percent_to_milli(t):g}m" for t in index_vals]


# -------------------- Stats helpers --------------------

def t_critical_975(df: float) -> float:
    """
    Two-sided 95% t critical (alpha=0.05 -> 0.975 quantile) from a small lookup table,
    nearest-match fallback to avoid SciPy dependency.
    """
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 80: 1.990, 100: 1.984
    }
    if not (isinstance(df, (int, float)) and math.isfinite(df)):
        return 1.96  # z
    if df <= 1:
        return table[1]
    keys = np.array(sorted(table.keys()), dtype=float)
    idx = (np.abs(keys - df)).argmin()
    return float(table[int(keys[idx])])


def welch_t_and_ci(a: np.ndarray, b: np.ndarray, alpha=0.05):
    """
    Welch's t-test components (no SciPy): returns (mean_diff, t, dof, ci_lower, ci_upper).
    Uses t critical from t_critical_975 for 95% CI.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    n1, n2 = a.size, b.size
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    m1, m2 = a.mean(), b.mean()
    s1, s2 = a.std(ddof=1) if n1 >= 2 else 0.0, b.std(ddof=1) if n2 >= 2 else 0.0
    mean_diff = m1 - m2
    se = math.sqrt((s1 ** 2) / n1 + (s2 ** 2) / n2) if (n1 > 0 and n2 > 0) else float("nan")
    if n1 >= 2 and n2 >= 2 and se > 0:
        denom = ((s1 ** 2) / n1) ** 2 / (n1 - 1) + ((s2 ** 2) / n2) ** 2 / (n2 - 1)
        df = ((s1 ** 2) / n1 + (s2 ** 2) / n2) ** 2 / denom if denom > 0 else float("nan")
        t = mean_diff / se
    else:
        df, t = float("nan"), float("nan")
    tcrit = t_critical_975(df)
    half = tcrit * se if (isinstance(se, float) and math.isfinite(se)) else float("nan")
    return mean_diff, t, df, mean_diff - half, mean_diff + half


def hedges_g(a: np.ndarray, b: np.ndarray):
    """
    Hedges' g (bias-corrected Cohen's d) for two independent samples.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    n1, n2 = a.size, b.size
    if n1 < 2 or n2 < 2:
        return float("nan")
    m1, m2 = a.mean(), b.mean()
    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    sp2 = ((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2)
    if sp2 <= 0:
        return float("nan")
    d = (m1 - m2) / math.sqrt(sp2)
    df = n1 + n2 - 2
    J = 1.0 - 3.0 / (4.0 * df - 1.0) if df > 1 else 1.0
    return J * d


# -------------------- Parsing & aggregation --------------------
def parse_name(fname: str):
    """
    Accepts:
      mubench_<algo>_<thr>.txt
      mubench_<algo>_<thr>_1.txt
      mubench_<algo>_<thr>_i1.txt
      mubench_<algo>_<thr>_iter2.txt
      mubench_<algo>_<thr>_rep03.txt
    Returns (ALGO_UPPER, threshold_float, iteration_or_None)
    """
    import os, re
    base = os.path.basename(fname)
    if not (base.startswith("mubench_") and base.endswith(".txt")):
        return None
    stem = base[:-4]  # drop .txt
    # algo = any non-underscore; thr = int or float; optional suffix: _<digits> or _i\d/_it\d/_iter\d/_rep\d
    m = re.match(
        r'^mubench_([^_]+)_([0-9]+(?:\.[0-9]+)?)(?:_(?:(?:i|it|iter|rep)?(\d+)))?$',
        stem, flags=re.IGNORECASE
    )
    if not m:
        return None
    algo = m.group(1).upper()
    threshold = float(m.group(2))
    it = int(m.group(3)) if m.group(3) else None
    return algo, threshold, it


def compute_iteration_metrics(df: pd.DataFrame):
    """
    Per-file (iteration) summary metrics.
    """
    elapsed = df["elapsed_time"].dropna().astype(float)
    return {
        "mean": float(elapsed.mean()) if len(elapsed) else float("nan"),
        "p90": float(elapsed.quantile(0.90)) if len(elapsed) else float("nan"),
        "p95": float(elapsed.quantile(0.95)) if len(elapsed) else float("nan"),
        "p99": float(elapsed.quantile(0.99)) if len(elapsed) else float("nan"),
    }


def aggregate_across_iterations(iter_vals: list):
    """
    Given a list of per-iteration metric values, return:
      mean, CI_half (95%), n, std (sample).
    """
    arr = np.asarray(iter_vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n == 0:
        return float("nan"), float("nan"), 0, float("nan")
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n >= 2 else float("nan")
    if n >= 2 and math.isfinite(std):
        tcrit = t_critical_975(n - 1)
        half = tcrit * std / math.sqrt(n)
    else:
        half = float("nan")
    return mean, half, n, std


# -------------------- Plotting helpers --------------------
# --- drop-in fixes: safe helpers + plotting functions ---

def _safe_mean(arr):
    """Return float mean of finite values, or None if empty/all-NaN."""
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else None

def _clean_yerr(ci, expected_len):
    """
    Return a numeric yerr array if it contains at least one finite value and
    matches the expected length; otherwise return None so Matplotlib skips errorbars.
    """
    if ci is None:
        return None
    ci = np.asarray(ci, dtype=float).ravel()
    # match length (defensively clip or pad with NaN)
    if ci.size != expected_len:
        if ci.size > expected_len:
            ci = ci[:expected_len]
        else:
            ci = np.pad(ci, (0, expected_len - ci.size), constant_values=np.nan)
    # only keep if there is at least one finite value
    return ci if np.isfinite(ci).any() else None

def legend_with_n(name, series_vals, n_series=None, sigma_series=None, show_var=False):
    """
    Legend label: μ, σ (and optionally Var), n≈ — all computed safely.
    If a component has no finite values, it is omitted (or shown as em dash for μ).
    """
    mu = _safe_mean(series_vals)
    n_mean = _safe_mean(n_series) if n_series is not None else None
    sigma_mean = _safe_mean(sigma_series) if sigma_series is not None else None

    parts = [f"μ={mu:.2f}" if mu is not None else "μ=—"]
    if sigma_mean is not None and math.isfinite(sigma_mean):
        parts.append(f"σ={sigma_mean:.2f}")
        if show_var:
            parts.append(f"Var={(sigma_mean**2):.2f}")
    if n_mean is not None and math.isfinite(n_mean):
        parts.append(f"n≈{int(round(n_mean))}")

    return f"{name} ({', '.join(parts)})"

def grouped_bar_ci(values_pivot, ci_half_pivot, counts_pivot, std_pivot,
                   title, ylabel, outfile, fs=14,
                   label_maker=None, xlabel="CPU Threshold (m)",
                   show_variance=True):
    thresholds = values_pivot.index.values
    xlabels = label_maker(thresholds) if label_maker else [str(x) for x in thresholds]
    x = np.arange(len(xlabels))
    algos = list(values_pivot.columns)

    plt.figure(figsize=(10, 6))
    bar_w = 0.8 / max(1, len(algos))

    for i, algo in enumerate(algos):
        vals = np.asarray(values_pivot[algo].values, dtype=float)
        ci   = ci_half_pivot[algo].values if ci_half_pivot is not None else None
        nser = counts_pivot[algo].values if counts_pivot is not None else None
        sser = std_pivot[algo].values    if std_pivot is not None else None
        yerr = _clean_yerr(ci, len(vals))
        xpos = x + (i - (len(algos) - 1) / 2) * bar_w

        plt.bar(
            xpos, vals, bar_w,
            label=legend_with_n(algo, vals, nser, sser, show_var=show_variance),
            yerr=yerr, capsize=6 if yerr is not None else 0,
        )
        for px, v in zip(xpos, vals):
            if isinstance(v, (int, float)) and math.isfinite(v):
                plt.text(px, v, f"{v:.2f}", ha="center", va="bottom", fontsize=fs-4)

    plt.title(title, fontsize=fs)
    plt.xlabel(xlabel, fontsize=fs)
    plt.ylabel(ylabel, fontsize=fs)
    plt.xticks(x, xlabels, fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.grid(axis="y", linestyle=":")
    plt.legend(title="Algorithm", fontsize=fs, title_fontsize=fs)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.show()

def multiline_ci(values_pivot, ci_half_pivot, counts_pivot, std_pivot,
                 title, ylabel, outfile, fs=14,
                 label_maker=None, xlabel="CPU Threshold (m)",
                 show_variance=True):
    thresholds = values_pivot.index.values
    xlabels = label_maker(thresholds) if label_maker else [str(x) for x in thresholds]
    x = np.arange(len(xlabels))
    algos = list(values_pivot.columns)

    plt.figure(figsize=(10, 6))
    for algo in algos:
        y    = np.asarray(values_pivot[algo].values, dtype=float)
        ci   = ci_half_pivot[algo].values if ci_half_pivot is not None else None
        nser = counts_pivot[algo].values if counts_pivot is not None else None
        sser = std_pivot[algo].values    if std_pivot is not None else None
        yerr = _clean_yerr(ci, len(y))

        # plot the line if there is at least one finite y
        if np.isfinite(y).any():
            plt.plot(x, y, marker="o", linewidth=2,
                     label=legend_with_n(algo, y, nser, sser, show_var=show_variance))
            # CI band only where both y and yerr are finite
            if yerr is not None:
                mask = np.isfinite(y) & np.isfinite(yerr)
                if mask.any():
                    lo = (y - yerr)[mask]
                    hi = (y + yerr)[mask]
                    plt.fill_between(x[mask], lo, hi, alpha=0.2, linewidth=0)

            for xi, yi in zip(x, y):
                if isinstance(yi, (int, float)) and math.isfinite(yi):
                    plt.text(xi, yi, f"{yi:.2f}", ha="center", va="bottom", fontsize=fs-4)
        else:
            # nothing finite -> skip plotting for this series
            continue

    plt.title(title, fontsize=fs)
    plt.xlabel(xlabel, fontsize=fs)
    plt.ylabel(ylabel, fontsize=fs)
    plt.xticks(x, xlabels, fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.grid(True, linestyle=":")
    plt.legend(title="Algorithm", fontsize=fs, title_fontsize=fs)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.show()



# -------------------- Main analysis --------------------

def plot_mean_std_elapsed_time(directory="."):
    """
    • Parses mubench_*.txt files, supporting multiple iterations per (algorithm, threshold).
    • Computes per-iteration metrics (mean, p90, p95, p99).
    • Aggregates across iterations → mean, 95% CI of mean, σ, n.
    • Plots:
        - Grouped bars for mean with CI error bars and legends showing μ, σ, Var (optional), n≈.
        - Lines for p90/p95/p99 with CI bands and legends showing μ, σ, Var (optional), n≈.
    • Writes pairwise significance stats (Welch's t, CI of difference, Hedges' g) to CSV.
    """
    # Collect per-iteration metrics
    metrics_store = {}  # {(ALGO, THRESH): {"mean":[...], "p90":[...], "p95":[...], "p99":[...]}}
    raw_by_threshold = {}  # optional raw for an example plot

    paths = glob.glob(os.path.join(directory, "mubench_*.txt"))
    if not paths:
        print("No files found matching mubench_*.txt in", directory)
        return

    for path in sorted(paths):
        parsed = parse_name(path)
        if not parsed:
            print("Skipping unrecognized file name:", os.path.basename(path))
            continue
        algo, thr, _ = parsed

        try:
            df = pd.read_csv(
                path,
                delim_whitespace=True,
                header=None,
                names=["timestamp", "elapsed_time", "http_status", "processed", "pending"],
            )
        except Exception as e:
            print(f"Failed to read {path}: {e}")
            continue

        if df.empty or "elapsed_time" not in df:
            print("No elapsed_time rows in:", os.path.basename(path))
            continue

        # per-file metrics
        runm = compute_iteration_metrics(df)
        key = (algo, thr)
        if key not in metrics_store:
            metrics_store[key] = {k: [] for k in ("mean", "p90", "p95", "p99")}
        for k in ("mean", "p90", "p95", "p99"):
            metrics_store[key][k].append(runm[k])

        # keep first raw series per (algo, thr) for optional raw plot
        raw_by_threshold.setdefault(thr, {})
        if algo not in raw_by_threshold[thr]:
            raw_by_threshold[thr][algo] = df["elapsed_time"].dropna().reset_index(drop=True)

    # Build aggregated table
    records = []
    for (algo, thr), m in metrics_store.items():
        for metric in ("mean", "p90", "p95", "p99"):
            avg, half, n, std = aggregate_across_iterations(m[metric])
            records.append({
                "algorithm": algo, "threshold": thr, "metric": metric,
                "value": avg, "ci_half": half, "n": n, "std": std
            })
    agg = pd.DataFrame.from_records(records)
    if agg.empty:
        print("No aggregated metrics; check your input files.")
        return

    # Pivot helpers
    def pivots_for(metric: str):
        dfm = agg[agg["metric"] == metric]
        values = dfm.pivot(index="threshold", columns="algorithm", values="value").sort_index()
        ci     = dfm.pivot(index="threshold", columns="algorithm", values="ci_half").reindex(values.index)
        counts = dfm.pivot(index="threshold", columns="algorithm", values="n").reindex(values.index)
        std    = dfm.pivot(index="threshold", columns="algorithm", values="std").reindex(values.index)
        return values, ci, counts, std

    # Label helpers
    def percent_to_micro(threshold_percent: float) -> int:
        # map % → micropods under 1% = 2000 μ assumption
        return int(round(threshold_percent * 2000))

    def labels_percent(index_vals):
        return [f"{t:g}%" for t in index_vals]

    def labels_micro(index_vals):
        return [f"{percent_to_micro(t)}\u03bc" for t in index_vals]

    # --- 1) Grouped BAR: mean (+95% CI) ---
    mean_vals, mean_ci, mean_counts, mean_std = pivots_for("mean")
    grouped_bar_ci(
        mean_vals, mean_ci, mean_counts, mean_std,
        title="Mean Elapsed Time by Algorithm and Threshold (95% CI across iterations)",
        ylabel="Mean Elapsed Time (ms)",
        outfile="mean_elapsed_time_bar_graph_percent_ci.png",
        fs=14,
        label_maker=labels_milli,
        xlabel="CPU Threshold (m)",
        show_variance=True  # include Var in legend; set False to hide
    )

    # --- 2) LINE plots with CI bands for p90/p95/p99 ---
    p90_vals, p90_ci, p90_counts, p90_std = pivots_for("p90")
    multiline_ci(
        p90_vals, p90_ci, p90_counts, p90_std,
        title="p90 Elapsed Time by Algorithm and CPU Threshold (95% CI across iterations)",
        ylabel="p90 Elapsed Time (ms)",
        outfile="p90_elapsed_time_line_graph_micro_ci.png",
        fs=14,
        label_maker=labels_milli,
        xlabel="CPU Threshold (m)",
        show_variance=True
    )

    p95_vals, p95_ci, p95_counts, p95_std = pivots_for("p95")
    multiline_ci(
        p95_vals, p95_ci, p95_counts, p95_std,
        title="p95 Elapsed Time by Algorithm and Threshold (95% CI across iterations)",
        ylabel="p95 Elapsed Time (ms)",
        outfile="p95_elapsed_time_line_graph_percent_ci.png",
        fs=14,
        label_maker=labels_milli,
        xlabel="CPU Threshold (m)",
        show_variance=True
    )

    p99_vals, p99_ci, p99_counts, p99_std = pivots_for("p99")
    multiline_ci(
        p99_vals, p99_ci, p99_counts, p99_std,
        title="p99 Elapsed Time by Algorithm and Threshold (95% CI across iterations)",
        ylabel="p99 Elapsed Time (ms)",
        outfile="p99_elapsed_time_line_graph_percent_ci.png",
        fs=14,
        label_maker=labels_milli,
        xlabel="CPU Threshold (m)",
        show_variance=True
    )

    # --- 3) Optional: 2×2 raw subplots (first-seen iteration per (algo,thr)) ---
    def raw_multiplot(raw_map, wanted_thresholds=(0.5, 1.0, 2.0, 4.0), outfile="raw_elapsed_2x2_micro.png"):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False, sharey=False)
        axes = axes.flatten()

        for idx, thr in enumerate(wanted_thresholds):
            ax = axes[idx]
            match_thr = None
            for available in raw_map.keys():
                if abs(available - thr) < 1e-9:
                    match_thr = available
                    break
            if match_thr is None:
                for available in raw_map.keys():
                    if abs(available - thr) < 1e-3:
                        match_thr = available
                        break

            micro_label = f"{percent_to_milli(thr)}m"
            ax.set_title(f"Request times at {milli_label} Threshold")
            ax.set_xlabel("Request Interval")
            ax.set_ylabel("Elapsed Time (ms)")
            ax.grid(True, linestyle=":")

            if match_thr is None or not raw_map.get(match_thr):
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=11, alpha=0.7)
                continue

            for algo in sorted(raw_map[match_thr].keys()):
                y = np.asarray(raw_map[match_thr][algo].values, dtype=float)
                x = np.arange(len(y))
                mu = float(np.nanmean(y)) if y.size else 0.0
                sigma = float(np.nanstd(y, ddof=1)) if y.size >= 2 else float("nan")
                label = f"{algo} (μ={mu:.2f}" + (f", σ={sigma:.2f}" if math.isfinite(sigma) else "") + ")"
                plt.plot(x, y, linewidth=1.5, label=label)

            ax.legend(fontsize=8)

        for j in range(len(wanted_thresholds), 4):
            fig.delaxes(axes[j])

        fig.suptitle("Raw Request Elapsed Times by CPU Threshold (m)")
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        fig.savefig(outfile, dpi=150)
        plt.show()

    # Uncomment to generate raw multiplot:
    # raw_multiplot(raw_by_threshold)

    # --- 4) Pairwise significance stats per threshold (based on per-iteration MEANS) ---
    rows = []
    per_thr_algo_means = {}
    for (algo, thr), m in metrics_store.items():
        per_thr_algo_means.setdefault(thr, {})[algo] = np.asarray(m["mean"], dtype=float)

    for thr, by_algo in sorted(per_thr_algo_means.items()):
        for a1, a2 in itertools.combinations(sorted(by_algo.keys()), 2):
            A = by_algo[a1]
            B = by_algo[a2]
            mean_diff, t, df, lo, hi = welch_t_and_ci(A, B, alpha=0.05)
            g = hedges_g(A, B)
            significant = (isinstance(lo, float) and isinstance(hi, float)) and (lo > 0 or hi < 0)
            rows.append({
                "threshold_percent": thr,
                "algos": f"{a1} vs {a2}",
                "mean_diff_ms": mean_diff,
                "welch_t": t,
                "welch_df": df,
                "diff_ci95_low": lo,
                "diff_ci95_high": hi,
                "hedges_g": g,
                "significant_0.05": significant,
                "n1": np.isfinite(A).sum(),
                "n2": np.isfinite(B).sum(),
            })

    if rows:
        sig_df = pd.DataFrame(rows).sort_values(["threshold_percent", "algos"])
        sig_df.to_csv("significance_stats.csv", index=False)
        with pd.option_context("display.max_rows", 200, "display.width", 160, "display.max_colwidth", 60):
            print("\nPairwise significance (mean diffs, Welch's t, 95% CI, Hedges' g):")
            print(sig_df[
                ["threshold_percent", "algos", "mean_diff_ms", "welch_t", "welch_df",
                 "diff_ci95_low", "diff_ci95_high", "hedges_g", "significant_0.05", "n1", "n2"]
            ])


if __name__ == "__main__":
    plot_mean_std_elapsed_time(directory=".")

