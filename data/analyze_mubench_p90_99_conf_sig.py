# analyze_mubench_p90_99_conf.py
# Aggregates multiple iterations per (algorithm, threshold), draws 95% CIs,
# prints per-threshold and global significance (Welch’s t, df, CI, Hedges’ g, p-value).

import os
import re
import glob
import math
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
#                     STATS HELPERS
# =========================================================

def t_critical_975(df: float) -> float:
    """Two-sided 95% t critical (alpha=0.05 -> 0.975 quantile)."""
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 80: 1.990, 100: 1.984
    }
    if not (isinstance(df, (int, float)) and math.isfinite(df)):
        return 1.96  # z approx
    if df <= 1:
        return table[1]
    keys = np.array(sorted(table.keys()), dtype=float)
    idx = (np.abs(keys - df)).argmin()
    return float(table[int(keys[idx])])


def welch_t_and_ci(a: np.ndarray, b: np.ndarray, alpha=0.05):
    """Welch's t-test components: (mean_diff, t, dof, ci_low, ci_high)."""
    a = np.asarray(a, dtype=float); a = a[np.isfinite(a)]
    b = np.asarray(b, dtype=float); b = b[np.isfinite(b)]
    n1, n2 = a.size, b.size
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    m1, m2 = a.mean(), b.mean()
    s1 = a.std(ddof=1) if n1 >= 2 else 0.0
    s2 = b.std(ddof=1) if n2 >= 2 else 0.0
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
    """Hedges' g (bias-corrected Cohen's d) for two independent samples."""
    a = np.asarray(a, dtype=float); a = a[np.isfinite(a)]
    b = np.asarray(b, dtype=float); b = b[np.isfinite(b)]
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


# --------- Tiny incomplete beta / t-CDF for p-values (no SciPy) ---------

def _log_beta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)

def _betacf(a, b, x, max_iter=200, eps=1e-12):
    am, bm = 1.0, 1.0
    az = 1.0
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    bz = 1.0 - qab * x / qap
    aold = 0.0
    for m in range(1, max_iter + 1):
        em = float(m)
        tem = em + em
        d = em * (b - em) * x / ((qam + tem) * (a + tem))
        ap = az + d * am
        bp = bz + d * bm
        d = -(a + em) * (qab + em) * x / ((a + tem) * (qap + tem))
        app = ap + d * az
        bpp = bp + d * bz
        aold, am, bm, az, bz = az, app / bpp, bp / bpp, app / bpp, 1.0
        if abs(az - aold) < eps * abs(az):
            return az
    return az

def _betainc_reg(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lnB = _log_beta(a, b)
    if x < (a + 1.0) / (a + b + 2.0):
        cf = _betacf(a, b, x)
        return math.exp(a * math.log(x) + b * math.log(1.0 - x) - lnB) * cf / a
    else:
        cf = _betacf(b, a, 1.0 - x)
        return 1.0 - math.exp(b * math.log(1.0 - x) + a * math.log(x) - lnB) * cf / b

def t_cdf(t, df):
    if not (isinstance(t, (int, float)) and isinstance(df, (int, float))):
        return float("nan")
    if not (math.isfinite(t) and math.isfinite(df)) or df <= 0:
        return float("nan")
    x = df / (df + t * t)
    a = df / 2.0
    b = 0.5
    if t >= 0:
        return 1.0 - 0.5 * _betainc_reg(a, b, x)
    else:
        return 0.5 * _betainc_reg(a, b, x)

def welch_pvalue_two_sided(t, df):
    c = t_cdf(t, df)
    if not math.isfinite(c):
        return float("nan")
    return 2.0 * min(c, 1.0 - c)


# =========================================================
#             FILENAME PARSING & AGGREGATION
# =========================================================

# Use within-run CI when there is only one iteration:
USE_WITHIN_RUN_CI_FALLBACK = True

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
    base = os.path.basename(fname)
    if not (base.startswith("mubench_") and base.endswith(".txt")):
        return None
    stem = base[:-4]
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
    """Per-file (iteration) summary + within-run stats for fallback."""
    y = df["elapsed_time"].dropna().astype(float).to_numpy()
    run_n = y.size
    run_std = float(np.std(y, ddof=1)) if run_n >= 2 else np.nan
    run_se = (run_std / np.sqrt(run_n)) if run_n >= 2 else np.nan
    return {
        "mean": float(np.mean(y)) if run_n else float("nan"),
        "p90": float(np.quantile(y, 0.90)) if run_n else float("nan"),
        "p95": float(np.quantile(y, 0.95)) if run_n else float("nan"),
        "p99": float(np.quantile(y, 0.99)) if run_n else float("nan"),
        "run_std": run_std,
        "run_se": run_se,
        "run_n": run_n,
    }


def aggregate_across_iterations(iter_vals: list):
    """Across-iteration mean, 95% CI half-width, n, and across-iteration σ."""
    arr = np.asarray(iter_vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n == 0:
        return float("nan"), float("nan"), 0, float("nan")
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n >= 2 else float("nan")
    if n >= 2 and np.isfinite(std):
        tcrit = t_critical_975(n - 1)
        half = tcrit * std / math.sqrt(n)
    else:
        half = float("nan")
    return mean, half, n, std


# =========================================================
#                    PLOTTING HELPERS
# =========================================================

def _safe_mean(arr):
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else None

def _clean_yerr(ci, expected_len):
    if ci is None:
        return None
    ci = np.asarray(ci, dtype=float).ravel()
    if ci.size != expected_len:
        if ci.size > expected_len:
            ci = ci[:expected_len]
        else:
            ci = np.pad(ci, (0, expected_len - ci.size), constant_values=np.nan)
    return ci if np.isfinite(ci).any() else None

def legend_with_n(name, series_vals, n_series=None, sigma_series=None, show_var=False):
    """Legend label with μ, σ (and Var optional), n≈; robust to NaNs/empties."""
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


def grouped_bar_ci(values_pivot, ci_half_pivot, counts_pivot, std_for_legend_pivot,
                   title, ylabel, outfile, fs=14,
                   label_maker=None, xlabel="Threshold (%)",
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
        sser = std_for_legend_pivot[algo].values if std_for_legend_pivot is not None else None
        yerr = _clean_yerr(ci, len(vals))
        xpos = x + (i - (len(algos) - 1)) * bar_w / 2 + i * bar_w

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


def multiline_ci(values_pivot, ci_half_pivot, counts_pivot, std_for_legend_pivot,
                 title, ylabel, outfile, fs=14,
                 label_maker=None, xlabel="CPU Threshold (μ)",
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
        sser = std_for_legend_pivot[algo].values if std_for_legend_pivot is not None else None
        yerr = _clean_yerr(ci, len(y))

        if np.isfinite(y).any():
            plt.plot(x, y, marker="o", linewidth=2,
                     label=legend_with_n(algo, y, nser, sser, show_var=show_variance))
            if yerr is not None:
                mask = np.isfinite(y) & np.isfinite(yerr)
                if mask.any():
                    plt.fill_between(x[mask], (y - yerr)[mask], (y + yerr)[mask], alpha=0.2, linewidth=0)
            for xi, yi in zip(x, y):
                if isinstance(yi, (int, float)) and math.isfinite(yi):
                    plt.text(xi, yi, f"{yi:.2f}", ha="center", va="bottom", fontsize=fs-4)

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


# =========================================================
#                     MAIN ANALYSIS
# =========================================================

def plot_mean_std_elapsed_time(directory="."):
    """
    • Parses mubench_*.txt files, supports multiple iterations per (algorithm, threshold).
    • Computes per-iteration metrics (mean, p90, p95, p99).
    • Aggregates across iterations → mean, 95% CI of mean, σ, n.
    • Plots grouped bars (mean) & lines (p90/p95/p99) with CI.
    • Prints significance per-threshold and globally (p-values, g, CI).
    """
    metrics_store = {}  # {(ALGO, THR): {"mean":[...], "p90":[...], "p95":[...], "p99":[...], "run_std":[...], "run_se":[...]}}
    raw_by_threshold = {}

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

        runm = compute_iteration_metrics(df)
        key = (algo, thr)
        if key not in metrics_store:
            metrics_store[key] = {k: [] for k in ("mean", "p90", "p95", "p99", "run_std", "run_se")}
        for k in ("mean", "p90", "p95", "p99", "run_std", "run_se"):
            metrics_store[key][k].append(runm[k])

        raw_by_threshold.setdefault(thr, {})
        if algo not in raw_by_threshold[thr]:
            raw_by_threshold[thr][algo] = df["elapsed_time"].dropna().reset_index(drop=True)

    # Aggregated table
    records = []
    for (algo, thr), m in metrics_store.items():
        for metric in ("mean", "p90", "p95", "p99"):
            avg, half, n, std_iter = aggregate_across_iterations(m[metric])
            std_for_legend = std_iter
            if (not np.isfinite(std_for_legend)) and len(m.get("run_std", [])):
                std_for_legend = _safe_mean(m["run_std"])
            # CI fallback for MEAN only, when n<2
            if (metric == "mean" and (not np.isfinite(half) or n < 2) and USE_WITHIN_RUN_CI_FALLBACK):
                if len(m.get("run_se", [])):
                    se_fallback = _safe_mean(m["run_se"])
                    if se_fallback is not None and math.isfinite(se_fallback):
                        half = 1.96 * se_fallback
            records.append({
                "algorithm": algo, "threshold": thr, "metric": metric,
                "value": avg, "ci_half": half, "n": n, "stdlegend": std_for_legend
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
        stdlg  = dfm.pivot(index="threshold", columns="algorithm", values="stdlegend").reindex(values.index)
        return values, ci, counts, stdlg

    # Label helpers
    def percent_to_micro(threshold_percent: float) -> int:
        return int(round(threshold_percent * 2000))

    def labels_percent(index_vals):
        return [f"{t:g}%" for t in index_vals]

    def labels_micro(index_vals):
        return [f"{percent_to_micro(t)}\u03bc" for t in index_vals]

    # --- Plots ---

    mean_vals, mean_ci, mean_counts, mean_stdlg = pivots_for("mean")
    grouped_bar_ci(
        mean_vals, mean_ci, mean_counts, mean_stdlg,
        title="Mean Elapsed Time by Algorithm and Threshold (95% CI across iterations)",
        ylabel="Mean Elapsed Time (ms)",
        outfile="mean_elapsed_time_bar_graph_percent_ci.png",
        fs=14, label_maker=labels_percent, xlabel="Threshold (%)", show_variance=True
    )

    p90_vals, p90_ci, p90_counts, p90_stdlg = pivots_for("p90")
    multiline_ci(
        p90_vals, p90_ci, p90_counts, p90_stdlg,
        title="p90 Elapsed Time by Algorithm and CPU Threshold (95% CI across iterations)",
        ylabel="p90 Elapsed Time (ms)",
        outfile="p90_elapsed_time_line_graph_micro_ci.png",
        fs=14, label_maker=labels_micro, xlabel="CPU Threshold (\u03bc)", show_variance=True
    )

    p95_vals, p95_ci, p95_counts, p95_stdlg = pivots_for("p95")
    multiline_ci(
        p95_vals, p95_ci, p95_counts, p95_stdlg,
        title="p95 Elapsed Time by Algorithm and Threshold (95% CI across iterations)",
        ylabel="p95 Elapsed Time (ms)",
        outfile="p95_elapsed_time_line_graph_percent_ci.png",
        fs=14, label_maker=labels_percent, xlabel="Threshold (%)", show_variance=True
    )

    p99_vals, p99_ci, p99_counts, p99_stdlg = pivots_for("p99")
    multiline_ci(
        p99_vals, p99_ci, p99_counts, p99_stdlg,
        title="p99 Elapsed Time by Algorithm and Threshold (95% CI across iterations)",
        ylabel="p99 Elapsed Time (ms)",
        outfile="p99_elapsed_time_line_graph_percent_ci.png",
        fs=14, label_maker=labels_percent, xlabel="Threshold (%)", show_variance=True
    )

    # --- Significance: per-threshold (pooled per-iteration means at each threshold) ---
    rows = []
    per_thr_algo_means = {}
    for (algo, thr), m in metrics_store.items():
        per_thr_algo_means.setdefault(thr, {})[algo] = np.asarray(m["mean"], dtype=float)

    for thr, by_algo in sorted(per_thr_algo_means.items()):
        for a1, a2 in itertools.combinations(sorted(by_algo.keys()), 2):
            A = by_algo[a1]; B = by_algo[a2]
            mean_diff, t, df, lo, hi = welch_t_and_ci(A, B, alpha=0.05)
            g = hedges_g(A, B)
            p = welch_pvalue_two_sided(t, df)
            significant = (isinstance(p, float) and math.isfinite(p) and p < 0.05)
            rows.append({
                "threshold_percent": thr,
                "algos": f"{a1} vs {a2}",
                "mean_diff_ms": mean_diff,
                "welch_t": t,
                "welch_df": df,
                "p_value": p,
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
        with pd.option_context("display.max_rows", 200, "display.width", 220, "display.max_colwidth", 60):
            print("\nPer-threshold significance (Welch's t, df, 95% CI, Hedges' g, p-value):")
            print(sig_df[[
                "threshold_percent", "algos", "mean_diff_ms", "welch_t", "welch_df",
                "p_value", "diff_ci95_low", "diff_ci95_high", "hedges_g",
                "significant_0.05", "n1", "n2"
            ]])

    # --- Significance: GLOBAL (all thresholds & iterations pooled by algo, using per-iteration means) ---
    global_by_algo = {}
    for (algo, thr), m in metrics_store.items():
        global_by_algo.setdefault(algo, []).extend(list(np.asarray(m["mean"], dtype=float)))

    global_rows = []
    algos_all = sorted(global_by_algo.keys())
    for a1, a2 in itertools.combinations(algos_all, 2):
        A = np.asarray(global_by_algo[a1], dtype=float); A = A[np.isfinite(A)]
        B = np.asarray(global_by_algo[a2], dtype=float); B = B[np.isfinite(B)]
        mean_diff, t, df, lo, hi = welch_t_and_ci(A, B, alpha=0.05)
        g = hedges_g(A, B)
        p = welch_pvalue_two_sided(t, df)
        significant = (isinstance(p, float) and math.isfinite(p) and p < 0.05)
        global_rows.append({
            "algos": f"{a1} vs {a2}",
            "mean_diff_ms": mean_diff,
            "welch_t": t,
            "welch_df": df,
            "p_value": p,
            "diff_ci95_low": lo,
            "diff_ci95_high": hi,
            "hedges_g": g,
            "significant_0.05": significant,
            "n1": A.size,
            "n2": B.size,
        })

    if global_rows:
        global_df = pd.DataFrame(global_rows).sort_values("algos")
        global_df.to_csv("significance_stats_global.csv", index=False)
        with pd.option_context("display.max_rows", 200, "display.width", 220):
            print("\nGLOBAL significance across ALL thresholds & iterations (per-iteration MEANS pooled):")
            print(global_df[[
                "algos", "mean_diff_ms", "welch_t", "welch_df", "p_value",
                "diff_ci95_low", "diff_ci95_high", "hedges_g", "significant_0.05",
                "n1", "n2"
            ]])


if __name__ == "__main__":
    plot_mean_std_elapsed_time(directory=".")

