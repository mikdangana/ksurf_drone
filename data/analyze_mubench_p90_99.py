import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_mean_std_elapsed_time(directory="."):
    """
    Parses mubench_*.txt files and generates:
      • Grouped BAR chart: mean elapsed time (stddev as error bars)  [14pt fonts, % thresholds]
      • LINE charts: p90 (μ thresholds), p95 [14pt, % thresholds], p99 [14pt, % thresholds]
      • 2×2 MULTI-PLOT: raw elapsed times for thresholds 0.5%, 1%, 2%, 4% (titles in μ)
    Thresholds are sorted numerically. Legends show per-algorithm μ and σ.
    """

    # Collect results + keep raw series by threshold for multi-plot
    results = {
        "algorithm": [],
        "threshold": [],              # numeric percentage (e.g., 0.5, 1.0, 2.0)
        "mean_elapsed_time": [],
        "stddev_elapsed_time": [],
        "p90_elapsed_time": [],
        "p95_elapsed_time": [],
        "p99_elapsed_time": [],
    }
    raw_by_threshold = {}  # {threshold_percent: {ALGO: pd.Series(elapsed_time)}}

    # Discover files like: mubench_<algo>_<threshold>.txt
    paths = glob.glob(os.path.join(directory, "mubench_*.txt"))
    if not paths:
        print("No files found matching mubench_*.txt in", directory)
        return

    for path in paths:
        fname = os.path.basename(path)
        parts = fname.split("_")  # mubench_<algo>_<threshold>.txt
        if len(parts) < 3 or not fname.endswith(".txt"):
            print("Skipping unrecognized file name:", fname)
            continue

        algo = parts[1].upper()
        thresh_str = parts[2].split(".txt")[0]
        try:
            thresh_val = float(thresh_str)  # percentage numeric
        except ValueError:
            print("Skipping (non-numeric threshold) file:", fname)
            continue

        # Read whitespace-delimited: timestamp elapsed http_status processed pending
        df = pd.read_csv(
            path,
            delim_whitespace=True,
            header=None,
            names=["timestamp", "elapsed_time", "http_status", "processed", "pending"],
        )

        elapsed = df["elapsed_time"].dropna()
        if elapsed.empty:
            print("No elapsed_time rows in:", fname)
            continue

        results["algorithm"].append(algo)
        results["threshold"].append(thresh_val)
        results["mean_elapsed_time"].append(float(elapsed.mean()))
        results["stddev_elapsed_time"].append(float(elapsed.std(ddof=1)))
        results["p90_elapsed_time"].append(float(elapsed.quantile(0.90)))
        results["p95_elapsed_time"].append(float(elapsed.quantile(0.95)))
        results["p99_elapsed_time"].append(float(elapsed.quantile(0.99)))

        # Stash raw series for the multi-plot
        raw_by_threshold.setdefault(thresh_val, {})[algo] = elapsed.reset_index(drop=True)

    results_df = pd.DataFrame(results)

    # Pivot by threshold (index) → algorithms (columns); ensure numeric sort
    pivot_means = (
        results_df.pivot(index="threshold", columns="algorithm", values="mean_elapsed_time")
        .sort_index()
    )
    pivot_stds = (
        results_df.pivot(index="threshold", columns="algorithm", values="stddev_elapsed_time")
        .reindex(pivot_means.index)
    )
    pivot_p90 = (
        results_df.pivot(index="threshold", columns="algorithm", values="p90_elapsed_time")
        .reindex(pivot_means.index)
    )
    pivot_p95 = (
        results_df.pivot(index="threshold", columns="algorithm", values="p95_elapsed_time")
        .reindex(pivot_means.index)
    )
    pivot_p99 = (
        results_df.pivot(index="threshold", columns="algorithm", values="p99_elapsed_time")
        .reindex(pivot_means.index)
    )

    # ---------- helpers ----------
    def percent_to_micro(threshold_percent: float) -> int:
        """Map percentage (e.g., 0.5) → micropods using 1% = 2000μ."""
        return int(round(threshold_percent * 2000))

    def labels_percent(index_vals):
        # Show 12 -> "12%", 12.5 -> "12.5%"
        return [f"{t:g}%" for t in index_vals]

    def labels_micro(index_vals):
        # Show 0.5 -> "1000μ", 2.0 -> "4000μ"
        return [f"{percent_to_micro(t)}μ" for t in index_vals]

    def legend_label_with_stats(name, series_vals):
        vals = np.asarray(series_vals, dtype=float)
        mu = np.nanmean(vals)
        sigma = float(np.nanstd(vals, ddof=1)) if np.isfinite(vals).sum() >= 2 else 0.0
        return f"{name} (μ={mu:.2f}, σ={sigma:.2f})"

    # ---------- 1) Grouped BAR: mean with stddev error bars (use % thresholds) ----------
    def grouped_bar_plot(values_pivot, yerr_pivot, title, ylabel, outfile, fs=12,
                         label_maker=labels_micro, xlabel="CPU Threshold (μ)"):
        thresholds = values_pivot.index.values
        xlabels = label_maker(thresholds)
        x = np.arange(len(xlabels))
        algos = list(values_pivot.columns)

        plt.figure(figsize=(10, 6))
        bar_w = 0.8 / max(1, len(algos))  # keep total width ~0.8

        for i, algo in enumerate(algos):
            vals = values_pivot[algo].values
            errs = yerr_pivot[algo].values if yerr_pivot is not None else None
            xpos = x + (i - (len(algos) - 1) / 2) * bar_w

            plt.bar(
                xpos,
                vals,
                bar_w,
                label=legend_label_with_stats(algo, vals),
                yerr=errs,
                capsize=5 if errs is not None else 0,
            )
            # value labels
            for px, v in zip(xpos, vals):
                if np.isfinite(v):
                    plt.text(px, v, f"{v:.2f}", ha="center", va="bottom", fontsize=fs-4)

        plt.title(title, fontsize=fs)
        plt.xlabel(xlabel, fontsize=fs)
        plt.ylabel(ylabel, fontsize=fs)
        plt.xticks(x, xlabels, fontsize=fs)
        plt.yticks(fontsize=fs)
        plt.grid(axis="y", linestyle=":")
        plt.legend(title="Algorithm", fontsize=fs, title_fontsize=fs)
        plt.tight_layout()
        plt.savefig(outfile)
        plt.show()

    # Mean plot (restore to % thresholds) with 14 pt fonts
    grouped_bar_plot(
        pivot_means,
        pivot_stds,
        title="Mean Elapsed Time by Algorithm and Threshold",
        ylabel="Mean Elapsed Time (ms)",
        outfile="mean_elapsed_time_bar_graph_percent.png",
        fs=14,
        label_maker=labels_percent,
        xlabel="Threshold (%)",
    )

    # ---------- 2) LINE plots ----------
    def multiline_plot(values_pivot, title, ylabel, outfile, fs=12,
                       label_maker=labels_micro, xlabel="CPU Threshold (μ)"):
        thresholds = values_pivot.index.values
        xlabels = label_maker(thresholds)
        x = np.arange(len(xlabels))
        algos = list(values_pivot.columns)

        plt.figure(figsize=(10, 6))
        for algo in algos:
            y = values_pivot[algo].values
            plt.plot(
                x,
                y,
                marker="o",
                linewidth=2,
                label=legend_label_with_stats(algo, y),
            )
            # value labels near markers
            for xi, yi in zip(x, y):
                if np.isfinite(yi):
                    plt.text(xi, yi, f"{yi:.2f}", ha="center", va="bottom", fontsize=fs-4)

        plt.title(title, fontsize=fs)
        plt.xlabel(xlabel, fontsize=fs)
        plt.ylabel(ylabel, fontsize=fs)
        plt.xticks(x, xlabels, fontsize=fs)
        plt.yticks(fontsize=fs)
        plt.grid(True, linestyle=":")
        plt.legend(title="Algorithm", fontsize=fs, title_fontsize=fs)
        plt.tight_layout()
        plt.savefig(outfile)
        plt.show()

    # p90 stays in μ (unchanged sizing)
    multiline_plot(
        pivot_p90,
        title="p90 Elapsed Time by Algorithm and CPU Threshold",
        ylabel="p90 Elapsed Time (ms)",
        outfile="p90_elapsed_time_line_graph_micro.png",
        fs=12,
        label_maker=labels_micro,
        xlabel="CPU Threshold (μ)",
    )

    # p95 and p99 restored to % thresholds with 14 pt fonts
    multiline_plot(
        pivot_p95,
        title="p95 Elapsed Time by Algorithm and Threshold",
        ylabel="p95 Elapsed Time (ms)",
        outfile="p95_elapsed_time_line_graph_percent.png",
        fs=14,
        label_maker=labels_percent,
        xlabel="Threshold (%)",
    )

    multiline_plot(
        pivot_p99,
        title="p99 Elapsed Time by Algorithm and Threshold",
        ylabel="p99 Elapsed Time (ms)",
        outfile="p99_elapsed_time_line_graph_percent.png",
        fs=14,
        label_maker=labels_percent,
        xlabel="Threshold (%)",
    )

    # ---------- 3) 2×2 multi-plot of RAW elapsed times (titles in μ) ----------
    def raw_multiplot(raw_map, wanted_thresholds=(0.5, 1.0, 2.0, 4.0), outfile="raw_elapsed_2x2_micro.png"):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False, sharey=False)
        axes = axes.flatten()

        for idx, thr in enumerate(wanted_thresholds):
            ax = axes[idx]

            # find exact threshold or best float-equal match (tolerant)
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

            micro_label = f"{percent_to_micro(thr)}μ"
            ax.set_title(f"Request times at {micro_label} Threshold")
            ax.set_xlabel("Request Interval")
            ax.set_ylabel("Elapsed Time (ms)")
            ax.grid(True, linestyle=":")

            if match_thr is None or not raw_map.get(match_thr):
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=11, alpha=0.7)
                continue

            # Plot one line per algorithm; legend shows μ and σ for this raw series
            for algo in sorted(raw_map[match_thr].keys()):
                y = np.asarray(raw_map[match_thr][algo].values, dtype=float)
                x = np.arange(len(y))
                mu = float(np.nanmean(y)) if y.size else 0.0
                sigma = float(np.nanstd(y, ddof=1)) if y.size >= 2 else 0.0
                label = f"{algo} (μ={mu:.2f}, σ={sigma:.2f})"
                ax.plot(x, y, linewidth=1.5, label=label)

            ax.legend(fontsize=8)

        # If fewer than 4 thresholds present, clear the remaining axes
        for j in range(len(wanted_thresholds), 4):
            fig.delaxes(axes[j])

        fig.suptitle("Raw Request Elapsed Times by CPU Threshold (μ)")
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        fig.savefig(outfile)
        plt.show()

    raw_multiplot(raw_by_threshold)

    # ---------- Optional: per-algorithm mean bar (kept in μ; change if you want %) ----------
    for algo in pivot_means.columns:
        vals = pivot_means[algo].values
        errs = pivot_stds[algo].values
        labels_micro = labels_micro_fn = labels_micro(pivot_means.index.values)

        plt.figure(figsize=(10, 6))
        x = np.arange(len(labels_micro))
        plt.bar(x, vals, yerr=errs, capsize=5)
        for xi, v in zip(x, vals):
            if np.isfinite(v):
                plt.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

        plt.title(f"{algo} - Mean Elapsed Time by CPU Threshold")
        plt.xlabel("CPU Threshold (μ)")
        plt.ylabel("Mean Elapsed Time (ms)")
        plt.xticks(x, labels_micro)
        plt.grid(axis="y", linestyle=":")
        plt.tight_layout()
        plt.savefig(f"{algo.lower()}_mean_elapsed_time_micro.png")
        plt.show()


if __name__ == "__main__":
    plot_mean_std_elapsed_time(directory=".")

