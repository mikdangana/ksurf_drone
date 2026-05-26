import numpy as np, os
import tarfile
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import re
from math import ceil
from typing import Optional, Tuple

# Extract tarball and filter valid folders
def extract_and_filter_tarball(tarball_path, extract_path):
    with tarfile.open(tarball_path, "r") as tar:
        tar.extractall(path=extract_path)
    # Filter folders containing underscores
    valid_folders = [
        os.path.join(extract_path, folder)
        for folder in os.listdir(extract_path)
        if os.path.isdir(os.path.join(extract_path, folder)) and "_" in folder
    ]
    return valid_folders

# Get file paths based on tokens
def get_files_from_folders(folders, tokens):
    files = {token: [] for token in tokens}
    for folder in folders:
        for file in os.listdir(folder):
            for token in tokens:
                if token in file:
                    files[token].append(os.path.join(folder, file))
    return files

# Load a CSV file
def load_csv(file_name):
    return pd.read_csv(file_name)

# Simplify node names
def simplify_node_name(node_name):
    name_parts = node_name.split("-")
    return "-".join(name_parts[:2])

# Filter out unwanted series
def filter_columns(df, unwanted_columns):
    return [col for col in df.columns if all(uc not in col for uc in unwanted_columns)]


# Generate a distinct color for each series
def get_unique_colors(num_colors):
    scale=20
    cmap = plt.cm.get_cmap("tab20", num_colors*scale)  # Use tab20 for better distinct colors
    return list(set([cmap(i) for i in range(num_colors*scale)]))


# Map autoscaler types to descriptive names
def format_title(metric_label):
    parts, autoscaler, utilization = metric_label.split("_"), "UNK", 0.0
    if len(parts) > 3:
        autoscaler, utilization = parts[1], parts[2]
        utilization_percent = float(utilization) / 100
        if autoscaler == "hpa":
            return f"HPA {utilization_percent*200000:.0f}µ Threshold", autoscaler, utilization
        elif autoscaler == "kf":
            return f"KF {utilization_percent*200000:.0f}µ Threshold", autoscaler, utilization
        elif autoscaler == "th":
            return f"TH {utilization_percent*200000:.0f}µ Threshold", autoscaler, utilization
        elif autoscaler == "na":
            return f"NA {utilization_percent*200000:.0f}µ Threshold", autoscaler, utilization
        elif autoscaler == "dr":
            return f"DR {utilization_percent*200000:.0f}µ Threshold", autoscaler, utilization
        elif autoscaler == "drkf":
            return f"DRKF {utilization_percent*200000:.0f}µ Threshold", autoscaler, utilization
        elif autoscaler == "drrq":
            return f"DRRQ {utilization_percent*200000:.0f}µ Threshold", autoscaler, utilization
    return f"Pod Metrics ({metric_label})", autoscaler, utilization

def parse_algo_and_threshold(path: str):
    """
    From a path like:
      extracted_data/_combined/data_drkf_0.5/node_metrics.csv
    return: ('drkf', 0.5)

    Falls back gracefully if pattern isn't matched.
    """
    base = os.path.basename(os.path.dirname(path))  # e.g. 'data_drkf_0.5'
    # Accept 'data_<algo>_<float>' or '<algo>_<float>'
    m = re.match(r'^(?:data_)?([A-Za-z0-9]+)_([0-9]*\.?[0-9]+)$', base)
    if m:
        algo = m.group(1).lower()
        thr = float(m.group(2))
        return algo, thr

    # Fallback: scan tokens to find the last numeric as threshold and
    # the token before it as algo.
    parts = base.split('_')
    thr = float('nan')
    algo = 'UNK'
    for i in range(len(parts)-1, -1, -1):
        try:
            thr = float(parts[i])
            if i-1 >= 0:
                algo = parts[i-1].lower()
            break
        except ValueError:
            continue
    return algo, thr



def parse_algo_and_threshold(path: str):
    """
    From a path like:
      extracted_data/_combined/data_drkf_0.5/node_metrics.csv
    return: ('drkf', 0.5)
    """
    base = os.path.basename(os.path.dirname(path))  # e.g. 'data_drkf_0.5'
    m = re.match(r'^(?:data_)?([A-Za-z0-9]+)_([0-9]*\.?[0-9]+)$', base)
    if m:
        return m.group(1).lower(), float(m.group(2))

    # Fallback: last numeric token is threshold; token before it is algo
    parts = base.split('_')
    thr, algo = float('nan'), 'UNK'
    for i in range(len(parts)-1, -1, -1):
        try:
            thr = float(parts[i])
            if i-1 >= 0:
                algo = parts[i-1].lower()
            break
        except ValueError:
            continue
    return algo, thr


def build_worker_metric_summary(files, metric_keyword="CPU (m)"):
    """
    Build {threshold: {algo: [series, ...], ...}, ...} for the given metric.
    - metric_keyword: "CPU (m)" or "Memory (MiB)"
    We consider columns containing the metric keyword and try to focus on worker pods.
    """
    by_thr = {}  # {thr: {algo: [np.array,...]}}
    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f"[mem/cpu summary] skip {fp}: {e}")
            continue

        algo, thr = parse_algo_and_threshold(fp)

        # Find candidate columns for the metric
        cols = [c for c in df.columns if metric_keyword in c]

        # Prefer 'worker' columns if present
        worker_like = [c for c in cols if "work" in c.lower() or "scaler" in c.lower()]
        picked = worker_like if worker_like else cols

        if not picked:
            continue

        # Convert to numeric and average across worker columns per timestamp
        for c in picked:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        series = df[picked].mean(axis=1)

        # Optional: use expanding mean, matching your CPU plot behavior
        series = series.expanding().mean()

        # Store
        by_thr.setdefault(thr, {})
        by_thr[thr].setdefault(algo, [])
        by_thr[thr][algo].append(series.to_numpy(dtype=float))

    return by_thr


def plot_worker_memory(autoscaler_data_by_threshold=None):
    """
    Expected:
      { thr: { 'dr': [arrays...], 'drkf': [arrays...], 'drrq': [arrays...] }, ... }
    Plots average worker memory per autoscaler at up to 4 thresholds (2x2).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if not isinstance(autoscaler_data_by_threshold, dict) or not autoscaler_data_by_threshold:
        print("[plot_worker_memory] no memory summary to plot; skipping.")
        return

    def _to_num(k):
        if isinstance(k, (int, float)): return float(k)
        try: return float(k)
        except Exception: return None

    thr_keys = [ _to_num(k) for k in autoscaler_data_by_threshold.keys() ]
    thr_keys = sorted([k for k in thr_keys if k is not None])
    if not thr_keys:
        print("[plot_worker_memory] no numeric thresholds; skipping.")
        return

    nplots = min(4, len(thr_keys))
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    axes = axes.flatten()

    colors  = {"DR": "tab:pink", "DRKF": "tab:purple", "DRRQ": "tab:brown",
               "HPA": "tab:red",  "KF": "tab:purple",  "NA": "tab:gray"}
    markers = {"DR": "x",        "DRKF": "v",         "DRRQ": ".",
               "HPA": "^",       "KF": "o",           "NA": ","}

    for idx in range(nplots):
        thr = thr_keys[idx]
        ax = axes[idx]
        data = autoscaler_data_by_threshold.get(thr) or autoscaler_data_by_threshold.get(str(thr), {})

        for autoscaler, series_list in (data or {}).items():
            if not series_list:
                continue
            # Align by minimum length
            arrs = [np.asarray(s, dtype=float) for s in series_list if len(s)]
            if not arrs:
                continue
            min_len = min(len(a) for a in arrs)
            if min_len == 0:
                continue
            arr = np.vstack([a[:min_len] for a in arrs])
            avg_values = np.nanmean(arr, axis=0)
            x = np.arange(min_len) * 2  # same 2s tick spacing

            akey = autoscaler.upper()
            color = colors.get(akey, "black")
            marker = markers.get(akey, "o")
            label = f"{akey}: {float(np.nanmean(avg_values)):.2f} +- {float(np.nanstd(avg_values)):.2f}"
            ax.plot(x, avg_values, marker=marker, linestyle="-", color=color, label=label)

        ax.set_ylabel("Worker & Scaler Memory (MiB)")
        if idx // 2 == 1:
            ax.set_xlabel("Time (s)")
        # Title still references CPU threshold units, which is fine (it's the scaling control)
        ax.set_title(f"Worker & Scaler Memory at {thr/100.0*200000:.0f}u CPU Threshold")
        ax.legend(title="Autoscaler Type")

    # Hide unused axes
    for j in range(nplots, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(f"gcloud_ksurf_drone_node_metrics_summary.png")
    plt.show()



def plot_worker_memory_overall(
    mem_summary: dict,
    title: str = "Worker & Scaler Memory over ALL Thresholds",
    xlabel: str = "CPU Threshold",
    ylabel: str = "Average Worker & Scaler Memory (MiB)",
    outfile: str = "gcloud_worker_memory_overall.png",
    fs: int = 14,
):
    """
    Multi-line plot across ALL thresholds.
    For each autoscaler (algo), at each threshold we compute:
      - per-series mean memory (one run = one series)
      - then mean and std across runs at that threshold.
    We plot mean vs threshold with error bars (std).
    """
    if not isinstance(mem_summary, dict) or not mem_summary:
        print("[plot_worker_memory_overall_lines] no memory summary to plot; skipping.")
        return

    # Collect union of thresholds (numeric, sorted)
    def _to_num(k):
        if isinstance(k, (int, float)):
            return float(k)
        try:
            return float(k)
        except Exception:
            return None

    thr_list = sorted([t for t in (_to_num(k) for k in mem_summary.keys()) if t is not None])
    if not thr_list:
        print("[plot_worker_memory_overall_lines] no numeric thresholds; skipping.")
        return

    # Collect union of algos across thresholds (keys may be lower-case from parse)
    algo_set = set()
    for thr in mem_summary:
        inner = mem_summary[thr]
        if isinstance(inner, dict):
            algo_set.update(inner.keys())
    if not algo_set:
        print("[plot_worker_memory_overall_lines] no algorithms found; skipping.")
        return
    algos = sorted(algo_set)

    # Colors/markers (fallback to black/'o' if missing)
    colors  = {"dr": "tab:blue", "drkf": "tab:orange", "drrq": "tab:green",
               "hpa": "tab:red", "kf": "tab:purple", "na": "tab:gray"}
    markers = {"dr": "x", "drkf": "v", "drrq": ".",
               "hpa": "^", "kf": "o", "na": ","}

    fig, ax = plt.subplots(figsize=(10, 5))

    for algo in algos:
        means_per_thr, stds_per_thr = [], []
        for thr in thr_list:
            # mem_summary may have str or float keys; try both
            data_at_thr = mem_summary.get(thr) or mem_summary.get(str(thr)) or {}
            series_list = data_at_thr.get(algo, [])
            if not series_list:
                means_per_thr.append(np.nan)
                stds_per_thr.append(np.nan)
                continue

            # Per-series means for this threshold
            per_run_means = []
            for series in series_list:
                arr = np.asarray(series, dtype=float)
                if arr.size:
                    per_run_means.append(float(np.nanmean(arr)))
            if not per_run_means:
                means_per_thr.append(np.nan)
                stds_per_thr.append(np.nan)
            else:
                m = float(np.nanmean(per_run_means))
                s = float(np.nanstd(per_run_means, ddof=1)) if len(per_run_means) > 1 else 0.0
                means_per_thr.append(m)
                stds_per_thr.append(s)

        means_per_thr = np.array(means_per_thr, dtype=float)
        stds_per_thr  = np.array(stds_per_thr, dtype=float)

        # Plot with error bars at each threshold
        akey = algo.upper()
        color = colors.get(algo.lower(), "black")
        marker = markers.get(algo.lower(), "o")
        # Mask out all-NaN series
        if np.all(np.isnan(means_per_thr)):
            continue
        ax.errorbar(
            thr_list, means_per_thr, yerr=stds_per_thr,
            fmt=f"-{marker}", color=color, capsize=4, label=akey
        )

    ax.set_title(title, fontsize=fs)
    ax.set_xlabel(xlabel, fontsize=fs)
    ax.set_ylabel(ylabel, fontsize=fs)
    ax.tick_params(axis="both", labelsize=fs-2)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=fs-2, title="Autoscaler", title_fontsize=fs-2)
    plt.tight_layout()
    plt.savefig(outfile, dpi=120)
    print("Saved overall memory multi-line chart ->", outfile)
    plt.show()


def build_memory_autoscaler_overall(mem_summary: dict) -> dict:
    """
    Convert mem_summary {thr: {algo: [arrays...]}} into autoscaler-level
    time series matching plot_summary() expectations:
        { 'dr': np.array([...]), 'drkf': np.array([...]), ... }
    We average across all runs & thresholds for each autoscaler,
    aligning series by the minimum common length.
    """
    import numpy as np

    agg = {}  # algo(lower) -> list of arrays
    if not isinstance(mem_summary, dict):
        return {}

    for _thr, algo_dict in mem_summary.items():
        if not isinstance(algo_dict, dict):
            continue
        for algo, series_list in algo_dict.items():
            if not series_list:
                continue
            key = str(algo).lower()
            for series in series_list:
                arr = np.asarray(series, dtype=float)
                if arr.size:
                    agg.setdefault(key, []).append(arr)

    autoscaler_data = {}
    for algo, arrs in agg.items():
        if not arrs:
            continue
        min_len = min(len(a) for a in arrs if len(a) > 0)
        if min_len <= 0:
            continue
        stacked = np.vstack([a[:min_len] for a in arrs])
        mean_ts = np.nanmean(stacked, axis=0)
        autoscaler_data[algo] = mean_ts

    return autoscaler_data


def plot_summary_memory(autoscaler_data):
    """
    Same multiline plot style as plot_summary(), but for Memory.
    autoscaler_data: { 'dr': np.array([...]), 'drkf': np.array([...]), ... }
    """
    import numpy as np
    import matplotlib.pyplot as plt

    fontsize = 14
    colors = {
        "na": "blue",
        "th": "green",
        "hpa": "red",
        "dr": "pink",
        "drkf": "purple",
        "kf": "orange",
        "drrq": "brown",
        "all": "yellow",
    }
    markers = {
        "NA": "o", "TH": "s", "HPA": "D", "KF": "^",
        "DR": "x", "DRKF": "v", "DRRQ": ".", "ALL": ","
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    for autoscaler in autoscaler_data.keys():
        avg_values = np.array(autoscaler_data[autoscaler])
        if np.size(avg_values) > 0 and np.any(np.atleast_1d(avg_values)) and autoscaler in colors.keys():
            ax.plot(
                [i * 2 for i in range(np.size(avg_values))],
                avg_values,
                marker=markers.get(autoscaler.upper(), "o"),
                linestyle="-",
                color=colors[autoscaler],
                label=f"{autoscaler.upper()}: {np.nanmean(avg_values):.2f} ± {np.nanstd(avg_values):.2f}",
            )
        else:
            # Keep the same behavior as plot_summary for unknown keys
            pass

    ax.set_ylabel("Worker & Scaler Pod Memory (MiB)", fontsize=fontsize)
    ax.set_xlabel("Time (s)", fontsize=fontsize)
    ax.set_title("Overall Worker & Scaler Memory Usage Across all Thresholds", fontsize=fontsize)
    ax.legend(title="Autoscaler Type", fontsize=fontsize)
    plt.savefig("gcloud_ksurf_drone_worker_memory_all.png")
    plt.show()



# Plot node metrics with mean and standard deviation in the subplot legends and series names in the global legend
def plot_with_mean_std(files, worker_cpus={"all": []}):
    # NEW: guard empty
    if not files:
        print("[plot] no files provided; skipping.")
        # return the same tuple arity your callers expect
        return np.array([]), np.array([]), worker_cpus, [], [], []

    fontsize = 12
    fig, axs = plt.subplots(ceil(len(files)/2), 2, figsize=(15, 10))
    fig.subplots_adjust(top=0.85)
    summary_legend_entries = {}
    all_series = []

    # Assign unique colors to all series
    for file_name in files:
        df = load_csv(file_name)
        unwanted_columns = ["ksurf-producer-0f35f8a8"]
        filtered_columns = filter_columns(df, unwanted_columns)[1:]  # Skip Timestamp
        all_series.extend(filtered_columns)
        all_series.extend(['Interval'])

    unique_colors = get_unique_colors(len(all_series))
    series_colors = {series: color for color, series in zip(unique_colors, all_series)}
    worker_means, worker_stds = [], []
    default_color = (0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 1.0)
    means, stds, cols = [], [], []

    print(f"files = {files}: {len(files)}, axs = {len(axs)}")
    for i, file_name in enumerate(files, 1):
        df = load_csv(file_name)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df["Interval"] = ((df["Timestamp"] - df["Timestamp"].iloc[0]).dt.total_seconds()).astype(int) # * 15
        metric_label = os.path.basename(os.path.dirname(file_name))
        formatted_title, autoscaler, threshold = format_title(metric_label)
        autoscaler, threshold = parse_algo_and_threshold(file_name)

        ax = axs[(i - 1) // 2, (i - 1) % 2]
        ax2 = ax.twinx()  # Create secondary y-axis for Memory (MiB)

        # Filter unwanted series
        #unwanted_columns = ["ksurf-producer-0f35f8a8", "ksurf-producer1-f61c05d4", "ksurf-scaler", "Interval"]
        unwanted_columns = ["Interval"]
        filtered_columns = filter_columns(df, unwanted_columns)

        worker_cpu_columns = [col for col in filtered_columns if "CPU (m)" in col]  # Worker pod CPU columns
        worker_mem_columns = [col for col in filtered_columns if "Mem (MiB)" in col]  # Worker pod CPU columns
        for col  in worker_cpu_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        avg_worker_cpu = df[worker_cpu_columns].mean(axis=1)  # Compute average CPU usage across all worker pods
        # ------------- CHANGE-4: cumulative mean of worker CPU -------------
        avg_worker_cpu = avg_worker_cpu.expanding().mean()
        # --------------------------------------------------------------------
        worker_cpus[autoscaler] = worker_cpus[autoscaler]+avg_worker_cpu if autoscaler in worker_cpus else avg_worker_cpu
        worker_cpus[threshold] = worker_cpus[threshold] if threshold in worker_cpus else {}
        worker_cpus[threshold][autoscaler] = worker_cpus[threshold][autoscaler] + [avg_worker_cpu] if autoscaler in worker_cpus[threshold] else [avg_worker_cpu]
        print(f"worker_cpus[{autoscaler}].len = {len(worker_cpus[autoscaler])}")
        print("worker_cpus.threshold, algo = ", threshold, autoscaler)


        for column in filtered_columns[1:]:  # Skip Timestamp
            numeric_col = pd.to_numeric(df[column], errors="coerce").fillna(0)
            mean_val = numeric_col.mean()
            std_val = numeric_col.std()
            means, stds, cols = means+[mean_val], stds+[std_val], cols+[autoscaler+"."+column]
            series_name = column
            summary_legend_entries[series_name] = series_colors[column] if column in series_colors else default_color
            legend_entry = f"{mean_val:.2f} ± {std_val:.2f}"
            color = series_colors[column] if column in series_colors else default_color

            # ------------- CHANGE-1: build cumulative mean -------------
            cum_avg = numeric_col.expanding().mean()
            # -----------------------------------------------------------


            if "CPU (m)" in column:
                ax.plot(df["Interval"], cum_avg, label=f"CPU-{legend_entry}", linestyle="-", color=color)
                ax.set_ylabel("CPU (m)", color="black")
            elif "Memory (MiB)" in column:
                ax2.plot(df["Interval"], cum_avg, label=f"Mem-{legend_entry}", linestyle="--", color=color)
                ax2.set_ylabel("Memory (MiB)", color="black")

        # **Add the average worker pod CPU usage line**
        avg_color = "red"  # Choose a distinct color for visibility
        #avg_line, = ax.plot(df["Interval"], avg_worker_cpu, label="Avg Worker CPU", linestyle="dotted", color=avg_color, linewidth=2)
        avg_label = f"{avg_worker_cpu.mean():.2f} ± {avg_worker_cpu.std():.2f}"
        worker_means.append(avg_worker_cpu.mean())
        worker_stds.append(avg_worker_cpu.std())


        ax.set_title(formatted_title, fontsize=fontsize)

        # Hide x-axis labels and ticks for all but the last row
        if (i - 1) // 2 != 3:  # Change "3" based on the number of rows (4 in this case)
            ax.set_xticklabels([])
            ax.set_xlabel("", fontsize=fontsize)
            ax.tick_params(axis='x', which='both', bottom=False, top=False)

        if (i-1) // 2 == 3:
            ax.set_xlabel("Time (s)", fontsize=fontsize)

        # Subplot legend with mean ± std only, limited to 6 entries
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:6], labels[:6], loc="upper left", fontsize=fontsize)
        #ax.legend(handles[:6]+[avg_line], labels[:6]+[avg_label], loc="upper left", fontsize=8)
        handles, labels = ax2.get_legend_handles_labels()
        ax2.legend(handles[:6], labels[:6], loc="upper right", fontsize=fontsize)
        #ax2.legend(handles[:6]+[avg_line], labels[:6]+[avg_label], loc="upper right", fontsize=8)
        ax.grid()

    # Add a global legend for the entire figure with series names
    global_legend_entries = [plt.Line2D([0], [0], color=color, lw=2, label=name) 
                              for name, color in summary_legend_entries.items()]
    fig.legend(handles=global_legend_entries, loc='upper center', ncol=4, fontsize=fontsize, frameon=True) #title="K3S Cluster Node Metrics", fontsize=8, frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for global legend
    plt.savefig(f"gcloud_ksurf_drone_node_metrics_{threshold}.png")
    #plt.show()
    return worker_means, worker_stds, worker_cpus, means, stds, cols


def plot_worker_cpus(autoscaler_data_by_threshold=None):
    """
    autoscaler_data_by_threshold: expected dict like:
      {
        0.5: { 'dr': [Series, Series, ...], 'drkf': [Series, ...], ... },
        1.0: { ... },
        ...
      }
    Falls back to sample data if None or empty.
    """
    # Fallback sample if nothing usable is passed
    print("plot_worker_cpus.data = ", autoscaler_data_by_threshold)
    if not isinstance(autoscaler_data_by_threshold, dict) or not autoscaler_data_by_threshold:
        autoscaler_data_by_threshold = {
            0.25: {"NA": [np.array([70,72,68,71])], "TH": [np.array([80,82,78,81])],
                   "HPA": [np.array([88,89,87,90])], "KF": [np.array([78,79,77,80])]},
            0.5:  {"NA": [np.array([75,78,72,74])], "TH": [np.array([85,87,83,86])],
                   "HPA": [np.array([90,91,92,89])], "KF": [np.array([80,81,82,79])]},
            1.0:  {"NA": [np.array([80,82,78,81])], "TH": [np.array([88,89,87,90])],
                   "HPA": [np.array([92,93,94,91])], "KF": [np.array([85,86,84,87])]},
            2.0:  {"NA": [np.array([85,87,83,86])], "TH": [np.array([90,92,88,91])],
                   "HPA": [np.array([95,96,97,94])], "KF": [np.array([90,91,89,92])]},
        }

    # Derive thresholds from keys (prefer numeric keys)
    keys = list(autoscaler_data_by_threshold.keys())
    # Coerce string number keys to float if needed
    def _to_num(k):
        if isinstance(k, (int, float)): return float(k)
        try: return float(k)
        except Exception: return None
    num_keys = [k for k in (_to_num(k) for k in keys) if k is not None]
    if not num_keys:
        print("[plot_worker_cpus] no numeric threshold keys; aborting plot.")
        return

    thresholds = sorted(num_keys)
    # Plot up to 4 thresholds (2x2 grid)
    nplots = min(4, len(thresholds))
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    axes = axes.flatten()

    colors = {"NA": "blue", "TH": "green", "HPA": "red", "KF": "orange",
              "DR": "pink", "DRKF": "purple", "DRRQ": "brown"}
    markers = {"NA": "o", "TH": "s", "HPA": "D", "KF": "^",
               "DR": "x", "DRKF": "v", "DRRQ": "."}

    for idx in range(nplots):
        thr = thresholds[idx]
        ax = axes[idx]
        data = autoscaler_data_by_threshold.get(thr, autoscaler_data_by_threshold.get(str(thr), {}))

        for autoscaler, series_list in data.items():
            # series_list is expected to be a list of pandas Series or 1-D arrays
            if not series_list:
                continue
            # Align by minimum length
            min_len = min(len(s) for s in series_list)
            if min_len == 0:
                continue
            arr = np.vstack([np.asarray(s[:min_len]) for s in series_list])
            avg_values = arr.mean(axis=0)
            # x-axis every 2 seconds as in original code
            x = [i * 2 for i in range(min_len)]

            akey = autoscaler.upper()
            color = colors.get(akey, "black")
            marker = markers.get(akey, "o")
            label = f"{akey}: {float(np.nanmean(avg_values)):.2f} ± {float(np.nanstd(avg_values)):.2f}"
            ax.plot(x, avg_values, marker=marker, linestyle="-", color=color, label=label)

        ax.set_ylabel("Worker CPU Usage (m)")
        if idx // 2 == 1:
            ax.set_xlabel("Time (s)")
        # Convert % threshold to micropods like original:
        ax.set_title(f"Worker Pod CPU Usage for {thr/100.0*200000:.0f}µ Threshold")
        ax.legend(title="Autoscaler Type")

    # Hide any unused subplots
    for j in range(nplots, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(f"gcloud_ksurf_drone_worker_cpu_{thresholds[-1]}.png")
    plt.show()



def plot_summary(autoscaler_data):
    fontsize = 14
    # Define colors for each autoscaler type
    colors = {
        "na": "blue",
        "th": "green",
        "hpa": "red",
        "dr": "pink",
        "drkf": "purple",
        "kf": "orange",
        "drrq": "brown",
        "all": "yellow"
    }
    markers = {"NA": "o", "TH": "s", "HPA": "D", "KF": "^", "DR": "x", "DRKF": "v", "DRRQ": ".", "ALL": ","}  # Different markers for each autoscaler type

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot each autoscaler type as a separate line
    for autoscaler in autoscaler_data.keys(): #colors.keys():
        avg_values = np.array(autoscaler_data[autoscaler])  # Extract values for each autoscaler type
        if np.size(avg_values) > 0 and np.any(np.atleast_1d(avg_values)) and autoscaler in colors.keys():
            print("plot_summary.autoscaler,avg_values =",autoscaler, avg_values)
            ax.plot([i*2 for i in range(np.size(avg_values))], avg_values, marker=markers[autoscaler.upper()], linestyle="-", color=colors[autoscaler], label=f"{autoscaler.upper()}: {np.nanmean(avg_values):.2f} ± {np.nanstd(avg_values):.2f}")

        else:
            print("plot_summary.skipped autoscaler =", autoscaler)
    # Labels and title
    ax.set_ylabel("Worker Pod CPU Usage (m)", fontsize=fontsize)
    ax.set_xlabel("Time (s)", fontsize=fontsize)
    ax.set_title("Overall Worker CPU Usage Across all Thresholds", fontsize=fontsize)
    ax.legend(title="Autoscaler Type", fontsize=fontsize)

    # Show the plot
    plt.savefig(f"gcloud_ksurf_drone_worker_cpu_all.png")
    plt.show()
 
# ---------------- NEW: group iteration folders and concatenate CSVs ----------------
def _parse_group_key(folder_name: str) -> Tuple[str, Optional[str]]:
    """
    If folder is a four-tuple (e.g., data_dr_0.5_1) treat the last token as iteration.
    Returns (group_key, iter_token). For non-iter folders, iter_token=None and
    group_key stays as the folder_name itself.
    """
    parts = folder_name.split("_")
    if len(parts) >= 4 and parts[-1].isdigit():
        # e.g., ["data", "dr", "0.5", "1"] -> group_key="data_dr_0.5", iter="1"
        return "_".join(parts[:-1]), parts[-1]
    return folder_name, None


def group_and_combine_by_iteration(folders, tokens, extract_path):
    """
    Build combined CSVs by grouping iteration folders (last underscore token)
    and concatenating CSVs of the same token across iterations.
    Writes to: extracted_data/_combined/<group_key>/<token>.csv
    Returns: {token: [combined_csv_paths...]}
    """
    # Group folders by their (algorithm,threshold) key
    grouped = {}  # key -> list of folder paths
    for folder in folders:
        fname = os.path.basename(folder)
        key, _iter = _parse_group_key(fname)
        grouped.setdefault(key, []).append(folder)

    output = {token: [] for token in tokens}
    base_out = os.path.join(extract_path, "_combined")

    for key, group_folders in grouped.items():
        out_dir = os.path.join(base_out, key)
        os.makedirs(out_dir, exist_ok=True)

        # For each token (e.g., "node_metrics"), gather CSVs across all iterations
        for token in tokens:
            candidate_files = []
            for folder in group_folders:
                for fn in os.listdir(folder):
                    if token in fn and fn.lower().endswith(".csv"):
                        candidate_files.append(os.path.join(folder, fn))

            if not candidate_files:
                continue

            # Concatenate CSVs from all iterations (ignore index, preserve columns)
            dfs = []
            for fp in candidate_files:
                try:
                    df = pd.read_csv(fp)
                    dfs.append(df)
                except Exception as e:
                    print(f"[combine] skip {fp}: {e}")

            if not dfs:
                continue

            df_all = pd.concat(dfs, ignore_index=True, sort=False)

            # Optional: sort by Timestamp if present
            if "Timestamp" in df_all.columns:
                try:
                    df_all["Timestamp"] = pd.to_datetime(df_all["Timestamp"])
                    df_all = df_all.sort_values("Timestamp")
                except Exception:
                    pass

            out_file = os.path.join(out_dir, f"{token}.csv")
            df_all.to_csv(out_file, index=False)
            output[token].append(out_file)
            print(f"[combine] wrote {out_file}  (from {len(candidate_files)} iter files)")

    return output
# -------------------------------------------------------------------------------

# ----------------------- main() — only preprocessing changed -------------------
def main():
    tarball_path = "data.tar"
    extract_path = "extracted_data"

    # 1) Extract and list folders as before
    folders = extract_and_filter_tarball(tarball_path, extract_path)

    # 2) NEW: group by (algo,threshold), concatenate iterations into combined CSVs
    tokens = ["node_metrics"]
    files = group_and_combine_by_iteration(folders, tokens, extract_path)

    # 3) From here on, plot the files
    mean, std = np.array([]), np.array([])
    mean1, std1, summary, m,s,c = plot_with_mean_std(
        list(filter(lambda f: ("dr" in f) and ("0.25" in f or "0.5" in f or "_1\\" in f), files["node_metrics"]))
    )
    means, stds, mcols = m, s, c

    for t in range(5):
      th = 2**(2*t+1)
      mean2, std2, summary, m,s,c = plot_with_mean_std(
        list(filter(lambda f: ("dr" in f) and (f"_{th}\\" in f or f"_{th*2}\\" in f), files["node_metrics"])),
        worker_cpus=summary
      )
      means, stds, mcols = means+m, stds+s, mcols+c

    #cols = [f.split("_")[2] for f in filter(lambda f: "_1" in f or "_2" in f, files["node_metrics"])]

    def show_stats1(label, autoscaler, metric, podtype="worker"):
        stats = np.array([m[0:2] for m in filter(lambda m: autoscaler+"." in m[2] and podtype in m[2] and metric in m[2],
                                                 zip(means, stds, mcols))])
        print(label, stats[:,0].mean(), "+-", stats[:,1].mean())
    
    def show_stats(label, autoscaler, metric, podtype="worker"):
      # Build a list of rows matching the filter
      rows = [
        (mm, ss, col)
        for (mm, ss, col) in zip(means, stds, mcols)
        if (f"{autoscaler}." in col) and (podtype in col) and (metric in col)
      ]
      if not rows:
        print(label, "n/a (no matching rows)")
        return

      # Force a 2-D shape: [[mean, std], ...]
      arr = np.array([[r[0], r[1]] for r in rows], dtype=float)
      # Now arr[:, 0] / arr[:, 1] are safe
      print(label, arr[:, 0].mean(), "+-", arr[:, 1].mean())


    show_stats("DR worker CPU \t\t= ", "dr", "CPU")
    show_stats("DRKF worker CPU \t= ", "drkf", "CPU")
    show_stats("DR worker Memory \t= ", "dr", "Mem")
    show_stats("DRKF worker Memory \t= ", "drkf", "Mem")
    show_stats("DR producer CPU \t= ", "dr", "CPU", "producer-1")
    show_stats("DRKF producer CPU \t= ", "drkf", "CPU", "producer-1")
    show_stats("DR producer Memory \t= ", "dr", "Mem", "producer-1")
    show_stats("DRKF producer Memory \t= ", "drkf", "Mem", "producer-1")
    show_stats("DR master CPU \t\t= ", "dr", "CPU", "master")
    show_stats("DRKF master CPU \t= ", "drkf", "CPU", "master")
    show_stats("DR master Memory \t= ", "dr", "Mem", "master")
    show_stats("DRKF master Memory \t= ", "drkf", "Mem", "master")
    show_stats("DR scaler CPU \t\t= ", "dr", "CPU", "scaler")
    show_stats("DRKF scaler CPU \t= ", "drkf", "CPU", "scaler")
    show_stats("DR scaler Memory \t= ", "dr", "Mem", "scaler")
    show_stats("DRKF scaler Memory \t= ", "drkf", "Mem", "scaler")

    data_for_worker = summary #if isinstance(summary, dict) and summary else sum1
    plot_worker_cpus(data_for_worker)
    # plot_summary expects autoscaler-level series; use sum1 (first call)
    plot_summary(summary if isinstance(summary, dict) else {})

    #plot_worker_cpus(summary)
    #plot_summary(summary)

    # Build memory summary from the same combined files and plot it
    combined_files = files["node_metrics"] 

    mem_summary = build_worker_metric_summary(
        combined_files,
        metric_keyword="Memory (MiB)"
    )
    plot_worker_memory(mem_summary)
    #plot_worker_memory_overall(mem_summary)  # NEW: uses ALL thresholds
    mem_overall = build_memory_autoscaler_overall(mem_summary)
    plot_summary_memory(mem_overall)


if __name__ == "__main__":
    main()

