import numpy as np, os
import tarfile
import pandas as pd
import matplotlib.pyplot as plt
import itertools

plt.rcParams.update({
    # --- text & label sizes ---
    "font.size": 18,            # base size for everything
    "axes.titlesize": 18,       # axes title
    "axes.labelsize": 14,       # x-/y-label
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 18,
    "legend.title_fontsize": 18,
    # --- optional: make the whole plot a bit larger ---
    "figure.figsize": (12, 7)   # width, height in inches
})

# Extract tarball and filter valid folders
def extract_and_filter_tarball(tarball_path, extract_path):
    with tarfile.open(tarball_path, "r") as tar:
        tar.extractall(path=extract_path)
    # Filter folders containing underscores
    valid_folders = [
        os.path.join(extract_path, folder)
        for folder in os.listdir(extract_path)
        if "_" in folder
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
    if len(parts) == 3:
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
    return f"Pod Metrics ({metric_label})", autoscaler, utilization


# Plot node metrics with mean and standard deviation in the subplot legends and series names in the global legend
def plot_with_mean_std(files, worker_cpus={"all": []}):
    fig, axs = plt.subplots(6, 2, figsize=(15, 10))
    fig.subplots_adjust(top=0.85)
    summary_legend_entries = {}
    all_series = []

    # Assign unique colors to all series
    for file_name in files:
        df = load_csv(file_name)
        unwanted_columns = ["ksurf-producer-0f35f8a8", "ksurf-scaler"]
        filtered_columns = filter_columns(df, unwanted_columns)[1:]  # Skip Timestamp
        all_series.extend(filtered_columns)
        all_series.extend(['Interval'])

    unique_colors = get_unique_colors(len(all_series))
    series_colors = {series: color for color, series in zip(unique_colors, all_series)}
    worker_means, worker_stds = [], []
    default_color = (0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 1.0)

    print(f"files = {files}: {len(files)}, axs = {len(axs)}")
    for i, file_name in enumerate(files, 1):
        df = load_csv(file_name)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df["Interval"] = ((df["Timestamp"] - df["Timestamp"].iloc[0]).dt.total_seconds()).astype(int) # * 15
        metric_label = os.path.basename(os.path.dirname(file_name))
        formatted_title, autoscaler, threshold = format_title(metric_label)

        ax = axs[(i - 1) // 2, (i - 1) % 2]
        ax2 = ax.twinx()  # Create secondary y-axis for Memory (MiB)

        # Filter unwanted series
        unwanted_columns = ["ksurf-producer-0f35f8a8", "ksurf-producer1-f61c05d4", "ksurf-scaler", "Interval"]
        filtered_columns = filter_columns(df, unwanted_columns)

        worker_cpu_columns = [col for col in filtered_columns if "CPU (m)" in col]  # Worker pod CPU columns
        for col  in worker_cpu_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        avg_worker_cpu = df[worker_cpu_columns].mean(axis=1)  # Compute average CPU usage across all worker pods
        worker_cpus[autoscaler] = worker_cpus[autoscaler]+avg_worker_cpu if autoscaler in worker_cpus else avg_worker_cpu
        worker_cpus[threshold] = worker_cpus[threshold] if threshold in worker_cpus else {}
        worker_cpus[threshold][autoscaler] = worker_cpus[threshold][autoscaler] + [avg_worker_cpu] if autoscaler in worker_cpus[threshold] else [avg_worker_cpu]
        print(f"worker_cpus[{autoscaler}].len = {len(worker_cpus[autoscaler])}")


        for column in filtered_columns[1:]:  # Skip Timestamp
            numeric_col = pd.to_numeric(df[column], errors="coerce").fillna(0)
            mean_val = numeric_col.mean()
            std_val = numeric_col.std()
            series_name = column
            summary_legend_entries[series_name] = series_colors[column] if column in series_colors else default_color
            legend_entry = f"{mean_val:.2f} ± {std_val:.2f}"
            color = series_colors[column] if column in series_colors else default_color

            if "CPU (m)" in column:
                ax.plot(df["Interval"], numeric_col, label=legend_entry, linestyle="-", color=color)
                ax.set_ylabel("CPU (m)", color="black")
            elif "Memory (MiB)" in column:
                ax2.plot(df["Interval"], numeric_col, label=legend_entry, linestyle="--", color=color)
                ax2.set_ylabel("Memory (MiB)", color="black")

        # **Add the average worker pod CPU usage line**
        avg_color = "red"  # Choose a distinct color for visibility
        #avg_line, = ax.plot(df["Interval"], avg_worker_cpu, label="Avg Worker CPU", linestyle="dotted", color=avg_color, linewidth=2)
        avg_label = f"{avg_worker_cpu.mean():.2f} ± {avg_worker_cpu.std():.2f}"
        worker_means.append(avg_worker_cpu.mean())
        worker_stds.append(avg_worker_cpu.std())


        ax.set_title(formatted_title, fontsize=10)

        # Hide x-axis labels and ticks for all but the last row
        if (i - 1) // 2 != 3:  # Change "3" based on the number of rows (4 in this case)
            ax.set_xticklabels([])
            ax.set_xlabel("")
            ax.tick_params(axis='x', which='both', bottom=False, top=False)

        if (i-1) // 2 == 3:
            ax.set_xlabel("Time (s)")

        # Subplot legend with mean ± std only, limited to 6 entries
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:6], labels[:6], loc="upper left", fontsize=8)
        #ax.legend(handles[:6]+[avg_line], labels[:6]+[avg_label], loc="upper left", fontsize=8)
        #handles, labels = ax2.get_legend_handles_labels()
        #ax2.legend(handles[:6]+[avg_line], labels[:6]+[avg_label], loc="upper right", fontsize=8)
        ax.grid()

    # Add a global legend for the entire figure with series names
    global_legend_entries = [plt.Line2D([0], [0], color=color, lw=2, label=name) 
                              for name, color in summary_legend_entries.items()]
    fig.legend(handles=global_legend_entries, loc='upper center', ncol=4, fontsize=8, frameon=True) #title="K3S Cluster Node Metrics", fontsize=8, frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for global legend
    plt.show()
    return worker_means, worker_stds, worker_cpus


def plot_worker_cpus(autoscaler_data_by_threshold = None):

    # Define threshold values
    thresholds = ['0.25', '0.5', '1', '2']

    # Sample data structure to hold results for each autoscaler type at each threshold
    autoscaler_data_by_threshold = {
        '0.25': {"NA": [70, 72, 68, 71], "TH": [80, 82, 78, 81], "HPA": [88, 89, 87, 90], "KF": [78, 79, 77, 80]},
        '0.5': {"NA": [75, 78, 72, 74], "TH": [85, 87, 83, 86], "HPA": [90, 91, 92, 89], "KF": [80, 81, 82, 79]},
        '1': {"NA": [80, 82, 78, 81], "TH": [88, 89, 87, 90], "HPA": [92, 93, 94, 91], "KF": [85, 86, 84, 87]},
        '2': {"NA": [85, 87, 83, 86], "TH": [90, 92, 88, 91], "HPA": [95, 96, 97, 94], "KF": [90, 91, 89, 92]},
    } if autoscaler_data_by_threshold is None else autoscaler_data_by_threshold 

    # Define colors for each autoscaler type
    colors = {"NA": "blue", "TH": "green", "HPA": "red", "KF": "orange"}
    markers = {"NA": "o", "TH": "s", "HPA": "D", "KF": "^"}  # Different markers for each autoscaler type


    # Create figure with subplots (one for each threshold)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    axes = axes.flatten()  # Flatten to 1D array for easy indexing

    # Iterate over each threshold and plot data in corresponding subplot
    for idx, threshold in enumerate(thresholds):
        ax = axes[idx]  # Select subplot
        data = autoscaler_data_by_threshold[threshold]

        # Plot each autoscaler type as a separate line
        for autoscaler in data.keys():
            if not autoscaler in ["hpa", "kf", "na", "th"]:
                continue
            avg_values = data[autoscaler][0]  # Extract values for each autoscaler type
            ax.plot([i*2 for i in range(len(avg_values))], avg_values, marker=markers[autoscaler.upper()], linestyle="-", color=colors[autoscaler.upper()], label=f"{autoscaler.upper()}: {avg_values.mean():.2f} ± {avg_values.std():.2f}")

        # Labels and title
        ax.set_ylabel("Worker CPU Usage (m)")
        if idx // 2 == 1:
            ax.set_xlabel("Time (s)")
        ax.set_title(f"Worker Pod CPU Usage for {float(threshold)/100*200000:.0f}µ Threshold", fontsize=14)
        ax.legend(title="Autoscaler Type", title_fontsize=14, fontsize=10)

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_summary(autoscaler_data):
    # Define colors for each autoscaler type
    colors = {
        "na": "blue",
        "th": "green",
        "hpa": "red",
        "kf": "orange"
    }
    markers = {"NA": "o", "TH": "s", "HPA": "D", "KF": "^"}  # Different markers for each autoscaler type

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot each autoscaler type as a separate line
    for autoscaler in colors.keys():
        avg_values = autoscaler_data[autoscaler]  # Extract values for each autoscaler type
        ax.plot([i*2 for i in range(len(avg_values))], avg_values, marker=markers[autoscaler.upper()], linestyle="-", color=colors[autoscaler], label=f"{autoscaler.upper()}: {avg_values.mean():.2f} ± {avg_values.std():.2f}")

    # Labels and title
    ax.set_ylabel("Worker Pod CPU Usage (m)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Overall Worker CPU Usage Across all Thresholds")
    ax.legend(title="Autoscaler Type")

    # Show the plot
    plt.show()
  

# Main function to process the tarball and generate plots
def main():
    tarball_path = "data.tar"
    extract_path = "extracted_data"

    # Extract and filter folders
    folders = extract_and_filter_tarball(tarball_path, extract_path)

    # Identify files by tokens
    tokens = ["node_metrics"]
    files = get_files_from_folders(folders, tokens)

    # Generate node metrics plot
    mean, std = np.array([]), np.array([])
    mean1, std1, sum1 = plot_with_mean_std(list(filter(lambda f: "0.25" in f or "0.5" in f, files["node_metrics"])))
    mean2, std2, summary = plot_with_mean_std(list(filter(lambda f: "_1" in f or "_2" in f, files["node_metrics"])), worker_cpus=sum1)
    cols = [f.split("_")[2] for f in filter(lambda f: "_1" in f or "_2" in f, files["node_metrics"])]
    mean, std = np.array(mean1)+np.array(mean2), np.array(std1)+np.array(std2)
    means, stds = {}, {}
    for col, m in zip(cols, mean):
        means[col] = means[col]+m if col in means else m
    for col, s in zip(cols, std):
        stds[col] = stds[col]+s if col in stds else s
    print(f"worker CPU = {list(zip(cols,mean))} ± {list(zip(cols,std))}")
    print(f"worker CPU = {means} +- {stds}")
    plot_worker_cpus(summary)
    plot_summary(summary)


if __name__ == "__main__":
    main()

