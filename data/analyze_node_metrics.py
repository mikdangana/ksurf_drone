import pandas as pd
import matplotlib.pyplot as plt
import os

# File paths (update with your actual folder path)
folder_path = "analysis"

# File mapping for node metrics
node_metrics_files = [
    "data_hpa_2_node_metrics.csv",
    "data_hpa_5_node_metrics.csv",
    "data_kf_2_node_metrics.csv",
    "data_kf_5_node_metrics.csv",
    "data_th_2_node_metrics.csv",
    "data_th_5_node_metrics.csv",
]

# Helper function to load data
def load_csv(file_name):
    return pd.read_csv(os.path.join(folder_path, file_name))

# Plotting node metrics with dual y-axes in a 2-column layout
def plot_node_metrics_dual_axis(files):
    # Create a 2-column layout with up to 3 rows per column
    n_cols = 2
    n_rows = -(-len(files) // n_cols)  # Ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4), sharex=False)

    # Flatten axes for easy iteration
    axes = axes.flatten()

    # Separate files into 2% and 5% threshold groups
    files_2 = [f for f in files if "_2_" in f]
    files_5 = [f for f in files if "_5_" in f]

    def compute_global_min_means(file_group):
        """Compute the minimum mean values for each column across the given file group."""
        min_means = {}
        for file_name in file_group:
            df = load_csv(file_name)
            memory_cols = [col for col in df.columns if "Memory" in col and "ksurf-producer-0f35f8a8" not in col]
            cpu_cols = [col for col in df.columns if "CPU" in col and "ksurf-producer-0f35f8a8" not in col]

            for col in memory_cols + cpu_cols:
                numeric_col = pd.to_numeric(df[col], errors="coerce").fillna(0)
                min_means[col] = min(min_means.get(col, float('inf')), numeric_col.mean())
        return min_means

    # Compute global minimum means for 2% and 5% threshold groups
    global_min_means_2 = compute_global_min_means(files_2)
    global_min_means_5 = compute_global_min_means(files_5)

    for i, file_name in enumerate(files):
        df = load_csv(file_name)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df["Time (s)"] = (df["Timestamp"] - df["Timestamp"].iloc[0]).dt.total_seconds()

        # Determine the threshold for the current file
        current_min_means = global_min_means_2 if "_2_" in file_name else global_min_means_5

        ax_memory = axes[i]
        ax_cpu = ax_memory.twinx()  # Create a second y-axis for CPU usage

        # Extract memory and CPU columns excluding "ksurf-producer-0f35f8a8"
        memory_cols = [col for col in df.columns if "Memory" in col and "ksurf-producer-0f35f8a8" not in col]
        cpu_cols = [col for col in df.columns if "CPU" in col and "ksurf-producer-0f35f8a8" not in col]

        memory_handles = []  # Store memory line handles for legend
        memory_labels = []   # Store memory labels for legend
        cpu_handles = []     # Store CPU line handles for legend
        cpu_labels = []      # Store CPU labels for legend

        # Plot Memory
        for col in memory_cols:
            numeric_col = pd.to_numeric(df[col], errors="coerce").fillna(0)
            mean_val = numeric_col.mean()
            variance_val = numeric_col.var()
            line, = ax_memory.plot(df["Time (s)"], numeric_col, label=col, linestyle="-")

            # Highlight if this subplot has the lowest mean value for the column in the current threshold group
            if mean_val == current_min_means[col]:
                memory_labels.append(f"**{col} ({mean_val:.2e} ± {variance_val:.2e})**")
            else:
                memory_labels.append(f"{col} ({mean_val:.2e} ± {variance_val:.2e})")

            memory_handles.append(line)

        ax_memory.set_ylabel("Memory (MiB)", fontsize=12)

        # Plot CPU
        for col in cpu_cols:
            numeric_col = pd.to_numeric(df[col], errors="coerce").fillna(0)
            mean_val = numeric_col.mean()
            variance_val = numeric_col.var()
            line, = ax_cpu.plot(df["Time (s)"], numeric_col, label=col, linestyle="--")

            # Highlight if this subplot has the lowest mean value for the column in the current threshold group
            if mean_val == current_min_means[col]:
                cpu_labels.append(f"**{col} ({mean_val:.2e} ± {variance_val:.2e})**")
            else:
                cpu_labels.append(f"{col} ({mean_val:.2e} ± {variance_val:.2e})")

            cpu_handles.append(line)

        ax_cpu.set_ylabel("CPU (m)", fontsize=12)

        # Add title and legend to each subplot
        ax_memory.set_title(f"Node Metrics ({file_name.split('_')[1]})", fontsize=14)
        ax_memory.grid()
        ax_memory.legend(
            memory_handles + cpu_handles,
            memory_labels + cpu_labels,
            loc="upper right",
            fontsize=6,  # Reduced legend font size
        )

    # Hide unused subplots
    for j in range(len(files), len(axes)):
        axes[j].set_visible(False)

    # Show x-axis label only on the bottom subplots
    for i in range(len(axes)):
        if i >= len(axes) - n_cols:  # Last row
            axes[i].set_xlabel("Time (s)", fontsize=12)
        else:
            axes[i].tick_params(labelbottom=False)  # Hide x-axis labels for non-bottom rows

    plt.tight_layout()
    plt.show()

# Run analysis
plot_node_metrics_dual_axis(node_metrics_files)

