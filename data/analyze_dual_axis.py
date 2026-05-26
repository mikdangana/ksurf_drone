import pandas as pd
import matplotlib.pyplot as plt
import os

# File paths (add your actual folder path)
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

# Plotting node metrics with dual y-axes and shared legend
def plot_node_metrics_dual_axis(files):
    fig, axes = plt.subplots(len(files), 1, figsize=(12, len(files) * 4), sharex=True)
    fig.suptitle("Node Metrics Over Time", fontsize=16)

    combined_legend_handles = []
    combined_legend_labels = []

    for i, file_name in enumerate(files):
        df = load_csv(file_name)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df["Time (s)"] = (df["Timestamp"] - df["Timestamp"].iloc[0]).dt.total_seconds()

        ax_memory = axes[i]
        ax_cpu = ax_memory.twinx()  # Create a second y-axis for CPU usage

        # Extract memory and CPU columns
        memory_cols = [col for col in df.columns if "Memory" in col]
        cpu_cols = [col for col in df.columns if "CPU" in col]

        # Plot Memory
        for col in memory_cols:
            line, = ax_memory.plot(df["Time (s)"], df[col], label=col, linestyle="-")
            combined_legend_handles.append(line)
            combined_legend_labels.append(col)

        ax_memory.set_ylabel("Memory (MiB)", fontsize=12)

        # Plot CPU
        for col in cpu_cols:
            line, = ax_cpu.plot(df["Time (s)"], df[col], label=col, linestyle="--")
            combined_legend_handles.append(line)
            combined_legend_labels.append(col)

        ax_cpu.set_ylabel("CPU (m)", fontsize=12)

        ax_memory.set_title(f"Node Metrics ({file_name.split('_')[1]})", fontsize=14)
        ax_memory.grid()

    # X-axis setup
    ax_memory.set_xlabel("Time (s)", fontsize=12)

    # Add a single legend for all subplots
    fig.legend(
        combined_legend_handles,
        combined_legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=4,
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# Run analysis
plot_node_metrics_dual_axis(node_metrics_files)

