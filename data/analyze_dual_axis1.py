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

# Plotting node metrics with all axes visible and legend at the top
def plot_node_metrics_dual_axis(files):
    # Create a 2-column layout with up to 3 rows per column
    n_cols = 2
    n_rows = -(-len(files) // n_cols)  # Ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 5))

    combined_legend_handles = []
    combined_legend_labels = []

    # Flatten axes for easy iteration
    axes = axes.flatten()

    for i, file_name in enumerate(files):
        df = load_csv(file_name)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df["Time (s)"] = (df["Timestamp"] - df["Timestamp"].iloc[0]).dt.total_seconds()

        ax_memory = axes[i]
        ax_cpu = ax_memory.twinx()  # Create a second y-axis for CPU usage

        # Extract memory and CPU columns
        memory_cols = [col for col in df.columns if "Memory" in col]
        cpu_cols = [col for col in df.columns if "CPU" in col]

        # Ensure columns are numeric and drop invalid data
        df[memory_cols] = df[memory_cols].apply(pd.to_numeric, errors="coerce")
        df[cpu_cols] = df[cpu_cols].apply(pd.to_numeric, errors="coerce")

        # Plot Memory
        for col in memory_cols:
            line, = ax_memory.plot(df["Time (s)"], df[col], label=col, linestyle="-")
            if col not in combined_legend_labels:
                combined_legend_handles.append(line)
                combined_legend_labels.append(col)

        ax_memory.set_ylabel("Memory (MiB)", fontsize=9, color="blue")
        ax_memory.tick_params(axis='y', labelcolor='blue', labelsize=8)

        # Plot CPU
        for col in cpu_cols:
            line, = ax_cpu.plot(df["Time (s)"], df[col], label=col, linestyle="--")
            if col not in combined_legend_labels:
                combined_legend_handles.append(line)
                combined_legend_labels.append(col)

        ax_cpu.set_ylabel("CPU (m)", fontsize=9, color="red")
        ax_cpu.tick_params(axis='y', labelcolor='red', labelsize=8)

        ax_memory.set_xlabel("Time (s)", fontsize=9)
        ax_memory.set_title(f"Node Metrics ({file_name.split('_')[1]})", fontsize=9)
        ax_memory.grid()

    # Hide unused subplots
    for j in range(len(files), len(axes)):
        axes[j].set_visible(False)

    # Add a single legend at the top center
    fig.legend(
        combined_legend_handles,
        combined_legend_labels,
        loc="upper center",
        #bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        fontsize=9,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout for the legend
    plt.show()

# Run analysis
plot_node_metrics_dual_axis(node_metrics_files)

