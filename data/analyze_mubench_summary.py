import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_mean_std_elapsed_time(directory="."):
    """
    Parses files and plots mean elapsed time as a bar graph with mean data labels and a legend showing mean ± stddev.

    Args:
        directory (str): Directory containing the data files.
    """
    # Initialize a dictionary to store results
    results = {"algorithm": [], "threshold": [], "mean_elapsed_time": [], "stddev_elapsed_time": []}

    # Find all matching files
    file_paths = glob.glob(os.path.join(directory, "mubench_*.txt"))
    if not file_paths:
        print("No files found matching the pattern!")
        return

    # Process each file
    for file_path in file_paths:
        # Extract algorithm and threshold from filename
        file_name = os.path.basename(file_path)
        parts = file_name.split("_")
        if len(parts) < 3:
            print(f"Skipping invalid file name: {file_name}")
            continue
        algorithm = parts[1]
        threshold = parts[2].split(".txt")[0]

        # Read the file into a DataFrame
        df = pd.read_csv(file_path, delim_whitespace=True, header=None,
                         names=["timestamp", "elapsed_time", "http_status", "processed", "pending"])

        # Calculate mean and standard deviation for elapsed time
        mean_elapsed_time = df["elapsed_time"].mean()
        stddev_elapsed_time = df["elapsed_time"].std()

        # Store results
        results["algorithm"].append(algorithm.upper())
        results["threshold"].append(threshold)
        results["mean_elapsed_time"].append(mean_elapsed_time)
        results["stddev_elapsed_time"].append(stddev_elapsed_time)

    # Create a DataFrame from results
    results_df = pd.DataFrame(results)

    # Pivot the DataFrame for plotting
    pivot_means = results_df.pivot(index="threshold", columns="algorithm", values="mean_elapsed_time")
    pivot_stds = results_df.pivot(index="threshold", columns="algorithm", values="stddev_elapsed_time")

    # Plot mean elapsed time as a bar graph with mean data labels
    plt.figure(figsize=(10, 6))
    bar_width = 0.2
    thresholds = pivot_means.index.astype(float).astype(str) + "%"
    x = np.arange(len(thresholds))

    for idx, algorithm in enumerate(pivot_means.columns):
        means = pivot_means[algorithm]
        stds = pivot_stds[algorithm]
        bar_positions = x + idx * bar_width
        plt.bar(bar_positions, means, bar_width, label=f"{algorithm} (Mean: {means.mean():.2f} ± {stds.mean():.2f})", yerr=stds, capsize=5)

        # Add mean data labels directly on the data point
        for pos, mean in zip(bar_positions, means):
            plt.text(pos, mean, f"{mean:.2f}", ha="center", va="bottom", fontsize=8)

    # Plot settings
    plt.title("Mean Elapsed Time by Algorithm and Threshold")
    plt.xlabel("Threshold (%)")
    plt.ylabel("Mean Elapsed Time (ms)")
    plt.xticks(x + bar_width * (len(pivot_means.columns) - 1) / 2, thresholds)
    plt.legend(title="Algorithm (Mean ± Stddev)", fontsize=9)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig("mean_elapsed_time_bar_graph_mean_labels_percent.png")
    plt.show()

    # Save individual plots for each algorithm
    for algorithm in pivot_means.columns:
        plt.figure(figsize=(10, 6))
        means = pivot_means[algorithm]
        stds = pivot_stds[algorithm]
        plt.bar(thresholds, means, yerr=stds, capsize=5, color="skyblue")
        plt.title(f"{algorithm} - Mean Elapsed Time by Threshold")
        plt.xlabel("Threshold (%)")
        plt.ylabel("Mean Elapsed Time (ms)")
        for i, mean in enumerate(means):
            plt.text(i, mean, f"{mean:.2f}", ha="center", va="bottom", fontsize=8)
        plt.grid(axis="y")
        plt.tight_layout()
        plt.savefig(f"{algorithm.lower()}_mean_elapsed_time.png")
        plt.show()

# Example usage
plot_mean_std_elapsed_time(directory=".")

