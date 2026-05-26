import os, re, re
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tarfile
import itertools


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
    df = pd.read_csv(file_name)
    df = compute_sum_column("All workers CPU (m)", df, list(df.columns), ".*worker.*CPU.*")
    df = compute_sum_column("All workers Memory (MiB)", df, list(df.columns), ".*worker.*Mem.*")
    return df


def coerce(df, col_name):
    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
    df = df.dropna()
    return df


def std(df, col_name):
    df = coerce(df, col_name)
    
    # Step 1: Compute mean
    mean_value = df[col_name].mean()

    # Step 2: Compute squared deviations
    squared_diffs = (df[col_name] - mean_value) ** 2

    # Step 3: Compute variance (sample standard deviation: divide by N-1, for population use len(df['values']))
    variance = squared_diffs.sum() / (len(df[col_name]) - 1)

    # Step 4: Compute standard deviation
    std_dev = np.sqrt(variance)

    print("Manual Standard Deviation:", std_dev)
    return std_dev


def plot_mean_std(file_paths, col_name="elapsed_time", col_label="Elapsed Time (ms)"):
    """
    Parses files and plots mean elapsed time as a bar graph with mean data labels and a legend showing mean ± stddev.

    Args:
        directory (str): Directory containing the data files.
    """
    # Initialize a dictionary to store results
    results = {"algorithm": [], "threshold": [], "mean_elapsed_time": [], "stddev_elapsed_time": []}
    print(f"plot_mean_std.col = {col_name}")

    # Find all matching files
    if not file_paths:
        print("No files found matching the pattern!")
        return

    # Process each file
    for file_path in file_paths:
        # Extract algorithm and threshold from filename
        #file_name = os.path.basename(file_path)
        #parts = file_name.split("_")
        metric_label = os.path.basename(os.path.dirname(file_path))
        parts = metric_label.split("_")
        if len(parts) < 3:
            print(f"Skipping invalid file name: {file_name}")
            continue
        algorithm, threshold = parts[1], parts[2]

        # Read the file into a DataFrame
        df = load_csv(file_path)

        # Calculate mean and standard deviation for elapsed time
        #print(f"mean.col_name = {col_name}, file_path = {file_path}")
        #df = df[pd.to_numeric(df[col_name], errors='coerce').notna()]
        df = coerce(df, col_name)
        if len(df.index) <= 0:
            continue;
        mean_elapsed_time = df[col_name].mean()
        try:
            stddev_elapsed_time = df[col_name].std()
        except:
            print(f"Error computing pd.std()")
            stddev_elapsed_time = std(df, col_name)
        #print(f"col = {col_name}, mean = {mean_elapsed_time}, std = {stddev_elapsed_time}, algo = {algorithm}, threshold = {threshold}, df = {len(df.index)}")

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

    tag = col_name.replace("(", "").replace(")", "").replace("/", "_")
    # Plot settings
    plt.title(f"Mean {col_label} by Algorithm and Threshold")
    plt.xlabel("Threshold (%)")
    plt.ylabel(f"Mean {col_label}")
    plt.xticks(x + bar_width * (len(pivot_means.columns) - 1) / 2, thresholds)
    plt.legend(title="Algorithm (Mean ± Stddev)", fontsize=9)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(f"kafka_summary_mean_{tag}_bar_graph.png")
    plt.show()

    # Save individual plots for each algorithm
    for algorithm in []: #pivot_means.columns:
        plt.figure(figsize=(10, 6))
        means = pivot_means[algorithm]
        stds = pivot_stds[algorithm]
        plt.bar(thresholds, means, yerr=stds, capsize=5, color="skyblue")
        plt.title(f"{algorithm} - Mean {col_label} by Threshold")
        plt.xlabel("Threshold (%)")
        plt.ylabel(f"Mean {col_label}")
        for i, mean in enumerate(means):
            plt.text(i, mean, f"{mean:.2f}", ha="center",va="bottom",fontsize=8)
        plt.grid(axis="y")
        plt.tight_layout()
        #plt.savefig(f"kafka_summary_{algorithm.lower()}_mean_{col_name}.png")
        #plt.show()


def compute_sum_column(col_name, df, columns, regex):
    filtered_cols = list(filter(lambda col: re.match(regex, col), columns))
    for col in filtered_cols:
        df = coerce(df, col)
    df[col_name] = df[filtered_cols].sum(axis=1)
    return df


def filter_columns(columns):
    unwanted_cols = ["Timestamp", "ksurf-producer-0f35f8a8 CPU (m)", "ksurf-producer-0f35f8a8 Memory (MiB)"]
    #print(f"filter.columns = {list(columns)}")
    return filter(lambda col: col not in unwanted_cols, list(columns))


def plot_mean_std_all_columns(files):
    df = load_csv(files[0])
    for col in filter_columns(df.columns[:]):
        plot_mean_std(files, col_name=col, col_label=col.replace("Throughput", "Queue Size"))


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
    #plot_with_mean_std(files["node_metrics"])
    
    # Example usage
    #plot_mean_std_all_columns(files["node_metrics"])
    
    files = get_files_from_folders(folders, ["throughput"])
    plot_mean_std_all_columns(files["throughput"])


if __name__ == "__main__":
    main()
