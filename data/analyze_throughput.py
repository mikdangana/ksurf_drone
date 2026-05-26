import pandas as pd
import matplotlib.pyplot as plt
import os, re
import tarfile


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


# Helper function to parse the autoscaler type and threshold from the filename
def parse_file_metadata(filename):
    # e.g. 'extracted_data\\data_na_5\\pod_counts.csv'
    filename = re.split("[\\\\]", filename)[-2]
    parts = filename.split("_")
    algorithm = parts[1]  # hpa, kf, or th
    threshold = parts[2]  # 2 or 5
    worker = "worker1" if "worker1" in filename else "worker2"
    #print("filename, algo, thresh, worker = ", filename, algorithm, threshold, worker)
    return algorithm, threshold, worker

# Plot throughput for each autoscaler and threshold
def plot_throughput(csv_files):
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))  # 3 rows (hpa, kf, th) x 2 columns (thresholds)
    axes = axes.flatten()  # Flatten the axes for easy indexing

    # Organize data by algorithm and threshold
    grouped_data = {"dr": {2: [], 5: []}, "drkf": {2: [], 5: []}, "hpa": {2: [], 5: []}, "kf": {2: [], 5: []}, "th": {2: [], 5: []}, "na": {2: [], 5: []}}
    algos = []

    # Read data and group by algorithm and threshold
    for file in csv_files:
        algorithm, threshold, worker = parse_file_metadata(file)
        algos.extend([algorithm])
        df = pd.read_csv(file)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])  # Convert timestamp to datetime
        df["Elapsed Time (s)"] = (df["Timestamp"] - df["Timestamp"].iloc[0]).dt.total_seconds()
        grouped_data[algorithm][int(threshold)].append((worker, df))

    # Plot each algorithm's throughput
    for i, (algorithm, thresholds) in zip(range(len(grouped_data)), filter(lambda item: item[0] in algos, list(grouped_data.items()))):
        for j, (threshold, data) in zip(range(len(thresholds)), list(thresholds.items())):
            ax = axes[i * 2 + j]
            legend_entries = []  # For custom legend entries
            for worker, df in data:
                # Plot throughput
                ax.plot(df["Elapsed Time (s)"], df["Throughput (Bytes/sec)"], label=worker)
                # Calculate mean and variance
                mean_throughput = df["Throughput (Bytes/sec)"].mean()
                variance_throughput = df["Throughput (Bytes/sec)"].var()
                # Add to legend with mean ± variance in scientific notation
                legend_entries.append(
                    f"{worker} ({mean_throughput:.2e} ± {variance_throughput:.2e})"
                )
            # Update legend with scientific notation
            ax.legend(legend_entries, fontsize=8)
            ax.set_title(f"{algorithm.upper()} ({threshold}% Threshold)", fontsize=10)
            ax.set_xlabel("Time (s)", fontsize=8)
            ax.set_ylabel("Throughput (Bytes/sec)", fontsize=8)
            ax.grid(True)

    plt.tight_layout()
    plt.show()



# Main function to process the tarball and generate plots
def main():
    tarball_path = "data.tar"
    extract_path = "extracted_data"

    # Extract and filter folders
    folders = extract_and_filter_tarball(tarball_path, extract_path)

    # Identify files by tokens
    tokens = ["throughput"]
    files = get_files_from_folders(folders, tokens)

    # Run the plot
    plot_throughput(files["throughput"])


if __name__ == "__main__":
    main()
