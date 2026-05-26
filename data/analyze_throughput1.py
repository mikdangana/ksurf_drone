import pandas as pd
import matplotlib.pyplot as plt
import os
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


# Helper function to parse the autoscaler type and threshold from the filename
def parse_file_metadata(filename):
    filename = filename.split("/")[-1]
    parts = filename.split("_")
    algorithm = parts[2]  # hpa, kf, or th
    threshold = parts[3].split("\\")[0]  # 2 or 5
    worker = parts[4].split("-")[-1].replace(".csv", "")  # 2 or 5
    #worker = "worker1" if "worker1" in filename else "worker2"
    print(f"algo, threshold, worker = {(algorithm, threshold, worker)}")
    return algorithm, threshold, worker


# Plot throughput for each autoscaler and threshold
def plot_throughput(csv_files, summary={}):
    fig, axes = plt.subplots(6, 2, figsize=(12, 12))  # 3 rows (hpa, kf, th) x 2 columns (thresholds)
    axes = axes.flatten()  # Flatten the axes for easy indexing

    # Organize data by algorithm and threshold
    grouped_data = {} 

    # Read data and group by algorithm and threshold
    for file in csv_files:
        algorithm, threshold, worker = parse_file_metadata(file)
        df = pd.read_csv(file)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])  # Convert timestamp to datetime
        df["Elapsed Time (s)"] = (df["Timestamp"] - df["Timestamp"].iloc[0]).dt.total_seconds()
        if not algorithm in grouped_data:
            grouped_data[algorithm] = {}
        if not threshold in grouped_data[algorithm]:
            grouped_data[algorithm][threshold] = []
        grouped_data[algorithm][threshold].append((worker, df))

    # Plot each algorithm's throughput
    for i, (algorithm, thresholds) in enumerate(grouped_data.items()):
        for j, (threshold, data) in enumerate(thresholds.items()):
            ax = axes[i * 2 + j]
            legend_entries = []  # For custom legend entries
            print(f"algo, thresh, data = {(algorithm, threshold)}")
            for worker, df in data:
                # Plot throughput
                ax.plot(df["Elapsed Time (s)"], df["Throughput (Bytes/sec)"], label=worker)
                # Calculate mean and variance
                mean_throughput = df["Throughput (Bytes/sec)"].mean()
                variance_throughput = df["Throughput (Bytes/sec)"].var()
                if not algorithm in summary:
                    summary[algorithm] = {"means": [], "vars": []}
                summary[algorithm]["means"].append(mean_throughput)
                summary[algorithm]["vars"].append(variance_throughput)
                # Add to legend with mean ± variance in scientific notation
                legend_entries.append(
                    f"{worker} ({mean_throughput:.2e} ± {variance_throughput:.2e})"
                )
            # Update legend with scientific notation
            ax.legend(legend_entries, fontsize=8)
            ax.set_title(f"{algorithm.upper()} {float(threshold)/100*200000:.0f}µ Threshold", fontsize=10)
            if i == len(grouped_data.items())-1:
                ax.set_xlabel("Time (s)", fontsize=8)
            ax.set_ylabel("Queue Size (Bytes/sec)", fontsize=8)
            ax.grid(True)

    plt.tight_layout()
    plt.show()
    return summary


def filter_tokens(tokens, items):
    res = []
    for token in tokens:
        res = res + list(filter(lambda f: token in f, items))
    return res


def main():
    # Run the plot
    tarball_path = "data.tar"
    extract_path = "extracted_data"

    # Extract and filter folders
    folders = extract_and_filter_tarball(tarball_path, extract_path)

    # Identify files by tokens
    tokens = ["throughput"]
    csv_files = get_files_from_folders(folders, tokens)
    print(f"csv_files = {csv_files}")
    summary=plot_throughput(filter_tokens(["0.25", "_2"], csv_files[tokens[0]]))
    #summary=plot_throughput(filter_tokens(["_2"], csv_files[tokens[0]]), summary)
    for algorithm in summary.keys():
        summary[algorithm]["mean"] = pd.DataFrame(summary[algorithm]["means"]).mean()
        summary[algorithm]["std"] = pd.DataFrame(summary[algorithm]["vars"]).std()
    print("summary = {}".format(summary))




if __name__ == "__main__":
    main()


