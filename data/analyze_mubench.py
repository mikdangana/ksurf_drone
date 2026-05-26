import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    # --- text & label sizes ---
    "font.size": 14,            # base size for everything
    "axes.titlesize": 14,       # axes title
    "axes.labelsize": 14,       # x-/y-label
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 18,
    "legend.title_fontsize": 18,
    # --- optional: make the whole plot a bit larger ---
    "figure.figsize": (12, 7)   # width, height in inches
})

def plot_mubench_files(directory="."):
    """
    Parses and plots data from files matching the pattern mubench_<algorithm>_<threshold>.txt.

    Args:
        directory (str): Directory containing the files.
    """
    # Find all matching files
    file_paths = glob.glob(os.path.join(directory, "mubench_*1.txt"))
    if not file_paths:
        print("No files found matching the pattern!")
        return

    # Combined figures
    elapsed_fig, elapsed_axs = plt.subplots(len(file_paths), 1, figsize=(12, 6 * len(file_paths)))
    status_fig, status_axs = plt.subplots(len(file_paths), 1, figsize=(12, 6 * len(file_paths)))
    requests_fig, requests_axs = plt.subplots(len(file_paths), 1, figsize=(12, 6 * len(file_paths)))

    algos = {"hpa": "HPA", "kf": "KF", "th": "TH", "na": "NA"}
    for idx, file_path in enumerate(file_paths):
        # Extract algorithm and threshold from filename
        file_name = os.path.basename(file_path)
        parts = file_name.split("_")
        if len(parts) < 3:
            print(f"Skipping invalid file name: {file_name}")
            continue
        algorithm = parts[1]
        threshold = parts[2].split(".")[0]

        # Read the file into a DataFrame
        df = pd.read_csv(file_path, delim_whitespace=True, header=None,
                         names=["timestamp", "elapsed_time", "http_status", "processed", "pending"])

        # Validate and preprocess timestamps
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

        # Skip if the DataFrame is empty after filtering
        if df.empty:
            print(f"No valid data in file: {file_name}")
            continue

        # Calculate relative time in seconds
        df["relative_time"] = df["timestamp"] 

        # Plot elapsed time
        elapsed_axs[idx].plot(df["relative_time"], df["elapsed_time"], label=f"Elapsed Time: {df['elapsed_time'].mean():.2f}±{df['elapsed_time'].std():.2f} (ms)")
        elapsed_axs[idx].set_title(f"Elapsed Time - Autoscaler: {algos[algorithm]}, Threshold: {threshold}")
        #elapsed_axs[idx].set_xlabel("Time (s)")
        elapsed_axs[idx].set_ylabel("Time (ms)")
        elapsed_axs[idx].legend()
        elapsed_axs[idx].grid()

        # Plot HTTP status counts
        status_counts = df["http_status"].value_counts()
        status_axs[idx].bar(status_counts.index, status_counts.values, color="skyblue")
        status_axs[idx].set_title(f"HTTP Status Codes - Autoscaler: {algos[algorithm]}, Threshold: {threshold}")
        status_axs[idx].set_xlabel("HTTP Status")
        status_axs[idx].set_ylabel("Count")
        status_axs[idx].grid(axis="y")

        # Plot processed and pending requests
        requests_axs[idx].plot(df["relative_time"], df["processed"], label=f"Processed Requests", color="green")
        requests_axs[idx].plot(df["relative_time"], df["pending"], label=f"Pending Requests", color="red")
        requests_axs[idx].set_title(f"Requests - Autoscaler: {algos[algorithm]}, Threshold: {threshold}")
        requests_axs[idx].set_xlabel("Time (s)")
        requests_axs[idx].set_ylabel("Requests")
        requests_axs[idx].legend()
        requests_axs[idx].grid()

    # Save combined figures
    elapsed_axs[len(elapsed_axs)-1].set_xlabel("Time (s)")
    elapsed_fig.tight_layout()
    elapsed_fig.savefig("combined_elapsed_time.png")
    plt.show()
    status_fig.tight_layout()
    status_fig.savefig("combined_http_status.png")
    requests_fig.tight_layout()
    requests_fig.savefig("combined_requests.png")

    print("Combined plots saved successfully!")

# Example usage
plot_mubench_files(directory=".")

