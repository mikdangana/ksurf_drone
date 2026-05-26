import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def plot_mubench_files(directory="."):
    """
    Parses and plots data from files matching the pattern mubench_<algorithm>_<threshold>.txt.

    Args:
        directory (str): Directory containing the files.
    """
    # Find all matching files
    file_paths = glob.glob(os.path.join(directory, "mubench_*.txt"))
    if not file_paths:
        print("No files found matching the pattern!")
        return

    # Combined figures
    timestamp_fig, timestamp_axs = plt.subplots(len(file_paths), 1, figsize=(12, 8), sharex=True)
    elapsed_fig, elapsed_axs = plt.subplots(len(file_paths), 1, figsize=(12, 8), sharex=True)
    requests_fig, requests_axs = plt.subplots(len(file_paths), 1, figsize=(12, 8), sharex=True)

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
        #df = df[df["timestamp"] > 0]
        #df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        #df = df.dropna(subset=["timestamp"])

        # Calculate mean and standard deviation for legend
        elapsed_mean = df["elapsed_time"].mean()
        elapsed_std = df["elapsed_time"].std()
        processed_mean = df["processed"].mean()
        processed_std = df["processed"].std()
        pending_mean = df["pending"].mean()
        pending_std = df["pending"].std()

        # Plot timestamps
        timestamp_axs[idx].plot(df["timestamp"], df["elapsed_time"], label=f"Elapsed Time (Mean: {elapsed_mean:.2f} ± {elapsed_std:.2f} ms)")
        timestamp_axs[idx].set_title(f"Timestamps - Algorithm: {algorithm}, Threshold: {threshold}")
        timestamp_axs[idx].legend()
        timestamp_axs[idx].grid()

        # Plot elapsed time
        elapsed_axs[idx].plot(df["timestamp"], df["elapsed_time"], label=f"Elapsed Time (Mean: {elapsed_mean:.2f} ± {elapsed_std:.2f} ms)")
        elapsed_axs[idx].set_title(f"Elapsed Time - Algorithm: {algorithm}, Threshold: {threshold}")
        elapsed_axs[idx].legend()
        elapsed_axs[idx].grid()

        # Plot processed and pending requests
        requests_axs[idx].plot(df["timestamp"], df["processed"], label=f"Processed (Mean: {processed_mean:.2f} ± {processed_std:.2f})", color="green")
        requests_axs[idx].plot(df["timestamp"], df["pending"], label=f"Pending (Mean: {pending_mean:.2f} ± {pending_std:.2f})", color="red")
        requests_axs[idx].set_title(f"Requests - Algorithm: {algorithm}, Threshold: {threshold}")
        requests_axs[idx].legend()
        requests_axs[idx].grid()

    # Save combined figures
    timestamp_fig.tight_layout()
    timestamp_fig.savefig("combined_timestamps.png")
    print("Saved combined_timestamps.png")
    elapsed_fig.tight_layout()
    elapsed_fig.savefig("combined_elapsed_time.png")
    print("Saved combined_elapsed_time.png")
    requests_fig.tight_layout()
    requests_fig.savefig("combined_requests.png")
    print("Saved combined_requests.png")

    print("Combined plots saved successfully!")

# Example usage
plot_mubench_files(directory=".")

