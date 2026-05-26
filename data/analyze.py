import pandas as pd
import matplotlib.pyplot as plt
import os

# File paths (add your actual folder path)
folder_path = "analysis"

# File mapping
files = {
    "node_metrics": [
        "data_hpa_2_node_metrics.csv",
        "data_hpa_5_node_metrics.csv",
        "data_kf_2_node_metrics.csv",
        "data_kf_5_node_metrics.csv",
        "data_th_2_node_metrics.csv",
        "data_th_5_node_metrics.csv",
    ],
    "pod_counts": [
        "data_hpa_2_pod_counts.csv",
        "data_hpa_5_pod_counts.csv",
        "data_kf_2_pod_counts.csv",
        "data_kf_5_pod_counts.csv",
        "data_th_2_pod_counts.csv",
        "data_th_5_pod_counts.csv",
    ],
    "throughput": [
        "data_hpa_2_throughput_worker1.csv",
        "data_hpa_5_throughput_worker1.csv",
        "data_kf_2_throughput_worker1.csv",
        "data_kf_5_throughput_worker1.csv",
        "data_th_2_throughput_worker1.csv",
        "data_th_5_throughput_worker1.csv",
    ],
}

# Helper function to load data
def load_csv(file_name):
    return pd.read_csv(os.path.join(folder_path, file_name))

# Plotting node metrics
def plot_node_metrics(files):
    for file_name in files:
        df = load_csv(file_name)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        metric_label = file_name.split("_")[1]  # Extract threshold (e.g., hpa_2)
        
        plt.figure(figsize=(12, 6))
        for column in df.columns[1:]:  # Skip Timestamp column
            plt.plot(df["Timestamp"], df[column], label=column)
        plt.title(f"Node Metrics Over Time ({metric_label})")
        plt.xlabel("Timestamp")
        plt.ylabel("Usage")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.grid()
        plt.show()

# Plotting pod counts
def plot_pod_counts(files):
    for file_name in files:
        df = load_csv(file_name)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        metric_label = file_name.split("_")[1]  # Extract threshold (e.g., hpa_2)

        plt.figure(figsize=(12, 6))
        for column in df.columns[1:]:  # Skip Timestamp column
            plt.plot(df["Timestamp"], df[column], label=column)
        plt.title(f"Pod Counts Over Time ({metric_label})")
        plt.xlabel("Timestamp")
        plt.ylabel("Pod Counts")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.grid()
        plt.show()

# Plotting throughput
def plot_throughput(files):
    for file_name in files:
        df = load_csv(file_name)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        metric_label = file_name.split("_")[1]  # Extract threshold (e.g., hpa_2)

        plt.figure(figsize=(12, 6))
        plt.plot(df["Timestamp"], df["Throughput (Bytes/sec)"], label="Throughput")
        plt.title(f"Throughput Over Time ({metric_label})")
        plt.xlabel("Timestamp")
        plt.ylabel("Throughput (Bytes/sec)")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.grid()
        plt.show()

# Run analysis
plot_node_metrics(files["node_metrics"])
plot_pod_counts(files["pod_counts"])
plot_throughput(files["throughput"])

