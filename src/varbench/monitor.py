import csv
import subprocess
import time
from datetime import datetime

# Configuration
CSV_FILE = "pod_metrics.csv"
REFRESH_INTERVAL = 5  # seconds
NAMESPACE = "kube-system"  # Target namespace
has_header = False

# Initialize CSV file and header
def initialize_csv():
    try:
        with open(CSV_FILE, mode="w", newline="") as file:
            writer = csv.writer(file)
            # Initialize the CSV header
            #writer.writerow(["Timestamp", "Pod Name", "CPU (m)", "Memory (MiB)"])
            print(f"Initialized CSV file: {CSV_FILE}")
    except Exception as e:
        print(f"Error initializing CSV: {e}")


# Function to collect metrics
def collect_metrics():
    try:
        result = subprocess.run(
            ["kubectl", "top", "pods", "-n", NAMESPACE, "--no-headers"],
            stdout=subprocess.PIPE,
            text=True,
        )

        if not result.stdout.strip():
            print("No metrics available. Retrying...")
            return []

        metrics = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metrics.append(["time", timestamp])

        # Extract pod metrics
        for line in result.stdout.strip().split("\n"):
            parts = line.split()
            pod_name, cpu, memory = parts[0], parts[1].replace("m", ""), parts[2].replace("Mi", "")
            metrics.append([pod_name, cpu])
            metrics.append([pod_name, memory])

        metrics.sort(key=lambda row: row[0])
        return [row[0] for row in metrics], [row[1] for row in metrics]

    except subprocess.CalledProcessError as e:
        print(f"Error fetching metrics: {e}")
        return []


# Function to append metrics to CSV
def update_csv(metrics):
    try:
        with open(CSV_FILE, mode="a", newline="") as file:
            writer = csv.writer(file)
            for metric in metrics:
                if not has_header:
                    writer.writerow(metric[0])
                    has_header = True
                writer.writerow(metric[1])
            print(f"Updated CSV at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        print(f"Error writing to CSV: {e}")


# Main Loop
if __name__ == "__main__":
    initialize_csv()

    try:
        while True:
            metrics = collect_metrics()
            if metrics:
                update_csv(metrics)
            time.sleep(REFRESH_INTERVAL)

    except KeyboardInterrupt:
        print("\nMonitoring stopped. CSV export complete.")

