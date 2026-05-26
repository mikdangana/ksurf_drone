import csv
import subprocess
import time
from datetime import datetime

# Configuration
NAMESPACE = "kube-system"   # Kubernetes namespace
CSV_FILE = "pod_metrics.csv"
INTERVAL = 5  # Polling interval in seconds
INTERVAL_MAX = 500 #300  # 10 min


# Initialize CSV File with Header Row
def initialize_csv(header=["Timestamp"]):
    with open(CSV_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write the initial header row
        #writer.writerow(header)


# Fetch All Pod Names
def get_all_pods():
    try:
        result = subprocess.run(
            ["kubectl", "get", "pods", "-n", NAMESPACE, "--no-headers"],
            stdout=subprocess.PIPE,
            text=True,
            check=True,
        )

        if not result.stdout.strip():
            print("No pods available.")
            return []

        pods = [line.split()[0] for line in result.stdout.strip().split("\n")]
        return pods

    except subprocess.CalledProcessError as e:
        #print(f"Error fetching pods: {e}")
        return []


# Fetch Metrics for a Single Pod
def fetch_pod_metrics(pod_name):
    try:
        result = subprocess.run(
            ["kubectl", "top", "pod", pod_name, "-n", NAMESPACE, "--no-headers"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
        )

        if not result.stdout.strip():
            return None

        pod_metrics = result.stdout.strip().split()
        cpu = pod_metrics[1].replace("m", "")
        memory = pod_metrics[2].replace("Mi", "")
        return (cpu, memory)

    except subprocess.CalledProcessError:
        return ("-", "-")


# Fetch Metrics for All Pods and Write Columns
def fetch_all_metrics_and_write(init=False):
    pods = get_all_pods()
    if not pods:
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [timestamp]
    header = ["Timestamp"]

    for pod in pods:
        cpu, memory = fetch_pod_metrics(pod)
        row.extend([cpu, memory])
        header.extend([f"{pod} CPU (m)", f"{pod} Memory (MiB)"])

    # Write Header if Not Present
    if init:
        initialize_csv(header)
        print(f"{','.join(['' if v=='-' else v for v in header])}")

    # Append Data to CSV
    with open(CSV_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(row)
        print(f"{','.join(['' if v=='-' else v for v in row])}")


# Main Execution Loop
if __name__ == "__main__":
    #initialize_csv()

    try:
        interval_count = 0
        while interval_count < INTERVAL_MAX:
            fetch_all_metrics_and_write(interval_count==0)
            interval_count = interval_count + 1
            time.sleep(INTERVAL)

    except KeyboardInterrupt:
        print("\nMonitoring stopped. CSV export complete.")

