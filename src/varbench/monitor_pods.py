import csv
import subprocess
import time
from datetime import datetime

# Configuration
CSV_FILE = "data/pod_counts.csv"
INTERVAL = 2  # Polling interval in seconds
INTERVAL_MAX = 2*300  # 10 min
NODES = sys.argv[sys.argv.index("-n")+1].split(",") if "-n" in sys.argv else ["worker1", "worker2"] 
NAMESPACE = sys.argv[sys.argv.index("-ns")+1] if "-ns" in sys.argv else ["worker1", "worker2"] 


# Initialize CSV File with Header Row
def initialize_csv():
    with open(CSV_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write initial header row
        header = ["Timestamp"] + [f"{node} Running Pods" for node in NODES]
        writer.writerow(header)
        print(f"Initialized CSV file: {CSV_FILE}: {','.join(header)}")


# Fetch Running Pods for a Specific Node
def get_running_pods_count(node_name):
    try:
        # Run the kubectl command to count running pods on the node
        cmd = f"kubectl get pods -n {NAMESPACE} --kubeconfig=/home/ubuntu/.kube/config | grep {node_name}.*Run | wc -l"
        result = subprocess.run(
            cmd, shell=True, stdout=subprocess.PIPE, text=True, check=True
        )
        return int(result.stdout.strip()) if result.stdout.strip().isdigit() else 0
    except subprocess.CalledProcessError as e:
        print(f"Error counting pods on {node_name}: {e}")
        return 0


# Fetch Metrics and Write to CSV
def fetch_and_log_pods_count():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [timestamp]

    for node in NODES:
        pod_count = get_running_pods_count(node)
        row.append(pod_count)

    # Append data to CSV
    with open(CSV_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(row)
        print(f"Updated metrics at {timestamp}: {row}")


# Main Execution Loop
if __name__ == "__main__":
    initialize_csv()

    try:
        interval_count = 0
        while interval_count < INTERVAL_MAX:
            fetch_and_log_pods_count()
            interval_count += 1
            time.sleep(INTERVAL)

    except KeyboardInterrupt:
        print("\nMonitoring stopped. CSV export complete.")

