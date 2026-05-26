import csv
import subprocess, sys
import time
from datetime import datetime

# Configuration
CSV_FILE = "data/node_metrics.csv"
INTERVAL = 2   # Polling interval in seconds
INTERVAL_MAX = 2*300  # Stop after 300 intervals (25 minutes)
ALGORITHM = sys.argv[sys.argv.index("--algorithm")+1] if "--algorithm" in sys.argv else ""
THRESHOLD = sys.argv[sys.argv.index("--threshold")+1] if "--threshold" in sys.argv else ""


# Initialize CSV File with Header Row
def initialize_csv():
    with open(CSV_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write the initial header row
        #writer.writerow(["Timestamp"])


# Fetch Metrics for All Nodes
def fetch_node_metrics():
    try:
        result = subprocess.run(
            ["kubectl", "top", "nodes", "--no-headers", "-n", "kube-system"],
            stdout=subprocess.PIPE,
            text=True,
            check=True,
        )

        if not result.stdout.strip():
            print("No nodes available.")
            return None

        return result.stdout.strip().replace('<unknown>', '-').split("\n")

    except subprocess.CalledProcessError as e:
        print(f"Error fetching node metrics: {e}")
        return None


# Fetch Metrics and Write to CSV
def fetch_metrics_and_write(interval):
    node_metrics = fetch_node_metrics()
    if not node_metrics:
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [timestamp]
    header = ["Timestamp"]

    for node in node_metrics:
        node_info = node.split()
        node_name = node_info[0]
        cpu = node_info[1].replace("m", "")
        memory = node_info[3].replace("Mi", "")

        row.extend([cpu, memory])
        header.extend([f"{node_name} CPU (m)", f"{node_name} Memory (MiB)"])

    # Read Existing Header
    with open(CSV_FILE, mode="r", newline="") as file:
        reader = csv.reader(file)
        existing_header = next(reader, None)

    # Append Data to CSV
    with open(CSV_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)

        # Write Header If It's Not Present
        if existing_header is None: # or header != existing_header:
            writer.writerow(header)
            print(f"Node CSV header updated: {','.join(header)}")

        writer.writerow(row)
        print(f"Nodes {ALGORITHM}-{THRESHOLD}, {interval}/{INTERVAL_MAX}: {','.join(row)}")


# Main Execution Loop
if __name__ == "__main__":
    initialize_csv()

    try:
        interval_count = 0
        while interval_count < INTERVAL_MAX:
            fetch_metrics_and_write(interval_count)
            interval_count += 1
            time.sleep(INTERVAL)

    except KeyboardInterrupt:
        print("\nMonitoring stopped. CSV export complete.")

