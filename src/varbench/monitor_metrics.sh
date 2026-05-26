#!/bin/bash

# Configuration
NAMESPACE="kube-system"       # Change this to your namespace
CSV_FILE="pod_metrics.csv"    # CSV output file
INTERVAL=2                    # Fetch interval in seconds
INTERVAL_COUNT=0	      
INTERVAL_MAX=300	      # 10 min

# Initialize CSV File
initialize_csv() {
    echo "Timestamp,Pod Name,CPU(m),Memory(MiB)" > "$CSV_FILE"
    echo "Initialized CSV file: $CSV_FILE"
}

# Fetch Metrics from All Pods in the Namespace
fetch_metrics() {
    local timestamp
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")

    # Get pod metrics
    kubectl top pods -n "$NAMESPACE" --no-headers | while read -r line; do
        pod_name=$(echo "$line" | awk '{print $1}')
        cpu=$(echo "$line" | awk '{print $2}' | sed 's/m//')
        memory=$(echo "$line" | awk '{print $3}' | sed 's/Mi//')

        # Append to CSV
        echo "$timestamp,$pod_name,$cpu,$memory" >> "$CSV_FILE"
    done

    echo "Updated CSV at $timestamp"
}

# Main Execution
initialize_csv

# Continuous Loop to Fetch Metrics
while [ $INTERVAL_COUNT -lt $INTERVAL_MAX ]; do
    fetch_metrics
    INTERVAL_COUNT=$((INTERVAL_COUNT + 1))
    sleep "$INTERVAL"
done

