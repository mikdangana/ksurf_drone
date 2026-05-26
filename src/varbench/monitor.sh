#!/bin/bash

# Log file for storing output
LOG_FILE="pod_metrics.log"
PLOT_FILE="pod_metrics_plot.png"

# Create or clear the log file
> $LOG_FILE

echo "Starting pod metrics monitoring..."

# Infinite monitoring loop
while true; do
    echo "Fetching metrics at $(date)..."

    # Collect metrics using kubectl
    METRICS=$(kubectl top pods -n kube-system --no-headers)

    # Save metrics to the log file
    echo "$METRICS" >> "$LOG_FILE"

    # Create a temporary CSV file for plotting
    echo "$METRICS" | awk '{print $1","$2","$3}' > metrics.csv

    # Generate the plot using Python
    python3 - <<EOF
EOF

    echo "Metrics fetched and logged at $(date). Waiting for 5 seconds..."
    sleep 5
done

