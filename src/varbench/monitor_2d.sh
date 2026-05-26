#!/bin/bash

# Log file for storing output
LOG_FILE="pod_metrics_text.log"

# Clear or create the log file
> "$LOG_FILE"

echo "Starting pod metrics monitoring..."

# Infinite monitoring loop
while true; do
    echo "Fetching metrics at $(date)..."

    # Collect pod metrics using kubectl
    METRICS=$(kubectl top pods -n kube-system --no-headers)

    # Check if metrics collection was successful
    if [ -z "$METRICS" ]; then
        echo "No metrics available. Retrying..." >> "$LOG_FILE"
        sleep 5
        continue
    fi

    # Save metrics to the log file
    echo "$METRICS" >> "$LOG_FILE"

    # Print header
    echo -e "\nKubernetes Pod Resource Usage (CPU/Memory) at $(date)\n"
    printf "%-30s | %-20s | %-20s\n" "Pod Name" "CPU (m)" "Memory (Mi)"
    echo "-------------------------------------------|----------------------|----------------------"

    # Process metrics and display a text-based 2D graph
    echo "$METRICS" | while read -r line; do
        POD_NAME=$(echo "$line" | awk '{print $1}')
        CPU=$(echo "$line" | awk '{print $2}' | sed 's/m//')
        MEMORY=$(echo "$line" | awk '{print $3}' | sed 's/Mi//')

        # Convert CPU and memory to graph bars
        CPU_BAR=$(printf "%-${CPU}s" "#" | sed "s/ /#/g")
        MEMORY_BAR=$(printf "%-${MEMORY}s" "#" | sed "s/ /#/g")

        # Print metrics with text-based graph
        printf "%-30s | %-20s | %-20s\n" "$POD_NAME" "$CPU_BAR" "$MEMORY_BAR"
    done

    echo -e "\nMetrics logged at $(date). Waiting for 5 seconds..."
    sleep 5
done

