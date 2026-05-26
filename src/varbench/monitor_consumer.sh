#!/bin/bash

# Configuration
NAMESPACE="kube-system"
CONTAINER_NAME="kafka-consumer"
LOG_FILE="/var/log/kafka-consumer.log"
POLL_INTERVAL=5  # In seconds
POLL_COUNT=0
MAX_POLL_COUNT=1000 #300 # 10 min
KAFKA_BROKER_ID="worker1"
if [ $# -gt 0 ]; then KAFKA_BROKER_ID=$1; fi
CSV_FILE="data/throughput_$KAFKA_BROKER_ID.csv"
echo "KAFKA_BROKER_ID = $KAFKA_BROKER_ID"

# Initialize CSV File
initialize_csv() {
    echo "Timestamp,File Size (Bytes),Throughput (Bytes/sec)" > "$CSV_FILE"
    echo "Initialized CSV file: $CSV_FILE"
}

# Get the size of the log file in the specified container
get_file_size() {
    #POD_NAME=$(kubectl get pod -n "$NAMESPACE" -l "app=kafka-$KAFKA_BROKER_ID-app" -o jsonpath="{.items[0].metadata.name}")
    POD_NAME=$(kubectl get pods -n "$NAMESPACE" | grep $KAFKA_BROKER_ID.*Run | awk '{print $1}' | xargs)
    kubectl exec -n "$NAMESPACE" "$POD_NAME" -c "$CONTAINER_NAME" -- sh -c "stat -c %s $LOG_FILE" 2>/dev/null
}

# Monitor and log throughput
track_throughput() {
    LAST_SIZE=0

    while [ $POLL_COUNT -lt $MAX_POLL_COUNT ]; do
        TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
        CURRENT_SIZE=$(get_file_size)
	echo CURRENT_SIZE = $CURRENT_SIZE

        if [ ! $CURRENT_SIZE -gt 0 ]; then
            echo "Error: Unable to retrieve file size. Retrying..."
            sleep "$POLL_INTERVAL"
            continue
        fi

        THROUGHPUT=$(( (CURRENT_SIZE - LAST_SIZE) / POLL_INTERVAL ))
        if [ $THROUGHPUT -lt 0 ]; then
            THROUGHPUT=0
        fi

        echo "$TIMESTAMP,$CURRENT_SIZE,$THROUGHPUT" | tee -a "$CSV_FILE"
        echo "$TIMESTAMP,$CURRENT_SIZE,$THROUGHPUT" 

        # Update last size
        LAST_SIZE=$CURRENT_SIZE

	POLL_COUNT=$(( POLL_COUNT + 1 ))
        # Wait for next interval
        sleep "$POLL_INTERVAL"
    done
}

# Main Execution
initialize_csv
track_throughput

