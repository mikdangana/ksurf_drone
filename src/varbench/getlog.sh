kubectl logs `cat pod.txt` -n kafka -c kafka > log.txt

kubectl logs `cat pod.txt` -n kafka -c kafka-consumer > log_consumer.txt

echo "Output in log.txt & log_consumer.txt"
