kubectl autoscale deployment kafka-worker1 -n kafka --cpu-percent=20 --min=1 --max=10
