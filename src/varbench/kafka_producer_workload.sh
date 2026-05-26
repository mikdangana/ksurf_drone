#!/bin/bash
POD_ID=`kubectl get pods -n kube-system --kubeconfig=/home/ubuntu/.kube/config | grep worker.*Run | head -1 | awk '{print $1}'`
POD_IP=`kubectl exec -it -n kube-system --kubeconfig=/home/ubuntu/.kube/config $POD_ID -- hostname -I | xargs`
echo POD_IP = $POD_IP, POD_ID = $POD_ID

echo "Running /root/kafka-env/bin/python3 /root/kfpca/src/kafka/producer.py --poisson --brokers $POD_IP:9092 --topic test-topic --rate 100 --duration 600..."
/root/kafka-env/bin/python3 /root/kfpca/src/kafka/producer.py --poisson --brokers $POD_IP:9092 --topic test-topic --rate 100 --duration 6
