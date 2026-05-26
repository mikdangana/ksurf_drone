
LOG=tests.log
echo "" > $LOG
NUM_WORKERS=5
NUM_PRODUCERS=2


run_test() {
  TEST_TYPE="kf"; if [ $# -gt 0 ]; then TEST_TYPE=$1; fi # hpa|kf|th
  THRESHOLD=5; if [ $# -gt 1 ]; then THRESHOLD=$2; fi # cpu threshold
  echo test_type = $TEST_TYPE threshold = $THRESHOLD args = $# 

  for w in $(seq 1 1 $NUM_WORKERS); do
    hpa=kafka-worker${w}-hpa
    host=ksurf-worker${w}
    echo hpa = $hpa, host = $host
    kubectl scale deployment ${host} -n kube-system --replicas=1
    kubectl rollout restart deployment ${host} -n kube-system
    sleep 60
    kubectl delete hpa ${hpa} -n kube-system
    suffix=""
    if [ $TEST_TYPE = "na" ]; then continue; fi
    if [ $TEST_TYPE = "kf" ]; then suffix=_kf; fi
    echo TYPE = $TEST_TYPE, THESHOLD = $THRESHOLD, hpa = $hpa, host = $host, suffix = $suffix >> $LOG
    if [ $TEST_TYPE = "hpa" ]; then
        sshpass -p "<VM_PASSWORD>" ssh ubuntu@$host "echo '<VM_PASSWORD>' | sudo sed -i 's/averageUtilization:.*/averageUtilization: ${THRESHOLD} # CPU utilization/' /root/hpa.yaml"
        sshpass -p "<VM_PASSWORD>" ssh ubuntu@$host "echo '<VM_PASSWORD>' | sudo kubectl apply -f /root/hpa.yaml --kubeconfig=/home/ubuntu/.kube/config"
	kubectl describe hpa ${hpa} -n kube-system >> $LOG
	kubectl describe hpa ${hpa} -n kube-system 
    else
        #sshpass -p "<VM_PASSWORD>" ssh ubuntu@ksurf-scaler "sudo /root/kafka-env/bin/python3 /root/kfpca/src/k3s/autoscaler${suffix}.py -d kafka-worker${w} -cpu $THRESHOLD "
        sshpass -p "<VM_PASSWORD>" ssh ubuntu@ksurf-scaler "sudo nohup /root/kafka-env/bin/python3 /root/kfpca/src/k3s/autoscaler${suffix}.py -d kafka-worker${w} -cpu $THRESHOLD > /home/ubuntu/logs/autoscaler${suffix}_${w}.log 2>&1 &"
        #nohup sshpass -p "<VM_PASSWORD>" ssh ubuntu@ksurf-scaler "echo '<VM_PASSWORD>' | sudo /root/kafka-env/bin/python3 /root/kfpca/src/k3s/autoscaler${suffix}.py -d kafka-worker${w} > /home/ubuntu/autoscaler${w}.log 2>&1" > auto${w}.log 2>&1 &
    fi
  done

  if [ ! -e logs ]; then mkdir /home/ubuntu/logs; fi

  for p in $(seq 1 1 $NUM_PRODUCERS); do 
    sshpass -p "<VM_PASSWORD>" ssh ubuntu@ksurf-producer${p} "sudo nohup sh /root/kafka_producer_workload.sh > /home/ubuntu/logs/workload.log 2>&1 &"
  done

  sh run_monitors.sh $NUM_WORKERS $TEST_TYPE $THRESHOLD

  sh stop_tests.sh

  cp -r data/ data_${TEST_TYPE}_${THRESHOLD}
  cp -r logs/ logs_${TEST_TYPE}_${THRESHOLD}
}


run_tests() {
  for threshold in 0.5 1 2 5; do
    for test_type in hpa kf th na; do
	echo Pre-test rest 180s...
	sleep 180
	run_test $test_type $threshold
    done
  done

  rm -fr data/
  tar cvf data.tar data_*
  sh summarize.sh > summary.txt
  echo Results in summary.txt
}


run_tests
