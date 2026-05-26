
./delete.sh pod prod

./zk_clear.sh

echo before:
kubectl get pods -n kafka | grep prod ; echo

kubectl apply -f kafka-deployment.yaml --validate=false --kubeconfig=/home/ubuntu/.kube/config

echo; echo after:
kubectl get pods -n kafka | grep prod ; echo

for p in `kubectl get pods -n kafka | grep prod | awk '{print $1}'`; do
    read -p "set POD=$p ? [y/n] " resp
    if [ $resp = "y" ]; then
	 echo $p > pod.txt; 
	 kubectl logs $p -n kafka > log.txt
	 echo Pod ID written to pod.txt, logs in log.txt
	 exit 0;
    fi
done
