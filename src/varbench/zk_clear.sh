ZPOD=`kubectl get pods -n kafka | grep zoo | grep Running | awk '{print $1}'`
echo Clearing broker 1 from ZPOD = $ZPOD

kubectl exec -it $ZPOD -n kafka -- bin/zkCli.sh delete /brokers/ids/1
