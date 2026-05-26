
if [ ! -e logs ]; then mkdir logs; fi
if [ ! -e data ]; then mkdir data; fi

for i in $(seq 1 1 20); do
    BROKER=worker$i
    nohup sh monitor_consumer.sh $BROKER > logs/consumer_$BROKER.log 2>&1 &
    echo launched consumer monitor: logs/consumer_$BROKER.log...
done

#nohup sh monitor_metrics.sh 2>&1 > metrics.log &
#echo launched metrics monitor: metrics.log...
nohup python3 monitor_pods.py > logs/pod_counts.log 2>&1 &
echo launched pod metrics monitor...

echo launching node metrics monitor...
python3 monitor_nodes.py
    
