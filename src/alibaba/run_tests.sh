
for e in 1 2 5 10; do
    echo Running Grid$e...
    python ~/src/kfpca/src/grid_lstm.py -f ~/src/alibaba/clusterdata/cluster-trace-microservices-v2021/data/Node/Node_0_15k.csv -x node_cpu_usage -y node_cpu_usage -e $e > grid${e}.txt 2>&1

    #continue

    echo Running Savgol$e...
    python ~/src/kfpca/src/grid_lstm.py -f ~/src/alibaba/clusterdata/cluster-trace-microservices-v2021/data/Node/Node_0_10k.csv -x node_cpu_usage -y node_cpu_usage -e $e --savgol > savgol${e}.txt 2>&1

    for t in AKF EKF; do
        echo Running $t$e...
        python ~/src/kfpca/src/grid_lstm.py -f ~/src/alibaba/clusterdata/cluster-trace-microservices-v2021/data/Node/Node_0_10k.csv -x node_cpu_usage -y node_cpu_usage -e $e --kalman -t ${t} > ${t}${e}.txt 2>&1
    done
done


echo Done
