#!/bin/sh


run_test() {
    if [ ${#1} -gt 0 ]; then method=$1; else method=savgol; fi
    if [ ${#2} -gt 0 ]; then epochs=$2; else epochs="5 10 30 50"; fi
    echo "method = $method, epochs = $epochs"

    for e in $epochs; do 
        echo "method = $method, epoch = $e"
	python ~/src/kfpca/src/grid_lstm.py -f trace.csv -x cpu -y cpu --$method -e $e > $method$e.txt
	#grep mean_squared_error $method$e.txt | awk '{print $11}' > $method$e.csv
	cp grid_results.pickle.csv $method$e.csv
	tail -4175 $method$e.csv > $method$((e+1)).csv
    done
    paste -d ',' ${method}5.csv ${method}11.csv ${method}51.csv ${method}101.csv > ${method}_all.csv
    echo "${method} tests done: ${method}_all.csv"
}


run_kalman() {
    for e in 50 100; do 
	python ~/src/kfpca/src/grid_lstm.py -f trace.csv -x cpu -y cpu --kalman -e $e > kalman$e.txt 
	grep accuracy kalman$e.txt | awk '{print $11}' > k$e.csv
	tail -4175 k$e.csv > k$((e+1)).csv
    done
    paste -d ',' k5.csv k11.csv k51.csv k101.csv > k_all.csv
    echo "kalman tests done: k_all.csv"
}

run_grid() {
    for e in 5 10 50 100; do 
	python ~/src/kfpca/src/grid_lstm.py -f trace.csv -x cpu -y cpu -e $e > grid$e.txt; 
	grep accuracy kalman$e.txt | awk '{print $11}' > k$e.csv
	tail -4175 k$e.csv > k$((e+1)).csv
    done
    paste -d ',' k5.csv k11.csv k51.csv k101.csv > k_all.csv
    echo "Grid tests done"
}


#run_test kalman
run_test $1 $2
