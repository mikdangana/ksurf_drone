# Creating datasets & analysis using Google Cloud trace file(s)


## Download Google benchmark data file
```
mkdir data; cd data
wget https://storage.googleapis.com/external-traces/charlie/trace-1/<MEMRACE_FILE>.gz
gunzip <MEMTRACE_FILE>.gz
cd ..
```


## Install DynamoRIO release software & Update script path variables
```
wget https://github.com/DynamoRIO/dynamorio/releases/download/release_10.0.0/DynamoRIO-Linux-10.0.0.tar.gz
tar xvf DynamoRIO*.tar.gz
vim view_data.sh
```


## Parse the trace file binary to txt using drrun software
```
sh view_data.sh
```


## Converting trace txt to csv (compute csv & mem variables)
```
python conv_trace_to_txt.py > trace.txt
```


## Run prediction tests with Grid LSTM, Savitzky-Gorlay, EKF & AKF
```
sh run_tests.sh [grid|kalman|savgol]
```


## (Optional) Update 'kf-type=EKF|AKF|EKF-PCA|AKF-PCA' global variable 
##   defined in <KFPCA_ROOT>/src/ekf.py for more Kalman filter type tests
## <KFPCA_ROOT> refers to the clone path to https://github.com/mikdangana/kfpca
```
sed -i 'kf_type = "AKF"' <KFPCA_ROOT>/src/ekf.py
```


