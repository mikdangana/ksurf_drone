
for dir in data_kf_scaler data_hpa data_thresh_scaler; do
  for csv in `find $dir`; do
    if [ -f $csv ]; then
        sudo /root/kafka-env/bin/python3 -c "import sys, pandas as pd; print(sys.argv); df = pd.read_csv(sys.argv[1]); num_df = df.select_dtypes(include=['number']); print(num_df.mean())" $csv
    fi
  done
done
