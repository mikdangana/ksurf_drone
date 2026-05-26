
csv=data/node_metrics.csv
if [ $# -gt 0 ]; then csv=$1; fi

col=6
if [ $# -gt 1 ]; then col=$2; fi

cname=`head -1 $csv | awk -F "," '{print $'$col'}' | tr '\n' ' ' | tr '\r' ' '`

sum=0
len=`wc -l $csv | awk '{print $1}'`

len=$(( len-1 ))

for cpu in `tail -${len} $csv | awk -F "," '{print $'$col'}' | tr '\n' ' ' | tr '\r' ' '`; do 
    #echo cpu = $cpu, sum = $sum
    sum=$(( sum + cpu ))
done

echo csv=$csv, col=$cname, sum=$sum, mean=$(( sum / len )), len=$len
