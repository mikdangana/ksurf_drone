
tag=monitor
if [ $# -gt 0 ]; then tag=$1; fi

echo Stopping these processes: 
ps aux | grep $tag; echo

for pid in `ps aux | grep $tag | awk '{print $2}'`; do 
    echo killing pid = $pid
    kill -9 $pid
done

echo done
