
for x in `ps aux | grep "start k3s-agent" | awk '{print $2}'`; do
    kill -9 $x;
    echo killed $x;
done
