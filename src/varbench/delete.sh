
type="$1s"
tag=$2
subtype=$1
echo type = $type, tag = $tag
ns=kafka
if [ $# -gt 2 ]; then ns=$3; fi
echo ns = $ns

kubectl get $type -n $ns | grep $tag ; echo

for p in `kubectl get $type -n $ns | grep $tag | awk '{print $1}'`; do
    read -p "Delete $subtype=$p ? [y/n] " resp
    if [ $resp = "y" ]; then
	 echo deleting $p ...
	 kubectl patch $subtype $p -n $ns -p '{"metadata":{"finalizers":[]}}' --type=merge
	 kubectl delete $subtype $p -n $ns
	 #kubectl delete $subtype $p -n $ns --force --grace-period=0
    fi
done

echo Done
