To add a worker node, run the following command on the worker node:
curl -sfL https://get.k3s.io | K3S_URL=https://<MASTER_IP>:6443 K3S_TOKEN=<K3S_TOKEN> sh -s - --with-node-id
