#!/bin/bash

set -e

PROM_VERSION="2.48.1"
PROM_USER="prometheus"
PROM_DIR="/etc/prometheus"
PROM_DATA_DIR="/var/lib/prometheus"
PROM_BIN_DIR="/usr/local/bin"
PROM_SERVICE="/etc/systemd/system/prometheus.service"
TMP_DIR="/tmp/prometheus_install"

echo "==> Creating Prometheus user"
useradd --no-create-home --shell /bin/false $PROM_USER || true

echo "==> Creating directories"
mkdir -p $PROM_DIR $PROM_DATA_DIR $TMP_DIR

echo "==> Downloading Prometheus $PROM_VERSION"
cd $TMP_DIR
curl -sLO "https://github.com/prometheus/prometheus/releases/download/v${PROM_VERSION}/prometheus-${PROM_VERSION}.linux-amd64.tar.gz"
tar -xzf prometheus-${PROM_VERSION}.linux-amd64.tar.gz
cd prometheus-${PROM_VERSION}.linux-amd64

echo "==> Installing binaries"
cp prometheus promtool $PROM_BIN_DIR
cp -r consoles console_libraries $PROM_DIR
cp prometheus.yml $PROM_DIR

echo "==> Setting permissions"
chown -R $PROM_USER:$PROM_USER $PROM_DIR $PROM_DATA_DIR
chown $PROM_USER:$PROM_USER $PROM_BIN_DIR/prometheus $PROM_BIN_DIR/promtool

echo "==> Creating systemd service"
cat <<EOF > $PROM_SERVICE
[Unit]
Description=Prometheus Monitoring
Wants=network-online.target
After=network-online.target

[Service]
User=$PROM_USER
Group=$PROM_USER
Type=simple
ExecStart=$PROM_BIN_DIR/prometheus \\
  --config.file=$PROM_DIR/prometheus.yml \\
  --storage.tsdb.path=$PROM_DATA_DIR \\
  --web.console.templates=$PROM_DIR/consoles \\
  --web.console.libraries=$PROM_DIR/console_libraries

[Install]
WantedBy=multi-user.target
EOF

echo "==> Reloading systemd and starting Prometheus"
systemctl daemon-reexec
systemctl daemon-reload
systemctl enable prometheus
systemctl start prometheus

echo "==> Cleanup"
rm -rf $TMP_DIR

echo "✅ Prometheus installed and running on port 9090"

