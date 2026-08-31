#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/scotty/bad_detector"
SERVICE_NAME="bad-detector.service"

if [ ! -x "$PROJECT_DIR/venv/bin/python" ]; then
  echo "Could not find $PROJECT_DIR/venv/bin/python"
  echo "Create the virtual environment first:"
  echo "  cd $PROJECT_DIR"
  echo "  python3 -m venv venv"
  echo "  source venv/bin/activate"
  echo "  pip install -r requirements.txt"
  exit 1
fi

sudo cp "$PROJECT_DIR/$SERVICE_NAME" "/etc/systemd/system/$SERVICE_NAME"
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
sudo systemctl restart "$SERVICE_NAME"

echo "Service installed and started."
echo "Status:"
sudo systemctl --no-pager status "$SERVICE_NAME"
