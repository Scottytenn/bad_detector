#!/usr/bin/env bash
set -euo pipefail

SSID="${1:-BadDetector}"
PASSWORD="${2:-badminton123}"
IFACE="${3:-wlan0}"

if [ "${#PASSWORD}" -lt 8 ] || [ "${#PASSWORD}" -gt 63 ]; then
  echo "Password must be 8-63 characters for WPA/WPA2."
  exit 1
fi

if ! command -v nmcli >/dev/null 2>&1; then
  echo "nmcli was not found. This script expects Raspberry Pi OS Bookworm/NetworkManager."
  exit 1
fi

sudo nmcli radio wifi on
sudo nmcli dev wifi hotspot ifname "$IFACE" ssid "$SSID" password "$PASSWORD"

echo "Hotspot started."
echo "SSID: $SSID"
echo "Password: $PASSWORD"
echo "Open from phone: http://10.42.0.1:5000"
