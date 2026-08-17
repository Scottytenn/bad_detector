# Bad Detector Camera Stream

Minimal Raspberry Pi camera streaming server for early badminton testing.

## Install on Raspberry Pi

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv v4l-utils
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Check camera

```bash
v4l2-ctl --list-devices
v4l2-ctl --device=/dev/video0 --list-formats-ext
```

## Run

```bash
source .venv/bin/activate
python stream_server.py --device /dev/video0 --width 640 --height 480 --fps 30
```

Then open this from a phone or laptop on the same network:

```text
http://<pi-ip>:5000
```

For example:

```text
http://192.168.137.42:5000
```

## Notes

Keep the preview modest at first. The detector can later process higher FPS internally while the phone receives a lower-FPS preview.
