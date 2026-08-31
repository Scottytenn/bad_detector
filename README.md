# Bad Detector

Simple Raspberry Pi camera stream with early shuttle candidate detection.

## 1. Copy To Pi

From Windows PowerShell:

```powershell
scp -r D:\Project\bad_detector scotty@<pi-ip>:~/
```

If replacing an old copy:

```powershell
ssh scotty@<pi-ip> "rm -rf ~/bad_detector"
scp -r D:\Project\bad_detector scotty@<pi-ip>:~/
```

## 2. Install On Pi

```bash
cd ~/bad_detector
sudo apt update
sudo apt install -y python3-opencv python3-flask v4l-utils
```

## 3. Check Camera

```bash
v4l2-ctl --list-devices
v4l2-ctl --device=/dev/video0 --list-formats-ext
```

## 4. Run Stream

```bash
python3 stream_server.py --device /dev/video0 --width 640 --height 480 --fps 120 --fourcc MJPG --jpeg-quality 50
```

Open from phone/laptop:

```text
http://<pi-ip>:5000
```

Debug streams:

```text
http://<pi-ip>:5000/video_feed/annotated
http://<pi-ip>:5000/video_feed/candidate
http://<pi-ip>:5000/video_feed/motion
```

## 5. Tune Thresholds

The web page has sliders for:

```text
Brightness  higher = only very white objects
Motion      higher = only faster/bigger changes
Min area    higher = ignore tiny noise
Max area    lower = ignore people/hands/large reflections
```

Startup defaults can also be changed:

```bash
python3 stream_server.py --device /dev/video0 --fourcc MJPG --brightness-threshold 170 --motion-threshold 22 --min-area 3 --max-area 500
```

## 6. Optional Pi Hotspot

This may disconnect Wi-Fi SSH. Use Ethernet SSH while testing hotspot mode.

```bash
chmod +x setup_hotspot.sh
./setup_hotspot.sh BadDetector badminton123
```

Phone connects to:

```text
SSID: BadDetector
Password: badminton123
```

Then open:

```text
http://10.42.0.1:5000
```

## 7. Autostart On Boot

This assumes:

```text
Project: /home/scotty/bad_detector
Venv:    /home/scotty/bad_detector/venv
User:    scotty
```

Create the venv if needed:

```bash
cd ~/bad_detector
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Install and start the service:

```bash
chmod +x install_service.sh
./install_service.sh
```

Useful commands:

```bash
sudo systemctl status bad-detector
sudo systemctl restart bad-detector
sudo systemctl stop bad-detector
journalctl -u bad-detector -f
```

## Local Laptop Test

```powershell
python .\test_shuttle.py --camera 0 --width 640 --height 480 --fps 60 --dshow
```

Keyboard controls:

```text
q      quit
[ / ]  lower / raise brightness threshold
- / =  lower / raise motion threshold
```
