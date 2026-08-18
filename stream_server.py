import argparse
import os
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, render_template_string


PAGE = """
<!doctype html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Bad Detector Camera</title>
  <style>
    body { margin: 0; font-family: system-ui, sans-serif; background: #101214; color: #f4f4f4; }
    header { padding: 14px 18px; background: #171a1f; border-bottom: 1px solid #2a2f37; }
    h1 { margin: 0; font-size: 18px; }
    main { padding: 16px; }
    img { width: 100%; max-width: 960px; background: #000; border: 1px solid #333; }
    .status { margin-top: 10px; color: #b8c0cc; font-size: 14px; }
  </style>
</head>
<body>
  <header><h1>Bad Detector Camera Stream</h1></header>
  <main>
    <img src="/video_feed" alt="camera stream">
    <div class="status">Stream: /video_feed</div>
  </main>
</body>
</html>
"""


class Camera:
    def __init__(self, device, width, height, fps):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.lock = threading.Lock()
        self.latest_frame = None
        self.running = False
        self.capture = None
        self.bright_threshold = 180
        self.min_area = 500
        self.max_area = 30000

    def start(self):
        device_value = self.device
        if isinstance(device_value, str) and device_value.isdigit():
            device_value = int(device_value)

        if os.name == "nt" and isinstance(device_value, int):
            self.capture = cv2.VideoCapture(device_value, cv2.CAP_DSHOW)
        else:
            self.capture = cv2.VideoCapture(device_value)

        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open camera device: {self.device}")

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)

        self.running = True
        thread = threading.Thread(target=self._capture_loop, daemon=True)
        thread.start()

    def _process_frame(self, frame):
        processed = frame.copy()
        processed = cv2.resize(processed, (self.width, self.height))

        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blur, self.bright_threshold, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue

            aspect_ratio = w / float(h)
            if not (0.7 < aspect_ratio < 1.5):
                continue

            roi = gray[y:y + h, x:x + w]
            if roi.size == 0:
                continue

            avg_intensity = float(np.mean(roi))
            if avg_intensity < self.bright_threshold:
                continue

            cv2.rectangle(processed, (x, y), (x + w, y + h), (0, 255, 255), 2)
            label = "又方又亮的东西"
            cv2.putText(
                processed,
                label,
                (x, max(0, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        timestamp = time.strftime("%H:%M:%S")
        cv2.putText(
            processed,
            timestamp,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        return processed

    def _capture_loop(self):
        while self.running:
            ok, frame = self.capture.read()
            if not ok:
                time.sleep(0.05)
                continue

            processed = self._process_frame(frame)

            with self.lock:
                self.latest_frame = processed

    def jpeg_frames(self, jpeg_quality):
        while True:
            with self.lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()

            if frame is None:
                time.sleep(0.05)
                continue

            ok, buffer = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
            )
            if not ok:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )

    def stop(self):
        self.running = False
        if self.capture is not None:
            self.capture.release()


def create_app(camera, jpeg_quality):
    app = Flask(__name__)

    @app.get("/")
    def index():
        return render_template_string(PAGE)

    @app.get("/video_feed")
    def video_feed():
        return Response(
            camera.jpeg_frames(jpeg_quality),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/health")
    def health():
        return {"ok": True}

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="Local MJPEG camera stream server")
    parser.add_argument("--device", default="0", help="Camera device index or path (default: 0 for Windows/local webcam)")
    parser.add_argument("--host", default="0.0.0.0", help="Server bind address")
    parser.add_argument("--port", default=5000, type=int, help="Server port")
    parser.add_argument("--width", default=640, type=int, help="Capture width")
    parser.add_argument("--height", default=480, type=int, help="Capture height")
    parser.add_argument("--fps", default=30, type=int, help="Requested camera FPS")
    parser.add_argument("--jpeg-quality", default=80, type=int, help="JPEG quality 1-100")
    return parser.parse_args()


def main():
    args = parse_args()
    camera = Camera(args.device, args.width, args.height, args.fps)
    camera.start()
    app = create_app(camera, args.jpeg_quality)

    try:
        app.run(host=args.host, port=args.port, threaded=True)
    finally:
        camera.stop()


if __name__ == "__main__":
    main()
