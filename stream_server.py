import argparse
import threading
import time

import cv2
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

    def start(self):
        self.capture = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open camera device: {self.device}")

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)

        self.running = True
        thread = threading.Thread(target=self._capture_loop, daemon=True)
        thread.start()

    def _capture_loop(self):
        while self.running:
            ok, frame = self.capture.read()
            if not ok:
                time.sleep(0.05)
                continue

            timestamp = time.strftime("%H:%M:%S")
            cv2.putText(
                frame,
                timestamp,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            with self.lock:
                self.latest_frame = frame

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
    parser.add_argument("--device", default="/dev/video0", help="Camera device path")
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
