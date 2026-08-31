import argparse
import os
import threading
import time
from collections import deque

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request


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
    a { color: #88c0ff; }
    .panel { width: 100%; max-width: 960px; margin-top: 12px; }
    .row { display: grid; grid-template-columns: 130px 1fr 54px; gap: 10px; align-items: center; margin: 10px 0; }
    input { width: 100%; }
    button { padding: 9px 12px; border: 0; border-radius: 6px; background: #2e7dd7; color: white; font-weight: 700; }
    .status { margin-top: 10px; color: #b8c0cc; font-size: 14px; line-height: 1.5; }
  </style>
</head>
<body>
  <header><h1>Bad Detector Camera Stream</h1></header>
  <main>
    <img src="/video_feed/annotated" alt="camera stream">
    <section class="panel">
      <div class="row">
        <label for="brightness_threshold">Brightness</label>
        <input id="brightness_threshold" type="range" min="0" max="255" step="1">
        <span id="brightness_value"></span>
      </div>
      <div class="row">
        <label for="motion_threshold">Motion</label>
        <input id="motion_threshold" type="range" min="0" max="255" step="1">
        <span id="motion_value"></span>
      </div>
      <div class="row">
        <label for="min_area">Min area</label>
        <input id="min_area" type="range" min="1" max="200" step="1">
        <span id="min_area_value"></span>
      </div>
      <div class="row">
        <label for="max_area">Max area</label>
        <input id="max_area" type="range" min="20" max="3000" step="10">
        <span id="max_area_value"></span>
      </div>
      <button id="reset">Reset test values</button>
    </section>
    <div class="status">
      Main stream: /video_feed/annotated<br>
      Debug streams:
      <a href="/video_feed/candidate">candidate mask</a> |
      <a href="/video_feed/motion">motion mask</a>
    </div>
  </main>
  <script>
    const ids = ["brightness_threshold", "motion_threshold", "min_area", "max_area"];

    function labelId(id) {
      return id === "brightness_threshold" ? "brightness_value" :
        id === "motion_threshold" ? "motion_value" :
        id === "min_area" ? "min_area_value" : "max_area_value";
    }

    async function loadSettings() {
      const response = await fetch("/settings");
      const settings = await response.json();
      ids.forEach((id) => {
        const input = document.getElementById(id);
        input.value = settings[id];
        document.getElementById(labelId(id)).textContent = settings[id];
      });
    }

    async function saveSetting(id, value) {
      document.getElementById(labelId(id)).textContent = value;
      await fetch("/settings", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({[id]: Number(value)})
      });
    }

    ids.forEach((id) => {
      const input = document.getElementById(id);
      input.addEventListener("input", () => saveSetting(id, input.value));
    });

    document.getElementById("reset").addEventListener("click", async () => {
      await fetch("/settings", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          brightness_threshold: 170,
          motion_threshold: 22,
          min_area: 3,
          max_area: 500
        })
      });
      await loadSettings();
    });

    loadSettings();
  </script>
</body>
</html>
"""


class ShuttleDetector:
    def __init__(self, brightness_threshold, motion_threshold, min_area, max_area, trail_length):
        self.brightness_threshold = brightness_threshold
        self.motion_threshold = motion_threshold
        self.min_area = min_area
        self.max_area = max_area
        self.prev_gray = None
        self.trail = deque(maxlen=trail_length)
        self.frame_count = 0
        self.last_time = time.time()
        self.measured_fps = 0.0
        self.settings_lock = threading.RLock()

    @staticmethod
    def to_gray(frame):
        if len(frame.shape) == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def build_mask(self, gray):
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        settings = self.get_settings()
        _, bright_mask = cv2.threshold(
            blur,
            settings["brightness_threshold"],
            255,
            cv2.THRESH_BINARY,
        )

        if self.prev_gray is None:
            motion_mask = np.zeros_like(gray)
        else:
            diff = cv2.absdiff(gray, self.prev_gray)
            _, motion_mask = cv2.threshold(
                diff,
                settings["motion_threshold"],
                255,
                cv2.THRESH_BINARY,
            )

        candidate_mask = cv2.bitwise_and(bright_mask, motion_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, kernel)
        candidate_mask = cv2.dilate(candidate_mask, kernel, iterations=1)
        return motion_mask, candidate_mask

    def find_candidates(self, candidate_mask):
        settings = self.get_settings()
        contours, _ = cv2.findContours(candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < settings["min_area"] or area > settings["max_area"]:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue

            aspect = w / float(h)
            if aspect < 0.2 or aspect > 5.0:
                continue

            cx = x + w // 2
            cy = y + h // 2
            candidates.append((area, x, y, w, h, cx, cy))

        candidates.sort(reverse=True, key=lambda item: item[0])
        return candidates

    def draw_trail(self, frame):
        points = list(self.trail)
        if not points:
            return

        for index, point in enumerate(points):
            if point is None:
                continue
            radius = max(2, int(6 * (index + 1) / len(points)))
            cv2.circle(frame, point, radius, (0, 180, 255), -1)

        for start, end in zip(points, points[1:]):
            if start is not None and end is not None:
                cv2.line(frame, start, end, (0, 180, 255), 2)

    def process(self, frame):
        settings = self.get_settings()
        gray = self.to_gray(frame)
        motion_mask, candidate_mask = self.build_mask(gray)
        candidates = self.find_candidates(candidate_mask)

        annotated = frame.copy()
        if len(annotated.shape) == 2:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

        best_point = None
        for index, (area, x, y, w, h, cx, cy) in enumerate(candidates[:8]):
            color = (0, 255, 255) if index == 0 else (255, 180, 0)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                annotated,
                f"{area:.0f}",
                (x, max(14, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
            if index == 0:
                best_point = (cx, cy)

        self.trail.append(best_point)
        self.draw_trail(annotated)

        self.frame_count += 1
        now = time.time()
        if now - self.last_time >= 1.0:
            self.measured_fps = self.frame_count / (now - self.last_time)
            self.frame_count = 0
            self.last_time = now

        cv2.putText(
            annotated,
            f"FPS {self.measured_fps:.1f}  bright>{settings['brightness_threshold']}  motion>{settings['motion_threshold']}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            "yellow = best moving bright small target",
            (10, annotated.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        self.prev_gray = gray.copy()
        return {
            "annotated": annotated,
            "candidate": cv2.cvtColor(candidate_mask, cv2.COLOR_GRAY2BGR),
            "motion": cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR),
        }

    def get_settings(self):
        with self.settings_lock:
            return {
                "brightness_threshold": int(self.brightness_threshold),
                "motion_threshold": int(self.motion_threshold),
                "min_area": float(self.min_area),
                "max_area": float(self.max_area),
            }

    def update_settings(self, updates):
        allowed = {
            "brightness_threshold": (0, 255, int),
            "motion_threshold": (0, 255, int),
            "min_area": (1, 10000, float),
            "max_area": (1, 100000, float),
        }
        with self.settings_lock:
            for key, value in updates.items():
                if key not in allowed:
                    continue
                low, high, caster = allowed[key]
                value = caster(value)
                value = max(low, min(high, value))
                setattr(self, key, value)
            if self.min_area > self.max_area:
                self.min_area = self.max_area
            return self.get_settings()


class Camera:
    def __init__(
        self,
        device,
        width,
        height,
        fps,
        fourcc,
        brightness_threshold,
        motion_threshold,
        min_area,
        max_area,
        trail_length,
    ):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = fourcc
        self.lock = threading.Lock()
        self.latest_frames = {}
        self.running = False
        self.capture = None
        self.actual_width = 0
        self.actual_height = 0
        self.actual_fps = 0.0
        self.actual_fourcc = ""
        self.detector = ShuttleDetector(
            brightness_threshold,
            motion_threshold,
            min_area,
            max_area,
            trail_length,
        )

    def start(self):
        device_value = self.device
        if isinstance(device_value, str) and device_value.isdigit():
            device_value = int(device_value)

        if os.name == "nt" and isinstance(device_value, int):
            self.capture = cv2.VideoCapture(device_value, cv2.CAP_DSHOW)
        else:
            self.capture = cv2.VideoCapture(device_value, cv2.CAP_V4L2)

        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open camera device: {self.device}")

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fourcc:
            self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        fourcc_value = int(self.capture.get(cv2.CAP_PROP_FOURCC))
        self.actual_fourcc = "".join(chr((fourcc_value >> 8 * i) & 0xFF) for i in range(4))
        print(
            "Camera opened:",
            f"{self.actual_width}x{self.actual_height}",
            f"{self.actual_fps:.1f}fps",
            self.actual_fourcc,
            flush=True,
        )

        self.running = True
        thread = threading.Thread(target=self._capture_loop, daemon=True)
        thread.start()

    def _capture_loop(self):
        while self.running:
            ok, frame = self.capture.read()
            if not ok:
                time.sleep(0.05)
                continue

            frame = cv2.resize(frame, (self.width, self.height))
            processed_frames = self.detector.process(frame)
            cv2.putText(
                processed_frames["annotated"],
                f"cam {self.actual_width}x{self.actual_height} {self.actual_fps:.1f}fps {self.actual_fourcc}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

            with self.lock:
                self.latest_frames = processed_frames

    def jpeg_frames(self, view, jpeg_quality):
        while True:
            with self.lock:
                latest_frame = self.latest_frames.get(view)
                frame = None if latest_frame is None else latest_frame.copy()

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
    @app.get("/video_feed/<view>")
    def video_feed(view="annotated"):
        if view not in {"annotated", "candidate", "motion"}:
            view = "annotated"
        return Response(
            camera.jpeg_frames(view, jpeg_quality),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/settings")
    def get_settings():
        return jsonify(camera.detector.get_settings())

    @app.post("/settings")
    def update_settings():
        updates = request.get_json(silent=True) or {}
        return jsonify(camera.detector.update_settings(updates))

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="Local MJPEG camera stream server")
    parser.add_argument("--device", default="0", help="Camera device index or path (default: 0 for Windows/local webcam)")
    parser.add_argument("--host", default="0.0.0.0", help="Server bind address")
    parser.add_argument("--port", default=5000, type=int, help="Server port")
    parser.add_argument("--width", default=640, type=int, help="Capture width")
    parser.add_argument("--height", default=480, type=int, help="Capture height")
    parser.add_argument("--fps", default=30, type=int, help="Requested camera FPS")
    parser.add_argument("--fourcc", default="MJPG", help="Requested camera format, usually MJPG or YUYV")
    parser.add_argument("--jpeg-quality", default=80, type=int, help="JPEG quality 1-100")
    parser.add_argument("--brightness-threshold", default=170, type=int, help="White/bright pixel threshold")
    parser.add_argument("--motion-threshold", default=22, type=int, help="Frame difference threshold")
    parser.add_argument("--min-area", default=3.0, type=float, help="Smallest candidate area")
    parser.add_argument("--max-area", default=500.0, type=float, help="Largest candidate area")
    parser.add_argument("--trail", default=20, type=int, help="Number of recent points to draw")
    return parser.parse_args()


def main():
    args = parse_args()
    camera = Camera(
        args.device,
        args.width,
        args.height,
        args.fps,
        args.fourcc,
        args.brightness_threshold,
        args.motion_threshold,
        args.min_area,
        args.max_area,
        args.trail,
    )
    camera.start()
    app = create_app(camera, args.jpeg_quality)

    try:
        app.run(host=args.host, port=args.port, threaded=True)
    finally:
        camera.stop()


if __name__ == "__main__":
    main()
