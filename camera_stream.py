import atexit
import os
import threading
import time

import cv2
from flask import Flask, Response, render_template_string

app = Flask(__name__)

camera = None
camera_lock = threading.Lock()


def open_camera(camera_index=0):
    """Open a USB camera using V4L2 if available."""
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)

    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at index {camera_index}. Check /dev/video0 or the USB camera connection.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def get_camera():
    global camera
    if camera is None:
        camera_index = int(os.environ.get("CAMERA_INDEX", "0"))
        camera = open_camera(camera_index)
    return camera


def generate_frames():
    cap = get_camera()

    while True:
        with camera_lock:
            success, frame = cap.read()

        if not success:
            time.sleep(0.1)
            continue

        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ret, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
        )
        time.sleep(0.03)


@app.route("/")
def index():
    return render_template_string(
        """
        <!doctype html>
        <html>
        <head>
            <title>USB Camera Stream</title>
            <style>
                body {
                    margin: 0;
                    background: #111;
                    color: white;
                    font-family: Arial, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    min-height: 100vh;
                }
                .panel {
                    text-align: center;
                    background: #1a1a1a;
                    padding: 20px;
                    border-radius: 16px;
                    box-shadow: 0 10px 25px rgba(0,0,0,0.35);
                }
                img {
                    width: min(90vw, 960px);
                    background: black;
                    border-radius: 10px;
                    border: 2px solid #444;
                }
            </style>
        </head>
        <body>
            <div class="panel">
                <h2>Raspberry Pi Camera</h2>
                <img src="{{ url_for('video_feed') }}" alt="Live camera stream" />
            </div>
        </body>
        </html>
        """
    )


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@atexit.register
def cleanup_camera():
    global camera
    if camera is not None:
        camera.release()


if __name__ == "__main__":
    try:
        get_camera()
    except RuntimeError as exc:
        print(exc)
        raise SystemExit(1)

    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)
