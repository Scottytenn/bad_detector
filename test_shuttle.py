import argparse
import time
from collections import deque

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Badminton shuttle candidate tester")
    parser.add_argument("--camera", type=int, default=0, help="Camera index, usually 0")
    parser.add_argument("--width", type=int, default=640, help="Display/capture width")
    parser.add_argument("--height", type=int, default=480, help="Display/capture height")
    parser.add_argument("--fps", type=int, default=60, help="Requested camera FPS")
    parser.add_argument("--brightness-threshold", type=int, default=170, help="White/bright pixel threshold")
    parser.add_argument("--motion-threshold", type=int, default=22, help="Frame difference threshold")
    parser.add_argument("--min-area", type=float, default=3.0, help="Smallest candidate area")
    parser.add_argument("--max-area", type=float, default=500.0, help="Largest candidate area")
    parser.add_argument("--trail", type=int, default=20, help="Number of recent points to draw")
    parser.add_argument("--dshow", action="store_true", help="Use DirectShow backend on Windows")
    return parser.parse_args()


def open_camera(args):
    backend = cv2.CAP_DSHOW if args.dshow else 0
    cap = cv2.VideoCapture(args.camera, backend)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Check index and whether another app is using it.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    return cap


def to_gray(frame):
    if len(frame.shape) == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def build_mask(gray, prev_gray, args):
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bright_mask = cv2.threshold(
        blur,
        args.brightness_threshold,
        255,
        cv2.THRESH_BINARY,
    )

    if prev_gray is None:
        motion_mask = np.zeros_like(gray)
    else:
        diff = cv2.absdiff(gray, prev_gray)
        _, motion_mask = cv2.threshold(
            diff,
            args.motion_threshold,
            255,
            cv2.THRESH_BINARY,
        )

    candidate_mask = cv2.bitwise_and(bright_mask, motion_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, kernel)
    candidate_mask = cv2.dilate(candidate_mask, kernel, iterations=1)

    return bright_mask, motion_mask, candidate_mask


def find_candidates(candidate_mask, args):
    contours, _ = cv2.findContours(candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < args.min_area or area > args.max_area:
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


def draw_trail(frame, points):
    for i, point in enumerate(points):
        if point is None:
            continue
        radius = max(2, int(6 * (i + 1) / len(points)))
        cv2.circle(frame, point, radius, (0, 180, 255), -1)

    for start, end in zip(points, list(points)[1:]):
        if start is not None and end is not None:
            cv2.line(frame, start, end, (0, 180, 255), 2)


def main():
    args = parse_args()
    cap = open_camera(args)
    prev_gray = None
    trail = deque(maxlen=args.trail)
    frame_count = 0
    last_time = time.time()
    measured_fps = 0.0

    print("Press q to quit.")
    print("Press [ / ] to lower/raise brightness threshold.")
    print("Press - / = to lower/raise motion threshold.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.resize(frame, (args.width, args.height))
        gray = to_gray(frame)
        bright_mask, motion_mask, candidate_mask = build_mask(gray, prev_gray, args)
        candidates = find_candidates(candidate_mask, args)

        best_point = None
        for index, (area, x, y, w, h, cx, cy) in enumerate(candidates[:8]):
            color = (0, 255, 255) if index == 0 else (255, 180, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
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

        trail.append(best_point)
        draw_trail(frame, trail)

        frame_count += 1
        now = time.time()
        if now - last_time >= 1.0:
            measured_fps = frame_count / (now - last_time)
            frame_count = 0
            last_time = now

        cv2.putText(
            frame,
            f"FPS {measured_fps:.1f}  bright>{args.brightness_threshold}  motion>{args.motion_threshold}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "yellow = best moving bright small target",
            (10, args.height - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("shuttle test - original", frame)
        cv2.imshow("candidate mask - bright AND moving", candidate_mask)
        cv2.imshow("motion mask", motion_mask)

        prev_gray = gray.copy()

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("["):
            args.brightness_threshold = max(0, args.brightness_threshold - 5)
        elif key == ord("]"):
            args.brightness_threshold = min(255, args.brightness_threshold + 5)
        elif key == ord("-"):
            args.motion_threshold = max(0, args.motion_threshold - 2)
        elif key == ord("="):
            args.motion_threshold = min(255, args.motion_threshold + 2)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
