import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("无法打开摄像头，请检查设备是否被其他程序占用")

frame_count = 0
skip_frames = 3

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % skip_frames != 0:
        continue

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500 or area > 30000:
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
        if avg_intensity < 180:
            continue

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(
            frame,
            "又方又亮的东西",
            (x, max(0, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        frame,
        time.strftime("%H:%M:%S"),
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow('实时画面', frame)
    cv2.imshow('白色方形掩膜', mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()