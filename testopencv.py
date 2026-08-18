import cv2
import numpy as np

cap = cv2.VideoCapture(0)
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
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 大部分是紫色，但允许摄像头/背景带一点其他颜色
    lower_purple = np.array([120, 35, 35])
    upper_purple = np.array([170, 255, 255])
    mask = cv2.inRange(hsv, lower_purple, upper_purple)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 1000:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue

        aspect_ratio = w / float(h)
        if not (1.2 < aspect_ratio < 4.0):
            continue

        if not (2000 < area < 30000):
            continue

        roi = mask[y:y + h, x:x + w]
        purple_pixels = cv2.countNonZero(roi)
        if purple_pixels == 0:
            continue

        purple_ratio = purple_pixels / float(w * h)
        if purple_ratio < 0.15:
            continue

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
        print(f"检测到主要为紫色的长方体，位置：({x}, {y})，宽高：({w}, {h})，紫色占比：{purple_ratio:.2f}")

    cv2.imshow('实时画面', frame)
    cv2.imshow('紫色掩膜', mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()