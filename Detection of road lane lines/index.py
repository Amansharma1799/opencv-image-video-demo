import cv2
import numpy as np
import os

# ---- Folder Path ----
folder_path = r"D:\Aman\Desktop\Detection of road lane lines"

# ---- Image File Name  ----
image_file = "pexels-sebastian-palomino-933481-1955134.jpg"

# ---- Video File Name  ----
video_file = "test_video.mp4"

# ================= IMAGE LANE DETECTION =================
image_path = os.path.join(folder_path, image_file)
if os.path.exists(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    mask = np.zeros_like(edges)
    height = image.shape[0]
    polygon = np.array([[
        (0, height),
        (image.shape[1], height),
        (image.shape[1] // 2, height // 2)
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)

    masked_edges = cv2.bitwise_and(edges, mask)
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=150)

    line_image = np.copy(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

    cv2.imshow("Lane Detection - Image", line_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print(f"[!] Image file not found: {image_path}")

# ================= VIDEO LANE DETECTION =================
video_path = os.path.join(folder_path, video_file)
if os.path.exists(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        mask = np.zeros_like(edges)
        height = frame.shape[0]
        polygon = np.array([[
            (0, height),
            (frame.shape[1], height),
            (frame.shape[1] // 2, height // 2)
        ]], np.int32)
        cv2.fillPoly(mask, polygon, 255)

        masked_edges = cv2.bitwise_and(edges, mask)
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=150)

        line_image = np.copy(frame)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

        cv2.imshow("Lane Detection - Video", line_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
else:
    print(f"[!] Video file not found: {video_path}")
