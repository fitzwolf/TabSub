import cv2
import numpy as np

def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return threshold

def find_handwriting_contour(threshold):
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    max_contour = max(contours, key=cv2.contourArea)
    return max_contour

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    threshold = preprocess(frame)
    contour = find_handwriting_contour(threshold)

    if contour is not None:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Webcam", frame)

    key = cv2.waitKey(1)
    if key == ord("c"):
        if contour is not None:
            handwriting = threshold[y : y + h, x : x + w]
            cv2.imwrite("handwriting.png", handwriting)
            print("Handwriting captured and saved.")
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()