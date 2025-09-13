import cv2

cap = cv2.VideoCapture(0)
cv2.namedWindow("Webcam Test", cv2.WINDOW_NORMAL)

print("📷 Press 'q' to quit webcam test...")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("⚠️ Failed to read frame")
        continue

    cv2.imshow("Webcam Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
