import cv2

cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

desired_width = 1920
desired_height = 1080
desired_fps = 60

cap.set(cv2.CAP_PROP_FPS, desired_fps)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)


actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
actual_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Resolution set to: {actual_width}x{actual_height}")
print(f"FPS set to: {actual_fps}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
