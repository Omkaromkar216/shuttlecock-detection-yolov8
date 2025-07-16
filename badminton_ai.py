import cv2
from ultralytics import YOLO

# Load your trained model
model = YOLO("best.pt")  # Make sure best.pt is in same folder

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.25)
    annotated = results[0].plot()

    cv2.imshow("Shuttle Live Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
