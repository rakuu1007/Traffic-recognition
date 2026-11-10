from ultralytics import YOLO
import cv2

# Load vehicle detection model
model = YOLO("yolo11n.pt")

# Load test image
image = cv2.imread("traffic_test.jpg.jpg")

if image is None:
    print("Could not find traffic_test.jpg.jpg")
    exit()

# Detect objects
results = model(image, conf=0.40)

# Vehicle classes in COCO
vehicle_classes = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# Draw detected vehicles
for box in results[0].boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])

    if class_id in vehicle_classes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        label = f"{vehicle_classes[class_id]} {confidence:.2f}"

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            image,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        print(
            f"Detected: {vehicle_classes[class_id]} "
            f"| Confidence: {confidence:.2f}"
        )

# Save result
cv2.imwrite("vehicle_detection_result.jpg", image)

print("\nVehicle detection finished.")
print("Result saved as vehicle_detection_result.jpg")