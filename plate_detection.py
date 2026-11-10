from ultralytics import YOLO
import cv2

# Load plate detector
model = YOLO("best.pt")

# Read ORIGINAL image
image = cv2.imread("test_car.jpg.png")

if image is None:
    print("Could not find test_car.jpg.png")
    exit()

# Detect plate
results = model(image, conf=0.25)

for i, box in enumerate(results[0].boxes.xyxy):
    x1, y1, x2, y2 = map(int, box)

    # Small padding around plate
    padding = 5

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.shape[1], x2 + padding)
    y2 = min(image.shape[0], y2 + padding)

    # Crop directly from ORIGINAL image
    plate = image[y1:y2, x1:x2]

    cv2.imwrite("plate_original_crop.jpg", plate)

    print("Plate saved as plate_original_crop.jpg")
    print("Plate size:", plate.shape[1], "x", plate.shape[0])