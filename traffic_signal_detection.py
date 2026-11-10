import cv2

# Load traffic signal image
image = cv2.imread("traffic_signal_test.png")

if image is None:
    print("Could not find traffic_signal_test.png")
    exit()

# Convert image to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Detect red color
lower_red1 = (0, 100, 100)
upper_red1 = (10, 255, 255)

lower_red2 = (170, 100, 100)
upper_red2 = (180, 255, 255)

mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

red_mask = mask1 + mask2

# Find red regions
contours, _ = cv2.findContours(
    red_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

red_count = 0

for contour in contours:
    area = cv2.contourArea(contour)

    if area > 20:
        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        red_count += 1

print("Red regions detected:", red_count)

if red_count > 0:
    print("TRAFFIC SIGNAL: RED")
else:
    print("TRAFFIC SIGNAL: NOT RED")

cv2.imwrite(
    "traffic_signal_detection_result.png",
    image
)

print("Result saved as traffic_signal_detection_result.png")