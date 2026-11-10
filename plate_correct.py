import cv2

image = cv2.imread("plate_original_crop.jpg")

if image is None:
    print("Could not find plate_original_crop.jpg")
    exit()

# Get image dimensions
height, width = image.shape[:2]

# The plate is already cropped, so we make a clean rectangular output
# by slightly trimming the edges.
margin_x = int(width * 0.03)
margin_y = int(height * 0.08)

x1 = margin_x
y1 = margin_y
x2 = width - margin_x
y2 = height - margin_y

corrected = image[y1:y2, x1:x2]

# Resize for OCR
corrected = cv2.resize(
    corrected,
    None,
    fx=2,
    fy=2,
    interpolation=cv2.INTER_CUBIC
)

cv2.imwrite("plate_corrected.jpg", corrected)

print("Saved: plate_corrected.jpg")
print("Size:", corrected.shape[1], "x", corrected.shape[0])