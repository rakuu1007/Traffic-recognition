import easyocr
import cv2

# Create OCR reader
reader = easyocr.Reader(['en'])

# Load the image
image = cv2.imread("test_car.jpg.png")

# Read text from the image
results = reader.readtext(image)

print("\nDetected text:")

for detection in results:
    text = detection[1]
    confidence = detection[2]

    print(f"{text}  |  Confidence: {confidence:.2f}")