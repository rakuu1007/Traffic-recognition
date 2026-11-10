from ultralytics import YOLO
import easyocr
import cv2
import re

# Load models
plate_model = YOLO("best.pt")
reader = easyocr.Reader(['en'])

# Load image
image = cv2.imread("test_car.jpg.png")

if image is None:
    print("Image not found!")
    exit()

results = plate_model(image, conf=0.25)

for result in results:

    for box in result.boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Add a little padding
        padding = 3
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.shape[1], x2 + padding)
        y2 = min(image.shape[0], y2 + padding)
         

        padding_x = 10
        padding_y = 10

        x1 = max(0, x1 - padding_x)
        y1 = max(0, y1 - padding_y)
        x2 = min(image.shape[1], x2 + padding_x)
        y2 = min(image.shape[0], y2 + padding_y)

        # Crop the plate with extra surrounding pixels
        plate = image[y1:y2, x1:x2]

        # Enlarge the crop for OCR
        plate = cv2.resize(
                plate,
                None,
                fx=3,
                fy=3,
                interpolation=cv2.INTER_CUBIC
        )

        # Save the improved crop
        cv2.imwrite("plate_crop_improved.jpg", plate)


        # --------------------------------
        # ENLARGE PLATE
        # --------------------------------

        plate = cv2.resize(
            plate,
            None,
            fx=12,
            fy=12,
            interpolation=cv2.INTER_LANCZOS4
        )

        # --------------------------------
        # GRAYSCALE
        # --------------------------------

        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

        # --------------------------------
        # CONTRAST ENHANCEMENT
        # --------------------------------

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        enhanced = clahe.apply(gray)

        # --------------------------------
        # SHARPEN
        # --------------------------------

        blurred = cv2.GaussianBlur(
            enhanced,
            (0, 0),
            3
        )

        sharpened = cv2.addWeighted(
            enhanced,
            1.8,
            blurred,
            -0.8,
            0
        )

        # --------------------------------
        # THRESHOLD
        # --------------------------------

        _, binary = cv2.threshold(
            sharpened,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Save for inspection
        cv2.imwrite(
            "plate_enhanced.jpg",
            sharpened
        )

        cv2.imwrite(
            "plate_binary.jpg",
            binary
        )

        # --------------------------------
        # OCR MULTIPLE VERSIONS
        # --------------------------------

        versions = [
            ("enhanced", sharpened),
            ("binary", binary)
        ]

        best_text = ""
        best_confidence = 0

        print("\n==============================")
        print("OCR RESULTS")
        print("==============================")

        for name, img in versions:

            ocr_results = reader.readtext(
                img,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                detail=1,
                paragraph=False
            )

            for detection in ocr_results:

                text = detection[1].upper()

                confidence = detection[2]

                cleaned = re.sub(
                    r'[^A-Z0-9]',
                    '',
                    text
                )
                # Correct common OCR mistakes in the numeric section
                match = re.match(
                    r'^([A-Z]{2})([A-Z0-9]{1,2})([A-Z]{1,3})([0-9]{1,4})$',
                    cleaned
                )

                if match:
                    state, number, letters, final_number = match.groups()

                    number = number.replace('Z', '2')
                    number = number.replace('S', '6')
                    number = number.replace('O', '0')
                    number = number.replace('I', '1')

                    cleaned = state + number + letters + final_number

                print(
                    f"{name}: {cleaned} | "
                    f"Confidence: {confidence:.2f}"
                )

                if len(cleaned) >= 6 and confidence > best_confidence:
                    best_text = cleaned
                    best_confidence = confidence

        print("\n==============================")
        print("BEST OCR RESULT")
        print("==============================")

        print("Plate:", best_text)
        print(
            "Confidence:",
            round(best_confidence, 2)
        )

        # Draw detection
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

# Save final result
cv2.imwrite(
    "anpr_result.jpg",
    image
)

print("\nFinished!")