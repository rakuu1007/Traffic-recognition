import easyocr
import cv2
import re

# ==============================
# EASY OCR SETUP
# ==============================

reader = easyocr.Reader(['en'])

# ==============================
# LOAD IMAGE
# ==============================

image = cv2.imread("plate_original_crop.jpg")

if image is None:
    print("ERROR: Could not find test_car.jpg.png")
    exit()

# ==============================
# CONVERT TO GRAYSCALE
# ==============================

if len(image.shape) == 3:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
else:
    gray = image

# ==============================
# RESIZE
# ==============================

gray = cv2.resize(
    gray,
    None,
    fx=3,
    fy=3,
    interpolation=cv2.INTER_CUBIC
)

# ==============================
# IMPROVE CONTRAST
# ==============================

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

enhanced = clahe.apply(gray)

# ==============================
# SHARPEN IMAGE
# ==============================

blur = cv2.GaussianBlur(
    enhanced,
    (0, 0),
    3
)

sharpened = cv2.addWeighted(
    enhanced,
    1.5,
    blur,
    -0.5,
    0
)

# ==============================
# OCR
# ==============================

results = reader.readtext(
    gray,
    detail=1,
    paragraph=False,
    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
    decoder='beamsearch'
)

# ==============================
# DISPLAY OCR RESULTS
# ==============================

print("\n==============================")
print("LICENSE PLATE OCR")
print("==============================")

best_text = ""
best_confidence = 0

for result in results:

    text = result[1].upper()
    confidence = result[2]

    # Remove spaces and special characters
    text = re.sub(
        r'[^A-Z0-9]',
        '',
        text
    )

    print("Detected:", text)
    print("Confidence:", round(confidence, 2))

    # ==============================
    # INDIAN NUMBER PLATE CORRECTION
    # ==============================

    match = re.match(
        r'^([A-Z]{2})([A-Z0-9]{1,2})([A-Z]{1,3})([0-9]{1,4})$',
        text
    )

    if match:

        state, number, letters, final_number = match.groups()

        # Common OCR mistakes in numeric section
        number = number.replace('Z', '2')
        number = number.replace('O', '0')
        number = number.replace('I', '1')
        number = number.replace('S', '6')
        number = number.replace('B', '8')

        text = (
            state
            + number
            + letters
            + final_number
        )

    # ==============================
    # SELECT BEST RESULT
    # ==============================

    if len(text) >= 4 and confidence > best_confidence:

        best_text = text
        best_confidence = confidence

# ==============================
# FINAL RESULT
# ==============================

print("\n==============================")
print("BEST RESULT")
print("==============================")

print("Plate:", best_text)
print(
    "Confidence:",
    round(best_confidence, 2)
)

print("==============================")