import cv2
import re
from collections import defaultdict, deque
from ultralytics import YOLO
import easyocr

# Models
vehicle_model = YOLO("yolo11n.pt")
plate_model = YOLO("best.pt")
reader = easyocr.Reader(["en"], gpu=False)

# Video
cap = cv2.VideoCapture("traffic.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    "vehicle_tracking_result.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

STOP_LINE = 400
RED_SIGNAL = True

# Keep previous vehicle images
HISTORY_SIZE = 20

previous_y = {}
history = defaultdict(lambda: deque(maxlen=HISTORY_SIZE))
violated = set()
searching = {}
best = {}


def valid_plate(text):
    text = re.sub(r"[^A-Z0-9]", "", text.upper())

    text = text.replace("O", "0")
    text = text.replace("I", "1")

    pattern = r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$"

    if re.match(pattern, text):
        return text

    return None


def ocr_plate(img):

    if img is None or img.size == 0:
        return None, 0

    img = cv2.resize(
        img,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    results = reader.readtext(
        gray,
        detail=1,
        paragraph=False,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )

    best_text = None
    best_conf = 0

    for r in results:

        plate = valid_plate(r[1])
        conf = float(r[2])

        if plate and conf > best_conf:
            best_text = plate
            best_conf = conf

    return best_text, best_conf


def find_plate(vehicle, vehicle_id):

    if vehicle is None or vehicle.size == 0:
        return None, 0

    # Enlarge vehicle
    big = cv2.resize(
        vehicle,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC
    )

    results = plate_model(
        big,
        conf=0.01,
        imgsz=960,
        verbose=False
    )

    best_plate = None
    best_conf = 0

    for box in results[0].boxes:

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(big.shape[1], x2)
        y2 = min(big.shape[0], y2)

        crop = big[y1:y2, x1:x2]

        plate, conf = ocr_plate(crop)

        if plate and conf > best_conf:

            best_plate = plate
            best_conf = conf

            cv2.imwrite(
                f"best_plate_{vehicle_id}.jpg",
                crop
            )

    return best_plate, best_conf


frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1
    original = frame.copy()

    results = vehicle_model.track(
        frame,
        persist=True,
        conf=0.4,
        classes=[2, 3, 5, 7],
        verbose=False
    )

    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().tolist()
        ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, ids):

            x1, y1, x2, y2 = map(int, box)

            current_y = y2

            # Keep vehicle history
            x1c = max(0, x1)
            y1c = max(0, y1)
            x2c = min(w, x2)
            y2c = min(h, y2)

            vehicle = original[
                y1c:y2c,
                x1c:x2c
            ]

            if vehicle.size > 0:
                history[track_id].append(
                    vehicle.copy()
                )

            # Draw vehicle
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            cv2.putText(
                frame,
                f"ID {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

            # Check stop-line crossing
            if track_id in previous_y:

                crossed = (
                    previous_y[track_id] < STOP_LINE
                    and current_y >= STOP_LINE
                )

                if (
                    crossed
                    and RED_SIGNAL
                    and track_id not in violated
                ):

                    violated.add(track_id)

                    # Search the frames BEFORE violation
                    searching[track_id] = list(
                        history[track_id]
                    )

                    # Also search current vehicle
                    searching[track_id].append(
                        vehicle.copy()
                    )

                    best[track_id] = (None, 0)

                    print()
                    print("==============================")
                    print(
                        f"RED LIGHT VIOLATION - "
                        f"Vehicle ID {track_id}"
                    )

                    print(
                        f"Searching {len(searching[track_id])} "
                        f"vehicle frames for plate..."
                    )

            previous_y[track_id] = current_y

            # Search saved frames
            if track_id in searching:

                frames = searching[track_id]

                if len(frames) > 0:

                    vehicle_image = frames.pop(0)

                    plate, conf = find_plate(
                        vehicle_image,
                        track_id
                    )

                    checked = (
                        HISTORY_SIZE + 1
                        - len(frames)
                    )

                    print(
                        f"Vehicle {track_id}: "
                        f"checking plate frame "
                        f"({checked})",
                        flush=True
                    )

                    if plate:

                        print(
                            f"Plate candidate: "
                            f"{plate} "
                            f"({conf:.2f})"
                        )

                        if conf > best[track_id][1]:

                            best[track_id] = (
                                plate,
                                conf
                            )

                else:

                    plate, conf = best[track_id]

                    print()
                    print("==============================")

                    if plate:

                        print(
                            f"FINAL PLATE: {plate}"
                        )

                        print(
                            f"CONFIDENCE: {conf:.2f}"
                        )

                    else:

                        print(
                            "FINAL PLATE: "
                            "Not reliably detected"
                        )

                    print("==============================")

                    del searching[track_id]

            # Display violation
            if track_id in violated:

                cv2.putText(
                    frame,
                    "RED LIGHT VIOLATION",
                    (x1, y2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

                if (
                    track_id in best
                    and best[track_id][0]
                ):

                    cv2.putText(
                        frame,
                        best[track_id][0],
                        (x1, y2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

    # Stop line
    cv2.line(
        frame,
        (0, STOP_LINE),
        (w, STOP_LINE),
        (0, 0, 255),
        2
    )

    # Signal
    cv2.putText(
        frame,
        "RED SIGNAL",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    out.write(frame)


cap.release()
out.release()

print()
print("==============================")
print("FINISHED")
print(f"Total violations: {len(violated)}")
print("Saved: vehicle_tracking_result.mp4")
print("==============================")