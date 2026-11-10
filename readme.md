# SmartTrafficSignal 🚦
### AI-Based Vehicle Detection, Traffic Signal Detection and Automatic Number Plate Recognition

## 📌 Overview

SmartTrafficSignal is a Python-based computer vision project that combines AI-powered vehicle detection, traffic signal detection and Automatic Number Plate Recognition (ANPR).

The project uses YOLO object detection models, OpenCV image processing and EasyOCR to analyse traffic images and extract useful vehicle-related information.

It explores the application of artificial intelligence and computer vision in intelligent transportation systems.

## ✨ Features

### 1. Vehicle Detection
- Uses YOLO11n for object detection.
- Identifies cars, motorcycles, buses and trucks.
- Draws bounding boxes around detected vehicles.
- Displays confidence scores for detected objects.

### 2. Traffic Signal Detection
- Uses HSV color-space conversion.
- Identifies red regions using color thresholding.
- Uses contour detection to locate red regions.
- Reports whether red regions are detected.

### 3. Number Plate Detection
- Uses a custom-trained YOLO model (`best.pt`).
- Detects vehicle registration plates.
- Crops detected number plates with additional padding.

### 4. Automatic Number Plate Recognition (ANPR)
- Uses EasyOCR to extract alphanumeric characters.
- Enlarges plate images to improve readability.
- Applies grayscale conversion and CLAHE contrast enhancement.
- Uses Gaussian sharpening and Otsu thresholding.
- Compares OCR results from enhanced and binary images.
- Applies basic character corrections to common OCR mistakes.
- Selects the highest-confidence OCR result.

## 🛠️ Technologies Used

- Python
- YOLO11n
- Ultralytics
- OpenCV
- EasyOCR
- NumPy
- Computer Vision
- Optical Character Recognition (OCR)

## 📂 Project Structure

```text
SmartTrafficSignal/
│
├── vehicle_detection.py
├── vehicle_tracking.py
├── traffic_signal_detection.py
├── plate_detection.py
├── plate_ocr.py
├── plate_correct.py
├── ocr_test.py
├── anpr.py
│
├── best.pt
│
├── vehicle_detection_result.jpg
├── vehicle_tracking_result.mp4
├── traffic_signal_detection_result.png
├── anpr_result.jpg
│
├── README.md
└── Other test images and videos
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/rakuu1007/SmartTrafficSignal.git
```

### 2. Navigate to the project directory

```bash
cd SmartTrafficSignal
```

### 3. Install dependencies

```bash
pip install ultralytics opencv-python easyocr numpy
```

### 4. Download or provide model weights

- `yolo11n.pt` is used for general vehicle detection. Ultralytics can download it automatically when required.
- `best.pt` contains the custom-trained number plate detection model and must be present in the project directory.

## ▶️ Running the Project

Run individual modules from the project directory:

**Vehicle Detection**
```bash
python vehicle_detection.py
```

**Traffic Signal Detection**
```bash
python traffic_signal_detection.py
```

**Automatic Number Plate Recognition**
```bash
python anpr.py
```

Ensure the corresponding input images are present in the project directory.

## 📊 Output

The project generates visual outputs, including:
- Vehicle detection bounding boxes.
- Traffic signal detection results.
- Cropped and enhanced number plate images.
- OCR-extracted registration numbers and confidence scores.
- Vehicle tracking output videos.

## 🚀 Future Enhancements

- Integrate all detection modules into one unified system.
- Extend traffic signal detection to red, yellow and green states.
- Implement vehicle density estimation.
- Introduce adaptive traffic signal timing.
- Improve number plate recognition under different lighting and weather conditions.
- Explore real-time camera-based traffic monitoring.
- Deploy the system on edge computing hardware.

## 👩‍💻 Author

**Raksha Aruna**

GitHub: [@rakuu1007](https://github.com/rakuu1007)

## 🏷️ Project Domain

Artificial Intelligence | Computer Vision | Intelligent Transportation Systems | Robotics