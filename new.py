import os
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import pytesseract
import re

# --- Torch override fix ---
orig_load = torch.load
def torch_load_override(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return orig_load(*args, **kwargs)
torch.load = torch_load_override

# --- Configure pytesseract for macOS Homebrew ---
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
os.environ["TESSDATA_PREFIX"] = "/opt/homebrew/share/tessdata/"

# --- Step 1: Initialize YOLO Model ---
try:
    yolo_model = YOLO("License-Plate-Detection/runs/detect/train/weights/best.pt")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()

# Debug folder
os.makedirs("debug_outputs", exist_ok=True)

# --- OCR function ---
def read_license_plate(crop):
    """
    Run Tesseract OCR on a cropped plate image and return cleaned text + confidence.
    """
    try:
        config = "-l eng --oem 1 --psm 7"  # Single-line OCR mode
        text = pytesseract.image_to_string(crop, config=config)
        text = text.strip().replace(" ", "").replace("\n", "")

        # Apply regex filtering for Indian number plates (basic cleanup)
        pattern = r"[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{3,4}"
        match = re.search(pattern, text.upper())
        if match:
            return match.group(), 1.0
        elif len(text) > 0:
            return text.upper(), 0.5  # low confidence
        else:
            return None, 0.0
    except Exception as e:
        print("OCR error:", e)
        return None, 0.0

# --- Step 2: OCR Extraction Function ---
def extract_text_from_plate(plate_image, plate_index=0):
    """
    Preprocess the license plate and extract text.
    """
    # Save raw cropped plate
    cv2.imwrite(f"debug_outputs/cropped_plate_{plate_index}.jpg", plate_image)

    # Convert to grayscale
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)

    # Binary inverse threshold
    _, thresh = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY_INV)

    # Save preprocessed plate
    cv2.imwrite(f"debug_outputs/preprocessed_plate_{plate_index}.jpg", thresh)

    # OCR
    plate_text, text_score = read_license_plate(thresh)

    if plate_text:
        print(f"--- OCR Results for Plate {plate_index} ---")
        print(f'Detected Text: "{plate_text}", Confidence: {text_score:.4f}')
        return plate_text
    return None

# --- Step 3: Detect & OCR ---
def detect_and_read_plate(image_path):
    """
    Detects license plates in an image and runs OCR on them.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # YOLO detection
    detections = yolo_model(img)[0]
    if len(detections.boxes) == 0:
        print("No license plates detected.")
        return

    for i, box in enumerate(detections.boxes):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        # Crop license plate
        plate_crop = img[y1:y2, x1:x2]

        # Draw detection box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Run OCR
        text = extract_text_from_plate(plate_crop, plate_index=i)

        # Display OCR text
        if text:
            print(f"==> FINAL DETECTED TEXT (Plate {i}): {text} <==")
            cv2.putText(img, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show and save result
    cv2.imshow("License Plate Detection", img)
    cv2.imwrite("debug_outputs/final_detection.jpg", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- Usage ---
if __name__ == "__main__":
    image_folder = "License-Plate-Detection/train/images"

    # Loop over all image files in the folder
    for filename in os.listdir(image_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            full_path = os.path.join(image_folder, filename)
            print(f"\n=== Processing {filename} ===")
            detect_and_read_plate(full_path)
