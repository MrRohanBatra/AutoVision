import cv2
import numpy as np
import torch
from PIL import Image
import re

from sympy import false
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO
from sort.sort import Sort
from collections import defaultdict, Counter

# -----------------------------
# Initialize Models
# -----------------------------
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)

YOLO_WEIGHTS_PATH = "License-Plate-Detection/runs/detect/train/weights/best.pt"
plate_model = YOLO(YOLO_WEIGHTS_PATH)


# -----------------------------
# OCR + Helper Functions
# -----------------------------
def clean_plate(text: str) -> str:
    """Keep only alphanumeric characters."""
    return "".join(re.findall(r"[A-Z0-9]", text.upper()))


def correct_plate_confusion(plate: str) -> str:
    """
    Correct common OCR misreads for Indian-style plates.
    """
    plate = plate.upper()
    # State code (first 2 letters)
    if len(plate) >= 2:
        plate = plate[:2].replace("0", "O").replace("1", "I").replace("5", "S") + plate[2:]
    # District code (next 2 digits)
    if len(plate) >= 4:
        plate = plate[:2] + plate[2:4].replace("O", "0").replace("I", "1") + plate[4:]
    # Series letters (next 1-2 letters)
    if len(plate) >= 6:
        plate = plate[:4] + plate[4:6].replace("0", "O").replace("1", "I").replace("5", "S") + plate[6:]
    # Last 4 digits
    if len(plate) >= 10:
        plate = plate[:-4] + plate[-4:].replace("O", "0").replace("I", "1").replace("S", "5").replace("B", "8")
    return plate


def ocr_image(img: np.ndarray,d=false) -> (str, str):
    """
    Run TrOCR on cropped plate image.
    Returns: (corrected_plate, raw_ocr_text)
    """
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values, max_new_tokens=20)
    raw_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    cleaned = clean_plate(raw_text)
    corrected = correct_plate_confusion(cleaned)
    if d:
        print(f"Raw text: {raw_text}\tCleaned: {cleaned}\tCorrected: {corrected}")
    return corrected


# -----------------------------
# Detection / Masking
# -----------------------------
def detect_plates(frame: np.ndarray) -> list:
    """Return YOLO detected boxes in [x1,y1,x2,y2,conf] format."""
    results = plate_model.predict(frame, verbose=False)
    dets = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        score = float(box.conf[0].cpu().numpy())
        dets.append([x1, y1, x2, y2, score])
    return dets


def crop_plate(frame: np.ndarray, box: list) -> np.ndarray:
    """Crop plate using bounding box."""
    x1, y1, x2, y2 = map(int, box[:4])
    return frame[y1:y2, x1:x2]


# -----------------------------
# SORT + Tracking
# -----------------------------
tracker = Sort()


def track_plates(detections: list) -> list:
    """
    Update SORT with current frame detections.
    Returns list of tracked objects: [[x1,y1,x2,y2,track_id], ...]
    """
    if len(detections) == 0:
        dets = np.empty((0, 5))
    else:
        dets = np.array(detections)
    tracked_objects = tracker.update(dets)
    return tracked_objects


# -----------------------------
# OCR aggregation per ID
# -----------------------------
id_to_plate_frames = defaultdict(list)


def aggregate_ocr(track_id: int, plate_text: str) -> str:
    """
    Store OCR results per track_id and return majority-vote corrected plate.
    """
    if plate_text:
        id_to_plate_frames[track_id].append(plate_text)

    plates = id_to_plate_frames[track_id]
    if not plates:
        return ""

    # Majority vote per character
    aggregated = ""
    for chars in zip(*plates):
        aggregated += Counter(chars).most_common(1)[0][0]

    return correct_plate_confusion(aggregated)
