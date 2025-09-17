import os
import cv2
from datetime import datetime
from collections import defaultdict, Counter
import cv2
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# ----------------------------
# Initialize TrOCR
# ----------------------------
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)


# ----------------------------
# Confusion correction
# ----------------------------
def correct_plate_confusion(plate: str) -> str:
    """
    Corrects common OCR misreads in Indian-style plates.
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

    # Numeric part (last 4 digits)
    if len(plate) >= 10:
        plate = plate[:-4] + plate[-4:].replace("O", "0").replace("I", "1").replace("S", "5").replace("B", "8")

    return plate


# ----------------------------
# OCR function
# ----------------------------
def ocr_plate(img):
    """
    img: np.ndarray cropped plate image (BGR)
    Returns: (final_plate, raw_text)
    """
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values, max_new_tokens=20)
    raw_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Clean plate text
    cleaned = "".join([c for c in raw_text.upper() if c.isalnum()])

    # Apply confusion correction
    corrected = correct_plate_confusion(cleaned)

    return corrected, raw_text
# Save vehicle crop to file
def save_detected_car(frame, plate_number, location):
    img_dir = os.path.join("database", "detections")
    os.makedirs(img_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{plate_number}_{timestamp}.jpg"
    filepath = os.path.join(img_dir, filename)
    cv2.imwrite(filepath, frame)
    return filepath
