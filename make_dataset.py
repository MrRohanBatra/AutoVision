import os
import cv2
import re
import pandas as pd
import numpy as np
import onnxruntime as ort

# -----------------------
# Paths
# -----------------------
CROPPED_IMAGES_DIR = "OCR_DATASET/images"
CSV_FILE_PATH = "OCR_DATASET/train.csv"
ONNX_MODEL_PATH = "us_lprnet_baseline18_deployable.onnx"

# -----------------------
# Reset CSV if exists
# -----------------------
if os.path.exists(CSV_FILE_PATH):
    os.remove(CSV_FILE_PATH)
    print("🗑️ Old CSV deleted.")

# -----------------------
# Character set
# -----------------------
characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# -----------------------
# Helper functions
# -----------------------
def clean_plate(text: str) -> str:
    """Keep only alphanumeric, uppercase."""
    text = text.replace(" ", "").upper()
    return "".join(re.findall(r"[A-Z0-9]", text))

def preprocess(img):
    """Resize, normalize, and convert to CHW for ONNX input."""
    img_resized = cv2.resize(img, (96, 48))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_chw = np.transpose(img_norm, (2,0,1))
    return np.expand_dims(img_chw, axis=0)

def decode(output, characters):
    """
    Decode LPRNet ONNX output.
    Assumes output shape: [1, seq_len] → class indices directly.
    """
    pred = output[0]  # already class indices
    plate = ""
    last_char = -1
    for c in pred:
        if c != last_char and c != 0:  # collapse repeated + skip blank
            plate += characters[c-1]
        last_char = c
    return plate

# -----------------------
# Load ONNX model
# -----------------------
session = ort.InferenceSession(ONNX_MODEL_PATH)
input_name = session.get_inputs()[0].name

# -----------------------
# Process images
# -----------------------
rows = []
for f in sorted(os.listdir(CROPPED_IMAGES_DIR)):
    if not f.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(CROPPED_IMAGES_DIR, f).replace("\\", "/")
    print(f"🔍 Processing {f}...")

    img = cv2.imread(path)
    if img is None:
        print(f"⚠️ Skipped {f}, could not read image.")
        continue

    # Preprocess
    pre_img = preprocess(img)

    # Run inference
    outputs = session.run(None, {input_name: pre_img})
    print(f"Output shape: {outputs[0].shape}")  # for debugging

    # Decode prediction
    plate_text = decode(outputs[0], characters)
    plate_text = clean_plate(plate_text)

    print(f"✅ Detected: {plate_text}")
    rows.append({"filepath": path, "label": plate_text})

# -----------------------
# Save CSV
# -----------------------
df = pd.DataFrame(rows)
df.to_csv(CSV_FILE_PATH, index=False)
print(f"🎉 New CSV generated with {len(df)} records → {CSV_FILE_PATH}")
