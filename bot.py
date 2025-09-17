import os
import cv2
import numpy as np
from telegram.ext import filters, ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes
from telegram import Update, InputFile
import io
import dotenv
dotenv.load_dotenv()
import util as helper  # your util.py

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Hello! I am an ANPR system made by Rohan Batra.\n"
        "📸 Send me the image of your car number plate for testing."
    )


async def handle_image_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("✅ Image received, processing...")

    # Get the highest resolution photo
    photo = update.message.photo[-1]
    file = await photo.get_file()
    file_bytes = await file.download_as_bytearray()

    # Convert to OpenCV image
    npImg = np.frombuffer(file_bytes, np.uint8)
    frame = cv2.imdecode(npImg, cv2.IMREAD_COLOR)

    if frame is None:
        await update.message.reply_text("❌ Error reading the image.")
        return

    # Detect plates
    detections = helper.detect_plates(frame)

    if len(detections) == 0:
        await update.message.reply_text("⚠️ No license plate detected.")
        return

    # Process the first detected plate
    box = detections[0]
    plate_img = helper.crop_plate(frame, box)
    plate_text = helper.ocr_image(plate_img)
    corrected_plate = helper.correct_plate_simple(plate_text)

    # Encode cropped plate to send back
    _, buffer = cv2.imencode(".jpg", plate_img)
    plate_bytes = io.BytesIO(buffer)

    await update.message.reply_photo(
        photo=InputFile(plate_bytes, filename="plate.jpg"),
        caption=f"🔍 Detected Plate: `{corrected_plate}`",
        parse_mode="Markdown"
    )


def main():
    token = os.getenv("TOKEN")
    print(token)# ⚠️ replace with your actual token
    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_image_message))

    print("🚀 Bot started...")
    app.run_polling()


if __name__ == "__main__":
    main()
