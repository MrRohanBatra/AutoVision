import os
from queue import Queue
from telegram.ext import ApplicationBuilder, CommandHandler
from utils.db_helper import init_db
from utils.util import ocr_plate
from workers.CameraWorker import CameraWorker
from workers.NotificationWorker import NotificationWorker
import dotenv
from handlers.handler import start_handler, stop_handler, list_handler, add_handler, remove_handler, search_handler, register_handler
import warnings
import logging
logging.getLogger("ultralytics").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")
print(f"Current working directory: {os.getcwd()}")
dotenv.load_dotenv()

# Simple Camera class
class Camera:
    def __init__(self, cam=0, location="unknown"):
        self.camera = cam
        self.location = location
    def getCamera(self):
        return self.camera
    def getLocation(self):
        return self.location

def main():
    print("Checking Database...")
    init_db()
    token = os.getenv("TOKEN")
    app = ApplicationBuilder().token(token).build()
    bot = app.bot
    print("Initializing Handlers...")

    app.add_handler(CommandHandler("start",start_handler))
    app.add_handler(CommandHandler("register",register_handler))
    app.add_handler(CommandHandler("stop",stop_handler))
    app.add_handler(CommandHandler("list",list_handler))
    app.add_handler(CommandHandler("add",add_handler))
    app.add_handler(CommandHandler("remove",remove_handler))
    app.add_handler(CommandHandler("search",search_handler))

    notify_queue = Queue()
    notifier = NotificationWorker(notify_queue, bot)
    notifier.start()

    cameras = [Camera(0,"Gate 1")]
    workers = []

    for cam in cameras:
        worker = CameraWorker(cam, "weights/LPR.pt", ocr_plate, notify_queue)
        worker.start()
        workers.append(worker)

    print("🚀 AutoVision system running...")
    app.run_polling()

    # Cleanup
    for w in workers:
        w.stop()
    notifier.stop()

if __name__ == "__main__":
    main()
