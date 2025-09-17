import threading
import asyncio
from utils.db_helper import get_user_chat_ids_for_plate

class NotificationWorker(threading.Thread):
    def __init__(self, notify_queue, bot):
        super().__init__()
        self.queue = notify_queue
        self.bot = bot
        self.running = True

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        while self.running:
            if not self.queue.empty():
                plate_number, img_path, location = self.queue.get()
                chat_ids = get_user_chat_ids_for_plate(plate_number)
                for chat_id in chat_ids:
                    loop.run_until_complete(
                        self.bot.send_photo(
                            chat_id=chat_id,
                            photo=open(img_path,"rb"),
                            caption=f"🚨 Plate {plate_number} detected at {location}"
                        )
                    )

    def stop(self):
        self.running = False
