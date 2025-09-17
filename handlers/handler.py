import datetime
import os
import sqlite3

from telegram import Update
from telegram.ext import *


def check_for_chatID(chat_id: int) -> bool:
    db_path = os.path.join(os.getcwd(), "database", "autovision.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT chat_id FROM users WHERE chat_id = ?", (chat_id,))
    data = cursor.fetchall()

    conn.close()
    return len(data) > 0


def add_user(chat_id: int, username: str) -> bool:
    db_path = os.path.join(os.getcwd(), "database", "autovision.db")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users(chat_id, username) VALUES (?, ?)", (chat_id, username)
        )
        conn.commit()
        return True
    except Exception as e:
        print("DB Error:", e)
        return False
    finally:
        conn.close()


async def register_handler(update: Update, context: ContextTypes.DEFAULT_TYPE,show=True):
    chat_id = update.effective_chat.id

    if check_for_chatID(chat_id):
        if show:
            await update.message.reply_text(
            "✅ <b>You are already registered!</b>\n"
            "Use <code>/list</code> to see your registered number plates.",
            parse_mode="HTML",
        )
    else:
        username = update.effective_user.username or ""
        first_name = update.effective_user.first_name or ""
        last_name = update.effective_user.last_name or ""

        # Prefer full name if possible, fallback to username
        if first_name or last_name:
            name = f"{first_name} {last_name}".strip()
        elif username:
            name = f"@{username}"
        else:
            name = "Unknown User"

        if add_user(chat_id, name):
            if show:
                await update.message.reply_text(
                f"🎉 <b>Welcome {name}!</b>\n"
                "You are now <i>successfully registered</i> in our system.\n\n"
                "👉 Use <code>/add</code> to register your number plate 🚗",
                parse_mode="HTML",
            )
        else:
            if show:
                await update.message.reply_text(
                "⚠️ <b>There was a problem adding you to the database.</b>\n"
                "Please try again later.",
                parse_mode="HTML",
            )

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await register_handler(update, context,False)  # ensure user is registered

    username = update.effective_user.first_name or "there"
    msg = (
        f"👋 Hello {username}!\n\n"
        "Welcome to <b>AutoVision Bot</b> 🚗\n"
        "I can help you track your car license plates, see registration dates, "
        "and notify you when a plate is detected.\n\n"
        "Here’s what you can do:\n"
        "• /add <code>plate_no</code> [plate_no2 …] — Register one or more plates\n"
        "• /list — See all your registered plates\n"
        "• /remove <code>plate_no</code> [plate_no2 …] — Stop tracking plates\n"
        "• /stop — Unregister yourself and all your plates\n\n"
        "Start by registering your first car with\n /add! usage /add plate_no"
    )
    await update.effective_message.reply_text(msg, parse_mode="HTML")


async def stop_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    db_path = os.path.join(os.getcwd(), "database", "autovision.db")

    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON;")  # important!
        cursor = conn.cursor()

        # fetch cars before deletion
        cursor.execute("SELECT plate_number FROM cars WHERE chat_id = ?", (chat_id,))
        cars = [row[0] for row in cursor.fetchall()]

        # delete user (cascade deletes cars)
        cursor.execute("DELETE FROM users WHERE chat_id = ?", (chat_id,))
        conn.commit()

        msg = f"😢 We’re sad to see you go, {update.effective_user.first_name}...\n\n"
        if cars:
            msg += "The following cars will no longer be tracked:\n"
            msg += "\n".join(f"• {plate}" for plate in cars)
            msg += "\n\n"
        msg += "If you change your mind, you can always come back with /register 💌"

        await update.message.reply_text(msg, parse_mode="HTML")

    except Exception as e:
        print("DB Error in stop_handler:", e)
        await update.message.reply_text(
            "⚠️ There was an error while removing you.\nPlease try again later.",
            parse_mode="HTML",
        )
    finally:
        conn.close()


async def list_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    db_path = os.path.join(os.getcwd(), "database", "autovision.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT plate_number, added_at FROM cars WHERE chat_id = ?", (chat_id,))
    data = cursor.fetchall()
    conn.close()

    if not data:
        await update.message.reply_text(
            "You don't have any registered cars.\nUse the /add command to register your car."
        )
        return

    # Build clean message
    msg = "🚗 <b>Your registered cars:</b>\n\n"
    for plate, added in data:
        date_str = datetime.datetime.strptime(added, "%Y-%m-%d %H:%M:%S").strftime("%d/%m/%Y")
        msg += f"• <b>{plate}</b> — registered on {date_str}\n"

    await update.message.reply_text(msg, parse_mode="HTML")

async def add_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    db_path = os.path.join(os.getcwd(), "database", "autovision.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Extract plate numbers from command arguments
    # Example: /add DL8CBD6844 1234FEA → ["DL8CBD6844", "1234FEA"]
    plates = context.args

    if not plates:
        await update.message.reply_text("Usage: /add <plate_no> [plate_no2 ...]")
        return

    added = []
    skipped = []

    for plate in plates:
        normalized_plate = plate.strip().upper()

        cursor.execute(
            "SELECT 1 FROM cars WHERE chat_id = ? AND plate_number = ?",
            (chat_id, normalized_plate)
        )
        if cursor.fetchone():
            skipped.append(normalized_plate)
            continue

        cursor.execute(
            "INSERT INTO cars (chat_id, plate_number) VALUES (?, ?)",
            (chat_id, normalized_plate)
        )
        added.append(normalized_plate)

    conn.commit()
    conn.close()


    msg = ""
    if added:
        msg += "✅ Added:\n" + "\n".join(added) + "\n\n"
    if skipped:
        msg += "⚠️ Already registered:\n" + "\n".join(skipped)

    await update.message.reply_text(msg.strip())

async def remove_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    db_path = os.path.join(os.getcwd(), "database", "autovision.db")

    plates_to_remove = context.args  # e.g. /remove DL8CBD6844 1234FEA

    if not plates_to_remove:
        await update.message.reply_text("Usage: /remove <plate_no> [plate_no2 ...]")
        return

    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON;")  # ensure cascade works if needed
        cursor = conn.cursor()

        removed = []
        not_found = []

        for plate in plates_to_remove:
            normalized_plate = plate.strip().upper()
            cursor.execute(
                "SELECT 1 FROM cars WHERE chat_id = ? AND plate_number = ?",
                (chat_id, normalized_plate)
            )
            if cursor.fetchone():
                cursor.execute(
                    "DELETE FROM cars WHERE chat_id = ? AND plate_number = ?",
                    (chat_id, normalized_plate)
                )
                removed.append(normalized_plate)
            else:
                not_found.append(normalized_plate)

        conn.commit()

        # Build response message
        msg = ""
        if removed:
            msg += "✅ Removed the following plates from tracking:\n"
            msg += "\n".join(f"• {plate}" for plate in removed)
            msg += "\n\n"
        if not_found:
            msg += "⚠️ These plates were not found in your list:\n"
            msg += "\n".join(f"• {plate}" for plate in not_found)

        await update.message.reply_text(msg.strip())

    except Exception as e:
        print("DB Error in remove_handler:", e)
        await update.message.reply_text(
            "⚠️ There was an error while removing plates. Please try again later."
        )
    finally:
        conn.close()

async def search_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    plates_to_search = context.args

    if not plates_to_search:
        await update.effective_message.reply_text("Usage: /search <plate_no> [plate_no2 ...]")
        return

    try:
        db_path = "database/autovision.db"
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        cursor = conn.cursor()

        registered = []
        not_registered = []

        for plate in plates_to_search:
            normalized_plate = plate.strip().upper()
            cursor.execute(
                "SELECT 1 FROM cars WHERE chat_id = ? AND plate_number = ?",
                (chat_id, normalized_plate)
            )
            if cursor.fetchone():
                registered.append(normalized_plate)
            else:
                not_registered.append(normalized_plate)

        msg = ""
        if registered:
            msg += "✅ Registered plates:\n" + "\n".join(f"• {p}" for p in registered) + "\n\n"
        if not_registered:
            msg += "⚠️ Not registered:\n" + "\n".join(f"• {p}" for p in not_registered)

        await update.effective_message.reply_text(msg.strip())

    except Exception as e:
        print("DB Error in search_handler:", e)
        if update.effective_message:
            await update.effective_message.reply_text(
                "⚠️ There was an error while searching plates."
            )
    finally:
        conn.close()
