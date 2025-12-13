import os
import time
import subprocess
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

BOT_TOKEN = os.environ.get("7299506136:AAH1X9eyeBCP5OkC91tEVpGB0GEGpU6-i3k")

CAMERA_PROCESS = None
AI_ENABLED = True


def keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("▶️ Камера ON", callback_data="cam_on"),
         InlineKeyboardButton("⏸ Камера OFF", callback_data="cam_off")],
        [InlineKeyboardButton("📸 Фото", callback_data="photo")],
        [InlineKeyboardButton("🧠 AI ON/OFF", callback_data="ai_toggle")],
        [InlineKeyboardButton("📊 Статус", callback_data="status")]
    ])


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 AI Assistant Cam\n\nКерування системою:",
        reply_markup=keyboard()
    )


async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global CAMERA_PROCESS, AI_ENABLED
    query = update.callback_query
    await query.answer()

    if query.data == "cam_on":
        if CAMERA_PROCESS is None:
            CAMERA_PROCESS = subprocess.Popen(
                ["python3", "recognize_me.py"],
                cwd=os.getcwd()
            )
            await query.edit_message_text("▶️ Камера запущена", reply_markup=keyboard())
        else:
            await query.edit_message_text("⚠️ Камера вже працює", reply_markup=keyboard())

    elif query.data == "cam_off":
        if CAMERA_PROCESS:
            CAMERA_PROCESS.terminate()
            CAMERA_PROCESS = None
            await query.edit_message_text("⏸ Камера зупинена", reply_markup=keyboard())
        else:
            await query.edit_message_text("ℹ️ Камера не запущена", reply_markup=keyboard())

    elif query.data == "photo":
        subprocess.run(["fswebcam", "shot.jpg"], check=False)
        if os.path.exists("shot.jpg"):
            await context.bot.send_photo(
                chat_id=query.message.chat_id,
                photo=open("shot.jpg", "rb")
            )
        await query.edit_message_text("📸 Фото зроблено", reply_markup=keyboard())

    elif query.data == "ai_toggle":
        AI_ENABLED = not AI_ENABLED
        state = "ON" if AI_ENABLED else "OFF"
        await query.edit_message_text(f"🧠 AI {state}", reply_markup=keyboard())

    elif query.data == "status":
        cam_state = "ON" if CAMERA_PROCESS else "OFF"
        ai_state = "ON" if AI_ENABLED else "OFF"
        await query.edit_message_text(
            f"📊 Статус:\nКамера: {cam_state}\nAI: {ai_state}",
            reply_markup=keyboard()
        )


def main():
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(on_button))

    print("🤖 Telegram bot started")
    app.run_polling()


if __name__ == "__main__":
    main()
