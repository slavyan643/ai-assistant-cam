#!/usr/bin/env python3
import os
import time
import asyncio
import subprocess
from pathlib import Path

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# --- CONFIG ---
REPO_DIR = Path(__file__).resolve().parent
RECOGNIZE_SCRIPT = REPO_DIR / "recognize_me.py"

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()

# AI (optional)
try:
    from ai_chat import ask_ai  # expects OPENAI_API_KEY in env
    AI_AVAILABLE = True
except Exception:
    ask_ai = None
    AI_AVAILABLE = False


# --- RUNTIME STATE (in-memory) ---
STATE = {
    "camera_on": False,
    "ai_on": True,
    "proc": None,          # subprocess.Popen for recognize_me.py
    "last_start_ts": 0.0,
}


def keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("▶ Камера ON", callback_data="cam_on"),
            InlineKeyboardButton("⏸ Камера OFF", callback_data="cam_off"),
        ],
        [
            InlineKeyboardButton("📸 Фото", callback_data="photo"),
        ],
        [
            InlineKeyboardButton("🧠 AI ON/OFF", callback_data="ai_toggle"),
        ],
        [
            InlineKeyboardButton("📊 Статус", callback_data="status"),
        ],
    ])


async def safe_send(update: Update, text: str, reply_markup=None):
    if update.message:
        await update.message.reply_text(text, reply_markup=reply_markup)
    elif update.callback_query:
        await update.callback_query.message.reply_text(text, reply_markup=reply_markup)


def start_camera_process() -> str:
    # already running?
    if STATE["proc"] and STATE["proc"].poll() is None:
        STATE["camera_on"] = True
        return "Камера вже працює ✅"

    if not RECOGNIZE_SCRIPT.exists():
        return f"Не знайдено {RECOGNIZE_SCRIPT.name} ❌ (перевір репо)"

    # start recognize_me.py
    proc = subprocess.Popen(
        ["python3", str(RECOGNIZE_SCRIPT)],
        cwd=str(REPO_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,  # detach from bot process
    )
    STATE["proc"] = proc
    STATE["camera_on"] = True
    STATE["last_start_ts"] = time.time()
    return "Камера запущена ✅"


def stop_camera_process() -> str:
    proc = STATE.get("proc")
    if proc and proc.poll() is None:
        try:
            proc.terminate()
        except Exception:
            pass

    STATE["proc"] = None
    STATE["camera_on"] = False
    return "Камера зупинена ✅"


def build_status() -> str:
    proc = STATE.get("proc")
    running = (proc is not None and proc.poll() is None)
    cam = "ON" if running else "OFF"
    ai = "ON" if STATE["ai_on"] else "OFF"
    ai_av = "OK" if AI_AVAILABLE else "NO ai_chat.py"
    key = "OK" if OPENAI_API_KEY else "NO KEY"
    return (
        f"📊 Статус:\n"
        f"Камера: {cam}\n"
        f"AI: {ai}\n"
        f"AI модуль: {ai_av}\n"
        f"OPENAI_API_KEY: {key}\n\n"
        f"💬 Можеш просто написати текстом задачу/план — я відповім."
    )


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await safe_send(update, "✅ Бот онлайн.\nНапиши задачу текстом або користуйся кнопками нижче.", reply_markup=keyboard())


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await safe_send(
        update,
        "Команди:\n"
        "/start — меню\n"
        "/status — статус\n"
        "/ai_on — AI ON\n"
        "/ai_off — AI OFF\n\n"
        "Або просто напиши повідомлення — це буде як 'завдання/план'.",
        reply_markup=keyboard()
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await safe_send(update, build_status(), reply_markup=keyboard())


async def cmd_ai_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE["ai_on"] = True
    await safe_send(update, "AI увімкнено ✅", reply_markup=keyboard())


async def cmd_ai_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE["ai_on"] = False
    await safe_send(update, "AI вимкнено ✅", reply_markup=keyboard())


async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    data = query.data

    if data == "cam_on":
        msg = start_camera_process()
        await safe_send(update, msg, reply_markup=keyboard())
        return

    if data == "cam_off":
        msg = stop_camera_process()
        await safe_send(update, msg, reply_markup=keyboard())
        return

    if data == "ai_toggle":
        STATE["ai_on"] = not STATE["ai_on"]
        await safe_send(update, f"AI: {'ON ✅' if STATE['ai_on'] else 'OFF ⛔'}", reply_markup=keyboard())
        return

    if data == "status":
        await safe_send(update, build_status(), reply_markup=keyboard())
        return

    if data == "photo":
        # quick snapshot using libcamera-still (works even if camera script is off, but camera must be available)
        tmp = REPO_DIR / "tg_photo.jpg"
        try:
            # timeout to avoid hanging
            subprocess.run(
                ["libcamera-still", "-n", "-o", str(tmp), "--width", "1280", "--height", "720", "--timeout", "300"],
                cwd=str(REPO_DIR),
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            await query.message.reply_photo(photo=open(tmp, "rb"), caption="📸 Фото")
        except Exception as e:
            await safe_send(update, f"Не вийшло зробити фото ❌\n{e}", reply_markup=keyboard())
        finally:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
        return


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    if not text:
        return

    # Treat any text as "task / plan"
    if not STATE["ai_on"] or (not AI_AVAILABLE) or (ask_ai is None):
        await update.message.reply_text(
            "📝 Прийняв задачу.\n"
            f"Текст: {text}\n\n"
            "AI зараз OFF або недоступний. Увімкни AI кнопкою або /ai_on.",
            reply_markup=keyboard()
        )
        return

    # Ask AI
    try:
        prompt = (
            "Ти короткий асистент. Користувач написав завдання/план. "
            "Відповідай українською коротко, по суті: 1) уточнювальне питання, 2) 2-3 кроки.\n\n"
            f"Текст користувача: {text}"
        )
        # run in thread to avoid blocking asyncio loop
        resp = await asyncio.to_thread(ask_ai, prompt)
        resp = (resp or "").strip() or "Ок. Уточни, будь ласка, що саме потрібно зробити?"
        await update.message.reply_text(resp, reply_markup=keyboard())
    except Exception as e:
        await update.message.reply_text(f"AI помилка ❌\n{e}", reply_markup=keyboard())


def main():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("ai_on", cmd_ai_on))
    app.add_handler(CommandHandler("ai_off", cmd_ai_off))

    app.add_handler(CallbackQueryHandler(on_button))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
