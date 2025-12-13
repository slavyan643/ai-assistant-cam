import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ====== CONFIG ======
PROJECT_DIR = Path(__file__).resolve().parent
RECOGNIZE_SCRIPT = PROJECT_DIR / "recognize_me.py"

# Якщо у тебе AI окремо в ai_chat.py:
# from ai_chat import ask_ai

BOT_TOKEN_ENV = "TELEGRAM_BOT_TOKEN"

# ====== RUNTIME STATE ======
camera_proc: subprocess.Popen | None = None
ai_enabled: bool = True


def _venv_python() -> str:
    """Повертає python з venv, якщо запущено з venv; інакше системний python."""
    # Якщо бот запущений з venv, sys.executable вже буде правильний
    return sys.executable


def is_process_alive(p: subprocess.Popen | None) -> bool:
    return p is not None and p.poll() is None


def start_camera_process() -> tuple[bool, str]:
    global camera_proc

    if is_process_alive(camera_proc):
        return True, "Камера вже запущена ✅"

    if not RECOGNIZE_SCRIPT.exists():
        return False, f"❌ Не знайдено файл: {RECOGNIZE_SCRIPT}"

    py = _venv_python()

    # ВАЖЛИВО: запускаємо unbuffered (-u), щоб логи були одразу
    # Працює у фоні, але логи можна буде дивитись через journalctl, якщо колись запустиш як сервіс
    try:
        camera_proc = subprocess.Popen(
            [py, "-u", str(RECOGNIZE_SCRIPT)],
            cwd=str(PROJECT_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,  # щоб можна було вбити всю групу процесів
        )
        time.sleep(0.4)
        if is_process_alive(camera_proc):
            return True, "Камера запущена ✅"
        return False, "❌ Камера не стартувала (процес одразу завершився)."
    except Exception as e:
        return False, f"❌ Помилка старту камери: {e}"


def stop_camera_process() -> tuple[bool, str]:
    global camera_proc

    if not is_process_alive(camera_proc):
        camera_proc = None
        return True, "Камера вже зупинена ✅"

    try:
        # Вбиваємо групу процесів
        os.killpg(os.getpgid(camera_proc.pid), signal.SIGTERM)
        time.sleep(0.5)

        if is_process_alive(camera_proc):
            os.killpg(os.getpgid(camera_proc.pid), signal.SIGKILL)

        camera_proc = None
        return True, "Камера зупинена ✅"
    except Exception as e:
        return False, f"❌ Помилка зупинки камери: {e}"


def keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("▶️ Камера ON", callback_data="CAM_ON"),
                InlineKeyboardButton("⏸️ Камера OFF", callback_data="CAM_OFF"),
            ],
            [InlineKeyboardButton("📸 Фото", callback_data="PHOTO")],
            [InlineKeyboardButton("🧠 AI ON/OFF", callback_data="AI_TOGGLE")],
            [InlineKeyboardButton("📊 Статус", callback_data="STATUS")],
        ]
    )


async def safe_send(update: Update, text: str, **kwargs):
    """Надійна відправка: і для message, і для callback."""
    if update.message:
        await update.message.reply_text(text, reply_markup=keyboard(), **kwargs)
    elif update.callback_query:
        await update.callback_query.message.reply_text(text, reply_markup=keyboard(), **kwargs)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await safe_send(update, "✅ Бот онлайн.\nНапиши задачу текстом або користуйся кнопками нижче.")


async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global ai_enabled
    q = update.callback_query
    await q.answer()

    data = q.data

    if data == "CAM_ON":
        ok, msg = start_camera_process()
        await safe_send(update, msg)
        return

    if data == "CAM_OFF":
        ok, msg = stop_camera_process()
        await safe_send(update, msg)
        return

    if data == "AI_TOGGLE":
        ai_enabled = not ai_enabled
        await safe_send(update, f"AI: {'ON ✅' if ai_enabled else 'OFF ⛔'}")
        return

    if data == "STATUS":
        cam_on = is_process_alive(camera_proc)
        await safe_send(
            update,
            "📊 Статус:\n"
            f"Камера: {'ON ✅' if cam_on else 'OFF ⛔'}\n"
            f"AI: {'ON ✅' if ai_enabled else 'OFF ⛔'}\n"
            f"Процес камери живий: {'так ✅' if cam_on else 'ні'}"
        )
        return

    if data == "PHOTO":
        # Тут можна підключити фото-зйомку з camera процесу, але це окрема логіка (IPC/файл/черга).
        await safe_send(update, "📸 Фото: функція буде додана (потрібен канал звʼязку з процесом камери).")
        return


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()

    # Якщо AI вимкнений — просто відповідаємо
    if not ai_enabled:
        await safe_send(update, "AI вимкнений ⛔ (увімкни кнопкою AI ON/OFF)")
        return

    # Якщо AI включений — тут можна викликати твій ask_ai()
    # Приклад:
    # try:
    #     reply = ask_ai(text)
    # except Exception as e:
    #     reply = f"AI помилка ❌ {e}"
    # await safe_send(update, reply)

    # Поки що просто тестова відповідь:
    await safe_send(update, "✅ Отримав текст. AI-обробку підключимо наступним кроком.")


def main():
    token = os.environ.get(BOT_TOKEN_ENV)
    if not token:
        raise RuntimeError(f"{BOT_TOKEN_ENV} is not set")

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    print("Telegram bot started")
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
