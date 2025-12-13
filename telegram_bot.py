import os
import sys
import time
import asyncio
import signal
import subprocess
from datetime import datetime

from telegram import ReplyKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from telegram.error import TimedOut, NetworkError
from telegram.request import HTTPXRequest

# --- AI (OpenAI) ---
try:
    from ai_chat import ask_ai
    AI_AVAILABLE = True
except Exception:
    ask_ai = None
    AI_AVAILABLE = False

# ======================
# CONFIG
# ======================
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()

# Камера/процеси
REPO_DIR = os.path.abspath(os.path.dirname(__file__))
RECOGNIZE_SCRIPT = os.path.join(REPO_DIR, "recognize_me.py")

# Фото (найнадійніше через libcamera-still)
PHOTO_PATH = "/tmp/ai_cam_photo.jpg"

# Глобальний стан
STATE = {
    "camera_on": False,
    "ai_on": True,
    "cam_proc": None,  # subprocess.Popen
    "last_ai_ts": 0.0,
}

# Клавіатура
def main_keyboard():
    return ReplyKeyboardMarkup(
        [
            ["▶️ Камера ON", "⏸️ Камера OFF"],
            ["📸 Фото"],
            ["🧠 AI ON/OFF"],
            ["📊 Статус"],
        ],
        resize_keyboard=True
    )

# ======================
# SAFE SEND (важливо!)
# ======================
async def safe_send(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str, kb=True):
    """
    Надійна відправка повідомлення: retries + не валить бота при TimedOut.
    """
    markup = main_keyboard() if kb else None
    for attempt in range(3):
        try:
            if update.message:
                await update.message.reply_text(text, reply_markup=markup)
            elif update.callback_query:
                await update.callback_query.message.reply_text(text, reply_markup=markup)
            return
        except (TimedOut, NetworkError) as e:
            # Telegram іноді "підвисає" — пробуємо ще раз
            await asyncio.sleep(1.2 * (attempt + 1))
            if attempt == 2:
                # остання спроба — просто не падаємо
                print(f"[safe_send] failed after retries: {e}")
                return
        except Exception as e:
            print(f"[safe_send] unexpected error: {e}")
            return

# ======================
# CAMERA CONTROL
# ======================
def _is_proc_alive(p: subprocess.Popen | None) -> bool:
    return p is not None and (p.poll() is None)

def start_camera_process():
    if _is_proc_alive(STATE["cam_proc"]):
        return True, "Камера вже запущена ✅"

    if not os.path.exists(RECOGNIZE_SCRIPT):
        return False, f"Не знайдено файл {RECOGNIZE_SCRIPT}"

    # Запускаємо тим самим python, що і бот (venv)
    py = sys.executable
    try:
        p = subprocess.Popen(
            [py, RECOGNIZE_SCRIPT],
            cwd=REPO_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid  # щоб убивати групу процесів
        )
        STATE["cam_proc"] = p
        STATE["camera_on"] = True
        return True, "🎥 Камера запущена ✅"
    except Exception as e:
        return False, f"Помилка запуску камери: {e}"

def stop_camera_process():
    p = STATE["cam_proc"]
    if not _is_proc_alive(p):
        STATE["cam_proc"] = None
        STATE["camera_on"] = False
        return True, "Камера вже зупинена ✅"

    try:
        # Вбиваємо всю групу процесів
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        time.sleep(0.7)
        if _is_proc_alive(p):
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        STATE["cam_proc"] = None
        STATE["camera_on"] = False
        return True, "⏸️ Камера зупинена ✅"
    except Exception as e:
        return False, f"Не зміг зупинити камеру: {e}"

def take_photo_libcamera() -> tuple[bool, str]:
    """
    Робимо фото через libcamera-still (не потрібно окремих python-модулів).
    """
    try:
        # -n: no preview, --timeout 1000: 1 сек
        cmd = ["libcamera-still", "-n", "--timeout", "1000", "-o", PHOTO_PATH]
        r = subprocess.run(cmd, cwd=REPO_DIR, capture_output=True, text=True, timeout=20)
        if r.returncode != 0:
            return False, f"Помилка фото (libcamera): {r.stderr[-400:]}"
        if not os.path.exists(PHOTO_PATH):
            return False, "Фото не створилось (файлу немає)."
        return True, PHOTO_PATH
    except FileNotFoundError:
        return False, "Команда libcamera-still не знайдена. Встанови: sudo apt install -y libcamera-apps"
    except Exception as e:
        return False, f"Помилка фото: {e}"

# ======================
# HANDLERS
# ======================
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = "✅ Бот онлайн.\nНапиши задачу текстом або користуйся кнопками нижче."
    await safe_send(update, context, text, kb=True)

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cam = "ON ✅" if STATE["camera_on"] else "OFF ⛔"
    ai = "ON ✅" if STATE["ai_on"] else "OFF ⛔"
    alive = "так" if _is_proc_alive(STATE["cam_proc"]) else "ні"
    msg = f"📊 Статус:\nКамера: {cam}\nAI: {ai}\nПроцес камери живий: {alive}"
    await safe_send(update, context, msg, kb=True)

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    t = (update.message.text or "").strip()

    # Кнопки (текстові)
    if t in ("▶️ Камера ON", "Camera ON", "Камера ON"):
        ok, msg = start_camera_process()
        await safe_send(update, context, msg, kb=True)
        return

    if t in ("⏸️ Камера OFF", "Camera OFF", "Камера OFF"):
        ok, msg = stop_camera_process()
        await safe_send(update, context, msg, kb=True)
        return

    if t in ("📸 Фото", "Фото"):
        ok, res = take_photo_libcamera()
        if not ok:
            await safe_send(update, context, f"❌ {res}", kb=True)
            return
        try:
            await update.message.reply_photo(photo=open(res, "rb"), caption="📸 Ось фото", reply_markup=main_keyboard())
        except (TimedOut, NetworkError) as e:
            await safe_send(update, context, f"❌ Telegram timeout при відправці фото: {e}", kb=True)
        except Exception as e:
            await safe_send(update, context, f"❌ Не зміг відправити фото: {e}", kb=True)
        return

    if t in ("🧠 AI ON/OFF", "AI ON/OFF"):
        STATE["ai_on"] = not STATE["ai_on"]
        await safe_send(update, context, f"AI: {'ON ✅' if STATE['ai_on'] else 'OFF ⛔'}", kb=True)
        return

    if t in ("📊 Статус", "Статус"):
        await cmd_status(update, context)
        return

    # Звичайний текст = "завдання"
    if not STATE["ai_on"]:
        await safe_send(update, context, "AI зараз вимкнений. Увімкни через 🧠 AI ON/OFF.", kb=True)
        return

    if not AI_AVAILABLE or not ask_ai:
        await safe_send(update, context, "AI модуль недоступний (ai_chat.py не підключився).", kb=True)
        return

    # щоб не спамив API
    now = time.time()
    if now - STATE["last_ai_ts"] < 2.0:
        await asyncio.sleep(0.2)

    try:
        answer = ask_ai(t)
        if not answer:
            answer = "Не отримав відповідь від AI."
        STATE["last_ai_ts"] = time.time()
        await safe_send(update, context, answer, kb=True)
    except Exception as e:
        # тут можуть бути quota/429 і т.д.
        await safe_send(update, context, f"AI помилка ❌\n{str(e)[:250]}", kb=True)

async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    # Головне: не вбиваємо бота при будь-якій помилці
    print(f"[ERROR] {context.error}")

def main():
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

    # ВАЖЛИВО: збільшуємо таймаути до Telegram (щоб не було TimedOut)
    request = HTTPXRequest(
        connect_timeout=20.0,
        read_timeout=30.0,
        write_timeout=30.0,
        pool_timeout=30.0,
    )

    app = Application.builder().token(BOT_TOKEN).request(request).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_error_handler(on_error)

    print("✅ Telegram bot started")
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
