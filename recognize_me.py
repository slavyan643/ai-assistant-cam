from picamera2 import Picamera2
import cv2
import time
import os
import numpy as np

DATA_DIR = "faces_data"
MODEL_FILE = "me_model.yml"
LABEL_ME = 1

# --- Налаштування розпізнавання ---
LBPH_THRESHOLD = 70
FRAME_SLEEP = 0.25
ME_STREAK_ON = 5
ME_STREAK_OFF = 5
TEXT_COOLDOWN_SEC = 30

# --- AI (ініціативний) ---
AI_ENABLED = True
AI_COOLDOWN_SEC = 180  # максимум 1 раз на 3 хв, поки ти в кадрі

# Спроба підключити ai_chat.py (має бути в репо)
try:
    from ai_chat import ask_ai
    AI_AVAILABLE = True
except Exception:
    ask_ai = None
    AI_AVAILABLE = False


def train_model():
    if not os.path.isdir(DATA_DIR):
        raise RuntimeError("No faces_data folder. Run enroll_me.py first.")

    images = []
    labels = []

    for fn in sorted(os.listdir(DATA_DIR)):
        fn_low = fn.lower()
        if not fn_low.endswith((".png", ".jpg", ".jpeg")):
            continue

        path = os.path.join(DATA_DIR, fn)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # LBPH краще, коли всі кадри одного розміру
        img = cv2.resize(img, (200, 200))
        images.append(img)
        labels.append(LABEL_ME)

    if len(images) < 10:
        raise RuntimeError(f"Need more samples. Found only {len(images)} images in {DATA_DIR}")

    labels_np = np.array(labels, dtype=np.int32)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, labels_np)
    recognizer.save(MODEL_FILE)
    print(f"✅ Model trained & saved: {MODEL_FILE} (samples={len(images)})")


def safe_ai_prompt() -> str:
    return "Привіт! Я бачу тебе. Що хочеш зробити зараз? Які плани?"


def get_proactive_ai_message() -> str:
    """
    1 коротке питання від AI українською/російською.
    Якщо AI недоступний або помилка — повертає локальний текст.
    """
    if not (AI_ENABLED and AI_AVAILABLE and ask_ai):
        return safe_ai_prompt()

    try:
        prompt = (
            "Ти асистент камери. Користувач з'явився в кадрі. "
            "Запитай 1 коротке питання (одне речення) українською або російською: "
            "що він хоче зробити зараз / які плани. Без зайвих пояснень."
        )
        msg = ask_ai(prompt).strip()
        return msg if msg else safe_ai_prompt()
    except Exception:
        return safe_ai_prompt()


def main():
    # Якщо моделі ще нема — тренуємо
    if not os.path.exists(MODEL_FILE):
        print("No model file, training now...")
        train_model()

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_FILE)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "XRGB8888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1)

    confirmed_me = False
    me_streak = 0
    not_me_streak = 0
    last_text_ts = 0.0
    last_ai_ts = 0.0

    print("✅ AI camera started (TEXT+AI)")
    if AI_ENABLED:
        print(f"AI: {'ON' if AI_AVAILABLE else 'OFF (ai_chat.py not available)'}")

    while True:
        try:
            frame = picam2.capture_array()
        except Exception:
            time.sleep(0.05)
            continue

        # XRGB8888 -> беремо RGB
        frame_rgb = frame[:, :, :3]
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))

        is_me_raw = False
        conf = None

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face_roi = gray[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, (200, 200))

            label, confidence = recognizer.predict(face_roi)
            conf = float(confidence)

            # У LBPH: менше confidence = краще
            if label == LABEL_ME and conf < LBPH_THRESHOLD:
                is_me_raw = True

        # Стабілізація (щоб не мигало)
        if is_me_raw:
            me_streak += 1
            not_me_streak = 0
        else:
            not_me_streak += 1
            me_streak = 0

        prev = confirmed_me
        if (not confirmed_me) and me_streak >= ME_STREAK_ON:
            confirmed_me = True
        if confirmed_me and not_me_streak >= ME_STREAK_OFF:
            confirmed_me = False

        now = time.time()

        # Подія: впізнав (NOT YOU -> YOU)
        if confirmed_me and (not prev):
            if now - last_text_ts > TEXT_COOLDOWN_SEC:
                print("👤 ТЕБЕ ВПІЗНАНО | YOU ARE RECOGNIZED")
                last_text_ts = now

            if AI_ENABLED and (now - last_ai_ts) > AI_COOLDOWN_SEC:
                msg = get_proactive_ai_message()
                print("AI:", msg)
                last_ai_ts = now

        # Оверлей на відео
        if confirmed_me:
            cv2.putText(frame_rgb, "YOU", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            if conf is not None:
                cv2.putText(frame_rgb, f"conf:{conf:.1f}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame_rgb, "NOT YOU", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        cv2.imshow("AI Assistant Cam (TEXT+AI)", frame_rgb)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

        time.sleep(FRAME_SLEEP)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
