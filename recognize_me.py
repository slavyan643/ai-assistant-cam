from picamera2 import Picamera2
import cv2
import time
import os
import subprocess

DATA_DIR = "faces_data"
MODEL_FILE = "me_model.yml"
LABEL_ME = 1

# --- Налаштування ---
LBPH_THRESHOLD = 70          # менше = суворіше (краще співпадіння)
FRAME_SLEEP = 0.25           # пауза між кадрами (стабільніше)
ME_STREAK_ON = 5             # скільки разів підряд "ти", щоб підтвердити
ME_STREAK_OFF = 5            # скільки разів підряд "не ти/нема", щоб скинути
GREET_COOLDOWN_SEC = 45      # не вітатись частіше ніж раз на N секунд

def say(voice: str, text: str, speed: int = 165):
    subprocess.run(["espeak-ng", "-v", voice, "-s", str(speed), text], check=False)

def greet_uk_ru():
    say("uk", "Привіт! Я тебе бачу.")
    time.sleep(0.2)
    say("ru", "Привет! Я тебя вижу.")

def train_model():
    if not os.path.isdir(DATA_DIR):
        raise RuntimeError("No faces_data folder. Run enroll_me.py first.")

    images = []
    labels = []

    for fn in sorted(os.listdir(DATA_DIR)):
        if not fn.lower().endswith(".png"):
            continue
        path = os.path.join(DATA_DIR, fn)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        images.append(img)
        labels.append(LABEL_ME)

    if len(images) < 10:
        raise RuntimeError("Not enough samples. Need at least ~10. Run enroll_me.py again.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, labels)
    recognizer.save(MODEL_FILE)
    print("Model trained & saved:", MODEL_FILE)

def main():
    if not os.path.exists(MODEL_FILE):
        print("No model file, training now...")
        train_model()

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_FILE)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    picam2 = Picamera2()
    # Стабільніший формат для capture_array()
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "XRGB8888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0)

    print("Recognition + voice started. Ctrl+C to stop.")

    confirmed_me = False
    me_streak = 0
    not_me_streak = 0
    last_greet_ts = 0.0
    last_status = ""

    while True:
        # ✅ РІШЕННЯ №1: інколи кадр приходить битий — пропускаємо
        try:
            frame = picam2.capture_array()
        except Exception as e:
            print("⚠️ frame error, skip")
            time.sleep(0.05)
            continue

        # frame може бути XRGB — беремо перші 3 канали
        if frame is None or frame.size == 0:
            time.sleep(0.05)
            continue

        frame_rgb = frame[:, :, :3]
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
        )

        is_me_raw = False
        conf = None

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (200, 200))

            label, confidence = recognizer.predict(face_roi)
            conf = confidence
            if label == LABEL_ME and confidence < LBPH_THRESHOLD:
                is_me_raw = True

        # стабілізація
        if is_me_raw:
            me_streak += 1
            not_me_streak = 0
        else:
            not_me_streak += 1
            me_streak = 0

        prev_confirmed = confirmed_me

        if (not confirmed_me) and me_streak >= ME_STREAK_ON:
            confirmed_me = True

        if confirmed_me and not_me_streak >= ME_STREAK_OFF:
            confirmed_me = False

        # привітання тільки при переході NOT_ME -> ME + cooldown
        now = time.time()
        if confirmed_me and (not prev_confirmed):
            if (now - last_greet_ts) >= GREET_COOLDOWN_SEC:
                greet_uk_ru()
                last_greet_ts = now

        # короткий статус без спаму
        if len(faces) == 0:
            status = "No face"
        else:
            if confirmed_me:
                status = f"✅ YOU (conf={conf:.1f})" if conf is not None else "✅ YOU"
            else:
                status = f"❌ Not you/unsure (conf={conf:.1f})" if conf is not None else "❌ Not you/unsure"

        if status != last_status:
            print(status)
            last_status = status

        time.sleep(FRAME_SLEEP)

if __name__ == "__main__":
    main()
