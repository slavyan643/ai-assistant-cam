 from picamera2 import Picamera2
import cv2
import time
import os
import subprocess

DATA_DIR = "faces_data"
MODEL_FILE = "me_model.yml"
LABEL_ME = 1

# --- TUNING ---
LBPH_THRESHOLD = 70          # менше = суворіше, більше = легше впізнати
FRAME_SLEEP = 0.15           # затримка циклу (менше = частіше перевіряє)
ME_STREAK_ON = 5             # скільки кадрів підряд "ME" щоб зафіксувати "це ти"
ME_STREAK_OFF = 5            # скільки кадрів підряд "NOT ME/NO FACE" щоб скинути стан
GREET_COOLDOWN_SEC = 45      # не вітатись частіше ніж раз на N секунд

def say(voice: str, text: str, speed: int = 165):
    # voices: "uk", "ru"
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
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0)

    print("Recognition + TTS started. Ctrl+C to stop.")

    # --- state machine ---
    confirmed_me = False
    me_streak = 0
    not_me_streak = 0
    last_greet_ts = 0.0
    last_status_print = ""  # щоб не спамити однаковими рядками

    while True:
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
        )

        # 1) raw decision per frame
        is_me_raw = False
        confidence = None

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (200, 200))

            label, confidence = recognizer.predict(face_roi)
            if label == LABEL_ME and confidence < LBPH_THRESHOLD:
                is_me_raw = True
        else:
            # no face -> treat as not-me for streak logic
            is_me_raw = False

        # 2) streak stabilization
        if is_me_raw:
            me_streak += 1
            not_me_streak = 0
        else:
            not_me_streak += 1
            me_streak = 0

        # 3) confirmed state transitions
        prev_confirmed_me = confirmed_me

        if (not confirmed_me) and me_streak >= ME_STREAK_ON:
            confirmed_me = True
        if confirmed_me and not_me_streak >= ME_STREAK_OFF:
            confirmed_me = False

        # 4) greet on NOT_ME -> ME (with cooldown)
        now = time.time()
        if confirmed_me and (not prev_confirmed_me):
            if (now - last_greet_ts) >= GREET_COOLDOWN_SEC:
                greet_uk_ru()
                last_greet_ts = now

        # 5) console status (no spam)
        if len(faces) == 0:
            status = "No face"
        else:
            if confirmed_me:
                status = f"✅ YOU (conf={confidence:.1f})" if confidence is not None else "✅ YOU"
            else:
                status = f"❌ Not you/unsure (conf={confidence:.1f})" if confidence is not None else "❌ Not you/unsure"

        if status != last_status_print:
            print(status)
            last_status_print = status

        time.sleep(FRAME_SLEEP)

if __name__ == "__main__":
    main()
