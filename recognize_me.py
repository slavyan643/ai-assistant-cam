from picamera2 import Picamera2
import cv2
import time
import os

DATA_DIR = "faces_data"
MODEL_FILE = "me_model.yml"
LABEL_ME = 1

# --- Налаштування ---
LBPH_THRESHOLD = 70
FRAME_SLEEP = 0.25
ME_STREAK_ON = 5
ME_STREAK_OFF = 5
TEXT_COOLDOWN_SEC = 30

def train_model():
    if not os.path.isdir(DATA_DIR):
        raise RuntimeError("No faces_data folder. Run enroll_me.py first.")

    images = []
    labels = []

    for fn in sorted(os.listdir(DATA_DIR)):
        if fn.lower().endswith(".png"):
            img = cv2.imread(os.path.join(DATA_DIR, fn), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(LABEL_ME)

    if len(images) < 10:
        raise RuntimeError("Need at least 10 samples. Run enroll_me.py again.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, labels)
    recognizer.save(MODEL_FILE)

def main():
    if not os.path.exists(MODEL_FILE):
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
    last_text_ts = 0

    print("AI camera started (TEXT MODE)")

    while True:
        try:
            frame = picam2.capture_array()
        except Exception:
            time.sleep(0.05)
            continue

        frame_rgb = frame[:, :, :3]
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

        faces = face_cascade.detectMultiScale(
            gray, 1.2, 5, minSize=(80, 80)
        )

        is_me_raw = False
        conf = None

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
            face_roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            label, confidence = recognizer.predict(face_roi)
            conf = confidence
            if label == LABEL_ME and confidence < LBPH_THRESHOLD:
                is_me_raw = True

        if is_me_raw:
            me_streak += 1
            not_me_streak = 0
        else:
            not_me_streak += 1
            me_streak = 0

        prev = confirmed_me
        if not confirmed_me and me_streak >= ME_STREAK_ON:
            confirmed_me = True
        if confirmed_me and not_me_streak >= ME_STREAK_OFF:
            confirmed_me = False

        now = time.time()
        if confirmed_me and not prev and now - last_text_ts > TEXT_COOLDOWN_SEC:
            print("👤 ТЕБЕ ВПІЗНАНО | YOU ARE RECOGNIZED")
            last_text_ts = now

        # Текст поверх відео
        if confirmed_me:
            cv2.putText(frame_rgb, "YOU", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        else:
            cv2.putText(frame_rgb, "NOT YOU", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        cv2.imshow("AI Assistant Cam (TEXT)", frame_rgb)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        time.sleep(FRAME_SLEEP)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
