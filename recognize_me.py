from picamera2 import Picamera2
import cv2
import time
import os
import numpy as np

# ====== PATHS ======
DATA_DIR = "faces_data"
MODEL_FILE = "me_model.yml"
CASCADE_PATH = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"

LABEL_ME = 1

# ====== SETTINGS ======
LBPH_THRESHOLD = 70
FRAME_SLEEP = 0.25
ME_STREAK_ON = 5
ME_STREAK_OFF = 5
TEXT_COOLDOWN_SEC = 30

# ====== TRAIN MODEL ======
def train_model():
    if not os.path.isdir(DATA_DIR):
        raise RuntimeError("faces_data folder not found")

    images = []
    labels = []

    for fn in sorted(os.listdir(DATA_DIR)):
        if fn.lower().endswith((".jpg", ".png")):
            path = os.path.join(DATA_DIR, fn)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (200, 200))
            images.append(img)
            labels.append(LABEL_ME)

    if len(images) < 10:
        raise RuntimeError("Need more samples")

    labels = np.array(labels, dtype=np.int32)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, labels)
    recognizer.save(MODEL_FILE)

    print(f"✓ Model trained & saved: {MODEL_FILE} (samples={len(images)})")

# ====== MAIN ======
def main():
    if not os.path.exists(MODEL_FILE):
        print("No model file, training now...")
        train_model()

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_FILE)

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        raise RuntimeError(f"Cannot load cascade: {CASCADE_PATH}")

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

    print("AI camera started (TEXT MODE)")

    while True:
        frame = picam2.capture_array()
        frame_rgb = frame[:, :, :3]
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
        )

        is_me_raw = False
        conf = None

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (200, 200))

            label, confidence = recognizer.predict(roi)
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

        if confirmed_me:
            cv2.putText(frame_rgb, "YOU", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            if conf is not None:
                cv2.putText(frame_rgb, f"conf:{conf:.1f}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame_rgb, "NOT YOU", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        cv2.imshow("AI Assistant Cam", frame_rgb)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        time.sleep(FRAME_SLEEP)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
