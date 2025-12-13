from picamera2 import Picamera2
import cv2
import time
import os

DATA_DIR = "faces_data"
MODEL_FILE = "me_model.yml"
LABEL_ME = 1

LBPH_THRESHOLD = 70
FRAME_SLEEP = 0.25
ME_STREAK_ON = 5
ME_STREAK_OFF = 5

def train_model():
    if not os.path.isdir(DATA_DIR):
        raise RuntimeError("Run enroll_me.py first")

    images, labels = [], []

    for fn in os.listdir(DATA_DIR):
        if fn.lower().endswith(".png"):
            img = cv2.imread(os.path.join(DATA_DIR, fn), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(LABEL_ME)

    if len(images) < 10:
        raise RuntimeError("Need more samples")

    r = cv2.face.LBPHFaceRecognizer_create()
    r.train(images, labels)
    r.save(MODEL_FILE)

def main():
    if not os.path.exists(MODEL_FILE):
        train_model()

    r = cv2.face.LBPHFaceRecognizer_create()
    r.read(MODEL_FILE)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(
        main={"size": (640, 480), "format": "XRGB8888"}
    ))
    cam.start()
    time.sleep(1)

    confirmed = False
    me_streak = 0
    not_me_streak = 0

    print("Camera started")

    while True:
        frame = cam.capture_array()
        rgb = frame[:, :, :3]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        is_me = False

        if len(faces):
            x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
            roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            label, conf = r.predict(roi)
            if label == LABEL_ME and conf < LBPH_THRESHOLD:
                is_me = True

        if is_me:
            me_streak += 1
            not_me_streak = 0
        else:
            not_me_streak += 1
            me_streak = 0

        if not confirmed and me_streak >= ME_STREAK_ON:
            confirmed = True
            print("YOU")

        if confirmed and not_me_streak >= ME_STREAK_OFF:
            confirmed = False
            print("LEFT")

        cv2.putText(
            rgb,
            "YOU" if confirmed else "NOT YOU",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0) if confirmed else (0, 255, 255),
            2
        )

        cv2.imshow("Cam", rgb)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        time.sleep(FRAME_SLEEP)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
