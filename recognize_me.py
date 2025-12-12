from picamera2 import Picamera2
import cv2
import time
import os

DATA_DIR = "faces_data"
MODEL_FILE = "me_model.yml"
LABEL_ME = 1

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
    config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0)

    print("Recognition started. Ctrl+C to stop.")

    while True:
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

        if len(faces) == 0:
            print("No face")
            time.sleep(0.3)
            continue

        x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (200, 200))

        label, confidence = recognizer.predict(face_roi)
        # У LBPH: менше confidence = краще співпадіння
        if label == LABEL_ME and confidence < 70:
            print(f"✅ It's YOU (confidence={confidence:.1f})")
        else:
            print(f"❌ Not you / unsure (label={label}, confidence={confidence:.1f})")

        time.sleep(0.3)

if __name__ == "__main__":
    main()
