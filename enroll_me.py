from picamera2 import Picamera2
import cv2
import time
import os

DATA_DIR = "faces_data"
LABEL = 1  # 1 = "you"

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0)

    count = 0
    target = 40  # скільки “облич” зібрати для навчання

    print("Enrollment started. Look at camera, move head a bit. Ctrl+C to stop.")

    while count < target:
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

        if len(faces) == 0:
            continue

        # беремо найбільше обличчя
        x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (200, 200))

        out = os.path.join(DATA_DIR, f"me_{count:03d}.png")
        cv2.imwrite(out, face_roi)
        count += 1
        print("Captured:", out)

        time.sleep(0.1)

    print("Done. Collected:", count, "samples")

if __name__ == "__main__":
    main()
