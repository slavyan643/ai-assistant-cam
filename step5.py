import cv2

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera not available")
        return

    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        return

    cv2.imwrite("snapshot.jpg", frame)
    print("Snapshot saved as snapshot.jpg")

    cap.release()

if __name__ == "__main__":
    main()
