import cv2

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera not available")
        return

    ret, frame = cap.read()
    if ret:
        print("Camera frame captured")
    else:
        print("Failed to read frame")

    cap.release()

if __name__ == "__main__":
    main()
