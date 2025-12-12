import cv2

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera not available")
        return

    print("Press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        cv2.imshow("AI Assistant Cam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
