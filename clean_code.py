import cv2


def load_classifier(path=None):
    if path is None:
        path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    clf = cv2.CascadeClassifier(path)
    if clf.empty():
        raise IOError(f"Failed to load classifier from: {path}")
    return clf


def detect_faces(gray_frame, classifier):
    return classifier.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )


def draw_faces(frame, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


def main():
    face_clf = load_classifier()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray, face_clf)
        draw_faces(frame, faces)

        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(10) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()