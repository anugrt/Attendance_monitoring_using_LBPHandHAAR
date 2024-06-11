import cv2

def recognize_faces_in_video(recognizer, label_mapping):
    # Load prebuilt classifier for Frontal Face detection
    cascadePath = 'Multiple-Face-Recognition-master (1)\\haarcascade_frontalface_default.xml'  # Replace with the actual path
    faceCascade = cv2.CascadeClassifier(cascadePath)

    if faceCascade.empty():
        raise Exception("Error: Unable to load Haar Cascade classifier.")

    video_capture = cv2.VideoCapture(1)  # Use 0 for the default webcam, or replace with the video file path

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = gray_frame[y:y + h, x:x + w]
            label, confidence = recognizer.predict(face_roi)

            if confidence < 100:  # You can adjust this threshold as per your confidence level
                person_name = label_mapping[label]
                cv2.putText(frame, f"{person_name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            2)
            else:
                cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Load the trained LBPH model from the file
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("lbph_model.yml")

    # Load the label mapping (person names to integer labels)
    label_mapping = {
        0: "Erin",
        1: "Next",
        2: "See",
        3: "To",
        4: "You",

        # Add more mappings here as per your training data
    }

    recognize_faces_in_video(recognizer, label_mapping)
