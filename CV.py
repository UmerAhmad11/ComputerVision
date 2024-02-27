import cv2 as cv
import numpy as np 
from keras.models import load_model

# Load the pre-trained mood detection model
model = load_model('ml.h5')

# Load the pre-trained face detection cascade classifier
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_frame(frame, target_size=(64, 64)):
    # Resize the frame to the target size
    resized_frame = cv.resize(frame, target_size)
    # Normalize pixel values to range [0, 1]
    normalized_frame = resized_frame / 255.0
    # Expand dimensions to match model input shape (batch size = 1)
    preprocessed_frame = np.expand_dims(normalized_frame, axis=0)
    return preprocessed_frame

def detect_mood(frame):
    # Preprocess the frame
    preprocessed_frame = preprocess_frame(frame)
    # Perform mood detection inference using the pre-trained model
    predictions = model.predict(preprocessed_frame)
    # Get the predicted mood label
    mood_labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    predicted_mood_index = np.argmax(predictions)
    predicted_mood = mood_labels[predicted_mood_index]
    return predicted_mood

def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot Open Camera")
        exit()

    while True:
        try:
            ret, frame = cap.read()

            # If frame is read correctly, ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Convert frame to grayscale for face detection
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=20, minSize=(30, 30))

            # Iterate over detected faces
            for (x, y, w, h) in faces:
                # Draw rectangle around the face
                cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                # Crop region of interest (ROI) containing the face
                face_roi = frame[y:y+h, x:x+w]
                # Perform mood detection on the face ROI
                mood = detect_mood(face_roi)
                # Draw mood label on the frame
                cv.putText(frame, "Mood: " + mood, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

            # Display the resulting frame
            cv.imshow('Camera', frame)

            if cv.waitKey(1) == ord('q'):
                break

        except Exception as e:
            print("Error:", e)
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

# Run the main function
if __name__ == "__main__":
    main()
