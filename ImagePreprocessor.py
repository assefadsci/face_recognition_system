import cv2
import numpy as np
import os


class ImagePreprocessor:
    def __init__(self, face_detection_model_path='haarcascade_frontalface_default.xml'):
        if not os.path.isfile(face_detection_model_path):
            raise FileNotFoundError(f"Face detection model not found at: {face_detection_model_path}")

        self.face_detection_model = cv2.CascadeClassifier(face_detection_model_path)

    def preprocess_image(self, input_image):
        if not isinstance(input_image, np.ndarray):
            raise ValueError("Input image should be a numpy array")

        gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

        faces = self.face_detection_model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        preprocessed_data = []

        for (x, y, w, h) in faces:
            x -= 10
            y -= 10
            w += 20
            h += 20

            x = max(0, x)
            y = max(0, y)
            w = min(w, input_image.shape[1] - x)
            h = min(h, input_image.shape[0] - y)

            face_cropped = input_image[y:y + h, x:x + w]

            face_resized = cv2.resize(face_cropped, (224, 224))

            preprocessed_data.append((face_resized, x, y, w, h))

        return preprocessed_data