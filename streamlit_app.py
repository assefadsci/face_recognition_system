# Import required libraries
import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
from ImagePreprocessor import ImagePreprocessor
import cv2

# Set up the Streamlit application
st.set_page_config(
    page_title='Face Recognition System',
    page_icon=':sunglasses:',
    layout='centered'
)

st.title('Face Recognition System')
st.text('This Application Recognizes  faces of 160 Hollywood actors and actresses.')
st.write("")

@st.cache_resource
def load_model():
    return YOLO('face_recognition_weights.pt')


model = load_model()

uploaded_file = st.sidebar.file_uploader("Upload a Hollywood Celebrity Image: ", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        input_image = np.array(image)

        # Preprocess the image using the ImagePreprocessor module
        preprocessor = ImagePreprocessor()
        preprocessed_image = preprocessor.preprocess_image(input_image)

        face_resized, x, y, w, h = preprocessed_image[0]
        col1, col2 = st.columns(2)
        with col1:
            st.image(face_resized, caption="Uploaded Face", use_column_width=True)

        run_recognition = st.sidebar.button('Who is this?')
        if run_recognition:

            results = model.predict(face_resized)
            for result in results:
                probs = result.probs
                class_index = probs.top1
                class_name = result.names[class_index]
                score = float(probs.top1conf.cpu().numpy())

                # cv2.rectangle(face_resized, (x , y ), (x + w, y + h ), (0, 255, 0), 3)

                # Add a label for the detected object
                text_size, _ = cv2.getTextSize(f'{class_name} {score:.2f}', cv2.FONT_HERSHEY_PLAIN, 1, 2)
                text_width, text_height = text_size

                x = 0
                y = text_height + 10
                rectangle_height = text_height + 10
                cv2.rectangle(face_resized, (x, y - rectangle_height), (x + text_width + 5, y), (0, 255, 0), -1)
                cv2.putText(face_resized, f'{class_name} {score:.2f}', (x, y - 5),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

            with col2:
                st.image(face_resized, caption="Predicted Face", use_column_width=True)
                #st.download_button(':green[Download image]',face_resized, file_name=f'{class_name}.jpg', mime = "image/jpg" )

    except Exception as e:
        st.error('Failed to load image: ' + str(e))