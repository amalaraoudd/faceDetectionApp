import cv2
import streamlit as st
import webcolors


# Create the face cascade classifier
face_cascade = cv2.CascadeClassifier('C:/Users/uber/Downloads/haarcascade_frontalface_default.xml')

# Function to detect faces and draw rectangles
def detect_faces(rectangle_color ,min_neighbors ,scaleFactor):
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Initialize a counter to track saved images
    counter = 0

    while True:
        # Read the frames from the webcam
        ret, frame = cap.read()
        # Convert the frames to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect the faces using the face cascade classifier
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=min_neighbors)
        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            #cv2.rectangle(frame, (x, y), (x + w, y + h), rectangle_color, 2)
           # cv2.rectangle(frame, (x, y), (x + w, y + h), tuple(int(rectangle_color[i:i + 2], 16) for i in (0, 2, 4)), 2)
            rgb_color = webcolors.hex_to_rgb(rectangle_color)
            cv2.rectangle(frame, (x, y), (x + w, y + h), rgb_color, 2)

        # Save the frame with detected faces
        if len(faces) > 0:
            image_with_faces_path = f"face_image_{counter}.jpg"
            cv2.imwrite(image_with_faces_path, frame)
            counter += 1

        # Display the frames
        cv2.imshow('Face Detection using Viola-Jones Algorithm', frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()


def app():
    st.title("Face Detection using Viola-Jones Algorithm")
    st.sidebar.title("Instructions")
    st.sidebar.write("Welcome to the Face Detection App!")
    st.sidebar.write("To use this app, follow these steps:")

    st.sidebar.write("1. Press the button below to start detecting faces from your webcam")
    st.sidebar.write("2. Click the 'Detect Faces' button to run the face detection algorithm.")
    st.sidebar.write("3. The detected faces will be highlighted in the image.")

    # Create a color picker widget to choose the rectangle color
    rectangle_color = st.color_picker("Choose rectangle color", "#00ff00")
    # Add a slider to adjust the minNeighbors parameter
    min_neighbors = st.slider("Adjust minNeighbors", 1, 10, 5)
    # Add a slider to adjust the scaleFactor parameter
    scaleFactor = st.slider("Adjust scaleFactor", 1.1, 2.0, 1.3, 0.1)

    # Add a button to start the face detection process
    if st.button('Start Face Detection'):
        detect_faces(rectangle_color, min_neighbors, scaleFactor)

if __name__ == "__main__":
    app()
