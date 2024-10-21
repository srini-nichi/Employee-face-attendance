import sys
import os
import cv2
import torch
from datetime import datetime
from pathlib import Path
from MTCNN.MTCNN import create_mtcnn_net  # MTCNN for face detection
from utils.align_trans import Face_alignment  # Utility for face alignment

# Constants for display and configurations
TEXT_POSITION = (10, 50)  # Position for displaying instructions on the frame
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX  # Font type for the text on the frame
FONT_SCALE = 1  # Font scale for the text
TEXT_COLOR = (0, 255, 0)  # Green color for the displayed text
THICKNESS = 2  # Thickness of the text
FACE_MIN_SIZE = 20  # Minimum face size for detection
FRAME_WINDOW_NAME = "My Capture"  # Window name for the video feed

# Function to get the resource path for model weights
# This is necessary when using PyInstaller or during development
def resource_path(relative_path):
    try:
        # If running as a PyInstaller package, use the temporary _MEIPASS directory
        base_path = sys._MEIPASS
    except Exception:
        # Otherwise, use the current directory for development
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Function to load MTCNN model weights for face detection
def initialize_mtcnn_weights():
    """Load model paths for P-Net, R-Net, and O-Net of MTCNN."""
    p_model_path = resource_path('MTCNN/weights/pnet_Weights')  # Path to P-Net weights
    r_model_path = resource_path('MTCNN/weights/rnet_Weights')  # Path to R-Net weights
    o_model_path = resource_path('MTCNN/weights/onet_Weights')  # Path to O-Net weights
    return p_model_path, r_model_path, o_model_path

# Function to save aligned face images to the person's folder
def save_aligned_face(frame, person_name, bboxes, landmarks, save_path):
    """Aligns detected face based on landmarks and saves it as an image."""
    # Align the face using the detected landmarks
    warped_face = Face_alignment(frame, default_square=True, landmarks=landmarks)
    
    # Generate a unique filename based on the current timestamp
    filename = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '.jpg'
    
    # Save the aligned face image to the specified path
    file_path = save_path / filename
    cv2.imwrite(str(file_path), warped_face[0])  # Save the aligned face image
    print(f"Saved face image to {file_path}")  # Notify user of the saved image

# Function to create a directory for storing the person's face images
def setup_person_directory(person_name):
    """Creates a directory for the person in 'facebank' if it doesn't exist."""
    data_path = Path('facebank')  # Base path for storing face images
    save_path = data_path / person_name  # Path for the specific person
    
    # Create the directory (and parent directories) if they don't exist
    save_path.mkdir(parents=True, exist_ok=True)
    return save_path

# Main function to capture video and perform face detection, alignment, and saving
def main():
    # Set up the Torch device: use GPU if available, otherwise use CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize the model weights for MTCNN face detection
    p_model_path, r_model_path, o_model_path = initialize_mtcnn_weights()

    # Prompt the user to input the name of the person to associate the face images with
    person_name = input("Enter the name of the person: ").strip()
    
    # Basic validation to ensure the person name is not empty
    if not person_name:
        print("Person name cannot be empty. Exiting.")
        return  # Exit the program if no valid name is provided

    # Set up a directory for the person in the 'facebank' folder to store images
    save_path = setup_person_directory(person_name)

    # Initialize video capture from the default camera (camera index 0)
    cap = cv2.VideoCapture(0)

    # Loop to continuously capture frames from the camera
    while cap.isOpened():
        isSuccess, frame = cap.read()  # Read a frame from the camera
        
        # If frame capture is successful, display it with instructions
        if not isSuccess:
            print("Failed to capture frame. Exiting.")
            break  # Exit if unable to capture a frame

        # Add instruction text to the frame
        frame_text = cv2.putText(frame, 'Press t to take a picture, q to quit...', 
                                 TEXT_POSITION, TEXT_FONT, FONT_SCALE, TEXT_COLOR, 
                                 THICKNESS, cv2.LINE_AA)
        cv2.imshow(FRAME_WINDOW_NAME, frame_text)  # Display the frame

        # Wait for keypress and process accordingly
        key = cv2.waitKey(1) & 0xFF
        if key == ord('t'):  # If 't' is pressed, capture the current frame
            try:
                # Perform face detection and alignment
                bboxes, landmarks = create_mtcnn_net(frame, FACE_MIN_SIZE, device,
                                                     p_model_path=p_model_path,
                                                     r_model_path=r_model_path,
                                                     o_model_path=o_model_path)
                if bboxes is not None:
                    # If a face is detected, save the aligned face image
                    save_aligned_face(frame, person_name, bboxes, landmarks, save_path)
                else:
                    # If no face is detected, notify the user
                    print("No face detected. Please try again.")
            except Exception as e:
                # Handle any errors that occur during face detection
                print(f"Error: {e}")

        elif key == ord('q') or cv2.getWindowProperty(FRAME_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            # If 'q' is pressed or the window is closed, exit the loop
            break

    # Release the video capture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Run the main function when the script is executed
if __name__ == "__main__":
    main()