import sys
import os
import cv2
import torch
from datetime import datetime
from pathlib import Path
from MTCNN.MTCNN import create_mtcnn_net  # MTCNN for face detection
from utils.align_trans import Face_alignment  # Utility for face alignment

# Function to get the resource path for model weights
# This works in both development environments and PyInstaller-based executables
def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller."""
    try:
        # When bundled using PyInstaller, resources are stored in the _MEIPASS temp directory
        base_path = sys._MEIPASS
    except Exception:
        # In development, use the current working directory
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Set the device to GPU if available, otherwise fallback to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Prompt the user to input the path to the image and the person's name
image_path = input("Enter the path to the image: ").strip()  # Strip whitespace for better input handling
person_name = input("Enter the name of the person: ").strip()  # Strip whitespace for better input handling

# Attempt to load the image using the provided path
image = cv2.imread(image_path)
if image is None:
    # Exit the program if the image cannot be loaded
    sys.exit(f"Error: Could not read image at {image_path}")

# Load the model weights for P-Net, R-Net, and O-Net from the resource path
p_model_path = resource_path('MTCNN/weights/pnet_Weights')
r_model_path = resource_path('MTCNN/weights/rnet_Weights')
o_model_path = resource_path('MTCNN/weights/onet_Weights')

# Perform face detection using MTCNN
try:
    bboxes, landmarks = create_mtcnn_net(image, min_face_size=20, device=device,
                                         p_model_path=p_model_path,
                                         r_model_path=r_model_path,
                                         o_model_path=o_model_path)
except Exception as e:
    # Handle any errors in face detection (e.g., missing model files)
    sys.exit(f"Error during face detection: {e}")

# If no face is detected, exit the program
if bboxes is None or landmarks is None:
    sys.exit("Error: No face detected in the image.")

# Align the detected face using the facial landmarks
aligned_face = Face_alignment(image, default_square=True, landmarks=landmarks)

# Define the path for saving the aligned face image inside the 'facebank' folder
face_bank_directory = Path(resource_path('facebank'))  # The base directory for storing face images
person_directory = face_bank_directory / person_name  # Create a directory for the person using their name

# Create the directory for the person if it doesn't already exist
if not person_directory.exists():
    person_directory.mkdir(parents=True)

# Generate a unique filename using the current date and time
filename = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '.jpg'  # Timestamped filename

# Save the aligned face image to the person's directory
save_path = person_directory / filename
cv2.imwrite(str(save_path), aligned_face[0])  # Save the aligned face image

# Notify the user that the image has been saved successfully
print(f"Saved aligned face image to {save_path}")
