import sys
import os
from pathlib import Path
import cv2
import torch
import numpy as np
from torchvision import transforms as trans
from MTCNN.MTCNN import create_mtcnn_net  # Import MTCNN for face detection
from utils.align_trans import Face_alignment  # Import face alignment utility
from face_model import MobileFaceNet, l2_norm  # Import MobileFaceNet model and normalization function

import warnings
# Suppress FutureWarnings from PyTorch
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Define transformation for input images
test_transform = trans.Compose([
    trans.ToTensor(),  # Convert image to tensor
    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize image
])

# Set device for model execution (GPU if available, otherwise CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def listdir_nohidden(path):
    """Generator function to list files in a directory, ignoring hidden files."""
    for filename in os.listdir(path):
        if not filename.startswith('.'):  # Exclude hidden files
            yield filename

def resource_path(relative_path):
    """Get the absolute path to a resource, works for development and PyInstaller."""
    try:
        # PyInstaller creates a temp folder and stores the path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")  # Use current directory if not in PyInstaller

    return os.path.join(base_path, relative_path)

# Define paths for model weights and facebank
facebank_path = resource_path('facebank')
pnet_model_path = resource_path('MTCNN/weights/pnet_Weights')
rnet_model_path = resource_path('MTCNN/weights/rnet_Weights')
onet_model_path = resource_path('MTCNN/weights/onet_Weights')
mobile_face_net_path = resource_path('Weights/MobileFace_Net')

def prepare_facebank(model, path=facebank_path, tta=True):
    """
    Prepare facebank by computing embeddings for faces in the given directory.

    Args:
        model: The face recognition model to generate embeddings.
        path: Directory where the images are stored (default is 'facebank').
        tta: Boolean indicating if test-time augmentation is used (default is True).

    Returns:
        Tuple containing:
            - embeddings: Tensor containing face embeddings.
            - names: Numpy array of names corresponding to the embeddings.
    """
    model.eval()  # Set the model to evaluation mode
    embeddings = []  # List to hold embeddings
    names = ['']  # List to hold names corresponding to embeddings
    data_path = Path(path)

    # Iterate through each directory in the facebank
    for person_dir in data_path.iterdir():
        if person_dir.is_file():  # Skip files, only process directories
            continue
        embs = []  # Temporary list to hold embeddings for current person

        # Iterate through each image file in the person's directory
        for filename in listdir_nohidden(person_dir):
            image_path = os.path.join(person_dir, filename)  # Construct full image path
            img = cv2.imread(image_path)  # Load image

            # Ensure image is of the required shape; otherwise, detect and align the face
            if img.shape != (112, 112, 3):
                bboxes, landmarks = create_mtcnn_net(img, 20, device, 
                    p_model_path=pnet_model_path, 
                    r_model_path=rnet_model_path, 
                    o_model_path=onet_model_path)
                img = Face_alignment(img, default_square=True, landmarks=landmarks)

            with torch.no_grad():  # Disable gradient computation for inference
                if tta:  # If test-time augmentation is enabled
                    mirror = cv2.flip(img, 1)  # Create mirrored version of the image
                    emb = model(test_transform(img).to(device).unsqueeze(0))  # Get embedding for original image
                    emb_mirror = model(test_transform(mirror).to(device).unsqueeze(0))  # Get embedding for mirrored image
                    embs.append(l2_norm(emb + emb_mirror))  # Average embeddings for augmentation
                else:
                    embs.append(model(test_transform(img).to(device).unsqueeze(0)))  # Get embedding for original image

        if len(embs) == 0:  # If no embeddings were computed, skip to the next person
            continue
        
        # Compute the mean embedding for the person and append to list
        embedding = torch.cat(embs).mean(0, keepdim=True)
        embeddings.append(embedding)
        names.append(person_dir.name)  # Append the directory name as the person's name

    # Save the computed embeddings and names to files
    embeddings = torch.cat(embeddings)  # Concatenate embeddings for all individuals
    names = np.array(names)  # Convert names to numpy array
    torch.save(embeddings, os.path.join(path, 'facebank.pth'))  # Save embeddings
    np.save(os.path.join(path, 'names.npy'), names)  # Save names

    return embeddings, names  # Return the computed embeddings and names

def load_facebank(path=facebank_path):
    """
    Load embeddings and names from the facebank.

    Args:
        path: Directory where the facebank files are stored (default is 'facebank').

    Returns:
        Tuple containing:
            - facebank.pth: Contains the computed embeddings for each face in the dataset. 
              This allows for quick access and comparison during face recognition tasks.
            - names.npy: Contains the names corresponding to the embeddings, 
              which helps in identifying who each embedding belongs to.
    """
    data_path = Path(path)
    embeddings = torch.load(data_path / 'facebank.pth', weights_only=True)  # Load embeddings
    names = np.load(data_path / 'names.npy')  # Load names
    return embeddings, names  # Return loaded embeddings and names

if __name__ == '__main__':
    # Initialize and load the face detection model
    detect_model = MobileFaceNet(512).to(device)  # Model for embedding size of 512
    detect_model.load_state_dict(
        torch.load(mobile_face_net_path, map_location=lambda storage, loc: storage, weights_only=True))
    

    detect_model.eval()  # Set model to evaluation mode

    # Prepare facebank with embeddings and names
    embeddings, names = prepare_facebank(detect_model, path=facebank_path, tta=True)
    print(f'Embeddings shape: {embeddings.shape}')  # Print shape of embeddings
    print(f'Names: {names}')  # Print names corresponding to embeddings
