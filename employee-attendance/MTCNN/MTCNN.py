from PIL import Image, ImageDraw, ImageFont  
import argparse  
import torch  
from MTCNN.MTCNN_nets import PNet, RNet, ONet  
import math
import numpy as np  
from MTCNN.utils.util import *  
import cv2  
import time  

# This function creates the MTCNN face detection pipeline (PNet -> RNet -> ONet)
def create_mtcnn_net(image, mini_face, device, p_model_path=None, r_model_path=None, o_model_path=None):

    # Initialize empty arrays to store bounding boxes and landmarks
    boxes = np.array([])  
    landmarks = np.array([])

    # If P-Net weights are provided, initialize and use P-Net for face detection
    if p_model_path is not None:
        pnet = PNet().to(device)  # Load PNet model onto device (CPU or GPU)
        pnet.load_state_dict(torch.load(p_model_path, map_location=lambda storage, loc: storage, weights_only=True))  # Load weights
        pnet.eval()  # Set PNet to evaluation mode (no training)
        
        # Perform face detection using P-Net and get bounding boxes
        bboxes = detect_pnet(pnet, image, mini_face, device)

    # If R-Net weights are provided, initialize and use R-Net to refine detection
    if r_model_path is not None:
        rnet = RNet().to(device)  # Load RNet model onto device
        rnet.load_state_dict(torch.load(r_model_path, map_location=lambda storage, loc: storage, weights_only=True))  # Load weights
        rnet.eval()  # Set RNet to evaluation mode
        
        # Refine the bounding boxes using R-Net
        bboxes = detect_rnet(rnet, image, bboxes, device)

    # If O-Net weights are provided, initialize and use O-Net for final refinement
    if o_model_path is not None:
        onet = ONet().to(device)  # Load ONet model onto device
        onet.load_state_dict(torch.load(o_model_path, map_location=lambda storage, loc: storage, weights_only=True))  # Load weights
        onet.eval()  # Set ONet to evaluation mode
        
        # Perform final bounding box refinement and detect facial landmarks using O-Net
        bboxes, landmarks = detect_onet(onet, image, bboxes, device)

    # Return the bounding boxes and detected facial landmarks
    return bboxes, landmarks

# P-Net face detection step
def detect_pnet(pnet, image, min_face_size, device):

    thresholds = 0.7  # Threshold for face detection confidence
    nms_thresholds = 0.7  # Threshold for Non-Maximum Suppression (NMS)

    # Get the dimensions of the input image
    height, width, channel = image.shape
    min_length = min(height, width)

    min_detection_size = 12  # Minimum detection size for faces
    factor = 0.707  # Scaling factor for image pyramid

    scales = []  # List of scales to resize the image

    # Calculate the scaling factor to adjust the image size
    m = min_detection_size / min_face_size
    min_length *= m

    # Generate scaled images until the smallest dimension falls below the minimum size
    factor_count = 0
    while min_length > min_detection_size:
        scales.append(m * factor ** factor_count)  # Store the scaling factor
        min_length *= factor
        factor_count += 1

    bounding_boxes = []  # List to store bounding boxes

    with torch.no_grad():  # No gradient calculation needed during inference
        # Iterate over each scale and apply P-Net to detect faces
        for scale in scales:
            # Resize the image according to the current scale
            sw, sh = math.ceil(width * scale), math.ceil(height * scale)
            img = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_LINEAR)
            img = torch.FloatTensor(preprocess(img)).to(device)  # Preprocess the image and move it to the device

            # Apply P-Net to get bounding box offsets and confidence scores
            offset, prob = pnet(img)
            probs = prob.cpu().data.numpy()[0, 1, :, :]  # Face confidence scores
            offsets = offset.cpu().data.numpy()  # Bounding box offsets

            # Get the indices of the regions where faces are detected with confidence > threshold
            inds = np.where(probs > thresholds)

            if inds[0].size == 0:  # No faces detected
                boxes = None
            else:
                # Extract bounding box coordinates and scores
                tx1, ty1, tx2, ty2 = [offsets[0, i, inds[0], inds[1]] for i in range(4)]
                offsets = np.array([tx1, ty1, tx2, ty2])
                score = probs[inds[0], inds[1]]

                # Calculate the bounding box positions in the original image
                bounding_box = np.vstack([
                    np.round((stride * inds[1] + 1.0) / scale),
                    np.round((stride * inds[0] + 1.0) / scale),
                    np.round((stride * inds[1] + 1.0 + cell_size) / scale),
                    np.round((stride * inds[0] + 1.0 + cell_size) / scale),
                    score, offsets
                ])
                boxes = bounding_box.T  # Convert to correct format
                keep = nms(boxes[:, 0:5], overlap_threshold=0.5)  # Apply Non-Maximum Suppression (NMS)
                boxes = boxes[keep]

            bounding_boxes.append(boxes)

        # Merge bounding boxes from all scales
        bounding_boxes = [i for i in bounding_boxes if i is not None]
        bounding_boxes = np.vstack(bounding_boxes)  # Stack into one array

        # Apply final NMS to remove overlapping boxes
        keep = nms(bounding_boxes[:, 0:5], nms_thresholds)
        bounding_boxes = bounding_boxes[keep]

        # Adjust bounding box coordinates using the offsets predicted by P-Net
        bboxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
        bboxes = convert_to_square(bboxes)  # Convert bounding boxes to squares
        bboxes[:, 0:4] = np.round(bboxes[:, 0:4])  # Round the bounding box coordinates

        return bboxes  # Return the bounding boxes

# Similar to P-Net, but for R-Net
def detect_rnet(rnet, image, bboxes, device):
    size = 24  # R-Net input size
    thresholds = 0.8  # Face detection confidence threshold for R-Net
    nms_thresholds = 0.7  # NMS threshold

    height, width, channel = image.shape  # Get image dimensions

    num_boxes = len(bboxes)  # Number of bounding boxes from P-Net
    # Adjust bounding boxes to fit within the image boundaries
    [dy, edy, dx, edx, y, ey, x, ex, w, h] = correct_bboxes(bboxes, width, height)

    img_boxes = np.zeros((num_boxes, 3, size, size))  # Initialize container for cropped face regions

    # Process each bounding box
    for i in range(num_boxes):
        img_box = np.zeros((h[i], w[i], 3))  # Create a placeholder for the face region

        # Extract the face region from the image and resize it to the R-Net input size
        img_box[dy[i]:(edy[i] + 1), dx[i]:(edx[i] + 1), :] = image[y[i]:(ey[i] + 1), x[i]:(ex[i] + 1), :]
        img_box = cv2.resize(img_box, (size, size), interpolation=cv2.INTER_LINEAR)
        img_boxes[i, :, :, :] = preprocess(img_box)  # Preprocess the face region

    img_boxes = torch.FloatTensor(img_boxes).to(device)  # Convert to tensor and move to device
    offset, prob = rnet(img_boxes)  # Run R-Net to get offsets and probabilities

    offsets = offset.cpu().data.numpy()  # Bounding box offsets
    probs = prob.cpu().data.numpy()  # Confidence scores

    keep = np.where(probs[:, 1] > thresholds)[0]  # Keep boxes with high confidence
    bboxes = bboxes[keep]
    bboxes[:, 4] = probs[keep, 1].reshape((-1,))  # Update bounding boxes with confidence scores
    offsets = offsets[keep]

    keep = nms(bboxes, nms_thresholds)  # Apply NMS to remove overlaps
    bboxes = bboxes[keep]
    bboxes = calibrate_box(bboxes, offsets[keep])  # Adjust bounding boxes with offsets
    bboxes = convert_to_square(bboxes)  # Convert bounding boxes to squares
    bboxes[:, 0:4] = np.round(bboxes[:, 0:4])  # Round bounding box coordinates

    return bboxes  # Return refined bounding boxes

# Similar approach for O-Net with facial landmark detection
def detect_onet(onet, image, bboxes, device):
    size = 48  # O-Net input size
    thresholds = 0.8  # Confidence threshold
    nms_thresholds = 0.7  # NMS threshold

    height, width, channel = image.shape  # Get image dimensions
    num_boxes = len(bboxes)  # Number of bounding boxes

    [dy, edy, dx, edx, y, ey, x, ex, w, h] = correct_bboxes(bboxes, width, height)  # Adjust bounding boxes

    img_boxes = np.zeros((num_boxes, 3, size, size))  # Placeholder for face regions

    # Process each bounding box for O-Net
    for i in range(num_boxes):
        img_box = np.zeros((h[i], w[i], 3))

        img_box[dy[i]:(edy[i] + 1), dx[i]:(edx[i] + 1), :] = image[y[i]:(ey[i] + 1), x[i]:(ex[i] + 1), :]
        img_box = cv2.resize(img_box, (size, size), interpolation=cv2.INTER_LINEAR)
        img_boxes[i, :, :, :] = preprocess(img_box)  # Preprocess face region

    img_boxes = torch.FloatTensor(img_boxes).to(device)  # Convert to tensor and move to device
    offset, prob, landmark = onet(img_boxes)  # Run O-Net to get offsets, probabilities, and landmarks

    offsets = offset.cpu().data.numpy()  # Bounding box offsets
    probs = prob.cpu().data.numpy()  # Confidence scores
    landmarks = landmark.cpu().data.numpy()  # Facial landmarks

    keep = np.where(probs[:, 1] > thresholds)[0]  # Keep boxes with high confidence
    bboxes = bboxes[keep]
    bboxes[:, 4] = probs[keep, 1].reshape((-1,))
    offsets = offsets[keep]
    landmarks = landmarks[keep]

    keep = nms(bboxes, nms_thresholds)  # Apply NMS to remove overlaps
    bboxes = bboxes[keep]
    bboxes = calibrate_box(bboxes, offsets[keep])  # Adjust bounding boxes with offsets
    landmarks = landmarks[keep]

    # Adjust landmark coordinates to original image size
    landmarks[:, 0::2] = (np.tile(bboxes[:, 0], [5, 1]).T + (landmarks[:, 0::2].T * (bboxes[:, 2] - bboxes[:, 0] + 1)).T).T
    landmarks[:, 1::2] = (np.tile(bboxes[:, 1], [5, 1]).T + (landmarks[:, 1::2].T * (bboxes[:, 3] - bboxes[:, 1] + 1)).T).T

    return bboxes, landmarks  # Return final bounding boxes and landmarks
