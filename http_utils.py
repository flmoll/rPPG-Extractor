

import base64
from io import BytesIO
from math import floor
import numpy as np
import cv2
import torch


import sys


from dataset.facedetection import facedetectors
from evaluation.gt_visualize import GTVisualizer
from tqdm import tqdm


MODEL_PATH_YUNET = "opencv_zoo/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

yunet_face_detector = facedetectors.YuNetFaceDetector(MODEL_PATH_YUNET, [1920, 1200], transpose=True, return_only_bb=True)

def read_video(video_path: str, max_frame: int) -> np.ndarray:
    video = []
    cap = cv2.VideoCapture(video_path)
    for i in tqdm(range(max_frame), desc="Reading video frames"):
        ret, frame = cap.read()
        if not ret:
            break
        video.append(frame)
    cap.release()
    return np.array(video)

def read_ground_truth_label(label_path: str, video_file_name: str, max_frame: int) -> np.ndarray:
    visualizer = GTVisualizer(label_path, video_file_name)
    dummy_predicted_ppg_values = np.zeros(900)
    _, new_ppg_values, _ = visualizer.resample_ppg(dummy_predicted_ppg_values, fps=30, num_frames=max_frame)
    return new_ppg_values

def face_detection(video: np.ndarray, large_face_box_coef=1.5) -> np.ndarray:

    cropped_frames_list = []
    correlation_face_detector = facedetectors.CorrelationFaceDetector(yunet_face_detector)

    for frame in tqdm(video, desc="Face detection"):
        face_box_coor = correlation_face_detector.detect_faces(frame)

        if face_box_coor is None:
            continue

        face_box_coor[0] = max(0, face_box_coor[0] - (large_face_box_coef - 1.0) / 2 * face_box_coor[2])
        face_box_coor[1] = max(0, face_box_coor[1] - (large_face_box_coef - 1.0) / 2 * face_box_coor[3])
        face_box_coor[2] = large_face_box_coef * face_box_coor[2]
        face_box_coor[3] = large_face_box_coef * face_box_coor[3]
        x, y, w, h = face_box_coor
        
        new_frame = frame[y:y+h, x:x+w]
        cropped_frames_list.append(new_frame)

    if len(cropped_frames_list) == 0:
        return video
    
    cropped_frames = np.array(cropped_frames_list, dtype=video.dtype)
    cropped_frames = np.reshape(cropped_frames, (-1, cropped_frames.shape[1], cropped_frames.shape[2], cropped_frames.shape[3]))
    return cropped_frames

def get_octetstream_tuple(data: np.ndarray) -> tuple:
    
    # Use BytesIO to simulate file upload without saving
    file_like = BytesIO(data)
    file_like.name = "tensor.zst"  # Optional but good to have filename

    return ("tensor.zst", file_like, "application/octet-stream")

def resize_video(video: np.ndarray, image_size: tuple) -> np.ndarray:
    """Resize video data from in_shape to out_shape."""
    if video.shape[1] == image_size[0] and video.shape[2] == image_size[1]:
        return video

    resized_video = np.zeros((video.shape[0], image_size[0], image_size[1], video.shape[3]), dtype=np.float32)
    for i in tqdm(range(video.shape[0]), desc="Resizing video frames"):
        frame = video[i].astype(np.float32)  # Ensure frame is in float32 format for resizing
        frame = frame / 255.0
        resized_frame = cv2.resize(frame, (image_size[1], image_size[0]), interpolation=cv2.INTER_AREA)
        resized_video[i] = resized_frame

    return resized_video

def diff_normalize_data(data):
    """Calculate discrete difference in video data along the time-axis and nornamize by its standard deviation."""
    n, h, w, c = data.shape
    diffnormalized_len = n - 1
    diffnormalized_data = np.zeros((diffnormalized_len, h, w, c), dtype=np.float32)
    diffnormalized_data_padding = np.zeros((1, h, w, c), dtype=np.float32)
    for j in range(diffnormalized_len):
        diffnormalized_data[j, :, :, :] = (data[j + 1, :, :, :] - data[j, :, :, :]) / (
                data[j + 1, :, :, :] + data[j, :, :, :] + 1e-7)
    diffnormalized_data = diffnormalized_data / np.std(diffnormalized_data)
    diffnormalized_data = np.append(diffnormalized_data, diffnormalized_data_padding, axis=0)
    diffnormalized_data[np.isnan(diffnormalized_data)] = 0
    return diffnormalized_data

def diff_normalize_label(label):
    """Calculate discrete difference in labels along the time-axis and normalize by its standard deviation."""
    diff_label = np.diff(label, axis=0)
    diffnormalized_label = diff_label / np.std(diff_label)
    diffnormalized_label = np.append(diffnormalized_label, np.zeros(1), axis=0)
    diffnormalized_label[np.isnan(diffnormalized_label)] = 0
    return diffnormalized_label

def standardized_data(data, mean=None, std=None):
    """Z-score standardization for video data."""
    data[np.isnan(data)] = 0

    if mean is None:
        mean = np.mean(data)
    if std is None:
        std = np.std(data)

    data = data - mean
    data = data / std
    data[np.isnan(data)] = 0
    return data

def standardized_label(label):
    """Z-score standardization for label signal."""
    label = label - np.mean(label)
    label = label / np.std(label)
    label[np.isnan(label)] = 0
    return label

def diff_normalized_to_standardized_label(label):
    """Convert diff normalized data to standardized data."""
    label = np.cumsum(label, axis=0)
    return standardized_label(label)
    
def standardized_to_diff_normalized_label(label):
    """Convert standardized data to diff normalized data."""
    return diff_normalize_label(label)


def get_model_image_size(model: str) -> tuple:
    if model.lower() in ["physformer", "rythmformer"]:
        return (128, 128)  # Example shape for PhysNet and RhythmFormer
    else:
        return (72, 72)  # Example shape for other models

def numpy_byteorder_to_endianness(byteorder):
    if byteorder == '<':
        return 'little'
    elif byteorder == '>':
        return 'big'
    elif byteorder == '=':
        return sys.byteorder  # Use system byte order
    else:
        raise ValueError(f"Unknown byte order: {byteorder}")

def format_video_tensor_for_network(tensor: np.ndarray, model_name: str, batch_len: int = 160) -> np.ndarray:
    """
    Format video tensor for network input.
    Resizes the tensor to the output shape and standardizes it.
    """

    num_batches = floor(tensor.shape[0] / batch_len)
    if num_batches == 0:
        raise ValueError("Input tensor is too short for the specified batch length.")
    
    tensor = tensor[:num_batches * batch_len]  # Trim to fit full batches
    tensor = tensor.reshape(num_batches, batch_len, tensor.shape[1], tensor.shape[2], tensor.shape[3])

    if model_name.lower() in ["tscan", "efficientphys", "rythmformer"]:
        tensor = np.transpose(tensor, (0, 1, 4, 2, 3))      # [batch, length, height, width, channel]
    elif model_name.lower() in ["physnet", "deepphys", "loglikelihood", "quantile", "physformer"]:
        tensor = np.transpose(tensor, (0, 4, 1, 2, 3))      # [batch, channel, length, height, width]
    
    return torch.from_numpy(tensor).to(torch.float32)

def encode_tensor_to_base64(arr: np.ndarray) -> str:
    # Serialize array to bytes
    raw_bytes = arr.tobytes()
    
    # Encode to base64 string
    b64_str = base64.b64encode(raw_bytes).decode('utf-8')
    return b64_str

def decode_tensor_from_base64(b64_str: str, dtype: np.dtype, shape: tuple) -> np.ndarray:
    # Decode base64 string to bytes
    raw_bytes = base64.b64decode(b64_str)
    
    # Convert back to numpy array
    arr = np.frombuffer(raw_bytes, dtype=dtype).reshape(shape)
    return arr