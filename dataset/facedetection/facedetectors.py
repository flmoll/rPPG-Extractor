from collections import deque
import cv2
import numpy as np
import importlib
import sys
import scipy
from sklearn.cluster import KMeans
import torch
from opencv_zoo.models.face_detection_yunet.yunet import YuNet
from dataset.facedetection.yolo5.YOLO5Face import YOLO5Face

import scipy.signal as signal
from scipy.signal import butter, lfilter
from math import ceil

from typing import Tuple

from dataset.facedetection.invariantTM import invariant_match_template
from mosse.utils import linear_mapping, pre_process, random_warp
import io
from PIL import Image

def rotate_image(image, angle, center = None, scale = 1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


class AbstractTimeFilter:
    def __call__(self, faces: np.ndarray, frames: np.ndarray):
        raise NotImplementedError("This should be overwritten!")
    
class MedianTimeFilter(AbstractTimeFilter):
    def __init__(self):
        pass

    def __call__(self, faces: np.ndarray, frames: np.ndarray):
        median_bb = np.median(faces, axis=0)
        return np.tile(median_bb, (faces.shape[0], 1))
    
class ButterLowTimeFilter(AbstractTimeFilter):
    def __init__(self, order: int, cutoff: float, fs: float, pad_len: int = 10):
        self.pad_len = pad_len
        self.sos = butter(order, cutoff, 'low', fs=fs, output='sos')

    def __call__(self, faces: np.ndarray, frames: np.ndarray):

        pad_front = int(self.pad_len / 2)
        pad_back = self.pad_len - pad_front
        padded_faces = np.pad(faces, ((pad_front, pad_back), (0, 0)), mode='edge')

        result = signal.sosfilt(self.sos, padded_faces, axis=0)
        unpadded_result = result[pad_front:-pad_back]
        
        return unpadded_result


class AbstractFaceDetector:

    def detect_faces(self, frame: np.ndarray): 
        """
        Detect faces in the given frame.

        Args:
            frame (np.ndarray): The input image in which to detect faces.

        Returns:
            array-like[int, int, int, int]: A list of bounding boxes, where each bounding box is represented 
            as a tuple of four integers (x, y, w, h).
        """
        raise NotImplementedError("This should be overwritten!")
    
    def get_last_bounding_box(self):
        raise NotImplementedError("This should be overwritten!")

    def _rotate_bounding_box(self, bounding_box: np.ndarray, frame: np.ndarray, rotation: int):
        """
        Rotate a bounding box according to the given rotation of the frame.

        Args:
            bounding_box (np.ndarray): Array of shape (4,) representing [x, y, w, h].
            frame (np.ndarray): Original frame as a 2D or 3D array.
            rotation (int): Number of 90-degree clockwise rotations (0-3).

        Returns:
            np.ndarray: Rotated bounding box as [x, y, w, h].
        """
        x, y, w, h = bounding_box

        if rotation == 1:  # 90 degrees rotated
            x, y = y, frame.shape[1] - x - w
            w, h = h, w
        elif rotation == 2:  # 180 degrees rotated
            x, y = frame.shape[0] - x - w, frame.shape[1] - y - h
        elif rotation == 3:  # 270 degrees rotated
            x, y = frame.shape[0] - y - h, x
            w, h = h, w

        return np.array([x, y, w, h])
    
    def get_corrected_frame(self, frame: np.ndarray):
        """
        Crop the detected face(s) from the input frame.

        If a single face is detected, returns the cropped face.
        If multiple faces are detected, returns a list of cropped faces.

        Args:
            frame (np.ndarray): Input frame.

        Returns:
            np.ndarray or list[np.ndarray] or None: Cropped face(s) or None if no face detected.
        """
        faces = self.detect_faces(frame)
        if faces is None:
            return None
        
        if len(faces.shape) == 1:   # Single face
            x, y, w, h = faces
            return frame[y:y+h, x:x+w]
        elif len(faces.shape) == 2: # Multiple faces
            corrected_frames = []
            for face in faces:
                x, y, w, h = face
                corrected_frames.append(frame[y:y+h, x:x+w])
            return corrected_frames
    
    def detect_rotated_faces(self, frame: np.ndarray):
        """
        Detect faces in a frame accounting for rotations in 90-degree increments.

        The method tries 0°, 90°, 180°, and 270° rotations and returns the first valid detection,
        along with the rotation index applied.

        Args:
            frame (np.ndarray): Input frame.

        Returns:
            tuple:
                - np.ndarray or None: Detected (and rotated) bounding box(es). Shape (4,) for single face, (N, 4) for multiple faces.
                - int or None: Rotation index (0-3) that resulted in successful detection, or None if no face detected.
        """
        for i in range(4):
            faces = self.detect_faces(frame)
            if faces is not None:

                if len(faces.shape) == 1:   # Single face
                    return self._rotate_bounding_box(faces, frame, i), i
                elif len(faces.shape) == 2: # Multiple faces
                    rotated_faces = np.zeros((len(faces), 4))
                    for j in range(len(faces)):
                        rotated_faces[j] = self._rotate_bounding_box(faces[j], frame, i)
                    return rotated_faces, i
                    
            frame = np.rot90(frame)
        return None, None


class CorrelationFaceDetector(AbstractFaceDetector):
    def __init__(self, base_detector: AbstractFaceDetector, face_template: np.ndarray = None, 
                 image_transform = None,
                 angle_range = [0], 
                 scale_range = [1],
                 blur_sigma: int = 5, radius_to_search: int = 10, 
                 reset_interval: int = None, match_method: str = 'TM_CCOEFF_NORMED',
                 matching_threshold: float = 0.9):
        
        """
        Face detector that uses a template-based approach for tracking faces.

        This class wraps a base detector and refines face detection using a template correlation
        method. It can track faces across frames, handle small rotations and scaling variations,
        and optionally apply Gaussian blurring to improve robustness.

        Attributes:
            base_detector (AbstractFaceDetector): The underlying face detector used for initial detection.
            template (np.ndarray): The face template for correlation-based detection.
            last_match (np.ndarray): Last matched bounding box and transformation [x, y, angle, scale].
            radius_to_search (int): Pixels around the last match to search for the face.
            blur_sigma (int): Sigma for optional Gaussian blur applied to template and search window.
            reset_interval (int or None): Number of frames after which the template is reset automatically.
            match_method (str): OpenCV template matching method, e.g., 'TM_CCOEFF_NORMED'.
            matching_threshold (float): Threshold for accepting a template match.
            angle_range (np.ndarray): Array of angles (in degrees) to consider for rotation.
            scale_range (np.ndarray): Array of scale factors to consider for resizing.
        """
        
        self.base_detector = base_detector
        self.image_transform = image_transform
        self.radius_to_search = radius_to_search
        self.blur_sigma = blur_sigma
        self.template = None
        self.template = face_template
        self.last_match = None
        self.reset_interval = reset_interval
        self.match_method = match_method
        self.reset_counter = 0
        self.matching_threshold = matching_threshold

        self.angle_range = np.array(angle_range)
        self.scale_range = np.array(scale_range)

    def reset(self):
        self.template = None
        self.last_match = None
        self.reset_counter = 0

    def get_last_landmarks(self):
        return self.base_detector.get_last_landmarks()

    def _detect_faces_no_template(self, frame: np.ndarray):
        face, rot = self.base_detector.detect_rotated_faces(frame)

        if face is None:
            self.corrected_frame = None
            return None
        
        x, y, w, h = face

        if w == 0 or h == 0:
            self.corrected_frame = None
            return None

        self.template = frame[y:y+h, x:x+w]
        self.corrected_frame = self.template

        if self.blur_sigma > 0:
            self.template = cv2.GaussianBlur(self.template, (self.blur_sigma, self.blur_sigma), 0)

        self.last_match = np.array([x, y, 0, 1])
        return face
    
    def _get_subframe(self, frame: np.ndarray, x_offset: int = None, y_offset: int = None, rotation_angle: int = None, scale_factor: float = None, relative_to_last_match: bool = True):
        if relative_to_last_match:
            x, y, angle, scale = self.last_match
            x = int(x)
            y = int(y)
            scale = 1/scale
            angle = -angle
        else:
            x, y = 0, 0
            angle = 0
            scale = 1

        if x_offset is not None:
            x += x_offset

        if y_offset is not None:
            y += y_offset

        if rotation_angle is not None:
            angle += rotation_angle

        if scale_factor is not None:
            scale *= scale_factor

        w, h = self.template.shape[1], self.template.shape[0]
        subframe_start_x = max(0, x - self.radius_to_search)
        subframe_start_y = max(0, y - self.radius_to_search)
        subframe_end_x = min(frame.shape[1], x + w + self.radius_to_search)
        subframe_end_y = min(frame.shape[0], y + h + self.radius_to_search)
        rot_center = (int(x + w//2), int(y + h//2))

        frame = rotate_image(frame, angle, scale=scale, center=rot_center)
        subframe = frame[subframe_start_y:subframe_end_y, subframe_start_x:subframe_end_x]
        return subframe, subframe_start_x, subframe_start_y
    
    def _detect_faces_with_template(self, frame: np.ndarray):
        
        w, h = self.template.shape[1], self.template.shape[0]

        for i in range(len(self.angle_range)):
            for j in range(len(self.scale_range)):
                angle = self.angle_range[i]
                scale = self.scale_range[j]

                subframe, subframe_start_x, subframe_start_y = self._get_subframe(frame, rotation_angle=angle, scale_factor=scale)

                if i == 0 and j == 0:
                    correlations_shape_x = subframe.shape[1] - w + 1
                    correlations_shape_y = subframe.shape[0] - h + 1
                    correlations = np.zeros((correlations_shape_y, correlations_shape_x, len(self.angle_range), len(self.scale_range)))

                if self.blur_sigma > 0:
                    subframe = cv2.GaussianBlur(subframe, (self.blur_sigma, self.blur_sigma), 0)

                correlations[:, :, i, j] = cv2.matchTemplate(subframe, self.template, cv2.TM_CCOEFF_NORMED)


        max_index = np.unravel_index(np.argmax(correlations, axis=None), correlations.shape)

        angle = self.last_match[2] - self.angle_range[max_index[2]]
        scale = self.last_match[3] / self.scale_range[max_index[3]]
        x = subframe_start_x + max_index[1]
        y = subframe_start_y + max_index[0]

        self.last_match = np.array([x, y, angle, scale])
        result = np.array([x, y, w, h])

        subframe, subframe_start_x, subframe_start_y = self._get_subframe(frame, x_offset=x, y_offset=y, rotation_angle=-angle, scale_factor=1/scale, relative_to_last_match=False)
        self.corrected_frame = subframe[self.radius_to_search:-self.radius_to_search, self.radius_to_search:-self.radius_to_search]

        return result

    def detect_faces(self, frame: np.ndarray):

        self.reset_counter += 1
        if self.reset_interval is not None and (self.reset_counter >= self.reset_interval):
            self.reset()

        if self.image_transform is not None:
            frame = self.image_transform(frame)

        frame = frame.astype(np.float32)

        if self.template is None:
            result = self._detect_faces_no_template(frame)
        else:
            result = self._detect_faces_with_template(frame)
        self.last_bounding_box = result
        return result
        
    def get_last_bounding_box(self):
        return self.last_bounding_box

    def get_corrected_frame(self, frame: np.ndarray):
        faces = self.detect_faces(frame)

        if faces is None:
            return None
        
        return self.corrected_frame
            
class OpticalFlowFaceDetector(AbstractFaceDetector):
    """
    Face detector that uses optical flow to track faces across frames.

    This detector wraps a base face detector and uses either Farneback dense optical flow 
    or Lucas-Kanade sparse optical flow to estimate face movement from the previous frame. 
    This allows tracking faces even when they move slightly between frames, reducing 
    the need for repeated full detections.

    Attributes:
        face_detector (AbstractFaceDetector): Base detector used for initial face detection.
        farneback (bool): Whether to use Farneback dense optical flow. If False, uses sparse Lucas-Kanade flow.
        max_corners (int): Maximum number of features to track for sparse optical flow.
        quality_level (float): Minimum accepted quality of features for sparse optical flow.
        min_distance (int): Minimum distance between tracked features.
        win_size (tuple): Window size for optical flow calculation.
        max_level (int): Maximum pyramid levels for optical flow calculation.
        radius_to_search (int): Number of pixels to pad around the bounding box for searching.
        last_bb (np.ndarray): Last detected or tracked bounding box [x, y, w, h].
        first_bb (np.ndarray): Initial bounding box for the current tracking sequence.
        first_frame (np.ndarray): Cropped grayscale frame corresponding to the initial bounding box.
        features_to_track (np.ndarray): Tracked points for sparse optical flow.
    """

    def __init__(self, face_detector: AbstractFaceDetector, farneback=True ,max_corners: int = 100, quality_level: float = 0.01, min_distance: int = 10, win_size: tuple = (15, 15), max_level: int = 0, radius_to_search: int = 10):
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.win_size = win_size
        self.max_level = max_level
        self.farneback = farneback
        self.radius_to_search = radius_to_search

        self.face_detector = face_detector
        self.features_to_track = None
        self.last_bb = None
        self.first_bb = None
        self.first_frame = None

    def get_last_bounding_box(self):
        return self.last_bb
    
    def get_last_landmarks(self):
        return self.features_to_track
    
    def _get_subframe(self, frame: np.ndarray):
        x, y, w, h = self.last_bb
        # Calculate the coordinates with padding if necessary
        subframe_start_x = x - self.radius_to_search
        subframe_start_y = y - self.radius_to_search
        subframe_end_x = x + w + self.radius_to_search
        subframe_end_y = y + h + self.radius_to_search

        pad_left = max(0, -subframe_start_x)
        pad_top = max(0, -subframe_start_y)
        pad_right = max(0, subframe_end_x - frame.shape[1])
        pad_bottom = max(0, subframe_end_y - frame.shape[0])

        # Crop the region inside the image
        crop_x1 = max(0, subframe_start_x)
        crop_y1 = max(0, subframe_start_y)
        crop_x2 = min(frame.shape[1], subframe_end_x)
        crop_y2 = min(frame.shape[0], subframe_end_y)

        subframe = frame[crop_y1:crop_y2, crop_x1:crop_x2]

        # Pad if needed
        if any([pad_left, pad_top, pad_right, pad_bottom]):
            subframe = np.pad(
            subframe,
            ((pad_top, pad_bottom), (pad_left, pad_right)) + ((0, 0),) * (subframe.ndim - 2),
            mode='edge'
            )

        return subframe, subframe_start_x, subframe_start_y

    def detect_faces_with_template(self, frame: np.ndarray):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray, subframe_start_x, subframe_start_y = self._get_subframe(frame_gray)

        if self.farneback:
            flow = cv2.calcOpticalFlowFarneback(self.first_frame, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow = flow[self.radius_to_search:-self.radius_to_search, self.radius_to_search:-self.radius_to_search]
            displacement = np.mean(flow, axis=(0, 1))

            new_bb = np.array([self.first_bb[0] + round(displacement[0]), self.first_bb[1] + round(displacement[1]), self.first_bb[2], self.first_bb[3]])
            self.last_bb = new_bb

            return new_bb
        else:
            next_points_estimate = [[self.first_bb[0] - self.last_bb[0], self.first_bb[1] - self.last_bb[1]]] * len(self.features_to_track)
            next_points_estimate = np.array(next_points_estimate, dtype=np.float32)
            next_points_estimate = next_points_estimate.reshape(-1, 1, 2)
            next_points_estimate += self.features_to_track
        
            next_points, status, error = cv2.calcOpticalFlowPyrLK(self.first_frame, frame_gray, self.features_to_track, next_points_estimate, winSize=self.win_size, maxLevel=self.max_level, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)

            if next_points is None: 
                return None
            
            good_new = next_points[status == 1]
            good_old = self.features_to_track[status == 1]

            # Perform RANSAC to find the best affine transformation
            matrix, inliers = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC)

        if matrix is None:
            return None

        # Apply the transformation to the bounding box
        x, y, w, h = self.last_bb
        points = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]], dtype=np.float32)
        transformed_points = cv2.transform(np.array([points]), matrix)[0]

        # Compute the new bounding box from the transformed points
        x_min, y_min = np.min(transformed_points, axis=0)
        x_max, y_max = np.max(transformed_points, axis=0)

        x_new = x_min + (x_max - x_min - w) / 2
        y_new = y_min + (y_max - y_min - h) / 2
        new_bb = np.array([int(x_new), int(y_new), int(w), int(h)])

        self.last_bb = new_bb
        return new_bb
    
    def detect_faces_no_template(self, frame: np.ndarray):
        bounding_box = self.face_detector.detect_faces(frame)

        if bounding_box is None:
            return None

        self.last_bb = bounding_box
        self.first_bb = bounding_box
        self.first_frame = self._get_subframe(frame)[0]
        self.first_frame = cv2.cvtColor(self.first_frame, cv2.COLOR_BGR2GRAY)
        self.features_to_track = cv2.goodFeaturesToTrack(self.first_frame, 
                                                         maxCorners=self.max_corners, 
                                                         qualityLevel=self.quality_level, 
                                                         minDistance=self.min_distance)
        return bounding_box

    def detect_faces(self, frame: np.ndarray):
        if self.first_frame is None:
            return self.detect_faces_no_template(frame)
        else:
            return self.detect_faces_with_template(frame)


class ViolaJonesFaceDetector(AbstractFaceDetector):
    """
    Face detector based on the classic Viola-Jones (Haar Cascade) algorithm.

    This detector wraps OpenCV's CascadeClassifier to detect faces in an image. 
    It can return either a single face (the first detected) or multiple faces.

    Attributes:
        face_cascade (cv2.CascadeClassifier): OpenCV Haar cascade classifier for face detection.
        scale_factor (float): Factor by which the image size is reduced at each image scale.
        min_neighbors (int): Minimum number of neighbor rectangles that makes a detection valid.
        multi_face (bool): If True, returns all detected faces; otherwise, returns only the first detected face.
    """

    def __init__(self, face_cascade: cv2.CascadeClassifier, scale_factor: float, min_neighbors: int, multi_face: bool = False):
        self.face_cascade = face_cascade
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.multi_face = multi_face

    def __init__(self, file_path: str, scale_factor: float = 1.3, min_neighbors: int = 5, multi_face: bool = False):
        self.face_cascade = cv2.CascadeClassifier(file_path)
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.multi_face = multi_face

    def __init__(self, scale_factor: float = 1.3, min_neighbors: int = 5, multi_face: bool = False):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.multi_face = multi_face

    def detect_faces(self, frame: np.ndarray):
        frame = frame.astype(np.uint8)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors)
        
        if len(faces) == 0:
            return None

        faces = faces.astype(np.int32)

        if self.multi_face:
            return faces
        else:
            return faces[0]
        
class Yolo5FaceDetector(AbstractFaceDetector):
    """
    Face detector using a YOLOv5-based face detection model.

    This detector wraps a YOLO5Face model for detecting faces in images. 
    It converts the input image from BGR to RGB and returns bounding boxes 
    in [x, y, width, height] format.

    Attributes:
        yolo5 (YOLO5Face): YOLOv5 face detection model.
        device (str): Device to run the model on ('cuda' or 'cpu').
    """

    def __init__(self, yolo5: YOLO5Face, device: str = 'cuda'):
        if yolo5 == None:
            yolo5 = YOLO5Face(backend="Y5F", device=device)

        self.yolo5 = yolo5
        self.device = device

    def detect_faces(self, frame: np.ndarray):
        frame = frame.astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.yolo5.detect_face(frame)
        
        if result is None or len(result) == 0:
            return None
        
        x1, y1, x2, y2 = result
        bounding_box = np.array([int(x1), int(y1), int(x2 - x1), int(y2 - y1)], dtype=np.int32)
        return bounding_box

class MOSSEFaceDetector(AbstractFaceDetector):
    """
    MOSSE (Minimum Output Sum of Squared Error) based face tracker.

    This detector uses a base face detector to detect faces in the first frame 
    and then tracks them across subsequent frames using a correlation filter 
    learned with the MOSSE algorithm. The filter is updated online to adapt 
    to changes in appearance.

    Attributes:
        face_detector (AbstractFaceDetector): Base detector used for initialization.
        init_frame (np.ndarray): Grayscale frame used for initializing the tracker.
        init_bb (np.ndarray): Initial bounding box of the detected face [x, y, w, h].
        num_pretrain (int): Number of pre-training iterations to stabilize the filter.
        sigma (float): Standard deviation used in Gaussian response map.
        rotate (bool): If True, applies random warp for pre-training to handle rotations.
        lr (float): Learning rate for online filter update.
        Ai (np.ndarray): Numerator of the MOSSE filter in the frequency domain.
        Bi (np.ndarray): Denominator of the MOSSE filter in the frequency domain.
        pos (np.ndarray): Current position of the tracked face [x, y, w, h].
        clip_pos (np.ndarray): Clipped bounding box to ensure it stays within frame bounds.

    Methods:
        detect_faces(frame): Detects or tracks a face in the given frame. Returns bounding box [x, y, w, h].
        _pre_training(init_frame, G): Pre-trains the MOSSE filter using the initial frame.
        _get_gauss_response(img, gt): Generates Gaussian response map for the given bounding box.
        _detect_faces_first_frame(frame): Detects face and initializes the MOSSE filter.
        _detect_faces_intermediate_frames(frame): Tracks the face in subsequent frames and updates the filter.
    """

    def __init__(self, base_detector: AbstractFaceDetector, num_pretrain: int = 10, sigma: float = 0.1, rotate: bool = False, lr: float = 0.1):
        self.face_detector = base_detector
        self.init_frame = None
        self.init_bb = None
        self.num_pretrain = num_pretrain
        self.sigma = sigma
        self.rotate = rotate
        self.lr = lr

    # pre train the filter on the first frame...
    def _pre_training(self, init_frame, G):
        height, width = G.shape
        fi = cv2.resize(init_frame, (width, height))
        # pre-process img..
        fi = pre_process(fi)
        Ai = G * np.conjugate(np.fft.fft2(fi))
        Bi = np.fft.fft2(init_frame) * np.conjugate(np.fft.fft2(init_frame))
        for _ in range(self.num_pretrain):
            if self.rotate:
                fi = pre_process(random_warp(init_frame))
            else:
                fi = pre_process(init_frame)
            Ai = Ai + G * np.conjugate(np.fft.fft2(fi))
            Bi = Bi + np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))
        
        return Ai, Bi

    # get the ground-truth gaussian reponse...
    def _get_gauss_response(self, img, gt):
        # get the shape of the image..
        height, width = img.shape
        # get the mesh grid...
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        # get the center of the object...
        center_x = gt[0] + 0.5 * gt[2]
        center_y = gt[1] + 0.5 * gt[3]
        # cal the distance...
        dist = (np.square(xx - center_x) + np.square(yy - center_y)) / (2 * self.sigma)
        # get the response map...
        response = np.exp(-dist)
        # normalize...
        response = linear_mapping(response)
        return response

    def _detect_faces_first_frame(self, frame: np.ndarray):
        init_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        init_frame = init_frame.astype(np.float32)

        self.init_bb = self.face_detector.detect_faces(frame)
        if self.init_bb is None or self.init_bb[2] == 0 or self.init_bb[3] == 0:
            self.init_bb = None
            return None

        self.init_frame = init_frame

        response_map = self._get_gauss_response(self.init_frame, self.init_bb)
        # start to create the training set ...
        # get the goal..
        g = response_map[self.init_bb[1]:self.init_bb[1]+self.init_bb[3], self.init_bb[0]:self.init_bb[0]+self.init_bb[2]]
        fi = self.init_frame[self.init_bb[1]:self.init_bb[1]+self.init_bb[3], self.init_bb[0]:self.init_bb[0]+self.init_bb[2]]
        self.G = np.fft.fft2(g)
        # start to do the pre-training...
        self.Ai, self.Bi = self._pre_training(fi, self.G)
        self.Ai = self.lr * self.Ai
        self.Bi = self.lr * self.Bi
        self.pos = self.init_bb.copy()
        self.clip_pos = np.array([self.pos[0], self.pos[1], self.pos[0]+self.pos[2], self.pos[1]+self.pos[3]]).astype(np.int64)

        return self.init_bb

    def _detect_faces_intermediate_frames(self, frame: np.ndarray):

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = frame_gray.astype(np.float32)

        Hi = self.Ai / self.Bi
        fi = frame_gray[self.clip_pos[1]:self.clip_pos[3], self.clip_pos[0]:self.clip_pos[2]]
        fi = pre_process(cv2.resize(fi, (self.init_bb[2], self.init_bb[3])))
        Gi = Hi * np.fft.fft2(fi)
        gi = linear_mapping(np.fft.ifft2(Gi))
        # find the max pos...
        max_value = np.max(gi)
        max_pos = np.where(gi == max_value)
        dy = int(np.mean(max_pos[0]) - gi.shape[0] / 2)
        dx = int(np.mean(max_pos[1]) - gi.shape[1] / 2)
        
        # update the position...
        self.pos[0] = self.pos[0] + dx
        self.pos[1] = self.pos[1] + dy

        # trying to get the clipped position [xmin, ymin, xmax, ymax]
        self.clip_pos[0] = np.clip(self.pos[0], 0, frame.shape[1])
        self.clip_pos[1] = np.clip(self.pos[1], 0, frame.shape[0])
        self.clip_pos[2] = np.clip(self.pos[0]+self.pos[2], 0, frame.shape[1])
        self.clip_pos[3] = np.clip(self.pos[1]+self.pos[3], 0, frame.shape[0])
        self.clip_pos = self.clip_pos.astype(np.int64)

        # object to track is lost...
        if self.clip_pos[2] == 0 or self.clip_pos[3] == 0:
            print("Warning: Object to track is lost! resetting to first frame.")
            return self._detect_faces_first_frame(frame)

        # get the current fi..
        fi = frame_gray[self.clip_pos[1]:self.clip_pos[3], self.clip_pos[0]:self.clip_pos[2]]
        fi = pre_process(cv2.resize(fi, (self.init_bb[2], self.init_bb[3])))
        # online update...
        self.Ai = self.lr * (self.G * np.conjugate(np.fft.fft2(fi))) + (1 - self.lr) * self.Ai
        self.Bi = self.lr * (np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))) + (1 - self.lr) * self.Bi

        return self.pos

    def detect_faces(self, frame: np.ndarray):
        if self.init_bb is None:
            return self._detect_faces_first_frame(frame)
        else:
            return self._detect_faces_intermediate_frames(frame)


        
        
class YuNetFaceDetector(AbstractFaceDetector):
    """
    Face detector using the YuNet deep learning model.

    This detector wraps the YuNet model for face detection and optionally 
    supports detecting multiple faces, rotated faces (via transposed model), 
    and returning landmarks. Bounding boxes can be scaled using a `large_box_factor`.

    Attributes:
        yunet (YuNet): The primary YuNet model used for inference.
        yunet_shape (Tuple[int, int]): Expected input shape for the model (width, height).
        yunet_transpose (YuNet): Optional transposed model for rotated face detection.
        yunet_transpose_shape (Tuple[int, int]): Shape of the transposed model input.
        transpose (bool): Whether to enable rotated face detection using transposed model.
        return_only_bb (bool): If True, only returns bounding boxes instead of full detection results.
        multi_face (bool): Whether to detect multiple faces or only the largest face.
        last_landmarks (np.ndarray): Landmarks of the last detected face.
        last_bb (np.ndarray): Bounding box(es) of the last detected face(s).
        landmarks_in_global_coordinates (bool): If True, landmarks are in global coordinates.
        device (str): Device for running the model (e.g., 'cuda' or 'cpu').
        large_box_factor (float): Factor to enlarge the returned bounding boxes.

    Methods:
        get_last_landmarks(): Returns the landmarks of the last detected face.
        get_last_bounding_box(): Returns the bounding box of the last detected face.
        get_yunet_shape(): Returns the expected input shape of the YuNet model.
        detect_faces(frame): Detects faces in the given frame. Returns either the largest face or all faces depending on `multi_face`.
        detect_rotated_faces(frame): Detects rotated faces using the transposed model. Requires `transpose=True`.
    """
    
    def __init__(self, yunet: YuNet, device: str = 'cuda'):
        self.yunet = yunet
        self.device = device

    def __init__(self, model_path: str, input_shape: Tuple[int, int], multi_face: bool = False, 
                 transpose: bool = False, return_only_bb: bool = True, landmarks_in_global_coordinates: bool = False, 
                 device: str = 'cuda', large_box_factor: float = 1.0):
        self.yunet_shape = tuple(input_shape)
        self.yunet = YuNet(model_path, self.yunet_shape)
        self.transpose = transpose
        self.return_only_bb = return_only_bb
        self.last_landmarks = None
        self.last_bb = None
        self.landmarks_in_global_coordinates = landmarks_in_global_coordinates
        self.device = device
        self.large_box_factor = large_box_factor

        if transpose:
            self.yunet_transpose_shape = (input_shape[1], input_shape[0])
            self.yunet_transpose = YuNet(model_path, self.yunet_transpose_shape)

        self.multi_face = multi_face

    def get_last_landmarks(self):
        return self.last_landmarks
    
    def get_last_bounding_box(self):
        return self.last_bb
    
    def get_yunet_shape(self):
        return self.yunet_shape

    def detect_faces(self, frame: np.ndarray):
        frame_shape = (frame.shape[1], frame.shape[0])

        if frame_shape == self.yunet_shape:
            result = self.yunet.infer(frame)
        elif self.transpose and (frame_shape == self.yunet_transpose_shape):
            result = self.yunet_transpose.infer(frame)
        else:
            raise ValueError(f"Frame shape {frame_shape} does not match the expected shape.")
        
        result = result.astype(np.int32)

        if len(result) == 0:
            return None

        biggest_face_index = np.argmax(result[:, 2] * result[:, 3])
        
        self.last_landmarks = result[biggest_face_index, 4:-1]
        self.last_landmarks = self.last_landmarks.reshape(-1, 2)

        if not self.landmarks_in_global_coordinates:
            self.last_landmarks -= result[biggest_face_index, 0:2]

        for i in range(len(result)):
            x, y, w, h = result[i, 0:4]
            increase_w = int(w * (self.large_box_factor - 1))
            increase_h = int(h * (self.large_box_factor - 1))
            x = max(0, x - increase_w // 2)
            y = max(0, y - increase_h // 2)
            w = min(frame.shape[1], x + w + increase_w) - x
            h = min(frame.shape[0], y + h + increase_h) - y
            result[i, 0:4] = np.array([x, y, w, h])
        
        if self.return_only_bb:
            result = result[:, 0:4]

        self.last_bb = result

        if self.multi_face:
            return result
        else:
            return result[biggest_face_index]
        
    def detect_rotated_faces(self, frame):
        if not self.transpose:
            raise ValueError("Enable transpose to use rotated face detection.")
        
        return super().detect_rotated_faces(frame)

class WindowedFaceDetector(AbstractFaceDetector):
    def __init__(self, face_detector: AbstractFaceDetector, window_size: int, frame_shape: Tuple[int, int, int]):
        self.face_detector = face_detector
        self.window_size = window_size
        self.frame_shape = frame_shape
        self.frames = np.zeros((window_size, *frame_shape))
        self.faces = np.zeros((window_size, 4), dtype=np.int32)
        self.frame_counter = 0

    def add_frame(self, frame: np.ndarray):
        if len(frame.shape) == 3:
            self.__append_frame(frame)
        elif len(frame.shape) == 4:
            for i in range(frame.shape[0]):
                self.__append_frame(frame[i])

    def __append_frame(self, frame: np.ndarray):
        if frame.shape != self.frame_shape:
            raise ValueError("Frame shape does not match the expected shape.")
        if len(self.frames) > self.window_size:
            self.frames.pop(0)

        self.frames = np.roll(self.frames, 1, axis=0)
        self.frames[0] = frame
        self.faces = np.roll(self.faces, 1, axis=0)
        self.faces[0] = self.face_detector.detect_faces(frame)
        self.frame_counter += 1

    def detect_faces(self, frame_range: Tuple[int, int] = None):
        if frame_range is None:
            frame_range = (0, len(self.frames))
        return np.squeeze(self.faces[frame_range[0]:frame_range[1]])

class FilteredFaceDetector(WindowedFaceDetector):
    def __init__(self, face_detector: AbstractFaceDetector, window_size: int, frame_shape: Tuple[int, int, int], time_filter):
        super().__init__(face_detector, window_size, frame_shape)
        self.time_filter = time_filter

    def detect_faces(self, frame_range: Tuple[int, int] = None):
        if self.frame_counter < self.window_size:
            return super().detect_faces(frame_range=frame_range)
        
        time_filtered_faces = self.time_filter(self.faces, self.frames).astype(np.uint32)
        return np.squeeze(time_filtered_faces[frame_range[0]:frame_range[1]])
        

        
class LandmarkFaceDetector(AbstractFaceDetector):
    def __init__(
        self, 
        base_detector: AbstractFaceDetector,
        smoothing_window: int = 5,
        polyorder: int = 2,
        recalc_interval: int = 10,
        margin: float = 1.5
    ):
        self.base_detector = base_detector
        self.smoothing_window = smoothing_window
        self.polyorder = polyorder
        self.recalc_interval = recalc_interval
        self.margin = margin

        self.frame_count = 0
        self.landmark_history = deque(maxlen=smoothing_window)
        self.last_smoothed_landmarks = None
        self.last_bounding_box = None

        self.fixed_crop_size = None  # Will be recalculated
        self.fixed_scale = None
        self.crop_half_size = None

        self.last_corrected_frame = None

    def detect_faces(self, frame: np.ndarray):
        bounding_box = self.base_detector.detect_faces(frame)
        if bounding_box is None:
            return None

        landmarks = self.base_detector.get_last_landmarks()
        landmarks = np.array(landmarks)

        self.landmark_history.append(np.array(landmarks))
        self.frame_count += 1

        if len(self.landmark_history) < self.smoothing_window:
            return bounding_box

        # Smooth landmarks
        landmark_array = np.stack(self.landmark_history, axis=0)  # (window, 5, 2)
        smoothed = scipy.signal.savgol_filter(landmark_array, self.smoothing_window, self.polyorder, axis=0)
        smoothed_landmarks = smoothed[-1]
        self.last_smoothed_landmarks = smoothed_landmarks

        # Estimate affine transform using all 5 landmarks
        src_pts = np.float32(landmarks)
        dst_pts = np.float32(smoothed_landmarks)
        affine_matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)

        if affine_matrix is None:
            self.last_bounding_box = bounding_box
            return bounding_box

        # Apply affine transform to stabilize the frame
        stabilized_frame = cv2.warpAffine(frame, affine_matrix, (frame.shape[1], frame.shape[0]))

        # Recalculate crop box size every N frames
        if self.fixed_crop_size is None or self.frame_count % self.recalc_interval == 0:
            # Compute bounding box around smoothed landmarks
            x_coords = smoothed_landmarks[:, 0]
            y_coords = smoothed_landmarks[:, 1]
            width = (x_coords.max() - x_coords.min()) * self.margin
            height = (y_coords.max() - y_coords.min()) * self.margin
            self.fixed_crop_size = int(max(width, height))
            self.crop_half_size = self.fixed_crop_size // 2

        # Crop around the center of smoothed landmarks
        center = np.mean(smoothed_landmarks, axis=0).astype(int)
        x1 = max(center[0] - self.crop_half_size, 0)
        y1 = max(center[1] - self.crop_half_size, 0)
        x2 = min(center[0] + self.crop_half_size, frame.shape[1])
        y2 = min(center[1] + self.crop_half_size, frame.shape[0])

        for (x, y) in smoothed_landmarks.astype(int):
            cv2.circle(stabilized_frame, (x, y), 2, (0, 255, 0), -1)

        self.last_corrected_frame = stabilized_frame[y1:y2, x1:x2]

        self.last_bounding_box = bounding_box
        return bounding_box

    def get_corrected_frame(self, frame):
        self.detect_faces(frame)
        return self.last_corrected_frame

    def get_last_bounding_box(self):
        return self.last_bounding_box

    def get_last_landmarks(self):
        return self.last_smoothed_landmarks

        