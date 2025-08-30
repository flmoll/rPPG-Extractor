
from ultralytics.models.sam import SAM2VideoPredictor, SAM
from ultralytics import FastSAM
from ultralytics.engine.results import Results

import cv2
import numpy as np
import importlib
import sys
from sklearn.cluster import KMeans
import torch
from facedetection.opencv_zoo.models.face_detection_yunet.yunet import YuNet

import scipy.signal as signal
from scipy.signal import butter, lfilter
from math import ceil

import matplotlib.pyplot as plt

from facedetection.invariantTM import invariant_match_template
import io
from PIL import Image


class AbstractImageTransform:
    def __call__(self, frame: np.ndarray, landmarks=None):
        raise NotImplementedError("This should be overwritten!")

class SkinMaskTransform(AbstractImageTransform):
    def __init__(self, dilate_kernel_size: int = 5, erode_kernel_size: int = 5):
        self.dilate_kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
        self.erode_kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)

    def _get_mask(self, frame: np.ndarray, landmarks=None):
        raise NotImplementedError("This should be overwritten!")

    def get_mask(self, frame: np.ndarray, landmarks=None):
        mask = self._get_mask(frame, landmarks=landmarks)
        mask = cv2.dilate(mask, self.dilate_kernel, iterations=1)
        mask = cv2.erode(mask, self.erode_kernel, iterations=1)
        return mask
    
    def __call__(self, frame: np.ndarray, landmarks=None):
        global_mask = self.get_mask(frame)
        frame *= (global_mask[:, :, np.newaxis] / 255).astype(np.uint8)
        return frame
    
class ColorSkinMaskTransform(SkinMaskTransform):
    def __init__(self, block_skin: bool = False, use_mask_of_first_frame: bool = False, logic_and_of_all_masks: bool = False):

        if use_mask_of_first_frame and logic_and_of_all_masks:
            raise ValueError("use_mask_of_first_frame and logic_and_of_all_masks cannot be True at the same time.")

        self.block_skin = block_skin
        self.use_mask_of_first_frame = use_mask_of_first_frame
        self.logic_and_of_all_masks = logic_and_of_all_masks
        self.global_mask = None

    def _get_mask(self, frame: np.ndarray, landmarks=None):
        #converting from gbr to hsv color space
        img_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #skin color range for hsv color space 
        HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17,170,255)) 
        HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

        #converting from gbr to YCbCr color space
        img_YCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

        #skin color range for hsv color space 
        YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255,180,135)) 
        YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

        #merge skin detection (YCbCr and hsv)
        global_mask=cv2.bitwise_and(YCrCb_mask,HSV_mask)
        global_mask=cv2.medianBlur(global_mask,3)
        global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))

        if self.block_skin:
            global_mask=cv2.bitwise_not(global_mask)

        return global_mask

    def __call__(self, frame: np.ndarray, landmarks=None):
        if self.use_mask_of_first_frame:
            if self.global_mask is not None:
                global_mask = self.global_mask
            else:
                self.global_mask = self._get_mask(frame)
                global_mask = self.global_mask
        else:
            global_mask = self._get_mask(frame)

        if self.logic_and_of_all_masks:
            if self.global_mask is not None:
                self.global_mask = cv2.bitwise_and(self.global_mask, global_mask)
            else:
                self.global_mask = global_mask

        frame *= (global_mask[:, :, np.newaxis] / 255).astype(np.uint8)
        return frame

class UNetSkinMaskTransform(SkinMaskTransform):
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.model = cfg.model.to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        self.model.eval()

        self.trans = cfg.test_transform
        self.device = device

    def _get_mask(self, frame: np.ndarray, landmarks=None):
        image_shape = frame.shape
        image = self.trans(frame)
        image = image.to(self.device)
        image = self.model(image)
        mask = image > 0.5
        mask = mask.squeeze(0).cpu().numpy().transpose(1,2,0).astype(int)*255

        mask_image = np.uint8(mask)
        mask_image = cv2.resize(mask_image, image_shape[:2][::-1])

        return mask_image

class EllipsisSkinMaskTransform(SkinMaskTransform):
    def __init__(self, yunet_detector):
        self.face_detector = yunet_detector
    
    def _get_mask(self, frame, landmarks=None):

        if landmarks is None:
            bounding_box = self.face_detector.detect_faces(frame)
            landmarks = bounding_box[4:-1]
            bounding_box = bounding_box[0:4]

        landmarks = np.array(landmarks).reshape(-1, 2)

        ellipse_center1 = (landmarks[0] + landmarks[1]) / 2
        ellipse_center2 = (landmarks[3] + landmarks[4]) / 2
        center = (ellipse_center1 + ellipse_center2) / 2
        center = (int(center[0]), int(center[1]))

        axisLengthY = int(np.linalg.norm(ellipse_center1 - ellipse_center2))
        axisLengthX = int(axisLengthY * 0.5)

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.ellipse(mask, center, (axisLengthX, axisLengthY), 0, 0, 360, 255, -1)

        return mask

class MedianSkinMaskTransform(SkinMaskTransform):
    def __init__(self, threshold: float, dilate_kernel_size: int = 5, erode_kernel_size: int = 5):
        super().__init__(dilate_kernel_size=dilate_kernel_size, erode_kernel_size=erode_kernel_size)
        self.threshold = threshold

    def _get_mask(self, frame, landmarks=None):
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        median_color = np.median(frame, axis=(0, 1))
        distances = np.linalg.norm(frame - median_color, axis=2)
        mask[distances < self.threshold] = 255
        return mask

class KMeansSkinMaskTransform(SkinMaskTransform):
    def __init__(self, n_clusters: int, dilate_kernel_size: int = 5, erode_kernel_size: int = 5):
        super().__init__(dilate_kernel_size=dilate_kernel_size, erode_kernel_size=erode_kernel_size)
        self.n_clusters = n_clusters

    def _get_mask(self, frame, landmarks=None):
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        pixel_values = frame.reshape(-1, 3)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(pixel_values)
        labels = kmeans.labels_
        unique, counts = np.unique(labels, return_counts=True)
        largest_cluster = unique[np.argmax(counts)]
        mask = labels.reshape(frame.shape[:2])
        mask = np.where(mask == largest_cluster, 255, 0)
        return mask
    
class EdgeDetectionSkinMaskTransform(SkinMaskTransform):
    def __init__(self, threshold1: int, threshold2: int, area_of_interest_generator: SkinMaskTransform=None, dilate_kernel_size: int = 5, erode_kernel_size: int = 5):
        super().__init__(dilate_kernel_size=dilate_kernel_size, erode_kernel_size=erode_kernel_size)
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.area_of_interest_generator = area_of_interest_generator

    def _get_mask(self, frame, landmarks=None):
        if self.area_of_interest_generator is not None:
            area_of_interest = self.area_of_interest_generator.get_mask(frame, landmarks=landmarks)
        else:
            area_of_interest = np.ones(frame.shape[:2], dtype=np.uint8) * 255

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply Canny edge detection
        edges = cv2.Canny(gray, self.threshold1, self.threshold2)

        #edges = cv2.bitwise_and(edges, area_of_interest)

        plt.imshow(edges, cmap='gray')
        plt.savefig("/workspaces/motion-magnification/edges.png")
        plt.close()
        input("Press any key. ")

        # Find contours from edges
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea) if contours else None

        # Create an empty mask
        mask = np.zeros_like(gray)

        # Draw the largest contour as a filled mask
        if largest_contour is not None:
            cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        return mask
    
class ConnectedComponentsSkinMaskTransform(SkinMaskTransform):
    def __init__(self, connectivity: int = 8, dilate_kernel_size: int = 5, erode_kernel_size: int = 5):
        super().__init__(dilate_kernel_size=dilate_kernel_size, erode_kernel_size=erode_kernel_size)
        self.connectivity = connectivity

    def _get_mask(self, frame, landmarks=None):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        threshold = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        output = cv2.connectedComponentsWithStats(threshold, self.connectivity, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output

        unique, counts = np.unique(labels, return_counts=True)
        largest_label = unique[np.argmax(counts)]
        mask = np.where(labels == largest_label, 255, 0)
        return mask

    
class PredefinedBoxSkinTransform(SkinMaskTransform):
    def __init__(self, yunet_detector):
        self.face_detector = yunet_detector

    def _get_mask(self, frame, landmarks=None):
        bounding_box = self.face_detector.detect_faces(frame)
        landmarks = bounding_box[4:-1]
        bounding_box = bounding_box[0:4]

        landmarks = np.array(landmarks).reshape(-1, 2)
        landmarks = landmarks.tolist()

        w = landmarks[1][0] - landmarks[0][0]
        h = w // 2
        x = landmarks[0][0]
        y = int(landmarks[0][1] - h*1.1)
        forehead_box = np.array([x, y, w, h])

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)

        return mask
    
class SAM2SkinMaskTransform(SkinMaskTransform):
    def __init__(self, model_path: str, face_detector):
        self.face_detector = face_detector
        # Load a model
        #self.model = SAM(model_path)
        self.model = FastSAM("FastSAM-s.pt")

        # Display model information (optional)
        self.model.info()


    def _get_mask(self, frame: np.ndarray, landmarks=None):

        bounding_box = self.face_detector.detect_faces(frame)
        landmarks = bounding_box[4:-1]
        bounding_box = bounding_box[0:4]

        landmarks = np.array(landmarks).reshape(-1, 2)
        landmarks = landmarks.tolist()

        """
        print(frame.dtype)
        frame = cv2.resize(frame, (640, 640))
        frame = np.transpose(frame, (2, 0, 1))
        frame = np.expand_dims(frame, axis=0)
        frame = frame.astype(np.float32)
        frame = torch.from_numpy(frame)
        print(frame.shape)
        print(frame.dtype)
        print(torch.finfo(frame.dtype))
        """
        # Convert the frame to a PIL image
        pil_image = Image.fromarray(frame)

        # Save the image to a temporary file on the filesystem
        temp_file_path = "/tmp/temp_image.png"
        pil_image.save(temp_file_path)
        results = self.model(temp_file_path, bboxes=[bounding_box.tolist()], points=landmarks)#, texts="Segment my facial skin"
        frame = results[0].plot()
        # Draw the bounding box on the frame
        x, y, w, h = bounding_box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        for landmark in landmarks:
            x, y = landmark
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        pil_image = Image.fromarray(frame)
        pil_image.save("/workspaces/motion-magnification/temp_image.png")
        exit(0)
