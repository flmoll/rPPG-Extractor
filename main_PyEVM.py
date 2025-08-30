from PyEVM.video_reader import FlatbufferVideoReader
from PyEVM.magnifyfactory import MagnifyFactory

from PyEVM.metadata import MetaData
from PyEVM.mode import Mode
from PyEVM.filtertype import FilterType
from PyEVM.colorspace import Colorspace
from PyEVM.util import plot_image_as_channels
import os
import sys
import cv2
import numpy as np
import gc
import importlib
from matplotlib import pyplot as plt
from tqdm import tqdm
from math import ceil
from dataset.facedetection import facedetectors
import logging
import shutil
import time
from evaluation.utils import get_rppg_toolbox_conversion_dict

from vitallens import Method as VitalLensMethod
import random


def detect_chroma_subsampling(frame):
    frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(frame_yuv)

    counter_right = 0
    counter_up = 0
    counter_down = 0
    counter_diag_down = 0
    counter_diag_up = 0

    for i in range(0, len(u), 2):
        for j in range(0, len(u[0]), 2):
            if u[i, j] == u[i, j+1] and v[i, j] == v[i, j+1]:
                counter_right += 1
            if u[i, j] == u[i+1, j] and v[i, j] == v[i+1, j]:
                counter_down += 1
            if u[i, j] == u[i+1, j+1] and v[i, j] == v[i+1, j+1]:
                counter_up += 1
            if u[i, j] == u[i+1, j+1] and v[i, j] == v[i+1, j+1]:
                counter_diag_down += 1
            if u[i, j] == u[i+1, j-1] and v[i, j] == v[i+1, j-1]:
                counter_diag_up += 1

    print("pixels with same value: ", counter_right, counter_down, counter_up, counter_diag_down, counter_diag_up)
    print("total pixel comparisons: ", int(len(u) * len(u[0]) / 4))

gt_file_name = None
conversion_dict = get_rppg_toolbox_conversion_dict()

video_file_name = "/mnt/data/vitalVideos/5fd35b20292842e3b552bafd83726fcf_2.mp4"

model_path_yunet = "opencv_zoo/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

#model_path_unet_skin_segmentation = "/workspaces/motion-magnification/SkinSegmentation/pretrained/Unet-epoch_25.pth"
#model_path_sam2 = "/workspaces/motion-magnification/sam2/checkpoints/sam2.1_hiera_tiny.pt"
model_path_sam2 = "sam2.1_t.pt"#"/workspaces/motion-magnification/sam2/checkpoints/sam2.1_t.pt"n
global_output_folder = "/mnt/results"
deepstab_preprocessed_folder = "/mnt/results/deepstab"


max_frame = np.inf
frame_offset = 0
samples_per_batch = 160
processing_windows_overlap = 0

mode = Mode.COLOR
vitallens_method = VitalLensMethod.VITALLENS
log_level = logging.WARNING
load_existing_data = False
skip_existing = False
vital_videos = True

rotate = None   # could be for example cv2.ROTATE_90_COUNTERCLOCKWISE

meta_data = MetaData(
    file_name=video_file_name,
    output_folder=os.path.join(global_output_folder, "preprocessed"),
    overlay_with_orig_vid=True,
    save_filtered_tensor=True,
    save_mask=True,
    max_frame=max_frame,
    frame_offset=frame_offset,
    samples_per_batch=samples_per_batch,
    processing_windows_overlap=processing_windows_overlap,
    mode=mode,
    vitallens_method=vitallens_method,
    color_space=Colorspace.NTSC,
    suffix="out",
    filter_type=FilterType.IDEAL,
    filter_order=3,
    low=1.0,
    high=1.4,
    levels=4,
    orientations=4,
    sigma=5,
    lambda_cutoff=16,
    filters_per_octave=2,
    transition_width=0.5,
    amplification=[40, 0, 0]
)

logger = logging.getLogger()
logger.setLevel(log_level)
logger.addHandler(logging.StreamHandler())

work = MagnifyFactory.from_metadata(meta_data, logger, vital_videos=vital_videos)

# Seed the random number generator for reproducibility
random.seed(42)
np.random.seed(42)

destabiliser = None # could be for example destabilisers.DeepstabDestabilizer(deepstab_preprocessed_folder, noise_amplitude=10.0, random=True)


face_cascade = facedetectors.ViolaJonesFaceDetector()
yunet_detector = facedetectors.YuNetFaceDetector(model_path_yunet, [1920, 1200], transpose=True, return_only_bb=True, large_box_factor=1.0, landmarks_in_global_coordinates=True)

skinMaskTransform = None    # could be used to do skin segmentation. examples are given below:
#skinMaskTransform = facedetectors.ColorSkinMaskTransform(logic_and_of_all_masks=False)
#skinMaskTransform = UNetSkinMaskTransform(model_path_unet_skin_segmentation)
#skinMaskTransform = SAM2SkinMaskTransform(model_path_sam2, yunet_detector)
#skinMaskTransform = PredefinedBoxSkinTransform(yunet_detector_landmarks)
#skinMaskTransform = EllipsisSkinMaskTransform(yunet_detector)
#skinMaskTransform = skinSegmentation.MedianSkinMaskTransform(curr_threshold)
#skinMaskTransform = KMeansSkinMaskTransform(3)
#skinMaskTransform = EdgeDetectionSkinMaskTransform(0, 120, skinMaskTransform)
#skinMaskTransform = ConnectedComponentsSkinMaskTransform(connectivity=4)


if not isinstance(video_file_name, list):
    video_file_name = [video_file_name]

if gt_file_name is not None and not isinstance(gt_file_name, list):
    gt_file_name = [gt_file_name]

if gt_file_name is None:
    gt_file_name = [None] * len(video_file_name)
    
print(f"Evaluating {len(video_file_name)} videos")

for i in tqdm(range(len(video_file_name)), desc="Processing videos"):
    curr_video_file_name = video_file_name[i]
    curr_gt_file_name = gt_file_name[i]
    logger.info(f"Processing video {i+1} of {len(video_file_name)}: {curr_video_file_name}")

    correlation_face_detector = facedetectors.CorrelationFaceDetector(yunet_detector, image_transform=None)

    result = work.do_magnify(curr_video_file_name, gt_file_name=curr_gt_file_name, frame_offset=frame_offset, 
                    max_frame=max_frame, face_detector=correlation_face_detector, destabiliser=destabiliser,
                    skinMaskTransform=skinMaskTransform, load_existing_data=load_existing_data, skip_existing=skip_existing, rotate=rotate)
    
    if result == "skipped":
        logger.info(f"Video {curr_video_file_name} already processed, skipping.")
        continue

    curr_output_folder = os.path.join(global_output_folder, "temp", os.path.basename(curr_video_file_name))
    
    os.makedirs(curr_output_folder, exist_ok=True)

    ppg_prediction_source_file = work.get_out_file_name(curr_video_file_name, "ppg_prediction.bin")
    ppg_prediction_dest_file = os.path.join(curr_output_folder, "ppg_prediction.bin")
    shutil.copyfile(ppg_prediction_source_file, ppg_prediction_dest_file)

    magnified_tensor_source_file = work.get_out_file_name(curr_video_file_name, "magnified.mp4")
    magnified_tensor_dest_file = os.path.join(curr_output_folder, "magnified.mp4")
    shutil.copyfile(magnified_tensor_source_file, magnified_tensor_dest_file)

    metadata_source_file = work.get_out_file_name(curr_video_file_name, "metadata.json")
    metadata_dest_file = os.path.join(curr_output_folder, "metadata.json")
    shutil.copyfile(metadata_source_file, metadata_dest_file)

    logger.info(f"Copied PPG prediction to {ppg_prediction_dest_file}")

