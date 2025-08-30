"""The Base Class for data-loading.

Provides a pytorch-style data-loader for end-to-end training pipelines.
Extend the class to support specific datasets.
Dataset already supported: UBFC-rPPG, PURE, SCAMPS, BP4D+, and UBFC-PHYS.

"""
import csv
import gc
import glob
import json
import os
import re
from math import ceil, floor
from matplotlib import pyplot as plt
from scipy import signal
from scipy import sparse
from unsupervised_methods.methods import POS_WANG
from unsupervised_methods import utils
import math
import multiprocessing as mp
from dataset.facedetection.yolo5.YOLO5Face import YOLO5Face

# To be used only for preparing data - for detecting face with YOLO5Face
try:
    mp.set_start_method('spawn', force=True)
    # print("spawned")
except RuntimeError:
    pass

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
#from retinaface import RetinaFace   # Source code: https://github.com/serengil/retinaface
from dataset.data_loader.video_transforms.videoTransforms import VideoTransformFactory
from dataset.data_loader.compression import PreprocessingCompressorFactory, NotEnoughDiskSpaceError
from neural_methods.model.PhysNetLifeness import PhysNet_Lifeness
from neural_methods.model.PhysNetQuantile import PhysNet_Quantile
from neural_methods.model.PhysNetUncertainty import PhysNet_Uncertainty

from dataset.facedetection import facedetectors
from dataset.artificial_destabilisation.destabilisers import GaussianNoiseDestabilizer, DeepstabDestabilizer, H264CompressionDestabiliser, RandomAffineDestabilizer
from tqdm import tqdm

from evaluation.utils import postprocess_rppg

import json
import shutil


MODEL_PATH_YUNET = "opencv_zoo/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"


def read_wave(bvp_file):
    """Reads a bvp signal file."""
    with open(bvp_file, "r") as f:
        str1 = f.read()
        str1 = str1.split("\n")
        bvp = [float(x) for x in str1[0].split()]
    return np.asarray(bvp)

def get_oldest_folder(parent_dir):
    """Finds the oldest folder in the given parent directory."""
    folders = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
    if not folders:
        return None
    return min(folders, key=os.path.getctime)

class PhysNetUncertainty_Wrapper(torch.nn.Module):
    def __init__(self, model):
        super(PhysNetUncertainty_Wrapper, self).__init__()
        self.model = model

    def forward(self, x):
        rPPG, uncertainty = self.model(x)
        return rPPG, uncertainty
    
class PhysNetQuantile_Wrapper(torch.nn.Module):
    def __init__(self, model):
        super(PhysNetQuantile_Wrapper, self).__init__()
        self.model = model

    def forward(self, x):
        lower, upper = self.model(x)
        rPPG = (lower + upper) / 2
        uncertainty = (upper - lower) / 2
        return rPPG, uncertainty


class BaseLoader(Dataset):
    """The base class for data loading based on pytorch Dataset.

    The dataloader supports both providing data for pytorch training and common data-preprocessing methods,
    including reading files, resizing each frame, chunking, and video-signal synchronization.
    """

    @staticmethod
    def add_data_loader_args(parser):
        """Adds arguments to parser for training process"""
        parser.add_argument(
            "--cached_path", default=None, type=str)
        parser.add_argument(
            "--preprocess", default=None, action='store_true')
        return parser

    def __init__(self, dataset_name, raw_data_path, config_data, device=None):
        """Inits dataloader with lists of files.

        Args:
            dataset_name(str): name of the dataloader.
            raw_data_path(string): path to the folder containing all data.
            config_data(CfgNode): data settings(ref:config.py).
        """
        self.inputs = list()
        self.labels = list()
        self.dataset_name = dataset_name
        self.raw_data_path = raw_data_path
        self.cached_path = config_data.CACHED_PATH
        self.file_list_path = config_data.FILE_LIST_PATH
        self.preprocessed_data_len = 0
        self.data_format = config_data.DATA_FORMAT
        self.do_preprocess = config_data.DO_PREPROCESS
        self.config_data = config_data
        self.device = device
        self.yunetObj = None

        self.transforms = []
        for transform in config_data.DATA_AUG:
            self.transforms.append(VideoTransformFactory.create_transform(transform, device=device, type_to_be_used=self.config_data.PREPROCESS.CONVERT_TO_TYPE))

        assert (config_data.BEGIN < config_data.END)
        assert (config_data.BEGIN > 0 or config_data.BEGIN == 0)
        assert (config_data.END < 1 or config_data.END == 1)
        
        self.data_compressor = PreprocessingCompressorFactory.create_compressor(config_data.COMPRESSION_FORMAT)

        if config_data.DO_PREPROCESS:
            self.raw_data_dirs = self.get_raw_data(self.raw_data_path)
            self.preprocess_dataset(self.raw_data_dirs, config_data.PREPROCESS, config_data.BEGIN, config_data.END)
        
        if not os.path.exists(self.cached_path):
            print('CACHED_PATH:', self.cached_path)
            raise ValueError(self.dataset_name,
                                'Please set DO_PREPROCESS to True. Preprocessed directory does not exist!')
        
        self.uncompress_cached_data()

        print(f"file list path: {self.file_list_path}")
        print(f"path exists: {os.path.exists(self.file_list_path)}")
        #if not os.path.exists(self.file_list_path):
        print('File list does not exist... generating now...')
        self.raw_data_dirs = self.get_raw_data(self.raw_data_path)
        self.build_file_list_retroactive(self.raw_data_dirs, config_data.BEGIN, config_data.END)
        print('File list generated.', end='\n\n')

        self.load_preprocessed_data()

        print('Cached Data Path', self.cached_path, end='\n\n')
        print('File List Path', self.file_list_path)
        print(f" {self.dataset_name} Preprocessed Dataset Length: {self.preprocessed_data_len}", end='\n\n')

        data_dirs_subset = self.split_raw_data(self.raw_data_dirs, config_data.BEGIN, config_data.END)
        self.peak_data, self.patient_hrs = self.read_peak_data(data_dirs_subset)  # read peak data from json file

        if self.config_data.PPG_INFERENCE_MODEL.NAME == 'PhysNetUncertainty':
            inference_model = PhysNet_Uncertainty(frames=self.config_data.PPG_INFERENCE_MODEL.PHYSNET.FRAME_NUM)
            inference_model.load_state_dict(torch.load(self.config_data.PPG_INFERENCE_MODEL.MODEL_PATH, map_location=self.device))
            self.inference_model = PhysNetUncertainty_Wrapper(inference_model.to(self.device))
        elif self.config_data.PPG_INFERENCE_MODEL.NAME == 'PhysNetQuantile':
            inference_model = PhysNet_Quantile(frames=self.config_data.PPG_INFERENCE_MODEL.PHYSNET.FRAME_NUM)
            inference_model.load_state_dict(torch.load(self.config_data.PPG_INFERENCE_MODEL.MODEL_PATH, map_location=self.device))
            self.inference_model = PhysNetQuantile_Wrapper(inference_model.to(self.device))
        elif self.config_data.PPG_INFERENCE_MODEL.NAME == '':
            self.inference_model = None
        else:
            raise ValueError(f"Unsupported PPG Inference Model: {self.config_data.PPG_INFERENCE_MODEL.NAME}")
        
        if self.inference_model is not None:
            self.rppg_buffer_file = f"rppg_buffers/{self.config_data.PPG_INFERENCE_MODEL.POSTPROCESS}/{self.config_data.PPG_INFERENCE_MODEL.MODEL_PATH}/rppg_buffer_{self.config_data.DATASET}_{self.config_data.BEGIN}_{self.config_data.END}.npy"
            self.inference_model.eval()

            os.makedirs(os.path.dirname(self.rppg_buffer_file), exist_ok=True)
            print(f"rPPG buffer file path: {self.rppg_buffer_file}")

            if not os.path.exists(self.rppg_buffer_file):
                self._preprocess_rppg_buffer()

            print("Loading rPPG buffer from file...")
            self.rppg_buffer = np.load(self.rppg_buffer_file)
            print(self.rppg_buffer.shape, self.rppg_buffer.dtype)

    def uncompress_cached_data(self):
        if self.config_data.CACHE_UNCOMPRESSED is not None and self.data_compressor is not None:
            uncompress_path = os.path.join(self.config_data.CACHE_UNCOMPRESSED, os.path.basename(self.cached_path))
            print('Uncompressing cached data...')
            print(f"Cached path: {self.cached_path}")
            print(f"Uncompressing to {uncompress_path}")

            while True:
                try:
                    self.data_compressor.decompress(self.cached_path, out_path=uncompress_path, delete_old=False)
                    break
                except NotEnoughDiskSpaceError as e:
                    print(f"Error not enough disk space")
                    print("Attempting to free up disk space...")
                    # Find the oldest folder in the cached_path directory and remove it
                    oldest_folder = get_oldest_folder(self.config_data.CACHE_UNCOMPRESSED)
                    if oldest_folder:
                        print(f"Removing oldest folder to free space: {oldest_folder}")
                        shutil.rmtree(oldest_folder)
                    else:
                        raise RuntimeError("No folders left to delete for freeing disk space.")

            self.cached_path = uncompress_path
            self.data_compressor = None

    def _preprocess_rppg_buffer(self):
        self.rppg_buffer = [None] * len(self.inputs)  # Initialize rPPG buffer with None for each input
        for index in tqdm(range(len(self.inputs)), desc="Preprocessing rPPG buffer"):
            data_dict = self.__getitem__(index)[0]  # Get the data dictionary for the current index
            data_unsqueezed = np.expand_dims(data_dict["data"], axis=0)  # Add batch dimension
            data_unsqueezed = torch.from_numpy(data_unsqueezed).to(self.device)  # Move to GPU if available
            
            rppg, uncertainty = self.inference_model(data_unsqueezed[:, :3, :, :, :])  # Run inference and move back to CPU
            
            rppg = rppg.detach().cpu().numpy()[0, :]
            uncertainty = uncertainty.detach().cpu().numpy()[0, :]
            
            curr_rppg = postprocess_rppg(rppg, self.config_data.PPG_INFERENCE_MODEL.POSTPROCESS, fs=self.config_data.FS)
            self.rppg_buffer[index] = [curr_rppg, uncertainty]  # Postprocess the rPPG signal
            
            del data_unsqueezed
            gc.collect()
            torch.cuda.empty_cache()

        np.save(self.rppg_buffer_file, np.array(self.rppg_buffer, dtype=np.float32))


    def _load_preprocessed_file(self, file_path):
        if self.data_compressor:
            data = self.data_compressor.load_compressed_data(file_path)
        else:
            data = np.load(file_path)

        return data
    
    def _save_preprocessed_file(self, file_path, data):
        if self.data_compressor:
            file_path = self.data_compressor.get_compressed_data_path(file_path)
            self.data_compressor.save_compressed_data(data, file_path)
        else:
            np.save(file_path, data)

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.inputs)

    def __getitem__(self, index):
        """Returns a clip of video(3,T,W,H) or a predicted rppg signal with its uncertainty and it's corresponding signals(T)."""

        data_dict = dict()
        # item_path is the location of a specific clip in a preprocessing output folder
        # For example, an item path could be /home/data/PURE_SizeW72_...unsupervised/501_input0.npy
        item_path = self.inputs[index]
        # item_path_filename is simply the filename of the specific clip
        # For example, the preceding item_path's filename would be 501_input0.npy
        item_path_filename = item_path.split(os.sep)[-1]
        # split_idx represents the point in the previous filename where we want to split the string 
        # in order to retrieve a more precise filename (e.g., 501) preceding the chunk (e.g., input0)
        split_idx = item_path_filename.rindex('_')
        # Following the previous comments, the filename for example would be 501
        filename = item_path_filename[:split_idx]
        # chunk_id is the extracted, numeric chunk identifier. Following the previous comments, 
        # the chunk_id for example would be 0
        chunk_id = item_path_filename[split_idx + 6:].split('.')[0]

        gt_hr = float(self.patient_hrs[index])
        peak_indices = np.array(self.peak_data[index])
        label = self._load_preprocessed_file(self.labels[index])

        if self.inference_model is not None and self.rppg_buffer[index] is not None:
            data_dict["rppg"] = self.rppg_buffer[index][0]
            data_dict["uncertainty"] = self.rppg_buffer[index][1]
            data_dict["gt_hr"] = gt_hr
            data_dict["label"] = label
            return data_dict, filename, chunk_id

        data = self._load_preprocessed_file(self.inputs[index]) #np.load(self.inputs[index])

        data_dict["shuffled"] = np.array([0], dtype=np.float32)
        info = {"factor": 0.0}

        if len(self.transforms) > 0:
            data = torch.from_numpy(data)
            data = data.to(device=self.device)  # Move to GPU if available

            label = torch.from_numpy(label) # leave on CPU

            if data.shape[3] == 6:
                data[:, :, :, 3:] = (data[:, :, :, 3:] - torch.min(data[:, :, :, 3:])) / (torch.max(data[:, :, :, 3:]) - torch.min(data[:, :, :, 3:]))
                data[:, :, :, :3] = (data[:, :, :, :3] - torch.min(data[:, :, :, :3])) / (torch.max(data[:, :, :, :3]) - torch.min(data[:, :, :, :3]))
            else:
                data = (data - torch.min(data)) / (torch.max(data) - torch.min(data))

            for transform in self.transforms:
                data, label, info = transform.apply(data, label)
                if "shuffled" in info:
                    data_dict["shuffled"] = np.array([1 if info["shuffled"] else 0], dtype=np.float32)

            data = data.cpu().numpy()
            label = label.numpy()

            if data.shape[3] == 6:
                data[:, :, :, 3:] = self.standardized_data(data[:, :, :, 3:])
                data[:, :, :, :3] = self.standardized_data(data[:, :, :, :3])
            else:
                data = self.standardized_data(data)

        if self.config_data.PREPROCESS.CONVERT_TO_TYPE == 'DiffNormalized':
            data = data[:, :, :, :3]
        elif self.config_data.PREPROCESS.CONVERT_TO_TYPE == 'Standardized':
            data = data[:, :, :, 3:]

        if self.config_data.PREPROCESS.RESIZE.RESIZE_IMAGE_AFTER_PREPROCESS_RESOLUTION != None:
            new_size = tuple(self.config_data.PREPROCESS.RESIZE.RESIZE_IMAGE_AFTER_PREPROCESS_RESOLUTION)
            #print("Resizing to: ", new_size)
            data_new = np.zeros((data.shape[0], new_size[0], new_size[1], data.shape[3]), dtype=data.dtype)
            for i in range(len(data)):
                for j in range(len(data[0, 0, 0, :])):
                    data_new[i, :, :, j] = cv2.resize(data[i, :, :, j], new_size, interpolation=cv2.INTER_CUBIC)

            data = data_new

        if self.config_data.PREPROCESS.LABEL_TYPE == 'Standardized':
            if self.config_data.PREPROCESS.CONVERT_LABEL_TO_TYPE == 'DiffNormalized':
                label = self.standardized_to_diff_normalized_label(label)
        elif self.config_data.PREPROCESS.LABEL_TYPE == 'DiffNormalized':
            if self.config_data.PREPROCESS.CONVERT_LABEL_TO_TYPE == 'Standardized':
                label = self.diff_normalized_to_standardized_label(label)


        if self.data_format == 'NDCHW':
            data = np.transpose(data, (0, 3, 1, 2))
        elif self.data_format == 'NCDHW':
            data = np.transpose(data, (3, 0, 1, 2))
        elif self.data_format == 'NDHWC':
            pass
        else:
            raise ValueError('Unsupported Data Format!')

        data = np.float32(data)
        label = np.float32(label)

        data_dict["data"] = data
        data_dict["label"] = label
        #data_dict["peak_indices"] = peak_indices       # Uncomment this line if you want to include peak_indices. Might be used for heart rate variability
        data_dict["gt_hr"] = gt_hr
        return data_dict, filename, chunk_id

    def get_raw_data(self, raw_data_path):
        """Returns raw data directories under the path.

        Args:
            raw_data_path(str): a list of video_files.
        """
        raise Exception("'get_raw_data' Not Implemented")

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values, 
        and ensures no overlapping subjects between splits.

        Args:
            data_dirs(List[str]): a list of video_files.
            begin(float): index of begining during train/val split.
            end(float): index of ending during train/val split.
        """
        raise Exception("'split_raw_data' Not Implemented")

    def read_npy_video(self, video_file):
        """Reads a video file in the numpy format (.npy), returns frames(T,H,W,3)"""
        frames = self._load_preprocessed_file(video_file[0]) #np.load(video_file[0])
        if np.issubdtype(frames.dtype, np.integer) and np.min(frames) >= 0 and np.max(frames) <= 255:
            processed_frames = [frame.astype(np.uint8)[..., :3] for frame in frames]
        elif np.issubdtype(frames.dtype, np.floating) and np.min(frames) >= 0.0 and np.max(frames) <= 1.0:
            processed_frames = [(np.round(frame * 255)).astype(np.uint8)[..., :3] for frame in frames]
        else:
            raise Exception(f'Loaded frames are of an incorrect type or range of values! '\
            + f'Received frames of type {frames.dtype} and range {np.min(frames)} to {np.max(frames)}.')
        return np.asarray(processed_frames)

    def generate_pos_psuedo_labels(self, frames, fs=30):
        """Generated POS-based PPG Psuedo Labels For Training

        Args:
            frames(List[array]): a video frames.
            fs(int or float): Sampling rate of video
        Returns:
            env_norm_bvp: Hilbert envlope normalized POS PPG signal, filtered are HR frequency
        """

        # generate POS PPG signal
        WinSec = 1.6
        RGB = POS_WANG._process_video(frames)
        N = RGB.shape[0]
        H = np.zeros((1, N))
        l = math.ceil(WinSec * fs)

        for n in range(N):
            m = n - l
            if m >= 0:
                Cn = np.true_divide(RGB[m:n, :], np.mean(RGB[m:n, :], axis=0))
                Cn = np.mat(Cn).H
                S = np.matmul(np.array([[0, 1, -1], [-2, 1, 1]]), Cn)
                h = S[0, :] + (np.std(S[0, :]) / np.std(S[1, :])) * S[1, :]
                mean_h = np.mean(h)
                for temp in range(h.shape[1]):
                    h[0, temp] = h[0, temp] - mean_h
                H[0, m:n] = H[0, m:n] + (h[0])

        bvp = H
        bvp = utils.detrend(np.mat(bvp).H, 100)
        bvp = np.asarray(np.transpose(bvp))[0]

        # filter POS PPG w/ 2nd order butterworth filter (around HR freq)
        # min freq of 0.7Hz was experimentally found to work better than 0.75Hz
        min_freq = 0.70
        max_freq = 3
        b, a = signal.butter(2, [(min_freq) / fs * 2, (max_freq) / fs * 2], btype='bandpass')
        pos_bvp = signal.filtfilt(b, a, bvp.astype(np.double))

        # apply hilbert normalization to normalize PPG amplitude
        analytic_signal = signal.hilbert(pos_bvp) 
        amplitude_envelope = np.abs(analytic_signal) # derive envelope signal
        env_norm_bvp = pos_bvp/amplitude_envelope # normalize by env

        return np.array(env_norm_bvp) # return POS psuedo labels
    
    def preprocess_dataset(self, data_dirs, config_preprocess, begin, end):
        """Parses and preprocesses all the raw data based on split.

        Args:
            data_dirs(List[str]): a list of video_files.
            config_preprocess(CfgNode): preprocessing settings(ref:config.py).
            begin(float): index of begining during train/val split.
            end(float): index of ending during train/val split.
        """
        data_dirs_split = self.split_raw_data(data_dirs, begin, end)  # partition dataset 
        # send data directories to be processed
        file_list_dict = self.multi_process_manager(data_dirs_split, config_preprocess) 
        self.build_file_list(file_list_dict)  # build file list
        self.load_preprocessed_data()  # load all data and corresponding labels (sorted for consistency)
        print("Total Number of raw files preprocessed:", len(data_dirs_split), end='\n\n')

    def preprocessing_convert_data_type(self, frames, bvps, config_preprocess):
        # Check data transformation type
        data = list()  # Video data
        for data_type in config_preprocess.DATA_TYPE:
            f_c = frames.copy()
            if data_type == "Raw":
                f_c = f_c.astype(np.float32)
                f_c /= 255.0
                data.append(f_c)
            elif data_type == "DiffNormalized":
                data.append(BaseLoader.diff_normalize_data(f_c))
            elif data_type == "Standardized":
                data.append(BaseLoader.standardized_data(f_c))
            else:
                raise ValueError("Unsupported data type!")
        data = np.concatenate(data, axis=-1)  # concatenate all channels
        if config_preprocess.LABEL_TYPE == "Raw":
            pass
        elif config_preprocess.LABEL_TYPE == "DiffNormalized":
            bvps = BaseLoader.diff_normalize_label(bvps)
        elif config_preprocess.LABEL_TYPE == "Standardized":
            bvps = BaseLoader.standardized_label(bvps)
        else:
            raise ValueError("Unsupported label type!")

        if config_preprocess.DO_CHUNK:  # chunk data into snippets
            frames_clips, bvps_clips = self.chunk(
                data, bvps, config_preprocess.CHUNK_LENGTH)
        else:
            frames_clips = np.array([data])
            bvps_clips = np.array([bvps])

        return frames_clips, bvps_clips


    def preprocess(self, frames, bvps, config_preprocess, additional_size=None):
        """Preprocesses a pair of data.

        Args:
            frames(np.array): Frames in a video.
            bvps(np.array): Blood volumne pulse (PPG) signal labels for a video.
            config_preprocess(CfgNode): preprocessing settings(ref:config.py).
        Returns:
            frame_clips(np.array): processed video data by frames
            bvps_clips(np.array): processed bvp (ppg) labels by frames
        """

        destabilizer = None

        if config_preprocess.ARTIFICIAL_DESTABILISATION_BACKEND == "GAUSSIAN_NOISE":
            destabilizer = GaussianNoiseDestabilizer(config_preprocess.ARTIFICIAL_DESTABILISATION_AMPLITUDE)
        elif config_preprocess.ARTIFICIAL_DESTABILISATION_BACKEND == "RANDOM_AFFINE":
            destabilizer = RandomAffineDestabilizer()
        elif config_preprocess.ARTIFICIAL_DESTABILISATION_BACKEND == "DEEPSTAB":
            destabilizer = DeepstabDestabilizer(config_preprocess.DEEPSTAB_PREPROCESSED_PATH, config_preprocess.ARTIFICIAL_DESTABILISATION_AMPLITUDE, random=True)
        elif config_preprocess.ARTIFICIAL_DESTABILISATION_BACKEND == "H264":
            destabilizer = H264CompressionDestabiliser(crf=int(config_preprocess.ARTIFICIAL_DESTABILISATION_AMPLITUDE))     # reusing the parameter from before because otherwise the path will be messy
        elif config_preprocess.ARTIFICIAL_DESTABILISATION_BACKEND == "H264_QP":
            destabilizer = H264CompressionDestabiliser(crf=int(config_preprocess.ARTIFICIAL_DESTABILISATION_AMPLITUDE), use_constant_qp=True)
        elif config_preprocess.ARTIFICIAL_DESTABILISATION_BACKEND == "H265_QP":
            destabilizer = H264CompressionDestabiliser(crf=int(config_preprocess.ARTIFICIAL_DESTABILISATION_AMPLITUDE), vcodec="libx265", use_constant_qp=True)

        if destabilizer is not None:
            # Apply the destabilizer to the frames
            frames = destabilizer.destabilise(frames, keep_frame_dims=True)

        # resize frames and crop for face region
        frames_resized = self.crop_face_resize(
            frames,
            config_preprocess.CROP_FACE.DO_CROP_FACE,
            config_preprocess.CROP_FACE.USE_LARGE_FACE_BOX,
            config_preprocess.CROP_FACE.LARGE_BOX_COEF,
            config_preprocess.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION,
            config_preprocess.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY,
            config_preprocess.CROP_FACE.DETECTION.USE_MEDIAN_FACE_BOX,
            config_preprocess.RESIZE.W, 
            config_preprocess.RESIZE.H)
        
        frames_clips, bvps_clips = self.preprocessing_convert_data_type(frames_resized, bvps, config_preprocess)

        # quick and dirty fix to speedup preprocessing
        if additional_size is not None:
            frames_additional_list = []
            bvps_additional_list = []

            for curr_size in additional_size:
                if not isinstance(curr_size, tuple) or len(curr_size) != 2:
                    raise ValueError("additional_size must be a list of tuples with two elements (width, height).")
                
                frames_additional = self.crop_face_resize(
                    frames,
                    config_preprocess.CROP_FACE.DO_CROP_FACE,
                    config_preprocess.CROP_FACE.USE_LARGE_FACE_BOX,
                    config_preprocess.CROP_FACE.LARGE_BOX_COEF,
                    config_preprocess.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION,
                    config_preprocess.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY,
                    config_preprocess.CROP_FACE.DETECTION.USE_MEDIAN_FACE_BOX,
                    curr_size[0],
                    curr_size[1])
                
                frames_additional_clips, bvps_additional_clips = self.preprocessing_convert_data_type(frames_additional, bvps, config_preprocess)
                frames_additional_list.append(frames_additional_clips)
                bvps_additional_list.append(bvps_additional_clips)

            return frames_clips, bvps_clips, frames_additional_list, bvps_additional_list

        return frames_clips, bvps_clips

    def face_detection(self, frame, use_larger_box=False, larger_box_coef=1.0):
        """Face detection on a single frame.

        Args:
            frame(np.array): a single frame.
            backend(str): backend to utilize for face detection.
            use_larger_box(bool): whether to use a larger bounding box on face detection.
            larger_box_coef(float): Coef. of larger box.
        Returns:
            face_box_coor(List[int]): coordinates of face bouding box.
        """
        frame_for_detector = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2BGR)
        result = self.face_crop_backend.detect_faces(frame_for_detector)

        if result is not None:
            x, y, width, height = result

            # Find the center of the face zone
            center_x = x + width // 2
            center_y = y + height // 2

            # Determine the size of the square (use the maximum of width and height)
            square_size = max(width, height)

            # Calculate the new coordinates for a square face zone
            new_x = center_x - (square_size // 2)
            new_y = center_y - (square_size // 2)
            face_box_coor = [new_x, new_y, square_size, square_size]
        else:
            print("ERROR: No Face Detected")
            face_box_coor = [0, 0, frame.shape[0], frame.shape[1]]

        if use_larger_box:
            face_box_coor[0] = max(0, face_box_coor[0] - (larger_box_coef - 1.0) / 2 * face_box_coor[2])
            face_box_coor[1] = max(0, face_box_coor[1] - (larger_box_coef - 1.0) / 2 * face_box_coor[3])
            face_box_coor[2] = larger_box_coef * face_box_coor[2]
            face_box_coor[3] = larger_box_coef * face_box_coor[3]
        
        return face_box_coor

    def create_yunet_object(self, shape):
        if self.yunetObj is None or self.yunetObj.get_yunet_shape() != shape:
            self.yunetObj = facedetectors.YuNetFaceDetector(MODEL_PATH_YUNET, shape, transpose=True, return_only_bb=True)

    def create_face_crop_backend(self, shape):
        if self.config_data.PREPROCESS.CROP_FACE.BACKEND == 'Y5F':
            yolo_obj = YOLO5Face(self.config_data.PREPROCESS.CROP_FACE.BACKEND)
            self.face_crop_backend = facedetectors.Yolo5FaceDetector(yolo_obj)
            return
        elif self.config_data.PREPROCESS.CROP_FACE.BACKEND == 'HC':
            self.face_crop_backend = facedetectors.ViolaJonesFaceDetector()
            return
        
        self.create_yunet_object(shape)
        
        if self.config_data.PREPROCESS.CROP_FACE.BACKEND == 'YUNET':
            self.face_crop_backend = self.yunetObj
        elif self.config_data.PREPROCESS.CROP_FACE.BACKEND == 'CORRYU':
            self.face_crop_backend = facedetectors.CorrelationFaceDetector(self.yunetObj)
        elif self.config_data.PREPROCESS.CROP_FACE.BACKEND == 'OPTYU':
            self.face_crop_backend = facedetectors.OpticalFlowFaceDetector(self.yunetObj)
        elif self.config_data.PREPROCESS.CROP_FACE.BACKEND == 'MOSSE':
            self.face_crop_backend = facedetectors.MOSSEFaceDetector(self.yunetObj)


    def crop_face_resize(self, frames, use_face_detection, use_larger_box, larger_box_coef, use_dynamic_detection, 
                         detection_freq, use_median_box, width, height):
        """Crop face and resize frames.

        Args:
            frames(np.array): Video frames.
            use_dynamic_detection(bool): If False, all the frames use the first frame's bouding box to crop the faces
                                         and resizing.
                                         If True, it performs face detection every "detection_freq" frames.
            detection_freq(int): The frequency of dynamic face detection e.g., every detection_freq frames.
            width(int): Target width for resizing.
            height(int): Target height for resizing.
            use_larger_box(bool): Whether enlarge the detected bouding box from face detection.
            use_face_detection(bool):  Whether crop the face.
            larger_box_coef(float): the coefficient of the larger region(height and weight),
                                the middle point of the detected region will stay still during the process of enlarging.
        Returns:
            resized_frames(list[np.array(float)]): Resized and cropped frames
        """

        self.create_face_crop_backend([frames.shape[1], frames.shape[2]])

        # Face Cropping
        if use_dynamic_detection:
            num_dynamic_det = ceil(frames.shape[0] / detection_freq)
        else:
            num_dynamic_det = 1
        face_region_all = []
        # Perform face detection by num_dynamic_det" times.
        for idx in range(num_dynamic_det):
            if use_face_detection:
                face_region_all.append(self.face_detection(frames[detection_freq * idx], use_larger_box, larger_box_coef))
            else:
                face_region_all.append([0, 0, frames.shape[1], frames.shape[2]])
        face_region_all = np.asarray(face_region_all, dtype='int')
        if use_median_box:
            # Generate a median bounding box based on all detected face regions
            face_region_median = np.median(face_region_all, axis=0).astype('int')

        # Frame Resizing
        total_frames, _, _, channels = frames.shape
        resized_frames = np.zeros((total_frames, height, width, channels))
        for i in range(0, total_frames):
            frame = frames[i]
            if use_dynamic_detection:  # use the (i // detection_freq)-th facial region.
                reference_index = i // detection_freq
            else:  # use the first region obtrained from the first frame.
                reference_index = 0
            if use_face_detection:
                if use_median_box:
                    face_region = face_region_median
                else:
                    face_region = face_region_all[reference_index]
                frame = frame[max(face_region[1], 0):min(face_region[1] + face_region[3], frame.shape[0]),
                        max(face_region[0], 0):min(face_region[0] + face_region[2], frame.shape[1])]
            
            resized_frames[i] = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        return resized_frames

    def chunk(self, frames, bvps, chunk_length):
        """Chunk the data into small chunks.

        Args:
            frames(np.array): video frames.
            bvps(np.array): blood volumne pulse (PPG) labels.
            chunk_length(int): the length of each chunk.
        Returns:
            frames_clips: all chunks of face cropped frames
            bvp_clips: all chunks of bvp frames
        """

        clip_num = frames.shape[0] // chunk_length
        frames_clips = [frames[i * chunk_length:(i + 1) * chunk_length] for i in range(clip_num)]
        bvps_clips = [bvps[i * chunk_length:(i + 1) * chunk_length] for i in range(clip_num)]
        return np.array(frames_clips), np.array(bvps_clips)

    def save(self, frames_clips, bvps_clips, filename, cached_path=None):
        """Save all the chunked data.

        Args:
            frames_clips(np.array): blood volumne pulse (PPG) labels.
            bvps_clips(np.array): the length of each chunk.
            filename: name the filename
        Returns:
            count: count of preprocessed data
        """

        if cached_path is None:
            cached_path = self.cached_path

        if not os.path.exists(cached_path):
            os.makedirs(cached_path, exist_ok=True)
        count = 0
        for i in range(len(bvps_clips)):
            assert (len(self.inputs) == len(self.labels))
            input_path_name = cached_path + os.sep + "{0}_input{1}.npy".format(filename, str(count))
            label_path_name = cached_path + os.sep + "{0}_label{1}.npy".format(filename, str(count))
            self.inputs.append(input_path_name)
            self.labels.append(label_path_name)
            #np.save(input_path_name, frames_clips[i])
            #np.save(label_path_name, bvps_clips[i])
            self._save_preprocessed_file(input_path_name, frames_clips[i])
            self._save_preprocessed_file(label_path_name, bvps_clips[i])
            count += 1
        return count

    def save_multi_process(self, frames_clips, bvps_clips, filename, cached_path=None):
        """Save all the chunked data with multi-thread processing.

        Args:
            frames_clips(np.array): blood volumne pulse (PPG) labels.
            bvps_clips(np.array): the length of each chunk.
            filename: name the filename
        Returns:
            input_path_name_list: list of input path names
            label_path_name_list: list of label path names
        """
        
        if cached_path is None:
            cached_path = self.cached_path

        if not os.path.exists(cached_path):
            os.makedirs(cached_path, exist_ok=True)
        count = 0
        input_path_name_list = []
        label_path_name_list = []
        for i in range(len(bvps_clips)):
            assert (len(self.inputs) == len(self.labels))
            input_path_name = cached_path + os.sep + "{0}_input{1}.npy".format(filename, str(count))
            label_path_name = cached_path + os.sep + "{0}_label{1}.npy".format(filename, str(count))
            input_path_name_list.append(input_path_name)
            label_path_name_list.append(label_path_name)
            #np.save(input_path_name, frames_clips[i])
            #np.save(label_path_name, bvps_clips[i])
            self._save_preprocessed_file(input_path_name, frames_clips[i])
            self._save_preprocessed_file(label_path_name, bvps_clips[i])
            count += 1
        return input_path_name_list, label_path_name_list

    def multi_process_manager(self, data_dirs, config_preprocess, multi_process_quota=1):
        """Allocate dataset preprocessing across multiple processes.

        Args:
            data_dirs(List[str]): a list of video_files.
            config_preprocess(Dict): a dictionary of preprocessing configurations
            multi_process_quota(Int): max number of sub-processes to spawn for multiprocessing
        Returns:
            file_list_dict(Dict): Dictionary containing information regarding processed data ( path names)
        """
        print('Preprocessing dataset...')
        file_num = len(data_dirs)
        choose_range = range(0, file_num)
        pbar = tqdm(list(choose_range))

        # shared data resource
        manager = mp.Manager()  # multi-process manager
        file_list_dict = manager.dict()  # dictionary for all processes to store processed files
        p_list = []  # list of processes
        running_num = 0  # number of running processes

        # in range of number of files to process
        for i in choose_range:
            process_flag = True
            while process_flag:  # ensure that every i creates a process
                if running_num < multi_process_quota:  # in case of too many processes
                    # send data to be preprocessing task
                    p = mp.Process(target=self.preprocess_dataset_subprocess, 
                                args=(data_dirs,config_preprocess, i, file_list_dict))
                    p.start()
                    p_list.append(p)
                    running_num += 1
                    process_flag = False
                for p_ in p_list:
                    if not p_.is_alive():
                        p_list.remove(p_)
                        p_.join()
                        running_num -= 1
                        pbar.update(1)
        # join all processes
        for p_ in p_list:
            p_.join()
            pbar.update(1)
        pbar.close()

        return file_list_dict

    def build_file_list(self, file_list_dict):
        """Build a list of files used by the dataloader for the data split. Eg. list of files used for 
        train / val / test. Also saves the list to a .csv file.

        Args:
            file_list_dict(Dict): Dictionary containing information regarding processed data ( path names)
        Returns:
            None (this function does save a file-list .csv file to self.file_list_path)
        """
        file_list = []
        # iterate through processes and add all processed file paths
        for process_num, file_paths in file_list_dict.items():
            file_list = file_list + file_paths

        if not file_list:
            raise ValueError(self.dataset_name, 'No files in file list')

        file_list_df = pd.DataFrame(file_list, columns=['input_files'])
        os.makedirs(os.path.dirname(self.file_list_path), exist_ok=True)
        file_list_df.to_csv(self.file_list_path)  # save file list to .csv

    def build_file_list_retroactive(self, data_dirs, begin, end):
        """ If a file list has not already been generated for a specific data split build a list of files 
        used by the dataloader for the data split. Eg. list of files used for 
        train / val / test. Also saves the list to a .csv file.

        Args:
            data_dirs(List[str]): a list of video_files.
            begin(float): index of begining during train/val split.
            end(float): index of ending during train/val split.
        Returns:
            None (this function does save a file-list .csv file to self.file_list_path)
        """

        # get data split based on begin and end indices.
        data_dirs_subset = self.split_raw_data(data_dirs, begin, end)

        # generate a list of unique raw-data file names
        filename_list = []
        for i in range(len(data_dirs_subset)):
            filename_list.append(data_dirs_subset[i]['index'])
        filename_list = list(set(filename_list))  # ensure all indexes are unique

        # generate a list of all preprocessed / chunked data files
        file_list = []
        for fname in filename_list:
            processed_file_data = list(glob.glob(self.cached_path + os.sep + "{0}_input*".format(fname)))
            file_list += processed_file_data

        if not file_list:
            print(self.cached_path)
            raise ValueError(self.dataset_name,
                             'File list empty. Check preprocessed data folder exists and is not empty.')

        file_list_df = pd.DataFrame(file_list, columns=['input_files'])
        os.makedirs(os.path.dirname(self.file_list_path), exist_ok=True)
        file_list_df.to_csv(self.file_list_path)  # save file list to .csv

    def load_preprocessed_data(self):
        """ Loads the preprocessed data listed in the file list.

        Args:
            None
        Returns:
            None
        """
        file_list_path = self.file_list_path  # get list of files in
        file_list_df = pd.read_csv(file_list_path)
        inputs = file_list_df['input_files'].tolist()
        if not inputs:
            raise ValueError(self.dataset_name + ' dataset loading data error!')
        

        # TODO make this block more readable
        subjects = []
        batch_idx = []
        for i in range(len(inputs)):
            file_name = os.path.basename(inputs[i])
            match = re.match(r".*?(\d+)_input(\d+)\.npy", file_name)

            if match:
                subjects.append(int(match.group(1)))
                batch_idx.append(int(match.group(2)))
            else:
                raise ValueError(f"Filename {file_name} does not match expected pattern.")

        idx_to_sort = [int(f"{subjects[i]:05d}{batch_idx[i]:05d}") for i in range(len(subjects))]
        idx = np.argsort(idx_to_sort)
        inputs = [inputs[i] for i in idx]


        labels = [input_file.replace("input", "label") for input_file in inputs]
        self.inputs = inputs
        self.labels = labels
        self.preprocessed_data_len = len(inputs)

    @staticmethod
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

    @staticmethod
    def diff_normalize_label(label):
        """Calculate discrete difference in labels along the time-axis and normalize by its standard deviation."""
        diff_label = np.diff(label, axis=0)
        diffnormalized_label = diff_label / np.std(diff_label)
        diffnormalized_label = np.append(diffnormalized_label, np.zeros(1), axis=0)
        diffnormalized_label[np.isnan(diffnormalized_label)] = 0
        return diffnormalized_label

    @staticmethod
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

    @staticmethod
    def standardized_label(label):
        """Z-score standardization for label signal."""
        label = label - np.mean(label)
        label = label / np.std(label)
        label[np.isnan(label)] = 0
        return label

    @staticmethod
    def diff_normalized_to_standardized_label(label):
        """Convert diff normalized data to standardized data."""
        label = np.cumsum(label, axis=0)
        return BaseLoader.standardized_label(label)
        
    @staticmethod
    def standardized_to_diff_normalized_label(label):
        """Convert standardized data to diff normalized data."""
        return BaseLoader.diff_normalize_label(label)

    @staticmethod
    def resample_ppg(input_signal, target_length):
        """Samples a PPG sequence into specific length."""
        return np.interp(
            np.linspace(
                1, input_signal.shape[0], target_length), np.linspace(
                1, input_signal.shape[0], input_signal.shape[0]), input_signal)
