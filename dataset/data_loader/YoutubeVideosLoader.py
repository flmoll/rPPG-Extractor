
import copy
import json
from math import ceil, floor
import pprint
import re
import cv2
import pandas as pd
import tqdm
from dataset.data_loader.BaseLoader import BaseLoader
import glob
import os
import numpy as np
import logging
#import mpld3
import scipy.signal

import matplotlib.pyplot as plt

NUMBER_OF_FRAMES_PER_VIDEO = 750

class YoutubeVideosLoader(BaseLoader):

    """
    This class was created to load and preprocess YouTube video data.
    The youtube videos need to be first downloaded and sliced to NUMBER_OF_FRAMES_PER_VIDEO frames per slice.
    This was not included in the study, since it did not show much success.
    """

    def __init__(self, name, data_path, config_data, device=None, logger=None):

        if logger is None:
            logger = logging.getLogger(__name__)

        self.logger = logger
        super().__init__(name, data_path, config_data, device)

    def preprocess_dataset(self, data_dirs, config_preprocess, begin, end):
        file_num = len(data_dirs)
        choose_range = range(int(begin * file_num), int(end * file_num))

        os.makedirs(self.cached_path, exist_ok=True)
        already_preprocessed = os.listdir(self.cached_path)
        print(f"Already preprocessed files: {already_preprocessed}")
        print("Choose range: ", choose_range)

        # Read Video Frames
        for i in tqdm.tqdm(choose_range, desc="Preprocessing videos"):
            video_file_path = data_dirs[i]['path']
            index = data_dirs[i]['index']

            folder_path = os.path.dirname(video_file_path)
            

            num_chunks_expected = floor(NUMBER_OF_FRAMES_PER_VIDEO / self.config_data.PREPROCESS.CHUNK_LENGTH)
            associated_files = [file for file in already_preprocessed if re.match(rf"^{index}_input.*\.npy", file) or re.match(rf"^{index}_label.*\.npy", file)]
            num_chunks_exists = len(associated_files) // 2

            if num_chunks_exists >= num_chunks_expected:
                print(f"Already preprocessed {video_file_path}_{index}. Skipping...")
                continue

            frames = self.read_video(video_file_path)
            print("Video frames shape: ", frames.shape)
            ppg = np.random.rand(frames.shape[0])  # Placeholder for PPG data, replace with actual PPG extraction logic

            

            if config_preprocess.RESIZE.ADDITIONAL_SIZES:
                additional_sizes = [(i, i) for i in config_preprocess.RESIZE.ADDITIONAL_SIZES]
                frames_clips, bvps_clips, frames_clips_additional, bvps_clips_additional = self.preprocess(frames, ppg, config_preprocess, additional_size=additional_sizes)

                for idx, curr_size in enumerate(additional_sizes):
                    curr_frames_clips = frames_clips_additional[idx]
                    curr_bvps_clips = bvps_clips_additional[idx]
                    curr_cached_path = copy.copy(self.cached_path)
                    curr_cached_path = curr_cached_path.replace(f"SizeW{config_preprocess.RESIZE.W}_SizeH{config_preprocess.RESIZE.H}", f"SizeW{curr_size[0]}_SizeH{curr_size[1]}")
                    self.save(curr_frames_clips, curr_bvps_clips, f"{index}", cached_path=curr_cached_path)

                self.preprocessed_data_len += self.save(frames_clips, bvps_clips, f"{index}")
            else:
                frames_clips, bvps_clips = self.preprocess(frames, ppg, config_preprocess)
                self.preprocessed_data_len += self.save(frames_clips, bvps_clips, f"{index}")

            if frames_clips.shape[0] != bvps_clips.shape[0] or frames_clips.shape[0] != num_chunks_expected:
                print(f"Error: frames and bvps clips do not match in shape for {video_file_path}_{index}. Skipping...")
                print(f"frames shape: {frames_clips.shape}, bvps shape: {bvps_clips.shape}, num_chunks_expected: {num_chunks_expected}")

    def read_peak_data(self, data_dirs, sampling_rate=30):
        return [[]]*len(self.inputs), [0]*len(self.inputs),
        
    def get_raw_data(self, data_path):
        """Returns data directories under the path(For Preprocessed Vital Videos dataset)."""
        data_dirs = glob.glob(os.path.join(data_path, "*.mp4"))
        print(data_dirs)
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = list()
        for data_dir in data_dirs:
            # Append the directory information to the list
            dirs.append({"index": 0, "path": data_dir})
        
        dirs = sorted(dirs, key=lambda x: (x['path']))

        for i in range(len(dirs)):
            dirs[i]['index'] = i

        return dirs
    
    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values, 
        and ensures no overlapping subjects between splits"""

        # sort data_dirs by subject
        data_dirs = sorted(data_dirs, key=lambda x: (x['index']))
        len_data_dirs = len(data_dirs)
        begin = int(begin * len_data_dirs)
        end = int(end * len_data_dirs)
        # Check if the begin and end indices are within the range
        if begin < 0 or end > len_data_dirs or begin >= end:
            raise ValueError("Invalid range for splitting data directories.")
        
        # Return the subset of data directories
        return data_dirs[begin:end]

    @staticmethod
    def get_video_sampling_rate(video_file):
        """Returns the sampling rate of the video file."""
        VidObj = cv2.VideoCapture(video_file)
        fps = VidObj.get(cv2.CAP_PROP_FPS)
        VidObj.release()
        return fps

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T,H,W,3) """
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = list()
        while (success):
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frame[np.isnan(frame)] = 0  # TODO: maybe change into avg
            frames.append(frame)
            success, frame = VidObj.read()

        return np.asarray(frames)