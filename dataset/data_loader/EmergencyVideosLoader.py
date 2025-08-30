
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


EMERGENCY_VIDEOS_PPG_BANDPASS_LOWCUT = 0.5
EMERGENCY_VIDEOS_PPG_BANDPASS_HIGHCUT = 3.3
ROTATE_FRAMES = cv2.ROTATE_90_CLOCKWISE

def convert_timestamp(timestamp, timestamps_current_series, timestamps_target_series):
    """
    Convert a timestamp from one series to another using linear interpolation.
    
    Args:
        timestamp (int): The timestamp to convert.
        timestamps_current_series (np.ndarray): The timestamps of the current series.
        timestamps_target_series (np.ndarray): The timestamps of the target series.
    
    Returns:
        int: The converted timestamp in the target series.
    """
    if len(timestamps_current_series) == 0 or len(timestamps_target_series) == 0:
        raise ValueError("Timestamps series cannot be empty.")
    
    # Interpolate the timestamp
    return np.interp(timestamp, timestamps_current_series, timestamps_target_series)

class EmergencyVideosLoader(BaseLoader):

    """
    This class was designed to load the RealisticVideos dataset as explained in the paper associated with this repository
    """

    def __init__(self, name, data_path, config_data, device=None, logger=None):

        if logger is None:
            logger = logging.getLogger(__name__)

        self.logger = logger
        new_config = config_data.clone()
        new_config.defrost()
        new_config.CACHED_PATH = new_config.CACHED_PATH + f"V_{new_config.PREPROCESS.VIDEO_SUBSET_TO_USE[0]}_{new_config.PREPROCESS.VIDEO_SUBSET_TO_USE[1]}/"
        new_config.freeze()
        super().__init__(name, data_path, new_config, device)

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
            ppg_file = os.path.join(folder_path, "waveform.npy")
            ppg = np.load(ppg_file)
            
            ppg_timestamps = ppg[:, 0]
            ppg_timestamps = ppg_timestamps - ppg_timestamps[0]  # Normalize timestamps to start from 0

            ppg = ppg[:, 1]

            if len(self.config_data.PREPROCESS.VIDEO_SUBSET_TO_USE) > 0:
                start_timestamp = self.config_data.PREPROCESS.VIDEO_SUBSET_TO_USE[0] * 1000  # Convert to milliseconds
                end_timestamp = self.config_data.PREPROCESS.VIDEO_SUBSET_TO_USE[1] * 1000  # Convert to milliseconds
            else:
                start_timestamp = 0.0
                end_timestamp = ppg_timestamps[-1]

            start_index = np.argmin(np.abs(ppg_timestamps - start_timestamp))
            end_index = np.argmin(np.abs(ppg_timestamps - end_timestamp))

            num_chunks_expected = floor((end_index - start_index) / self.config_data.PREPROCESS.CHUNK_LENGTH)
            associated_files = [file for file in already_preprocessed if re.match(rf"^{index}_input.*\.npy", file) or re.match(rf"^{index}_label.*\.npy", file)]
            num_chunks_exists = len(associated_files) // 2

            if num_chunks_exists == num_chunks_expected:
                print(f"Already preprocessed {video_file_path}_{index}. Skipping...")
                continue

            frames = self.read_video(video_file_path)

            if len(frames) != len(ppg):
                print(f"Warning: Number of frames and PPG samples do not match for {video_file_path}_{index}. Skipping...")
                continue

            # Bandpass filter the PPG signal (e.g., 0.7-4 Hz for heart rate)
            milliseconds_per_sample = (ppg_timestamps[-1] - ppg_timestamps[0]) / len(ppg_timestamps)
            fs = 1000.0 / milliseconds_per_sample  # Sampling frequency in Hz
            nyq = 0.5 * fs
            low = EMERGENCY_VIDEOS_PPG_BANDPASS_LOWCUT / nyq
            high = EMERGENCY_VIDEOS_PPG_BANDPASS_HIGHCUT / nyq
            b, a = scipy.signal.butter(1, [low, high], btype='band')
            ppg = scipy.signal.filtfilt(b, a, ppg)

            frames = frames[start_index:end_index]
            ppg = ppg[start_index:end_index]

            """
            ppg_norm = (ppg - np.min(ppg)) / (np.max(ppg) - np.min(ppg))  # Normalize PPG values to [0, 1]
            fig, ax = plt.subplots()
            ax.plot(ppg_timestamps, ppg, label='PPG Signal')
            html_str = mpld3.fig_to_html(fig)
            with open("ppg_signal.html", "w") as f:
                f.write(html_str)
            plt.close(fig)
            
            # Prepare video writer
            output_path = "video_with_ppg_overlay.mp4"
            height, width, _ = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

            rect_w, rect_h = 40, 40  # Size of the rectangle
            rect_x, rect_y = 10, 10  # Top-left corner

            # For each frame, overlay rectangle colored by corresponding PPG value
            for idx, frame in enumerate(frames):
                color_val = int(ppg_norm[idx] * 255)
                color = (color_val, 0, 255 - color_val)  # From blue (low) to red (high)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.rectangle(frame_bgr, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), color, -1)
                cv2.putText(frame_bgr, f"PPG: {ppg[idx]:.2f}", (rect_x + 5, rect_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                out.write(frame_bgr)

            out.release()
            print(f"Overlay video saved to: {output_path}")

            input("Press Enter to continue...")
            """

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
        data_dirs = glob.glob(os.path.join(data_path, "*", "video.mp4"))
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
    def read_video(video_file, rotate=ROTATE_FRAMES):
        """Reads a video file, returns frames(T,H,W,3) """
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = list()
        while (success):
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = cv2.rotate(frame, rotate)  # Rotate frame if needed
            frame = np.asarray(frame)
            frame[np.isnan(frame)] = 0  # TODO: maybe change into avg
            frames.append(frame)
            success, frame = VidObj.read()

        return np.asarray(frames)