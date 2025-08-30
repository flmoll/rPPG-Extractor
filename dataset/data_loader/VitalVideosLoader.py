import copy
from math import ceil, floor
import pprint
import re
import cv2
from matplotlib import pyplot as plt
from dataset.data_loader.BaseLoader import BaseLoader
import glob
import os
import numpy as np
import logging
import sys

from evaluation.gt_visualize import GTVisualizer
import json



def get_raw_data_vital_videos(data_path):
    """Returns data directories under the path(For Preprocessed Vital Videos dataset)."""
    data_dirs = glob.glob(data_path + os.sep + "*.mp4")
    if not data_dirs:
        raise ValueError("data paths empty!")
    dirs = list()
    for data_dir in data_dirs:
        folder_name = os.path.basename(data_dir)
        subject = folder_name.split("_")[0]
        index = int(folder_name.split("_")[1].split(".")[0])
        gt_file_name = os.path.join(data_path, f"{subject}.json")
        gt_visualizer = GTVisualizer(gt_file_name, folder_name)

        location = gt_visualizer.get_location()['location']

        # Check if the corresponding JSON file exists
        json_file = os.path.join(data_path, f"{subject}.json")
        if not os.path.exists(json_file):
            raise ValueError(f"JSON file {json_file} does not exist for subject {subject}. Skipping this directory.")

        # Append the directory information to the list
        dirs.append({"index": index, "path": data_dir, "subject": subject, "location": location})
    
    dirs = sorted(dirs, key=lambda x: (x['location'] + x['path']))

    for i in range(len(dirs)):
        dirs[i]['index'] = i

    return dirs

class VitalVideosLoader(BaseLoader):
    """
    Data loader for the Preprocessed Vital Videos dataset.
    The structure of the dataset is as follows:
    005e7070c2c3466aab7bac185e7fbd29_1.mp4
    005e7070c2c3466aab7bac185e7fbd29_2.mp4
    ...
    005e7070c2c3466aab7bac185e7fbd29.json

    01d1d2f99bba437aa9442c0a52d59d1e_1.mp4
    01d1d2f99bba437aa9442c0a52d59d1e_2.mp4
    ...
    01d1d2f99bba437aa9442c0a52d59d1e.json
    ...

    The dataset contains video files and corresponding flatbuffer files.
    The flatbuffer files contain the preprocessed video data.
    The JSON files contain the ground truth data.
    """

    def __init__(self, name, data_path, config_data, device=None, logger=None):

        if logger is None:
            logger = logging.getLogger(__name__)

        self.logger = logger
        super().__init__(name, data_path, config_data, device)

    def preprocess_dataset(self, data_dirs, config_preprocess, begin, end):
        """Preprocesses the raw data."""

        file_num = len(data_dirs)
        choose_range = range(int(begin * file_num), int(end * file_num))

        os.makedirs(self.cached_path, exist_ok=True)
        already_preprocessed = os.listdir(self.cached_path)
        print(f"Already preprocessed files: {already_preprocessed}")

        print("Choose range: ", choose_range)

        # Read Video Frames
        for i in choose_range:
            video_file_path = data_dirs[i]['path']
            video_file_name = os.path.basename(video_file_path)
            subject = data_dirs[i]['subject']
            index = data_dirs[i]['index']
            parent_dir = os.path.dirname(video_file_path)
            json_file = os.path.join(parent_dir, f"{subject}.json")
            print(f"Processing {i + 1} {end} {begin} files")
            print(f"Processing {data_dirs[i]['path']}")
            print(f"Processing {json_file} {index}")

            # Check if the video file has already been preprocessed
            associated_files = [file for file in already_preprocessed if re.match(rf"^{index}_input.*\.npy", file) or re.match(rf"^{index}_label.*\.npy", file)]
            num_chunks_exists = len(associated_files)
            num_chunks_expected = floor(900 / self.config_data.PREPROCESS.CHUNK_LENGTH) * 2        # 2 for labels and frames

            if config_preprocess.DATA_TYPE == ['Raw']:
                num_chunks_expected = 1

            if num_chunks_exists == num_chunks_expected:
                print(f"Already preprocessed {video_file_name} {subject}_{index}. Skipping...")
                continue

            print(f"Num chunks exists: {num_chunks_exists}, expected: {num_chunks_expected}")
            frames = self.read_video(video_file_path)

            # Read Labels
            gt_visualizer = GTVisualizer(json_file, video_file_name)
            dummy_ppg = np.zeros((frames.shape[0]))
            time, ppg, _ = gt_visualizer.resample_ppg(dummy_ppg, num_frames=frames.shape[0])

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


    def read_peak_data(self, data_dirs, sampling_rate=30):
        data_paths = [d['path'] for d in data_dirs]
        
        loaded_json = json.load(open(os.path.join(self.raw_data_path, "00_peaks.json"), "r"))

        # Match all data_paths with loaded_json["filenames"]
        matched_indices = []
        for curr_dir in data_dirs:
            path = curr_dir['path']
            filename = os.path.basename(path)
            if filename in loaded_json["filenames"]:
                idx = loaded_json["filenames"].index(filename)
                matched_indices.append(idx)
            else:
                self.logger.warning(f"Filename {filename} not found in loaded_json['filenames'].")

        # Gather peaks for matched indices
        print(matched_indices)
        peaks = []
        patient_hrs = []
        counter = 0
        for idx in matched_indices:
            curr_peaks = loaded_json["peaks"][idx]
            num_chunks = floor(900 / self.config_data.PREPROCESS.CHUNK_LENGTH)
            curr_chunks = [[] for _ in range(num_chunks)]
            curr_patient_hrs = []
            for peak in curr_peaks:
                idx_peak = floor(peak / self.config_data.PREPROCESS.CHUNK_LENGTH)
                if idx_peak >= num_chunks:
                    continue

                curr_chunks[idx_peak].append(peak - idx_peak * self.config_data.PREPROCESS.CHUNK_LENGTH)

            for chunk in curr_chunks:
                hr = (sampling_rate / np.mean(np.diff(chunk))) * 60
                curr_patient_hrs.append(hr)

            peaks.extend(curr_chunks)
            patient_hrs.extend(curr_patient_hrs)

            counter += 1

        
        print(f"Found {len(peaks)} peaks in total.")
        patient_hrs = np.squeeze(patient_hrs)
        return peaks, patient_hrs

    def get_raw_data(self, data_path):
        """Returns data directories under the path(For Preprocessed Vital Videos dataset)."""
        return get_raw_data_vital_videos(data_path)
    
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
    def read_video(video_file):
        """Reads a video file, returns frames(T,H,W,3) """
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = list()
        idx = 0
        while (success):
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frame[np.isnan(frame)] = 0  # TODO: maybe change into avg
            frames.append(frame)
            
            """if idx == 30:
                # If we have enough frames, break the loop
                break
            idx += 1"""
    
            success, frame = VidObj.read()

        return np.asarray(frames)