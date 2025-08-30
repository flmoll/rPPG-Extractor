import copy
import json
from math import ceil, floor
import pprint
import re
import cv2
from dataset.data_loader.BaseLoader import BaseLoader
import glob
import os
import numpy as np
import logging

class OwnVideosLoader(BaseLoader):

    """
    This class was created to load and preprocess videos of the AdBannerVideos and DistributionShift videos.
    For more information on that check the paper associated to this study
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
        for i in choose_range:
            video_file_path = data_dirs[i]['path']
            index = data_dirs[i]['index']
            properties_json_file = video_file_path.replace(".mp4", ".json").replace(".avi", ".json")

            if not os.path.exists(properties_json_file):
                print(f"Warning: Properties JSON file not found: {properties_json_file}")
                input_str = input("Generate properties JSON file? (y/n): ")
                
                if input_str.lower() == 'y':
                    # Generate properties JSON file
                    print(f"Generating properties JSON file: {properties_json_file}")
                    with open(properties_json_file, 'w') as f:
                        json.dump({"frames_cut_front": 0, "frames_cut_back": 0, "rotate": 90, "crop_box": [0, 0, 999999, 999999]}, f, indent=4)
                else:
                    print(f"Skipping {video_file_path} due to missing properties JSON file.")
                    continue

            cap = cv2.VideoCapture(os.path.join(data_dirs[i]['path']))
            file_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            print(f"File length: {file_len}")

            properties = json.load(open(properties_json_file, "r"))
            frames_cut_front = abs(properties["frames_cut_front"]) if "frames_cut_front" in properties else 0
            frames_cut_back = abs(properties["frames_cut_back"]) if "frames_cut_back" in properties else 0
            rotate = properties["rotate"] if "rotate" in properties else 0
            crop_box = properties["crop_box"] if "crop_box" in properties else [0, 0, 999999, 999999]

            print(f"Processing {i + 1} {end} {begin} files")
            print(f"Processing {data_dirs[i]['path']}")
            print(f"Processing {video_file_path} {index}")

            # Check if the video file has already been preprocessed
            associated_files = [file for file in already_preprocessed if re.match(rf"^{index}_input.*\.npy", file) or re.match(rf"^{index}_label.*\.npy", file)]
            num_chunks_exists = len(associated_files)
            num_chunks_expected = floor((file_len - frames_cut_front - frames_cut_back) / self.config_data.PREPROCESS.CHUNK_LENGTH) * 2

            if num_chunks_expected <= 0:
                print(f"Warning: num_chunks_expected <= 0 for {video_file_path}_{index}. Skipping...")
                print(f"num_chunks_expected: {num_chunks_expected}, file_len: {file_len}, frames_cut_front: {frames_cut_front}, frames_cut_back: {frames_cut_back}")
                continue

            if config_preprocess.DATA_TYPE == ['Raw']:
                num_chunks_expected = 1

            print("Chunks expected: ", num_chunks_expected)
            print("Chunks exists: ", num_chunks_exists)

            if num_chunks_exists == num_chunks_expected:
                print(f"Already preprocessed {video_file_path}_{index}. Skipping...")
                continue

            frames_cut_back = -frames_cut_back if frames_cut_back > 0 else None
            frames_cut_front = frames_cut_front if frames_cut_front > 0 else None

            frames = self.read_video(video_file_path, rotate=rotate)
            print(frames.shape, frames_cut_front, frames_cut_back)
            frames = frames[frames_cut_front:frames_cut_back, crop_box[1]:crop_box[3], crop_box[0]:crop_box[2], :]

            print(frames.shape)

            ppg = np.zeros((frames.shape[0]))

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

            if frames_clips.shape[0] != bvps_clips.shape[0] or frames_clips.shape[0] != num_chunks_expected // 2:
                print(f"Error: frames and bvps clips do not match in shape for {video_file_path}_{index}. Skipping...")
                print(f"frames shape: {frames_clips.shape}, bvps shape: {bvps_clips.shape}")
                exit(1)

    def read_peak_data(self, data_dirs, sampling_rate=30):
        return [[]]*len(self.inputs), [0]*len(self.inputs),
        
    def get_raw_data(self, data_path):
        """Returns data directories under the path(For Preprocessed Vital Videos dataset)."""
        data_dirs = glob.glob(data_path + os.sep + "*.mp4") + glob.glob(data_path + os.sep + "*.avi")
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
    def read_video(video_file, rotate=None):
        """Reads a video file, returns frames(T,H,W,3) """
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = list()
        while (success):
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frame[np.isnan(frame)] = 0  # TODO: maybe change into avg

            if rotate is not None:
                if rotate == 90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif rotate == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif rotate == 270:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            frames.append(frame)
            success, frame = VidObj.read()

        return np.asarray(frames)