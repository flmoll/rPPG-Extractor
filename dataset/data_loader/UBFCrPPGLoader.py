"""The dataloader for UBFC-rPPG dataset.

Details for the UBFC-rPPG Dataset see https://sites.google.com/view/ybenezeth/ubfcrppg.
If you use this dataset, please cite this paper:
S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri, J. Dubois, "Unsupervised skin tissue segmentation for remote photoplethysmography", Pattern Recognition Letters, 2017.
"""
import copy
import glob
import json
from math import floor
import os
import re
from multiprocessing import Pool, Process, Value, Array, Manager

import cv2
from matplotlib import pyplot as plt
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm


class UBFCrPPGLoader(BaseLoader):
    """The data loader for the UBFC-rPPG dataset."""

    def __init__(self, name, data_path, config_data, device=None):
        """Initializes an UBFC-rPPG dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "RawData" for below dataset structure:
                -----------------
                     RawData/
                     |   |-- subject1/
                     |       |-- vid.avi
                     |       |-- ground_truth.txt
                     |   |-- subject2/
                     |       |-- vid.avi
                     |       |-- ground_truth.txt
                     |...
                     |   |-- subjectn/
                     |       |-- vid.avi
                     |       |-- ground_truth.txt
                -----------------
                name(string): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        super().__init__(name, data_path, config_data, device)

    def get_raw_data(self, data_path):
        """Returns data directories under the path(For UBFC-rPPG dataset)."""
        data_dirs = glob.glob(data_path + os.sep + "subject*")
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = [{"index": re.search(
            'subject(\d+)', data_dir).group(0), "path": data_dir} for data_dir in data_dirs]
        dirs = sorted(dirs, key=lambda x: int(x['index'].replace('subject', '')))
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values."""
        if begin == 0 and end == 1:  # return the full directory if begin == 0 and end == 1
            return data_dirs

        file_num = len(data_dirs)
        choose_range = range(int(begin * file_num), int(end * file_num))
        data_dirs_new = []

        for i in choose_range:
            data_dirs_new.append(data_dirs[i])

        return data_dirs_new
    

    def read_peak_data(self, data_dirs, sampling_rate=30):

        data_paths = [d['path'] for d in data_dirs]

        loaded_json = json.load(open(os.path.join(self.raw_data_path, "00_peaks.json"), "r"))

        # Match all data_paths with loaded_json["filenames"]
        matched_indices = []
        for path in data_paths:
            filename = os.path.basename(path)
            if filename in loaded_json["filenames"]:
                idx = loaded_json["filenames"].index(filename)
                matched_indices.append(idx)
                #print(f"Matched filename: {filename} at index {idx}")
            else:
                self.logger.warning(f"Filename {filename} not found in loaded_json['filenames'].")

        # Gather peaks for matched indices
        #print(matched_indices)
        peaks = []
        patient_hrs = []
        counter = 0
        for idx in matched_indices:
            ppg_file_path = os.path.join(data_dirs[counter]['path'], "ground_truth.txt")
            ppg = self.read_wave(ppg_file_path)
            ppg_len = len(ppg)

            saved_filename = data_dirs[counter]['index']
            already_preprocessed = os.listdir(self.cached_path)
            associated_files = [file for file in already_preprocessed if re.match(rf"^{saved_filename}_input.*\.npy", file)]
            num_chunks_exists = len(associated_files)

            curr_peaks = loaded_json["peaks"][idx]
            num_chunks = floor(ppg_len / self.config_data.PREPROCESS.CHUNK_LENGTH)
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
            #print(f"Patient HR {current_patient_hr} for {data_dirs[counter]['path']}")
            #print(f"Found {num_chunks_exists} chunks for {num_chunks}")
            #print(f"len peaks: {len(peaks)}")

            """if counter > 10:
                video_file_path = data_dirs[counter]['path']
                print(f"Visualizing peaks for {video_file_path}...")
                ppg = self.read_wave(video_file_path + "/ground_truth.txt")
                peaks_to_plot = np.array(peaks[-1]) + 160*(num_chunks-1)

                plt.plot(ppg)
                plt.scatter(peaks_to_plot, ppg[peaks_to_plot], color='red', label='Peaks')
                plt.savefig("test_peaks.png")
                plt.close()
                input("Press Enter to continue...")"""



            counter += 1

        patient_hrs = np.squeeze(patient_hrs)
        print(f"Found {len(peaks)} peaks in total.")
        return peaks, patient_hrs


    def preprocess_dataset(self, data_dirs, config_preprocess, begin, end):
        for i in range(len(data_dirs)):
            self.preprocess_dataset_subprocess(data_dirs, config_preprocess, i, file_list_dict={})
            

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """ invoked by preprocess_dataset for multi_process."""
        filename = os.path.split(data_dirs[i]['path'])[-1]
        saved_filename = data_dirs[i]['index']

        os.makedirs(self.cached_path, exist_ok=True)
        already_preprocessed = os.listdir(self.cached_path)

        associated_files = [file for file in already_preprocessed if re.match(rf"^{saved_filename}_input.*\.npy", file) or re.match(rf"^{saved_filename}_label.*\.npy", file)]

        cap = cv2.VideoCapture(os.path.join(data_dirs[i]['path'], "vid.avi"))
        file_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        print(f"File length: {file_len}")

        num_chunks_exists = len(associated_files)
        num_chunks_expected = floor(file_len / self.config_data.PREPROCESS.CHUNK_LENGTH) * 2  # 2 for labels and frames

        if config_preprocess.DATA_TYPE == ['Raw']:
            num_chunks_expected = 1

        print("Chunks expected: ", num_chunks_expected)
        print("Chunks exists: ", num_chunks_exists)

        if num_chunks_exists == num_chunks_expected:
            print(f"Already preprocessed {data_dirs[i]['path']} {saved_filename}. Skipping...")
            return

        # Read Frames
        if 'None' in config_preprocess.DATA_AUG:
            # Utilize dataset-specific function to read video
            frames = self.read_video(
                os.path.join(data_dirs[i]['path'],"vid.avi"))
        elif 'Motion' in config_preprocess.DATA_AUG:
            # Utilize general function to read video in .npy format
            frames = self.read_npy_video(
                glob.glob(os.path.join(data_dirs[i]['path'],'*.npy')))
        else:
            raise ValueError(f'Unsupported DATA_AUG specified for {self.dataset_name} dataset! Received {config_preprocess.DATA_AUG}.')

        # Read Labels
        if config_preprocess.USE_PSUEDO_PPG_LABEL:
            bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
        else:
            bvps = self.read_wave(
                os.path.join(data_dirs[i]['path'],"ground_truth.txt"))
            

        # Quick and dirty way to speedup preprocessing
        if config_preprocess.RESIZE.ADDITIONAL_SIZES:
            additional_sizes = [(i, i) for i in config_preprocess.RESIZE.ADDITIONAL_SIZES]
            frames_clips, bvps_clips, frames_clips_additional, bvps_clips_additional = self.preprocess(frames, bvps, config_preprocess, additional_size=additional_sizes)

            for idx, curr_size in enumerate(additional_sizes):
                curr_frames_clips = frames_clips_additional[idx]
                curr_bvps_clips = bvps_clips_additional[idx]
                curr_cached_path = copy.copy(self.cached_path)
                curr_cached_path = curr_cached_path.replace(f"SizeW{config_preprocess.RESIZE.W}_SizeH{config_preprocess.RESIZE.H}", f"SizeW{curr_size[0]}_SizeH{curr_size[1]}")
                input_name_list, label_name_list = self.save_multi_process(curr_frames_clips, curr_bvps_clips, saved_filename, cached_path=curr_cached_path)
            
            self.save_multi_process(frames_clips, bvps_clips, saved_filename)
        else:
            frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
            input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)

        """if "SizeW72_SizeH72" in self.cached_path:
            cached_path_128 = copy.copy(self.cached_path)
            cached_path_128 = cached_path_128.replace("SizeW72_SizeH72", "SizeW128_SizeH128")
            frames_clips, bvps_clips, frames_clips_128, bvps_clips_128 = self.preprocess(frames, bvps, config_preprocess, additional_size=(128, 128))
            input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
            self.save_multi_process(frames_clips_128, bvps_clips_128, saved_filename, cached_path=cached_path_128)
            print(f"clips shape: {frames_clips.shape}, {bvps_clips.shape}")
        else:
            frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
            input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)"""



        file_list_dict[i] = input_name_list

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T, H, W, 3) """
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = list()
        while success:
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frames.append(frame)
            success, frame = VidObj.read()
        return np.asarray(frames)

    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        with open(bvp_file, "r") as f:
            str1 = f.read()
            str1 = str1.split("\n")
            bvp = [float(x) for x in str1[0].split()]
        return np.asarray(bvp)
