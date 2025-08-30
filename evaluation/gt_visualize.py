import json
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import logging

class GTVisualizer:
    def __init__(self, gt_data_file, filename, logger=None, vital_videos=True):

        if logger is None:
            logger = logging.getLogger(__name__)

        if '/' in filename:
            filename = os.path.basename(filename)

        self.logger = logger
        self.gt_data_file = gt_data_file
        self.video_file_name = filename

        if vital_videos:
            self._load_gt_data_vital_videos()
        else:
            self.load_gt_data_ubfc()

    def load_gt_data_ubfc(self):
        print(self.gt_data_file)
        ppg = GTVisualizer.read_wave(self.gt_data_file)
        self.ppg_recording = {
            "ppg_values": ppg,
            "timestamps": np.arange(0, len(ppg) * 1000 / 30, 1000 / 30)
        }
        self.rgb_recording = {
            "timestamps": np.arange(0, len(ppg) * 1000 / 30, 1000 / 30)
        }

    def _load_gt_data_vital_videos(self):
        with open(self.gt_data_file) as json_file:

            data = json.load(json_file)
            for scenario in data['scenarios']:

                if not scenario['recordings']['RGB']['filename'] == self.video_file_name:
                    continue

                self.scenario_settings = scenario['scenario_settings']
                self.location = data['location']

                # Construct the object for the ppg recording
                self.ppg_recording = {
                    "participant_metadata": {
                        **data['participant'],
                        'GUID': data['GUID']
                    },
                    # timeseries contains a list of [timestamp, value] pairs. We only want the values so we extract the 2nd item from the inner list.
                    "ppg_values": [item[1] for item in scenario['recordings']['ppg']['timeseries']],
                    "timestamps": [item[0] for item in scenario['recordings']['ppg']['timeseries']],
                    "recording_link": scenario['recordings']['RGB']['filename']
                }

                self.rgb_recording = {
                    "timestamps": [item[0] for item in scenario['recordings']['RGB']['timeseries']]
                }

                # Check if bp_sys and bp_dia exist in the recordings
                if 'bp_sys' in scenario['recordings'] and 'bp_dia' in scenario['recordings']:
                    # Construct the object for the blood pressure recording
                    self.bp_recording = {
                        "participant_metadata": {
                            **data['participant'],
                            'GUID': data['GUID']
                        },
                        "ppg_values": [item[1] for item in scenario['recordings']['ppg']['timeseries']],
                        "bp_values": {
                            "bp_sys": scenario['recordings']['bp_sys']['value'],
                            "bp_dia": scenario['recordings']['bp_dia']['value'],
                        },
                        "recording_link": scenario['recordings']['RGB']['filename']
                    }

    def get_video_file_name(self):
        return self.video_file_name

    def get_scenario_settings(self):
        return self.scenario_settings
    
    def get_metadata(self):
        return self.ppg_recording['participant_metadata']
    
    def get_subject_id(self):
        return self.ppg_recording['participant_metadata']['GUID']
    
    def get_gender(self):
        return self.ppg_recording['participant_metadata']['gender']
    
    def get_age(self):
        return self.ppg_recording['participant_metadata']['age']
    
    def get_fitzpatrick(self):
        return self.ppg_recording['participant_metadata']['fitzpatrick']
    
    def get_location(self):
        return self.location
    
    def get_ppg_recording(self):
        return self.ppg_recording
    
    def get_bp_recording(self):
        return self.bp_recording

    def resample_ppg(self, predicted_ppg_values, fps=30, num_frames=None):
        ppg_values = np.array(self.ppg_recording['ppg_values'])
        timestamps = np.array(self.ppg_recording['timestamps'])
        timestamps_predicted = np.array(self.rgb_recording['timestamps'])

        if num_frames is not None:
            max_time = 1000 * num_frames / fps
        else:
            max_time = timestamps[-1]

        new_timestamps = np.arange(0, max_time, 1000 / fps)
        new_ppg_values = np.interp(new_timestamps, timestamps, ppg_values)

    
        if len(timestamps_predicted) > predicted_ppg_values.shape[0]:
            self.logger.warning("The predicted timestamps are longer than the PPG values. Truncating the predicted timestamps.")
            timestamps_predicted = timestamps_predicted[:predicted_ppg_values.shape[0]]

        new_predicted_ppg_values = np.interp(new_timestamps, timestamps_predicted, predicted_ppg_values)

        return new_timestamps, new_ppg_values, new_predicted_ppg_values

    def plot_gt_ppg(self, filename="plot.png", interactive=True):
        ppg_values = np.array(self.ppg_recording['ppg_values'])
        timestamps = np.array(self.ppg_recording['timestamps'])

        plt.plot(timestamps, ppg_values)
        plt.xlabel('Time (ms)')
        plt.ylabel('PPG Value')
        plt.title('Ground Truth PPG')
        plt.grid()
        plt.savefig(filename)

        if interactive:
            input("Press any key. ")

    def calculate_hr(self, time_interval_ms=[0, np.inf]):
        ppg_values = np.array(self.ppg_recording['ppg_values'])
        timestamps = np.array(self.ppg_recording['timestamps'])

        time_interval_ms = [max(time_interval_ms[0], timestamps[0]), min(time_interval_ms[1], timestamps[-1])]
        time_interval_idx = np.where((timestamps >= time_interval_ms[0]) & (timestamps <= time_interval_ms[1]))[0]

        timestamps = timestamps[time_interval_idx]
        ppg_values = ppg_values[time_interval_idx]

        peaks = find_peaks(ppg_values, height=0)[0]
        peak_times = timestamps[peaks]

        time_diff = np.diff(peak_times)
        hr = 60000 / time_diff
        hr = np.mean(hr)

        return hr

    def visualize_ppg_pulsating_rect(self, video_tensor, rect_size=(20, 20), rect_pos=(0, 0), rect_color=(0.5, 0.5, 0.5), color_change_scale=0.5, fps=30, frame_offset=0):
        time_series = np.array(self.ppg_recording['timestamps'])
        ppg_values = np.array(self.ppg_recording['ppg_values'])

        ppg_values = 2 * (ppg_values - np.min(ppg_values)) / (np.max(ppg_values) - np.min(ppg_values)) - 1
        ppg_values = (ppg_values * color_change_scale).astype(np.int16)

        for i in range(len(video_tensor)):
            frame = video_tensor[i]
            curr_time = 1000 * (i + frame_offset) / fps
            idx = np.argmin(np.abs(time_series - curr_time))
            current_color = rect_color + ppg_values[idx]
            current_color = tuple(current_color.tolist())
            cv2.rectangle(frame, (rect_pos[0], rect_pos[1]), (rect_pos[0] + rect_size[0], rect_pos[1] + rect_size[1]), current_color, thickness=-1)
            video_tensor[i] = frame

        return video_tensor
    
    
    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        with open(bvp_file, "r") as f:
            str1 = f.read()
            str1 = str1.split("\n")
            bvp = [float(x) for x in str1[0].split()]
        return np.asarray(bvp)