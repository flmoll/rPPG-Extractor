import numpy as np
import matplotlib.pyplot as plt

import glob
import os

import sys

import scipy

from evaluation.gt_visualize import GTVisualizer
import json
import shutil



vital_videos_path = '/mnt/data/vitalVideos'
ubfc_path = '/mnt/data/ubfc'

def read_wave(bvp_file):
    """Reads a bvp signal file."""
    with open(bvp_file, "r") as f:
        str1 = f.read()
        str1 = str1.split("\n")
        bvp = [float(x) for x in str1[0].split()]
    return np.asarray(bvp)

def get_vital_videos_gt_labels():

    video_files = glob.glob(os.path.join(vital_videos_path, '*.mp4'))
    video_files.sort()  # Ensure consistent order
    gt_labels = []
    for video_file in video_files:
        folder_name = os.path.basename(video_file)
        subject = folder_name.split("_")[0]
        index = int(folder_name.split("_")[1].split(".")[0])
        gt_file_name = os.path.join(vital_videos_path, f"{subject}.json")
        gt_visualizer = GTVisualizer(gt_file_name, folder_name)

        dummy_ppg = np.zeros((900))  # Dummy PPG signal
        new_timestamps, new_ppg_values, new_predicted_ppg_values = gt_visualizer.resample_ppg(dummy_ppg)
        
        new_ppg_values = (new_ppg_values - np.mean(new_ppg_values)) / np.std(new_ppg_values)  # Normalize PPG values
        gt_labels.append(new_ppg_values)

    video_files = [os.path.basename(video_file) for video_file in video_files]  # Keep only filenames
    return gt_labels, video_files

def get_ubfc_gt_labels():
    subject_files = glob.glob(os.path.join(ubfc_path, 'subject*'))
    subject_files.sort()  # Ensure consistent order
    gt_labels = []
    for subject_file in subject_files:
        bvp_file = os.path.join(subject_file, 'ground_truth.txt')
        if os.path.exists(bvp_file):
            print(f"Reading BVP signal from {bvp_file}")
            bvp_signal = read_wave(bvp_file)
            bvp_signal = (bvp_signal - np.mean(bvp_signal)) / np.std(bvp_signal)  # Normalize PPG values
            gt_labels.append(bvp_signal)

    subject_files = [os.path.basename(subject_file) for subject_file in subject_files]  # Keep only folder names
    return gt_labels, subject_files

class PPGPeakEditor:
    def __init__(self, ppg_signals, initial_peaks, filenames):
        self.ppg_signals = ppg_signals
        self.num_samples = len(ppg_signals)
        self.current_idx = 0
        self.peaks_per_sample = [peaks.tolist() if isinstance(peaks, np.ndarray) else list(peaks) for peaks in initial_peaks]
        self.filenames = filenames

        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], label='PPG')
        self.peak_dots, = self.ax.plot([], [], 'ro', label='Peaks')

        self.text = self.ax.text(0.02, 0.95, "", transform=self.ax.transAxes)
        self.update_plot()

        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.onkey)
        plt.legend()
        plt.show()

    def update_plot(self):
        signal = self.ppg_signals[self.current_idx]
        peaks = self.peaks_per_sample[self.current_idx]

        self.line.set_data(np.arange(len(signal)), signal)
        self.peak_dots.set_data(peaks, signal[peaks])

        self.ax.relim()
        self.ax.autoscale_view()
        self.text.set_text(f"Sample {self.current_idx + 1}/{self.num_samples}")
        self.fig.canvas.draw()

    def onclick(self, event):
        if event.inaxes != self.ax:
            return
        x = int(event.xdata)
        peaks = self.peaks_per_sample[self.current_idx]

        # Check if close to an existing peak (delete if within 5 samples)
        for i, p in enumerate(peaks):
            if abs(p - x) < 5:
                del peaks[i]
                self.update_plot()
                return
            
        # Search for local max in the 5-point window centered at x
        window = np.arange(max(0, x - 5), min(len(self.ppg_signals[self.current_idx]), x + 5))
        signal = self.ppg_signals[self.current_idx]
        if len(window) > 0:
            local_max_idx = window[np.argmax(signal[window])]
            x = local_max_idx

        # Otherwise, add peak
        peaks.append(int(x))
        peaks.sort()
        self.update_plot()

    def onkey(self, event):
        if event.key == 'right':
            if self.current_idx < self.num_samples - 1:
                self.current_idx += 1
                self.update_plot()
        elif event.key == 'left':
            if self.current_idx > 0:
                self.current_idx -= 1
                self.update_plot()
        elif event.key == 'u':
            self.save()

    def save(self):
        json_data = {
            "peaks": self.peaks_per_sample,
            "filenames": self.filenames
        }
        with open("edited_peaks.json", "w") as f:
            json.dump(json_data, f)

        print("Saved to 'edited_peaks.json'")


vital_videos_labels, vital_videos_files = get_vital_videos_gt_labels()
ubfc_labels, ubfc_files = get_ubfc_gt_labels()

initial_peaks = [scipy.signal.find_peaks(signal, height=None, distance=30/2.5, prominence=1.0)[0] for signal in ubfc_labels]
PPGPeakEditor(ubfc_labels, initial_peaks, ubfc_files)
shutil.move("edited_peaks.json", os.path.join(ubfc_path, "00_peaks.json"))

initial_peaks = [scipy.signal.find_peaks(signal, height=None, distance=30/2.5, prominence=1.0)[0] for signal in vital_videos_labels]
PPGPeakEditor(vital_videos_labels, initial_peaks, vital_videos_files)
shutil.move("edited_peaks.json", os.path.join(vital_videos_path, "00_peaks.json"))