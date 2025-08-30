import os
import sys

from matplotlib import pyplot as plt
import numpy as np

from record import display_first_video_frame, get_video_sampling_rate, get_video_length

import androidTimeLogger
from read_wireshark_timestamps import get_ppg_from_wireshark_recording
from convert_timestamp import to_unix_timestamp_ms

data_dir = "data"

folders = sys.argv[1:]
folders = [os.path.join(data_dir, folder) for folder in folders]

for folder in folders:
    video_file_path = os.path.join(folder, "video.mp4")
    timestamps_file = os.path.join(folder, "timestamps.csv")
    recording_file = os.path.join(folder, "recording.pcapng")

    # Display the first frame of the video
    display_first_video_frame(video_file_path)

    while True:
        first_frame_timestamp = input("Enter the timestamp of the first frame (YYYY-MM-DD HH:MM:SS,fff): ")
        first_frame_timestamp = to_unix_timestamp_ms(first_frame_timestamp)
        
        if first_frame_timestamp is not None:
            break

    android_times, windows_times = androidTimeLogger.parse_timestamps(log_file=os.path.join(folder, "timestamps.csv"))
    first_frame_timestamp_windows = androidTimeLogger.convert_timestamp(first_frame_timestamp, android_times, windows_times)

    video_sampling_rate = get_video_sampling_rate(video_file_path)
    video_length = get_video_length(video_file_path)
    video_sample_times = np.arange(first_frame_timestamp_windows, first_frame_timestamp_windows + int(video_length * 1000 / video_sampling_rate), 1000 / video_sampling_rate)
    print(f"Video sampling rate: {video_sampling_rate} FPS")


    # Wait for the Wireshark file to be ready
    ppg_samples, ppg_timestamps = get_ppg_from_wireshark_recording(recording_file)

    plt.plot(ppg_timestamps, ppg_samples, label='PPG Data Samples')
    plt.show()

    ppg_resampled = np.interp(video_sample_times, ppg_timestamps, ppg_samples)
    vstacked_data = np.vstack((video_sample_times, ppg_resampled)).T

    plt.plot(video_sample_times, ppg_resampled, label='Resampled PPG Data')
    plt.show()
    
    np.save(os.path.join(folder, "waveform.npy"), vstacked_data)