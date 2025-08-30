
import os
import re
import threading
import time
import pandas as pd
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import cv2
import msvcrt
from datetime import datetime

import clock
import androidTimeLogger
import videoCopy
from read_wireshark_timestamps import get_ppg_from_wireshark_recording
from convert_timestamp import to_unix_timestamp_ms


data_folder = "C:/Users/flori/AppData/Local/VirtualStore/Program Files (x86)/SpO2 Assistant/Data"
store_folder = "C:/Users/flori/DATA/Uni/MotionMagnification/Code/dataset_recorder/data"
adb_command = 'adb'

data_folder = os.path.normpath(data_folder)
store_folder = os.path.normpath(store_folder)



def wait_for_key(msg="Press any key to continue..."):
    input(msg)

def get_video_sampling_rate(video_file):
    """Returns the sampling rate of the video file."""
    VidObj = cv2.VideoCapture(video_file)
    fps = VidObj.get(cv2.CAP_PROP_FPS)
    VidObj.release()
    return fps

def get_video_length(video_file):
    """Returns the length of the video file in seconds."""
    VidObj = cv2.VideoCapture(video_file)
    if not VidObj.isOpened():
        print("Error: Cannot open video file.")
        return 0
    length = int(VidObj.get(cv2.CAP_PROP_FRAME_COUNT))
    VidObj.release()
    return length

def get_newest_waveform(data_folder):
    # Define the regex pattern for matching the files
    pattern = re.compile(r'^[^_]+_\d{14}_wave\.csv$')

    # Get all files in the directory
    files = [f for f in os.listdir(data_folder) if pattern.match(f)]

    # If there are files that match the pattern, find the newest one
    if files:
        # Sort the files by the timestamp extracted from the filename
        files.sort(key=lambda x: x.split('_')[1], reverse=True)  # Sort by the timestamp (second part of the filename)
        
        newest_file = files[0]
        print(f"The newest file is: {newest_file}")
    else:
        print("No files matching the pattern were found.")

    csv_path = os.path.join(data_folder, os.path.join(data_folder, newest_file))
    df = pd.read_csv(csv_path)
    waveform = df.to_numpy()
    return waveform

def adb_devices():
    result = subprocess.run([adb_command, 'devices'], capture_output=True, text=True)
    lines = result.stdout.strip().splitlines()
    print(lines)
    devices = [line for line in lines[1:] if 'device' in line and 'unauthorized' not in line]
    return devices, result.stdout

def restart_adb_server():
    print("No devices found. Restarting ADB server...")
    subprocess.run([adb_command, 'kill-server'])
    subprocess.run([adb_command, 'start-server'])
    time.sleep(2)  # wait for adb to restart

def wait_for_device():
    while True:
        devices, output = adb_devices()
        if devices:
            print("Devices found:\n", output)
            break
        else:
            print("No devices found.\n", output)
            #restart_adb_server()
            input("Please connect your Android device and press Enter to retry...")


    print("Connected devices:\n", output)
    return devices

def wait_for_file(file_path):
    while not os.path.exists(file_path):
        print(f"Waiting for file: {file_path}")
        wait_for_key("Press Enter when the file is ready...")
    print(f"File found: {file_path}")

def display_first_video_frame(video_file):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    ret, frame = cap.read()
    if ret:
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.axis('off')  # Hide axes
        plt.title('First Frame of Video')
        plt.show()
    else:
        print("Error: Could not read the first frame.")

    cap.release()

if __name__ == "__main__":
    wait_for_device()

    #clock_thread, stop_event = clock.start_clock_in_new_thread()

    new_folder = input("Enter the name of the new folder to store the data: ")
    new_folder = os.path.join(store_folder, new_folder)
    os.makedirs(new_folder, exist_ok=True)
    print(f"Data will be stored in: {new_folder}")

    android_logger_thread, android_stop_event = androidTimeLogger.start_logging_in_new_thread(log_file=os.path.join(new_folder, "timestamps.csv"))

    wait_for_key("Start Wireshark recording and press Enter to continue...")
    wait_for_key("Start PPG recording and press Enter to continue...")
    wait_for_key("Start filming and press Enter when finished...")
    wait_for_key("Stop PPG and Wireshark recording and press Enter to continue...")

    #waveform = get_newest_waveform(data_folder)
    #np.save(os.path.join(new_folder, "waveform.npy"), waveform)

    android_stop_event.set()  # Stop the Android logger thread
    android_logger_thread.join()

    android_times, windows_times = androidTimeLogger.parse_timestamps(log_file=os.path.join(new_folder, "timestamps.csv"))

    while True:
        video_file_path = os.path.join(new_folder, "video.mp4")
        newest_video = videoCopy.find_newest_video()
        if newest_video:
            videoCopy.pull_video(newest_video, video_file_path)
            break
        else:
            print("No video files found to pull.")

    # Display the first frame of the video
    display_first_video_frame(video_file_path)

    while True:
        first_frame_timestamp = input("Enter the timestamp of the first frame (YYYY-MM-DD HH:MM:SS,fff): ")
        first_frame_timestamp = to_unix_timestamp_ms(first_frame_timestamp)

        if first_frame_timestamp is not None:
            break

    first_frame_timestamp_windows = androidTimeLogger.convert_timestamp(first_frame_timestamp, android_times, windows_times)

    video_sampling_rate = get_video_sampling_rate(video_file_path)
    video_length = get_video_length(video_file_path)
    video_sample_times = np.arange(first_frame_timestamp_windows, first_frame_timestamp_windows + int(video_length * 1000 / video_sampling_rate), 1000 / video_sampling_rate)
    print(f"Video sampling rate: {video_sampling_rate} FPS")


    # Wait for the Wireshark file to be ready
    pcap_file = os.path.join(new_folder, "recording.pcapng")
    wait_for_file(pcap_file)
    ppg_samples, ppg_timestamps = get_ppg_from_wireshark_recording(pcap_file)

    plt.plot(ppg_timestamps, ppg_samples, label='PPG Data Samples')
    plt.show()

    ppg_resampled = np.interp(video_sample_times, ppg_timestamps, ppg_samples)
    vstacked_data = np.vstack((video_sample_times, ppg_resampled)).T
    
    plt.plot(video_sample_times, ppg_resampled, label='Resampled PPG Data')
    plt.show()
    
    np.save(os.path.join(new_folder, "waveform.npy"), vstacked_data)

    #stop_event.set()  # Signal the clock thread to stop
    #clock_thread.join()



    #android_stop_event.set()  # Stop the Android logger thread
    #android_logger_thread.join()




"""timestamps = []

for i in range(1000):

    win_ts = int(time.time() * 1000)
    diff = win_ts - timestamps[-1] if timestamps else 0
    timestamps.append(win_ts)
    print(f"Logging timestamp {i} {win_ts} ms (diff: {diff} ms)")

    time.sleep(0.1)  # Log every 100 milliseconds


plt.plot(timestamps, label='Windows Timestamps')
plt.xlabel('Sample Index')
plt.ylabel('Timestamp (ms)')
plt.title('Timestamps Comparison')
plt.legend()
plt.savefig("timestamps_plot.png")
plt.close()"""
