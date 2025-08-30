import datetime
import subprocess
import time
import threading
import numpy as np
import pandas as pd

adb_command = 'adb'

def get_android_timestamp():
    try:
        result = subprocess.run(
            [adb_command, "shell", "date", "+%s+%N"],
            stdin=subprocess.DEVNULL,  # otherwise the input reading will fail
            capture_output=True,
            text=True,
            timeout=2
        )
        ts = result.stdout.strip()
        if not ts:
            return None
        
        seconds, nanoseconds = ts.split('+')
        seconds = int(seconds)
        nanoseconds = int(nanoseconds)
        return seconds * 1000 + nanoseconds // 1000000
    except subprocess.TimeoutExpired:
        return None

def get_windows_timestamp():
    return int(time.time() * 1000)

def log_timestamps():
    android_ts = get_android_timestamp()
    win_ts_ms = get_windows_timestamp()

    if android_ts is None:
        print("Failed to get Android timestamp.")
        return
    
    print(f"Android Timestamp: {android_ts}")
    print(f"Windows Timestamp: {win_ts_ms}")

def parse_timestamps(log_file="timestamps.csv"):
    df = pd.read_csv(log_file, header=0, names=["Android Timestamp", "Windows Timestamp"])
    android_times = df["Android Timestamp"].astype(int).to_numpy()
    windows_times = df["Windows Timestamp"].astype(int).to_numpy()
    return android_times, windows_times

def log_loop(stop_event, log_file="timestamps.csv"):
    with open(log_file, "w") as f:
        f.write("Android Timestamp, Windows Timestamp\n")
        
        while not stop_event.is_set():
            android_ts = get_android_timestamp()
            win_ts = get_windows_timestamp()
            
            if android_ts is not None:
                f.write(f"{android_ts}, {win_ts}\n")
            
            f.flush()
            time.sleep(0.1)  # Log every 100 milliseconds

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

def start_logging_in_new_thread(log_file="timestamps.csv"):
    stop_event = threading.Event()
    log_thread = threading.Thread(target=log_loop, args=(stop_event, log_file))
    log_thread.start()
    return log_thread, stop_event
