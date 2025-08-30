import os

import cv2
from matplotlib import pyplot as plt
import numpy as np
import scipy

def read_video(video_reader):
    frames = []
    while True:
        ret, frame = video_reader.read()
        if not ret:
            break
        frames.append(frame)
    return np.array(frames)

if __name__ == "__main__":

    deepstab_stable_folder = "/mnt/data/deepstab/stable"
    deepstab_unstable_folder = "/mnt/data/deepstab/unstable"
    results_file = "/mnt/results/deepstab/"

    template_size = 10

    files = os.listdir(deepstab_stable_folder)

    for file in files:
        if file.endswith(".avi"):
            curr_stable_file = os.path.join(deepstab_stable_folder, file)
            curr_unstable_file = os.path.join(deepstab_unstable_folder, file)
            curr_results_file = os.path.join(results_file, f"{file.split(".")[0]}.npy")
            print(f"Results file: {curr_results_file}")

            if not os.path.exists(curr_unstable_file) or not os.path.exists(curr_stable_file):
                print(f"Missing file: {curr_stable_file} or {curr_unstable_file}")
                continue

            if os.path.exists(curr_results_file):
                print(f"Results file already exists: {curr_results_file}")
                continue

            stable_video_reader = cv2.VideoCapture(curr_stable_file)
            unstable_video_reader = cv2.VideoCapture(curr_unstable_file)

            if not stable_video_reader.isOpened() or not unstable_video_reader.isOpened():
                print(f"Error opening video files: {curr_stable_file}, {curr_unstable_file}")
                continue

            stable_video = read_video(stable_video_reader)
            unstable_video = read_video(unstable_video_reader)

            len_diff = len(unstable_video) - len(stable_video)
            if abs(len_diff) > 50:
                print(f"Length difference too large: {len_diff}")
                continue

            if len_diff > 0:
                unstable_video = unstable_video[:len(stable_video)]
            elif len_diff < 0:
                stable_video = stable_video[:len(unstable_video)]

            print(f"Stable video shape: {stable_video.shape}")
            print(f"Unstable video shape: {unstable_video.shape}")

            jitters = np.zeros((len(stable_video), 2), dtype=np.int32)

            for i in range(len(stable_video)):
                gray_stable = cv2.cvtColor(stable_video[i], cv2.COLOR_BGR2GRAY)
                corners = cv2.goodFeaturesToTrack(gray_stable, maxCorners=20, qualityLevel=0.01, minDistance=10)
                
                locations = []
                for corner in corners:
                    x, y = corner.ravel()
                    start_x = max(0, int(x) - template_size)
                    start_y = max(0, int(y) - template_size)
                    end_x = min(stable_video[i].shape[1], int(x) + template_size)
                    end_y = min(stable_video[i].shape[0], int(y) + template_size)
                    template = stable_video[i][start_y:end_y, start_x:end_x]

                    correlation = cv2.matchTemplate(unstable_video[i], template, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(correlation)
                    max_loc = np.array(max_loc, dtype=np.int32) - np.array([x, y], dtype=np.int32) + np.array([template_size, template_size], dtype=np.int32)
                    locations.append(max_loc)

                locations = np.array(locations)
                jitters[i] = np.median(locations, axis=0)
                print(f"Frame {i}: Jitter location: {jitters[i]}")

            print(f"Jitter stats: {np.mean(jitters, axis=0)}, {np.std(jitters, axis=0)}")
            np.save(curr_results_file, jitters)