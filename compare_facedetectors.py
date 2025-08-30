

import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import tqdm
from evaluation.utils import ask_and_load_results, load_rppg_toolbox_results, save_results, get_mean_SNR_from_results, get_MAE_from_results, get_pickle_path, get_results
from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from tabulate import tabulate
from dataset.facedetection import facedetectors
import time


MODEL_PATH_YUNET = "opencv_zoo/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

def get_metric_from_results(results):
    #return get_mean_SNR_from_results(results)
    return get_MAE_from_results(results)

def read_video(video_file, max_frame=None):
    """Reads a video file, returns frames(T, H, W, 3) """
    VidObj = cv2.VideoCapture(video_file)
    VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
    success, frame = VidObj.read()
    frames = list()
    idx = 1
    while success:
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
        frame = np.asarray(frame)
        frames.append(frame)
        success, frame = VidObj.read()
        if max_frame is not None and idx >= max_frame:
            break
        idx += 1
    return np.asarray(frames)

if __name__ == "__main__":

    face_detectors = ["Y5F", "YUNET", "HC"]
    results_vector = []
    
    for dataset in ["test", "validation"]:
        curr_results_vector = []

        for face_detector in face_detectors:
            path = get_pickle_path("FaceDetection", dataset_name=dataset, stabilizer=face_detector)
            results = get_results(path, dataset=dataset)
            curr_results_vector.append(get_metric_from_results(results))

        results_vector.append(curr_results_vector)

    detectors_to_evaluate = [
        facedetectors.Yolo5FaceDetector(facedetectors.YOLO5Face(device="cpu")),
        facedetectors.YuNetFaceDetector(MODEL_PATH_YUNET, input_shape=(1920, 1200), device="cpu"),
        facedetectors.ViolaJonesFaceDetector(),
    ]

    face_detector_times = []

    time_count_num_operations = 10
    num_repeats = 3
    video_frames = read_video("/mnt/data/vitalVideos/0a687dbdecde4cf1b25e00e5f513a323_1.mp4", max_frame=time_count_num_operations)

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # Evaluierung
    for detector in detectors_to_evaluate:
        print(f"Evaluating {detector} ...")
        frame_times = []

        for idx in tqdm.tqdm(range(time_count_num_operations)):
            repeat_times = []
            for _ in range(num_repeats):
                start_time = time.time()
                detector.detect_faces(video_frames[idx])
                end_time = time.time()
                repeat_times.append(end_time - start_time)
            frame_times.append(np.mean(repeat_times))

        mean_time = np.mean(frame_times)
        std_time = np.std(frame_times)
        face_detector_times.append(mean_time)

    # Generate and print a table of face detector results using tabulate
    table = [[detector, validation_snr, test_snr, exec_time] for detector, validation_snr, test_snr, exec_time in zip(face_detectors, results_vector[0], results_vector[1], face_detector_times)]
    print(results_vector)
    print(f"\nResults for dataset: {dataset}")
    print(tabulate(table, headers=["Detector", "Validation SNR", "Test SNR", "Execution Time"], tablefmt="latex", floatfmt=".2f"))
