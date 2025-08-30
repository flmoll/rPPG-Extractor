
import pickle
import scipy
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

import glob
import pickle
import numpy as np
import os
import scipy
import torch
import tqdm
import json


from evaluation.post_process import _detrend
from evaluation.post_process import calculate_metric_per_video
from evaluation.metrics import calculate_metrics
from config import _C
from evaluation.BlandAltmanPy import BlandAltman
from neural_methods.loss.RythmFormerLossComputer import RhythmFormer_Loss

from vitallens.signal import windowed_freq, windowed_mean

from evaluation.gt_visualize import GTVisualizer
from PyEVM.video_reader import FlatbufferVideoReader

from prpy.numpy.signal import interpolate_cubic_spline
from prpy.numpy.signal import detrend, moving_average, standardize

from vitallens.enums import Mode
from vitallens.methods.simple_rppg_method import SimpleRPPGMethod
from vitallens.signal import detrend_lambda_for_hr_response
from vitallens.signal import moving_average_size_for_hr_response

from evaluation.heart_rate_filter import HeartRateFilter, AutocorrelationFilter


def calculate_gt_hr(labels_cpu, sampling_rate=30, bandpass_upper=2.5):
    """
    Calculate the ground-truth heart rate from label PPG signals by detecting peaks.

    Parameters
    ----------
    labels_cpu : np.ndarray
        Array of shape [num_samples, time_steps] containing the PPG signals for which to compute heart rate.
    sampling_rate : float, optional
        Sampling rate of the PPG signals in Hz. Default is 30.
    bandpass_upper : float, optional
        Upper cutoff frequency (in Hz) for minimum peak distance calculation. Used to avoid detecting peaks too close together. Default is 2.5.

    Returns
    -------
    np.ndarray
        Array of shape [num_samples] containing the estimated heart rate in beats per minute (BPM) for each sample.
        If no valid peaks are detected, the heart rate is set to 0 for that sample.
    """

    hr_labels = []
    for i in range(labels_cpu.shape[0]):
        peaks_gt = scipy.signal.find_peaks(labels_cpu[i], height=None, distance=sampling_rate/bandpass_upper, prominence=1.5)[0]
        #gt_hr = len(peaks_gt) * fs * 60 / len(label_ppG)
        gt_hr = (sampling_rate / np.mean(np.diff(peaks_gt))) * 60

        if np.isnan(gt_hr):
            gt_hr = 0

        hr_labels.append(gt_hr)

    return np.array(hr_labels)

def read_from_pickle_0_flat(data, data_to_load="predictions", normalize=True):
    """
    Load data from a dictionary or pickle file and flatten it into a single array per subject.

    Parameters
    ----------
    data : str or dict
        Path to a pickle file or already loaded dictionary containing the data.
    data_to_load : str, optional
        The key in the dictionary to extract (e.g., "predictions" or "labels"). Default is "predictions".
    normalize : bool, optional
        If True, normalize each sequence to zero mean and unit variance. Default is True.

    Returns
    -------
    np.ndarray
        Array of shape [num_subjects*num_batches, 1, sequence_length], flattened per subject.
    """

    if isinstance(data, str):
        with open(data, "rb") as f:
            data = pickle.load(f)

    data_dict = data[data_to_load]
    data_list = []

    for key in data_dict.keys():
        for i in range(len(data_dict[key])):
            curr_data = np.array(data_dict[key][i].cpu().numpy())
            curr_data = curr_data.reshape(-1)

            if normalize:
                curr_data = (curr_data - np.mean(curr_data)) / np.std(curr_data)

            data_list.append(curr_data)

    return np.array(data_list).reshape(len(data_dict.keys()), 1, -1)

def read_from_pickle_0_batched(data, data_to_load="predictions", normalize=True):
    """
    Load data from a dictionary or pickle file and return it in a batched format.

    Parameters
    ----------
    data : str or dict
        Path to a pickle file or already loaded dictionary containing the data.
    data_to_load : str, optional
        The key in the dictionary to extract. Default is "predictions".
    normalize : bool, optional
        If True, normalize each batch to zero mean and unit variance along the sequence dimension. Default is True.

    Returns
    -------
    np.ndarray
        Array of shape [num_subjects, num_batches, sequence_length].
    """

    if isinstance(data, str):
        with open(data, "rb") as f:
            data = pickle.load(f)

    data_dict = data[data_to_load]
    data_list = []

    min_num_batches = min([len(data_dict[key]) for key in data_dict.keys()])

    for key in data_dict.keys():
        curr_data = np.array([data_dict[key][i].cpu().numpy() for i in range(min_num_batches)])
        curr_data = curr_data.reshape(min_num_batches, -1)

        if normalize:
            curr_data = (curr_data - np.mean(curr_data, axis=-1, keepdims=True)) / np.std(curr_data, axis=-1, keepdims=True)
        
        data_list.append(curr_data)

    return np.array(data_list).reshape(len(data_dict.keys()), min_num_batches, -1)

def read_from_pickle_0(data, data_to_load="predictions", normalize=True, read_batched=True):
    """
    Wrapper function to read either batched or flat data from a dictionary or pickle file.

    Parameters
    ----------
    data : str or dict
        Path to pickle file or loaded dictionary.
    data_to_load : str
        Key to extract from the data dictionary.
    normalize : bool
        Whether to normalize sequences.
    read_batched : bool
        If True, use batched format; otherwise, flatten the sequences.

    Returns
    -------
    np.ndarray
        Processed data array.
    """

    if read_batched:
        return read_from_pickle_0_batched(data, data_to_load=data_to_load, normalize=normalize)
    else:
        return read_from_pickle_0_flat(data, data_to_load=data_to_load, normalize=normalize)


def read_from_pickle(results_file, data_to_load="rppg", read_batched=True):
    """
    Load rPPG or heart rate data from a pickle file, returning predictions, uncertainties, and labels.

    Parameters
    ----------
    results_file : str
        Path to the pickle file containing the results.
    data_to_load : str or list of str, optional
        Either "rppg", "heart_rate", or a list of keys to extract. Default is "rppg".
    read_batched : bool, optional
        Whether to read the data in batched format or flattened per subject. Default is batched.

    Returns
    -------
    tuple of np.ndarray
        Returns (uncertainties, predictions, labels). If a list of keys is provided, returns a tuple of arrays.
    """

    with open(results_file, "rb") as f:
        data = pickle.load(f)

    if isinstance(data_to_load, list):
        data_to_return = []
        for curr_data_to_load in data_to_load:
            data_to_return.append(read_from_pickle_0(data, data_to_load=curr_data_to_load, read_batched=read_batched))
        return tuple(data_to_return)
    else:
        if data_to_load == "rppg":
            uncertainties = read_from_pickle_0(data, data_to_load="uncertainties", normalize=False, read_batched=read_batched)
            preds = read_from_pickle_0(data, data_to_load="predictions", normalize=True, read_batched=read_batched)
            labels = read_from_pickle_0(data, data_to_load="labels", normalize=True, read_batched=read_batched)
        elif data_to_load == "heart_rate":
            uncertainties = read_from_pickle_0(data, data_to_load="heart_rates_uncertainty", normalize=False, read_batched=read_batched)
            preds = read_from_pickle_0(data, data_to_load="heart_rates", normalize=False, read_batched=read_batched)
            labels = read_from_pickle_0(data, data_to_load="heart_rates_labels", normalize=False, read_batched=read_batched)

        if data_to_load == "heart_rate":
            preds = preds * 240
            uncertainties = uncertainties * 240
            labels = labels * 240

    return uncertainties, preds, labels

def read_from_multiple_pickles(results_files, data_to_load="predictions", read_batched=True):
    """
    Load data from multiple pickle files and combine them into a single array.
    This assumes that all files have the same structure, keys and the arrays have compatible shapes.

    Parameters
    ----------
    results_files : list of str
        List of paths to pickle files.
    data_to_load : str, optional
        Key to extract from each file. Default is "predictions".
    read_batched : bool, optional
        Whether to read each file in batched format. Default is True.

    Returns
    -------
    np.ndarray
        Array of shape [num_files, num_subjects, ...] containing the combined data.
    """

    deserialized_list = []

    for results_file in results_files:
        deserialized = read_from_pickle_0(results_file, data_to_load=data_to_load, read_batched=read_batched)
        deserialized_list.append(deserialized)

    deserialized_array = np.array(deserialized_list)
    return deserialized_array

def get_interval_predictions(uncertainties, preds, mode="neg_log_likelihood", probability_in_interval=0.95):
    """
    Compute prediction intervals from model outputs and uncertainties.

    Parameters
    ----------
    uncertainties : np.ndarray
        Array of predicted uncertainties with the same shape as `preds`.
    preds : np.ndarray
        Array of model predictions.
    mode : str, optional
        Method to compute intervals. 
        "neg_log_likelihood" scales the interval by the uncertainty assuming Gaussian errors.
        "quantile_regression" treats `uncertainties` as a direct distance from predictions.
        Default is "neg_log_likelihood".
    probability_in_interval : float, optional
        Desired coverage probability of the interval (e.g., 0.95 for 95% intervals).

    Returns
    -------
    interval_lower : np.ndarray
        Lower bound of the prediction interval.
    interval_upper : np.ndarray
        Upper bound of the prediction interval.
    """

    assert uncertainties.shape == preds.shape, f"Shape mismatch: uncertainties {uncertainties.shape}, preds {preds.shape}"

    normalized_interval_lower = norm.ppf((1 - probability_in_interval) / 2)
    normalized_interval_upper = norm.ppf(1 - (1 - probability_in_interval) / 2)

    if mode == "neg_log_likelihood":
        interval_lower = normalized_interval_lower * uncertainties + preds
        interval_upper = normalized_interval_upper * uncertainties + preds
    elif mode == "quantile_regression":
        interval_lower = preds - uncertainties
        interval_upper = preds + uncertainties
    
    return interval_lower, interval_upper


def get_quantile(interval_lower, interval_upper, labels, alpha=0.1):
    """
    Compute the empirical quantile for conformalized prediction intervals.

    Parameters
    ----------
    interval_lower : np.ndarray
        Lower bounds of the initial prediction intervals.
    interval_upper : np.ndarray
        Upper bounds of the initial prediction intervals.
    labels : np.ndarray
        Ground truth labels corresponding to the predictions.
    alpha : float, optional
        Miscoverage level (1 - desired coverage probability). Default is 0.1 for 90% coverage.

    Returns
    -------
    empirical_quantile : float
        Quantile value used to adjust intervals to achieve the desired coverage.
    """
    errors = np.stack([interval_lower - labels, labels - interval_upper], axis=0)
    errors = np.max(errors, axis=0)

    n = labels.shape[0]*labels.shape[1]
    empirical_quantile_level = np.clip((1 - alpha) * (1 + 1/n), 0, 1)
    empirical_quantile = np.quantile(errors, empirical_quantile_level, method="linear", axis=(0, 1))
    return empirical_quantile
    
def calculate_conformalized_intervals(interval_lower, interval_upper, empirical_quantile):
    """
    Calculate the conformalized intervals based on the empirical quantile.
    """
    intervals_lower = interval_lower - empirical_quantile
    intervals_upper = interval_upper + empirical_quantile
    return intervals_lower, intervals_upper

def calculate_violations_percentage(interval_lower, interval_upper, curr_labels):
    """
    Calculate the percentage of violations based on the given intervals and labels.
    """
    num_violations = np.sum(np.logical_or(curr_labels < interval_lower, curr_labels > interval_upper))
    violations_percentage = (num_violations / np.prod(curr_labels.shape)) * 100
    return violations_percentage

def conformal_prediction_0(uncertainties_valid, preds_valid, labels_valid, uncertainties_test, preds_test, labels_test, mode, probability_in_interval):
    """
    Perform conformal prediction interval adjustment.

    Returns
    -------
    interval_lower_valid, interval_upper_valid, violations_valid,
    interval_lower_test, interval_upper_test, violations_test
    """

    interval_lower_test, interval_upper_test = get_interval_predictions(uncertainties_test, preds_test, mode, probability_in_interval)
    interval_lower_valid, interval_upper_valid = get_interval_predictions(uncertainties_valid, preds_valid, mode, probability_in_interval)

    quantile = get_quantile(interval_lower_valid, interval_upper_valid, labels_valid, alpha=(1 - probability_in_interval))

    interval_lower_valid, intervals_upper_valid = calculate_conformalized_intervals(interval_lower_valid, interval_upper_valid, quantile)
    interval_lower_test, intervals_upper_test = calculate_conformalized_intervals(interval_lower_test, interval_upper_test, quantile)

    violations_valid = calculate_violations_percentage(interval_lower_valid, intervals_upper_valid, labels_valid)
    violations_test = calculate_violations_percentage(interval_lower_test, intervals_upper_test, labels_test)

    return interval_lower_valid, intervals_upper_valid, violations_valid, interval_lower_test, intervals_upper_test, violations_test


def conformal_prediction(calib_set_outputs_path, test_set_outputs_path, mode, data_to_load, probability_in_interval):
    """
    Computes conformal prediction intervals for heart rate (or other rPPG outputs) 
    on a calibration and test set, using either preloaded outputs or files.

    Parameters
    ----------
    calib_set_outputs_path : str or tuple
        Path to the calibration set outputs file, or a tuple 
        (uncertainties_valid, preds_valid, labels_valid) if already loaded.
    test_set_outputs_path : str or tuple
        Path to the test set outputs file, or a tuple 
        (uncertainties_test, preds_test, labels_test) if already loaded.
    mode : str
        Method for computing prediction intervals (e.g., 'mean', 'quantile').
    data_to_load : str
        Identifier for which part of the data to load from the files (only used if paths are given).
    probability_in_interval : float
        Desired coverage probability for the conformal interval (e.g., 0.95 for 95% intervals).

    Returns
    -------
    interval_lower_valid : np.ndarray
        Lower bounds of the conformal prediction intervals on the calibration set.
    interval_upper_valid : np.ndarray
        Upper bounds of the conformal prediction intervals on the calibration set.
    violations_valid : float
        Percentage of calibration samples that fall outside their conformal interval.
    interval_lower_test : np.ndarray
        Lower bounds of the conformal prediction intervals on the test set.
    interval_upper_test : np.ndarray
        Upper bounds of the conformal prediction intervals on the test set.
    violations_test : float
        Percentage of test samples that fall outside their conformal interval.

    Notes
    -----
    - If the inputs are file paths, the function will automatically load them using `read_from_pickle`.
    - Conformalization is performed using the calibration set to adjust the intervals of the test set.
    """

    if isinstance(calib_set_outputs_path, tuple):   # if a tuple was supplied assume that it already contains the outputs
        uncertainties_valid, preds_valid, labels_valid = calib_set_outputs_path
        uncertainties_test, preds_test, labels_test = test_set_outputs_path
    else:
        uncertainties_valid, preds_valid, labels_valid = read_from_pickle(calib_set_outputs_path, data_to_load)
        uncertainties_test, preds_test, labels_test = read_from_pickle(test_set_outputs_path, data_to_load)

    return conformal_prediction_0(uncertainties_valid, preds_valid, labels_valid, uncertainties_test, preds_test, labels_test, mode, probability_in_interval)


def ask_and_load_results(results_file, dont_ask=False):
    """
    Check whether a results file exists, and optionally load it.

    Parameters
    ----------
    results_file : str
        Path to the JSON file.
    dont_ask : bool, optional
        If True, skip the overwrite prompt and return None (default: False).

    Returns
    -------
    dict or None
        Loaded results dictionary if user chose not to overwrite,
        otherwise None to signal a fresh start.
    """

    results = None
    if os.path.exists(results_file):
        if dont_ask:
            choice = 'n'
        else:
            choice = input(f"Results file {results_file} already exists. Do you want to overwrite it? (y/n): ")

        if choice.lower() != 'y':
            results = json.load(open(results_file, "r"))

    return results
        
def save_results(results, results_file):
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_file}")

def empty_results(num_subjects):
    """
    Initialize a results dictionary with preallocated fields.

    Parameters
    ----------
    num_subjects : int
        Number of subjects (or batches) to allocate results for.

    Returns
    -------
    dict
        Nested dictionary with metrics initialized to zero/empty values.
    """

    results = {
            "subject_id": [""] * num_subjects,
            "FFT":{ 
                "hr_label": [0] * num_subjects,
                "hr_pred": [0] * num_subjects,
                "SNR": [0] * num_subjects,
                "macc": [0] * num_subjects,
                },
            "Peak":{
                "hr_label": [0] * num_subjects,
                "hr_pred": [0] * num_subjects,
                "SNR": [0] * num_subjects,
                "macc": [0] * num_subjects,
                },
            "Vitallens":{
                "hr_label": [0] * num_subjects,
                "hr_pred": [0] * num_subjects,
                },
            "own_method":{
                "hr_label": [0] * num_subjects,
                "hr_pred": [0] * num_subjects,
                },
            "autocorrelation":{
                "hr_label": [0] * num_subjects,
                "hr_pred": [0] * num_subjects,
                },
            "Rythmformer_loss": [0] * num_subjects,
            "fitzpatrick": [0] * num_subjects,
            "age": [0] * num_subjects,
            "location": [""] * num_subjects,
            "lifenesses": [0.0] * num_subjects,
            "network_hr_prediction": [0.0] * num_subjects,
        }
    return results

def process_subject_prediction(results, i, ppg_prediction, label_ppg, gt_hr=None, bandpass_upper=None, bandpass_lower=None, fs=30):
    """
    Process predicted PPG signal against ground truth and compute heart rate metrics.

    Parameters
    ----------
    results : dict
        Results container (nested dicts with keys like "FFT", "Peak", "Vitallens", etc.)
        where computed metrics are stored at index `i`.
    i : int
        Index into results arrays.
    ppg_prediction : np.ndarray
        Predicted PPG signal (1D).
    label_ppg : np.ndarray
        Ground-truth PPG signal (1D).
    gt_hr : float, optional
        Ground-truth heart rate in BPM. If None, estimated from peaks in `label_ppg`.
    bandpass_upper : float, optional
        Upper cutoff frequency in Hz for bandpass filter (default: 3.3 Hz).
    bandpass_lower : float, optional
        Lower cutoff frequency in Hz for bandpass filter (default: 0.6 Hz).
    fs : int, optional
        Sampling frequency in Hz (default: 30).

    Notes
    -----
    - Normalizes both signals (zero mean, unit variance).
    - Applies multiple HR estimation methods:
        * FFT-based
        * Peak-based
        * Vitallens (windowed frequency domain)
        * Own method (HeartRateFilter)
        * Autocorrelation
        * RhythmFormer loss
    - Populates results dict with HR label, HR prediction, SNR, and macc
      depending on the method.
    """

    if bandpass_upper is None:
        bandpass_upper = 3.3
    if bandpass_lower is None:
        bandpass_lower = 0.6

    [b, a] = scipy.signal.butter(1, [bandpass_lower / fs * 2, bandpass_upper / fs * 2], btype='bandpass')
    ppg_prediction = scipy.signal.filtfilt(b, a, np.double(ppg_prediction))

    ppg_prediction = (ppg_prediction - np.mean(ppg_prediction)) / np.std(ppg_prediction)
    label_ppg = (label_ppg - np.mean(label_ppg)) / np.std(label_ppg)

    peaks = scipy.signal.find_peaks(ppg_prediction, height=None, distance=fs/bandpass_upper, prominence=1.5)[0]

    if gt_hr is None:
        peaks_gt = scipy.signal.find_peaks(label_ppg, height=None, distance=fs/bandpass_upper, prominence=1.5)[0]
        gt_hr = (fs / np.mean(np.diff(peaks_gt))) * 60

        if np.isnan(gt_hr):
            gt_hr = 0

    hr_label, hr_pred, SNR, macc = calculate_metric_per_video(ppg_prediction, label_ppg, fs=fs, diff_flag=True, use_bandpass=False, hr_method='FFT')
    results["FFT"]["hr_label"][i] = gt_hr
    results["FFT"]["hr_pred"][i] = float(hr_pred)
    results["FFT"]["SNR"][i] = float(SNR)
    results["FFT"]["macc"][i] = float(macc)

    hr_label, hr_pred, SNR, macc = calculate_metric_per_video(ppg_prediction, label_ppg, fs=fs, diff_flag=True, use_bandpass=False, hr_method='Peak', height=None, distance=fs/bandpass_upper, prominence=1.5)
    results["Peak"]["hr_label"][i] = gt_hr
    results["Peak"]["hr_pred"][i] = float(hr_pred)
    results["Peak"]["SNR"][i] = float(SNR)
    results["Peak"]["macc"][i] = float(macc)

    window_size = min(int(10*fs), len(ppg_prediction)//2)
    freq_estimate = windowed_freq(ppg_prediction, window_size=window_size, overlap=window_size//2, f_s=fs, f_res=0.005, f_range=(40/60, 240/60))
    freq_estimate = windowed_mean(freq_estimate, window_size=window_size, overlap=window_size//2)
    freq_estimate = np.mean(freq_estimate)

    freq_label = windowed_freq(label_ppg, window_size=window_size, overlap=window_size//2, f_s=fs, f_res=0.005, f_range=(40/60, 240/60))
    freq_label = windowed_mean(freq_label, window_size=window_size, overlap=window_size//2)
    freq_label = np.mean(freq_label)

    results["Vitallens"]["hr_label"][i] = gt_hr#60 * float(freq_label)
    results["Vitallens"]["hr_pred"][i] = 60 * float(freq_estimate)

    filter_class = HeartRateFilter()
    heart_rates = filter_class.apply(ppg_prediction)
    heart_rate = np.mean(heart_rates) * 60

    results["own_method"]["hr_label"][i] = gt_hr
    results["own_method"]["hr_pred"][i] = float(heart_rate)

    autocorrelation_filter = AutocorrelationFilter()
    autocorrelation_hr = autocorrelation_filter.apply(ppg_prediction)
    results["autocorrelation"]["hr_pred"][i] = float(autocorrelation_hr * 60)
    results["autocorrelation"]["hr_label"][i] = gt_hr

    criterion = RhythmFormer_Loss(device="cpu")
    results["Rythmformer_loss"][i] = float(criterion(torch.tensor(ppg_prediction, device="cpu"), 
                                               torch.tensor(label_ppg, device="cpu"), 
                                               1, fs, diff_flag=True))


def set_participant_metadata(results, i, gt_visualizer, subject_name="", batch_idx=0):
    """
    Populate participant metadata for a given results entry.

    Parameters
    ----------
    results : dict
        Dictionary holding results arrays/lists. Must contain keys
        "subject_id", "fitzpatrick", "age", "location".
    i : int
        Index into `results` where metadata should be stored.
    gt_visualizer : GTVisualizer or None
        If provided, metadata (Fitzpatrick skin type, age, location, video name)
        is extracted from the GTVisualizer. If None, defaults are used.
    subject_name : str, optional
        Used only if `gt_visualizer` is None. Defaults to empty string.
    batch_idx : int, optional
        Batch index appended to the subject ID. Defaults to 0.
    """
    if gt_visualizer is None:
        results["subject_id"][i] = f"{subject_name}_{batch_idx}"
        results["fitzpatrick"][i] = 0
        results["age"][i] = 0
        results["location"][i] = ""
    else:
        results["subject_id"][i] = f"{gt_visualizer.get_video_file_name()}_{batch_idx}"
        results["fitzpatrick"][i] = gt_visualizer.get_metadata()["fitzpatrick"]
        results["age"][i] = gt_visualizer.get_metadata()["age"]
        results["location"][i] = gt_visualizer.get_location()['location']

def evaluate_folder_structure(
    results_parent_folder,
    gt_files_parent_folder,
    expected_recording_length=900,
    batch_len=None,
    bandpass_upper=None,
    bandpass_lower=None,
    fs=30,
    vital_videos=True
):
    """
    Evaluate predicted PPG signals stored in a folder structure against ground truth.
    This function can create the same evaluations as load_rppg_toolbox_results for the Motion Magnification algorithms

    This function:
    - Iterates through subfolders of `results_parent_folder` (either `.mp4`-named folders for Vital Videos,
      or `subject_*` folders for other datasets).
    - Loads predicted PPG signals from `ppg_prediction.bin` using FlatbufferVideoReader.
    - Aligns predictions with ground truth signals using GTVisualizer.
    - Trims/pads predicted signals to `expected_recording_length`.
    - Splits signals into batches (if `batch_len < expected_recording_length`).
    - Processes each batch to compute evaluation metrics and stores metadata.

    Parameters
    ----------
    results_parent_folder : str
        Path to the folder containing prediction results.
    gt_files_parent_folder : str
        Path to the folder containing ground truth files.
    expected_recording_length : int, optional
        Expected number of frames per recording (default: 900).
    batch_len : int or None, optional
        Length of batches to split signals into. If None, the entire sequence is one batch.
    bandpass_upper : float or None, optional
        Upper cutoff frequency for bandpass filtering (Hz). Default: 3.3 Hz if None.
    bandpass_lower : float or None, optional
        Lower cutoff frequency for bandpass filtering (Hz). Default: 0.6 Hz if None.
    fs : int, optional
        Sampling frequency in Hz. Default: 30.
    vital_videos : bool, optional
        Whether the dataset is Vital Videos (`.mp4` folders) or a custom dataset (`subject_*` folders).
        Default: True.

    Returns
    -------
    dict
        Results dictionary with metrics, predictions, labels, and metadata
        for all subjects and batches.

    Notes
    -----
    - Predictions longer than `expected_recording_length` are trimmed.
    - Predictions shorter are padded by repeating the last frame.
    - If batching is enabled, results are stored per-batch with consistent indexing.
    """

    if bandpass_upper is None:
        bandpass_upper = 3.3
    if bandpass_lower is None:
        bandpass_lower = 0.6

    if batch_len is None:
        batch_len = expected_recording_length

    num_batches = int(np.floor(expected_recording_length / batch_len))

    folders = []
    for file in os.listdir(results_parent_folder):
        if vital_videos:
            if os.path.isdir(os.path.join(results_parent_folder, file)) and file.endswith(".mp4"):
                folders.append(os.path.join(results_parent_folder, file))
        else:
            if os.path.isdir(os.path.join(results_parent_folder, file)) and file.startswith("subject"):
                folders.append(os.path.join(results_parent_folder, file))

    num_folders = len(folders)
    results = empty_results(num_folders*num_batches)

    for i in tqdm.tqdm(range(num_folders)):
        curr_folder = folders[i]
        ppg_prediction_file = os.path.join(curr_folder, "ppg_prediction.bin")
        video_file_name = os.path.basename(curr_folder)
        ppg_reader = FlatbufferVideoReader(ppg_prediction_file)
        predicted_ppg = ppg_reader.read_n(ppg_reader.get_total_frames()).squeeze((1, 2, 3))

        if vital_videos:
            participant_id = video_file_name.split("_")[0]
            gt_file = os.path.join(gt_files_parent_folder, f"{participant_id}.json")
        else:
            gt_file = os.path.join(gt_files_parent_folder, f"{video_file_name}/ground_truth.txt")

        gt_visualizer = GTVisualizer(gt_file, video_file_name, vital_videos=vital_videos)

        if predicted_ppg.shape[0] > expected_recording_length:
            print(f"Warning: trimming because of length {predicted_ppg.shape[0]} > {expected_recording_length}")
            predicted_ppg = predicted_ppg[:expected_recording_length]
        elif predicted_ppg.shape[0] < expected_recording_length:
            print(f"Warning: padding because of length {predicted_ppg.shape[0]} < {expected_recording_length}")
            predicted_ppg = np.pad(predicted_ppg, ((0, expected_recording_length - predicted_ppg.shape[0]), (0, 0)), mode='edge')

        timestamps, ppg_values, ppg_prediction = gt_visualizer.resample_ppg(predicted_ppg, fps=30, num_frames=predicted_ppg.shape[0])
        
        if num_batches > 1:
            for batch_idx in range(num_batches):
                current_results_index = i*num_batches + batch_idx
                start_idx = batch_idx * batch_len
                end_idx = (batch_idx + 1) * batch_len

                ppg_prediction_batch = ppg_prediction[start_idx:end_idx]
                ppg_values_batch = ppg_values[start_idx:end_idx]

                process_subject_prediction(
                    results,
                    current_results_index,
                    ppg_prediction_batch,
                    ppg_values_batch,
                    bandpass_upper=bandpass_upper,
                    bandpass_lower=bandpass_lower,
                    fs=fs
                )

                if vital_videos:
                    set_participant_metadata(results, current_results_index, gt_visualizer, batch_idx=batch_idx)
                else:
                    set_participant_metadata(results, current_results_index, None, subject_name=str(i), batch_idx=batch_idx)
        else:
            process_subject_prediction(
                results,
                i,
                ppg_prediction,
                ppg_values,
                bandpass_upper=bandpass_upper,
                bandpass_lower=bandpass_lower,
                fs=fs
            )

            if vital_videos:
                set_participant_metadata(results, i, gt_visualizer)
            else:
                set_participant_metadata(results, i, None)
        
    return results

def get_rppg_toolbox_conversion_dict():
    """
    Build a conversion dictionary for the rPPG toolbox by scanning the vitalVideos dataset.
    The purpose is to create a mapping between the index in the Dataloader and the corresponding video file metadata.

    Returns
    -------
    list of dict
        A list of dictionaries, each containing:
        - "index" : int
            Sequential index of the entry after sorting.
        - "path" : str
            Full path to the `.mp4` file.
        - "subject" : str
            Subject identifier extracted from the filename.
        - "location" : str
            Location information extracted using GTVisualizer.

    Raises
    ------
    ValueError
        If no `.mp4` files are found in the dataset directory.
    """
    data_path = "/mnt/data/vitalVideos"
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
            print(f"JSON file {json_file} does not exist for subject {subject}. Skipping this directory.")
            continue

        # Append the directory information to the list
        dirs.append({"index": index, "path": data_dir, "subject": subject, "location": location})
    
    dirs = sorted(dirs, key=lambda x: (x['location'] + x['path']))

    for i in range(len(dirs)):
        dirs[i]['index'] = i

    return dirs

def apply_index_to_dict(dictonary, idx):
    """
    Apply an index or slice to all list or numpy.ndarray elements within a nested dictionary.

    This function traverses the dictionary recursively. For each key:
    - If the value is a list, it converts it to a NumPy array, applies the index, and converts back to list.
    - If the value is a NumPy array, it applies the index directly.
    - If the value is another dictionary, it recursively applies the indexing.
    - Other data types are not supported.

    Parameters
    ----------
    dictionary : dict
        A (possibly nested) dictionary containing lists, numpy arrays, or nested dictionaries.
    idx : int, slice, or array-like
        Index or slice to apply to each array/list element.

    Returns
    -------
    dict
        The same dictionary structure, with each list/array element indexed according to `idx`.

    Raises
    ------
    ValueError
        If a value type is not list, np.ndarray, or dict.
    """
    
    for key in dictonary.keys():
        print(key, type(dictonary))
        if isinstance(dictonary[key], list):
            dictonary[key] = (np.array(dictonary[key])[idx]).tolist()
        elif isinstance(dictonary[key], np.ndarray):
            dictonary[key] = dictonary[key][idx]
        elif isinstance(dictonary[key], dict):
            dictonary[key] = apply_index_to_dict(dictonary[key], idx)
        else:
            raise ValueError(f"Unsupported type: {type(dictonary[key])}")

    return dictonary

def all_tensors_to_list(dictonary):
    """
    Convert all PyTorch tensors in a nested dictionary or list to Python lists.

    This function traverses the dictionary recursively. For each key:
    - If the value is a torch.Tensor, it is converted to a list.
    - If the value is a list containing tensors, each tensor is converted to a list.
    - If the value is a dictionary, the function is applied recursively.
    - Other types remain unchanged.

    Parameters
    ----------
    dictionary : dict
        A (possibly nested) dictionary containing torch tensors, lists, or nested dictionaries.

    Returns
    -------
    dict
        The same dictionary structure, with all torch.Tensor objects converted to Python lists.
    """
    if dictonary is None:
        return None
    
    for key, value in dictonary.items():
        if isinstance(value, dict):
            all_tensors_to_list(value)
        elif isinstance(value, list):
            dictonary[key] = [v.tolist() if isinstance(v, torch.Tensor) else v for v in value]
        elif isinstance(value, torch.Tensor):
            dictonary[key] = value.tolist()

    return dictonary

def load_rppg_toolbox_results(
    results_file,
    gt_files_parent_folder,
    expected_recording_length=900,
    bandpass_upper=None,
    bandpass_lower=None,
    fs=30,
    vital_videos=True,
    vital_videos_offset=0,
    vital_videos_gt_peaks_file=None
):
    """
    Load and process rPPG predictions from the rPPG Toolbox results file.

    This function reads a pickle file containing rPPG predictions and optional
    network outputs such as lifeness or predicted heart rates. It aligns these
    predictions with ground-truth heart rate data, applies optional bandpass
    filtering, computes HR metrics, and stores results in a structured format.

    Parameters
    ----------
    results_file : str
        Path to the pickle file containing rPPG Toolbox predictions.
    gt_files_parent_folder : str
        Path to the folder containing ground truth JSON files for visualization
        and metadata.
    expected_recording_length : int, optional
        Number of samples per video to consider (default is 900).
    bandpass_upper : float or None, optional
        Upper frequency for bandpass filtering (Hz). If None, no upper limit is applied.
    bandpass_lower : float or None, optional
        Lower frequency for bandpass filtering (Hz). If None, no lower limit is applied.
    fs : float, optional
        Sampling frequency of the video/PPG signal (default is 30 Hz).
    vital_videos : bool, optional
        Whether the dataset follows the VitalVideos format (default is True).
    vital_videos_offset : int or float, optional
        Index offset for the VitalVideos dataset when processing (default is 0).
        If float, it is interpreted as a fraction of the dataset length.
    vital_videos_gt_peaks_file : str or None, optional
        Path to a JSON file containing ground-truth peak locations for VitalVideos.
        Required if `vital_videos=True`.

    Returns
    -------
    dict
        A dictionary containing processed rPPG predictions, optional lifeness,
        network HR predictions, and associated metadata. The structure is
        compatible with the rest of the analysis pipeline.

    Raises
    ------
    ValueError
        If `vital_videos_gt_peaks_file` is not provided when `vital_videos=True`.

    Notes
    -----
    - The output dictionary is initialized using `empty_results` and updated batch by batch.
    - Metadata such as participant ID, dataset index, and batch index is set for each sample.
    - Bandpass filtering is applied to the PPG signals if `bandpass_upper` and/or `bandpass_lower` are provided.
    """

    if not hasattr(torch.serialization, "_package_registered"):
        torch.serialization.register_package(0, lambda x: x.device.type, lambda x, _: x.cpu())
        torch.serialization._package_registered = True

    with open(results_file, "rb") as f:
        data = pickle.load(f)

    if vital_videos_gt_peaks_file is None and vital_videos:
        raise ValueError("vital_videos_gt_peaks_file must be provided for vital videos dataset")

    if vital_videos:
        conversion_dict = get_rppg_toolbox_conversion_dict()

        if isinstance(vital_videos_offset, float):
            vital_videos_offset = int(vital_videos_offset * len(conversion_dict))
    
        gt_peaks_times = json.load(open(vital_videos_gt_peaks_file, "r"))["peaks"]
        gt_peaks_names = json.load(open(vital_videos_gt_peaks_file, "r"))["filenames"]

    labels = data["labels"]
    num_data_points = len(labels)
    num_batches_all = 0
    current_index = 0

    for i, label in enumerate(labels):
        num_batches_all += len(data["predictions"][label])

    results = empty_results(num_batches_all)
    lifenesses = None
    heart_rates_network = None

    if 'lifenesses' in data:
        lifenesses = all_tensors_to_list(data['lifenesses'])
    else:
        del results["lifenesses"]

    if 'heart_rates' in data:
        heart_rates_network = all_tensors_to_list(data['heart_rates'])
    else:
        del results["network_hr_prediction"]
        
    for i, label in enumerate(tqdm.tqdm(labels, desc="Processing subjects")):
        if vital_videos:
            curr_data_dir = conversion_dict[i + vital_videos_offset]
            video_file_name = os.path.basename(curr_data_dir["path"])
            curr_subject_id = video_file_name.split("_")[0]
            curr_dataset_index = i

            curr_gt_peaks_index = gt_peaks_names.index(video_file_name)
            curr_gt_peaks = np.array(gt_peaks_times[curr_gt_peaks_index])

        for batch_idx in range(len(data["predictions"][label])):

            if vital_videos:
                start_idx = batch_idx * len(data["predictions"][label][0])
                end_idx = (batch_idx + 1) * len(data["predictions"][label][0])
                peaks = curr_gt_peaks[(curr_gt_peaks >= start_idx) & (curr_gt_peaks < end_idx)] - start_idx
                curr_gt_hr = (fs / np.mean(np.diff(peaks))) * 60
            else:
                video_file_name = str(i)
                curr_gt_hr = None

            ppg_prediction = data["predictions"][label][batch_idx]
            ppg_label = data["labels"][label][batch_idx]

            ppg_prediction = np.array(ppg_prediction).flatten()
            ppg_label = np.array(ppg_label).flatten()

            if lifenesses is not None:
                results["lifenesses"][current_index] = lifenesses[label][batch_idx]

            if heart_rates_network is not None:
                results["network_hr_prediction"][current_index] = heart_rates_network[label][batch_idx]

            process_subject_prediction(
                results,
                current_index,
                ppg_prediction,
                ppg_label,
                bandpass_upper=bandpass_upper,
                bandpass_lower=bandpass_lower,
                fs=fs,
                gt_hr=curr_gt_hr
            )

            if vital_videos:
                gt_file = os.path.join(gt_files_parent_folder, f"{curr_subject_id}.json")
                gt_visualizer = GTVisualizer(gt_file, video_file_name)
                set_participant_metadata(results, current_index, gt_visualizer, batch_idx=batch_idx)
            else:
                set_participant_metadata(results, current_index, None, subject_name=f"subject_{i}", batch_idx=batch_idx)

            current_index += 1


    
    return results


def get_MAE_from_results(results, method="Peak"):
    """
    Compute the Mean Absolute Error (MAE) between predicted and labeled heart rates.

    Parameters
    ----------
    results : dict
        Dictionary containing prediction results as returned by get_results. Expected structure:
        results[method]["hr_label"] and results[method]["hr_pred"].
    method : str, optional
        Key in the results dictionary specifying the method to extract (default is "Peak").

    Returns
    -------
    float
        Mean Absolute Error between hr_label and hr_pred. Returns np.nan if the key is missing or results is None.
    """
    if results is None or method not in results:
        print(f"No '{method}' key found in results. Returning NaN.")
        return np.nan

    results_method = np.stack([results[method]["hr_label"], results[method]["hr_pred"]], axis=0)
    results_method = np.nan_to_num(results_method, nan=0.0)

    mae_method = np.mean(np.abs(results_method[0] - results_method[1]))
    return mae_method

def get_mean_SNR_from_results(results):
    """
    Compute the mean Signal-to-Noise Ratio (SNR) from FFT results.

    Parameters
    ----------
    results : dict
        Dictionary containing prediction results as returned by get_results. Expected structure:
        results['FFT']["SNR"].

    Returns
    -------
    float
        Mean SNR across all entries. Returns np.nan if 'SNR' is missing.
    """
    all_snrs = results['FFT']["SNR"]
    if all_snrs is None:
        print("No 'SNR' key found in 'FFT'. Returning NaN.")
        return np.nan

    return np.mean(all_snrs)


def postprocess_rppg(rppg_signal, postprocessor, fs=30):
    """
    Postprocesses the rPPG (remote photoplethysmography) signal using the specified postprocessing method.
    This corresponds to the postprocessing step described in the paper

    Parameters
    ----------
    rppg_signal : np.ndarray
        The input rPPG signal to be postprocessed.
    postprocessor : str or None
        The postprocessing method to apply. Supported values are:
            - "cumsum": Cumulative sum of the signal.
            - "detrend": Detrending the signal using a window of 100 samples.
            - "butter": Bandpass Butterworth filtering (0.6 Hz to 3.3 Hz).
            - "cumsum_detrend": Cumulative sum followed by detrending.
            - "cumsum_detrend_butter": Cumulative sum, detrending, then Butterworth filtering.
            - "None" or None: No postprocessing.
    fs : int, optional
        Sampling frequency of the signal in Hz (default is 30).

    Returns
    -------
    np.ndarray
        The postprocessed rPPG signal, reshaped to the original input shape.

    Raises
    ------
    ValueError
        If an unknown postprocessor is specified.
    """

    orig_shape = rppg_signal.shape

    if postprocessor == "cumsum":
        result = np.cumsum(rppg_signal)
    elif postprocessor == "detrend":
        result = _detrend(rppg_signal, 100)
    elif postprocessor == "butter":
        [b, a] = scipy.signal.butter(1, [0.6 / fs * 2, 3.3 / fs * 2], btype='bandpass')
        result = scipy.signal.filtfilt(b, a, np.double(rppg_signal))
    elif postprocessor == "cumsum_detrend":
        result = _detrend(np.cumsum(rppg_signal), 100)
    elif postprocessor == "cumsum_detrend_butter":
        rppg_signal = np.cumsum(rppg_signal)
        rppg_signal = _detrend(rppg_signal, 100)
        [b, a] = scipy.signal.butter(1, [0.6 / fs * 2, 3.3 / fs * 2], btype='bandpass')
        result = scipy.signal.filtfilt(b, a, np.double(rppg_signal))
    elif postprocessor == "None" or postprocessor is None:
        return rppg_signal
    else:
        raise ValueError(f"Unknown postprocessor: {postprocessor}")

    return result.reshape(orig_shape)

def get_data_loader_name_for_dataset(dataset_name):
    if dataset_name == "train" or dataset_name == "validation" or dataset_name == "test" or dataset_name == "validation_shuffle" or dataset_name == "test_shuffle":
        return "VitalVideos_and_UBFC"
    elif dataset_name == "dead" or dataset_name == "own_videos":
        return "Own_Videos"
    elif dataset_name == "emergency":
        return "Emergency_Videos"
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    
def get_size_from_model_name(model_name):
    if model_name == "RythmFormer" or model_name == "PhysFormer":
        return 128
    else:
        return 72

def get_dd_frequency_from_stabilizer(stabilizer):
    if stabilizer == "Y5F" or stabilizer == "YUNET" or stabilizer == "HC":
        return 30
    else:
        return 1
    
def get_pickle_path(
        exp_name,
        dataset_name="validation",
        model_name="Physnet",
        stabilizer="YUNET",
        destabilizer="None",
        destabilizer_amp=10.0,
        size=None,
        dd_frequency=None,
        data_loader=None,
        ppg_preprocessor=None
    ):
    """
    Constructs the file path for the pickle file containing saved test outputs for a given experiment configuration.
    Parameters:
        exp_name (str): Name of the experiment.
        dataset_name (str, optional): Name of the dataset. Defaults to "validation".
        model_name (str, optional): Name of the model. Defaults to "Physnet".
        stabilizer (str, optional): Name of the stabilizer used. Defaults to "YUNET".
        destabilizer (str, optional): Name of the destabilizer used. Defaults to "None".
        destabilizer_amp (float, optional): Amplitude value for the destabilizer. Defaults to 10.0.
        size (int, optional): Size parameter for the model. If None, determined from model_name.
        dd_frequency (int, optional): Dynamic detection frequency. If None, determined from stabilizer.
        data_loader (str, optional): Name of the data loader. If None, determined from dataset_name.
        ppg_preprocessor (str, optional): Name of the PPG preprocessor. If provided, appended to model_name.
    Returns:
        str: The constructed file path to the pickle file for the specified experiment configuration.
    """

    if dd_frequency is None:
        dd_frequency = get_dd_frequency_from_stabilizer(stabilizer)

    if data_loader is None:
        data_loader = get_data_loader_name_for_dataset(dataset_name)

    if size is None:
        size = get_size_from_model_name(model_name)

    if ppg_preprocessor is not None:
        model_name = f"{model_name}_{ppg_preprocessor}"

    return f"runs/{exp_name}/{dataset_name}/{data_loader}_SizeW{size}_SizeH{size}_ClipLength160_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_Backend{stabilizer}_Large_boxTrue_Large_size1.5_Dyamic_DetTrue_det_len{dd_frequency}_Median_face_boxFalse_D{destabilizer}_amp{str(float(destabilizer_amp))}/saved_test_outputs/VitalLens_{model_name}_best_{data_loader}_outputs.pickle"

def get_results(
    results_file,
    dataset="validation",
    gt_files_parent_folder="/mnt/data/vitalVideos",
    vital_videos_gt_peaks_file="/mnt/data/vitalVideos/00_peaks.json",
    expected_recording_length=900,
    batch_len=160,
    bandpass_lower=0.6,
    bandpass_upper=3.3,
    vital_videos=True,
    always_recalculate=False
):
    """
    Loads or calculates rPPG Toolbox results for a given dataset and configuration.
    Depending on the dataset type, sets appropriate offsets and loads ground truth files.
    If results have already been calculated and saved as a JSON file, loads them unless
    `always_recalculate` is True. Otherwise, recalculates results using the provided parameters
    and saves them for future use.
    Args:
        results_file (str): Path to the results file (pickle format).
        dataset (str, optional): Dataset type, one of "validation", "test", "train". Defaults to "validation".
        gt_files_parent_folder (str, optional): Parent folder containing ground truth files. Defaults to "/mnt/data/vitalVideos".
        vital_videos_gt_peaks_file (str, optional): Path to vital videos ground truth peaks file. Defaults to "/mnt/data/vitalVideos/00_peaks.json".
        expected_recording_length (int, optional): Expected length of recordings in seconds. Defaults to 900.
        batch_len (int, optional): Length of each batch for processing. Defaults to 160.
        bandpass_lower (float, optional): Lower frequency bound for bandpass filtering. Defaults to 0.6.
        bandpass_upper (float, optional): Upper frequency bound for bandpass filtering. Defaults to 3.3.
        vital_videos (bool, optional): Whether to use vital videos ground truth. Defaults to True.
        always_recalculate (bool, optional): If True, always recalculate results even if cached results exist. Defaults to False.
    Returns:
        dict: Results dictionary containing evaluation metrics and processed data.
    """

    if dataset == "test":
        vital_videos_offset = 0.7
    elif dataset == "validation":
        vital_videos_offset = 0.86
    elif dataset == "train":
        vital_videos_offset = 0.0
    else:
        #print(f"Unknown dataset {dataset}, setting vital_videos to False and vital_videos_offset to 0.0")
        vital_videos = False
        vital_videos_offset = 0.0

    results_file_json = results_file.split(".pickle")[0] + ".json"

    if always_recalculate:
        results = None
    else:
        results = ask_and_load_results(results_file_json, dont_ask=True) 

    if results is None:
        results = load_rppg_toolbox_results(
                results_file,
                gt_files_parent_folder,
                expected_recording_length=expected_recording_length,
                bandpass_upper=bandpass_upper,
                bandpass_lower=bandpass_lower,
                vital_videos=vital_videos,
                vital_videos_offset=vital_videos_offset,
                vital_videos_gt_peaks_file=vital_videos_gt_peaks_file
            )
        save_results(results, results_file_json)

    return results