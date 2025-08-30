import itertools
import matplotlib
import torch
matplotlib.use("Agg")  # Use a non-interactive backend for matplotlib

from matplotlib import pyplot as plt
from PyEVM.evaluation import heart_rate_filter
from evaluation.utils import read_from_pickle_0, get_pickle_path

from evaluation.post_process import _detrend

from neural_methods.loss.PhysNetUncertaintyLoss import neg_log_likelihood_function

import numpy as np

from scipy import signal
from tqdm import tqdm
import optuna

from scipy.signal import butter, filtfilt
from tabulate import tabulate
import pandas as pd

def neg_log_likelihood_function_numpy(preds, uncertainty_pred, labels):
    preds = torch.tensor(preds)
    uncertainty_pred = torch.tensor(uncertainty_pred)
    labels = torch.tensor(labels)

    result = neg_log_likelihood_function(preds, uncertainty_pred, labels)
    return result.to("cpu").detach().numpy()

def get_avg_error_custom_filter(predictions, uncertainties, hr_labels, params):

    avg_error = 0
    avg_loss = 0
    f_res, blur_sigma, peaks_prior_sigma, window_size, num_iterations, height, prominence = params

    hr_filter = heart_rate_filter.HeartRateFilter(
        f_res=f_res,
        blur_sigma=blur_sigma,
        peaks_prior_sigma=peaks_prior_sigma,
        window_size=int(window_size),
        num_iterations=num_iterations,

    )
    for pred, uncertainty, hr_label in zip(predictions, uncertainties, hr_labels):
        heart_rates = hr_filter.apply(pred)
        hr = np.mean(heart_rates) * 60
        avg_error += abs(hr - hr_label)
        avg_loss += neg_log_likelihood_function_numpy(hr, uncertainty, hr_label)

    avg_error /= len(predictions)
    avg_loss /= len(predictions)
    return avg_error, avg_loss

def avg_error_find_peaks(predictions, uncertainties, hr_labels, params, sampling_rate=30):

    avg_error = 0
    avg_loss = 0
    height, prominence, distance = params

    for pred, uncertainty, hr_label in zip(predictions, uncertainties, hr_labels):
        peaks = signal.find_peaks(pred, height=height, prominence=prominence, distance=distance)[0]

        if len(peaks) < 2:
            hr = 0
        else:
            hr = (sampling_rate / np.mean(np.diff(peaks))) * 60

        avg_error += abs(hr - hr_label)
        avg_loss += neg_log_likelihood_function_numpy(hr, uncertainty, hr_label)

    avg_error /= len(predictions)
    avg_loss /= len(predictions)
    return avg_error, avg_loss

def avg_error_fft(predictions, uncertainties, hr_labels, sampling_rate=30, min_freq=0.5, max_freq=3.3):
    avg_error = 0
    avg_loss = 0
    for pred, hr_label, uncertainty in zip(predictions, hr_labels, uncertainties):

        fft_values = np.abs(np.fft.fft(pred))
        freqs = np.fft.fftfreq(len(pred), d=1/sampling_rate)

        # Filter frequencies within the desired range
        valid_freqs = (freqs >= min_freq) & (freqs <= max_freq)
        fft_values = fft_values[valid_freqs]
        freqs = freqs[valid_freqs]

        peak_freq = freqs[np.argmax(fft_values)]
        hr = peak_freq * 60  # Convert frequency to beats per minute
        avg_error += abs(hr - hr_label)

        avg_loss += neg_log_likelihood_function_numpy(hr, uncertainty, hr_label)

    avg_error /= len(predictions)
    avg_loss /= len(predictions)
    return avg_error, avg_loss

def avg_error_autocorrelation(predictions, uncertainties, hr_labels, params, sampling_rate=30, min_freq=0.5, max_freq=3.3):
    
    height, prominence = params
    
    filter = heart_rate_filter.AutocorrelationFilter(
        sampling_rate=sampling_rate,
        min_freq=min_freq,
        max_freq=max_freq,
        height=height,
        prominence=prominence
    )

    avg_error = 0
    avg_loss = 0
    for pred, hr_label, uncertainty in zip(predictions, hr_labels, uncertainties):
        hr = filter.apply(pred)
        avg_error += abs(hr - hr_label)
        avg_loss += neg_log_likelihood_function_numpy(hr, uncertainty, hr_label)

    avg_error /= len(predictions)
    avg_loss /= len(predictions)
    return avg_error, avg_loss

def avg_error_fft_autocorrelation(predictions, uncertainties, hr_labels, params, sampling_rate=30, min_freq=0.5, max_freq=3.3):
    
    height, prominence = params

    filter = heart_rate_filter.AutocorrelationFFTFilter(
        sampling_rate=sampling_rate,
        min_freq=min_freq,
        max_freq=max_freq,
        height=height,
        prominence=prominence
    )

    avg_error = 0
    avg_loss = 0
    for pred, hr_label, uncertainty in zip(predictions, hr_labels, uncertainties):
        hr = filter.apply(pred)
        avg_error += abs(hr - hr_label)
        avg_loss += neg_log_likelihood_function_numpy(hr, uncertainty, hr_label)

    avg_error /= len(predictions)
    avg_loss /= len(predictions)
    return avg_error, avg_loss

def custom_filter_search(predictions_valid, uncertainties, hr_labels_valid, height=0.01, prominence=0.01, n_trials=50):

    def objective(trial):
        # Parameter suggestions within your original ranges
        f_res = trial.suggest_float("f_res", 0.001, 0.1)
        blur_sigma = trial.suggest_float("blur_sigma", 0.01, 0.5)
        peaks_prior_sigma = trial.suggest_float("peaks_prior_sigma", 0.01, 0.5)
        window_size = trial.suggest_int("window_size", 50, 1000, step=1)
        num_iterations = trial.suggest_int("num_iterations", 1, 3)

        params = (f_res, blur_sigma, peaks_prior_sigma, window_size, num_iterations, height, prominence)
        avg_error, _ = get_avg_error_custom_filter(predictions_valid, uncertainties, hr_labels_valid, params)

        return avg_error  # we are minimizing the average error

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_params_full = (
        best_params["f_res"],
        best_params["blur_sigma"],
        best_params["peaks_prior_sigma"],
        best_params["window_size"],
        best_params["num_iterations"],
        height,
        prominence
    )
    
    avg_error, avg_loss = get_avg_error_custom_filter(predictions_valid, uncertainties, hr_labels_valid, best_params_full)
    return best_params_full, avg_error, avg_loss

def peaks_height_search(predictions_valid, uncertainties, hr_labels_valid, n_trials=50):
    def objective(trial):
        height = trial.suggest_float("height", 0.01, 3.0)
        prominence = trial.suggest_float("prominence", 0.01, 3.0)
        distance = trial.suggest_int("distance", 1, 100)

        params = (height, prominence, distance)
        avg_error, _ = avg_error_find_peaks(predictions_valid, uncertainties, hr_labels_valid, params)

        return avg_error  # minimizing error

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_error, best_loss = avg_error_find_peaks(predictions_valid, uncertainties, hr_labels_valid, (best_params["height"], best_params["prominence"], best_params["distance"]))

    return (best_params["height"], best_params["prominence"], best_params["distance"]), best_error, best_loss

def autocorrelation_search(predictions_valid, uncertainties, hr_labels_valid, n_trials=50):
    def objective(trial):
        height = trial.suggest_float("height", 0.01, 3.0)
        prominence = trial.suggest_float("prominence", 0.01, 3.0)

        params = (height, prominence)
        avg_error, _ = avg_error_autocorrelation(predictions_valid, uncertainties, hr_labels_valid, params)
        return avg_error  # minimizing error

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_error, best_loss = avg_error_autocorrelation(predictions_valid, uncertainties, hr_labels_valid, (best_params["height"], best_params["prominence"]))

    return (best_params["height"], best_params["prominence"]), best_error, best_loss

def autocorrelation_fft_search(predictions_valid, uncertainties, hr_labels_valid, n_trials=50):
    def objective(trial):
        height = trial.suggest_float("height", 0.01, 0.1)
        prominence = trial.suggest_float("prominence", 0.01, 0.1)

        params = (height, prominence)
        avg_error, _ = avg_error_fft_autocorrelation(predictions_valid, uncertainties, hr_labels_valid, params)
        return avg_error  # minimizing error

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_error, best_loss = avg_error_fft_autocorrelation(predictions_valid, uncertainties, hr_labels_valid, (best_params["height"], best_params["prominence"]))

    return (best_params["height"], best_params["prominence"]), best_error, best_loss

def do_tests(predictions_valid, hr_labels_valid, uncertainties_valid, predictions_test, hr_labels_test, uncertainties_test):
    
    best_params, best_error, best_loss = peaks_height_search(predictions_valid, uncertainties_valid, hr_labels_valid)
    best_height, best_prominence, best_distance = best_params

    test_error, test_loss = avg_error_find_peaks(predictions_test, uncertainties_test, hr_labels_test, best_params)
    test_error_peaks, valid_error_peaks = test_error, best_error
    test_loss_peaks, valid_loss_peaks = test_loss, best_loss
    peaks_result = (test_error_peaks, valid_error_peaks, test_loss_peaks, valid_loss_peaks)

    best_params, best_error, best_loss = custom_filter_search(predictions_valid, uncertainties_valid, hr_labels_valid, best_height, best_prominence)

    test_error, test_loss = get_avg_error_custom_filter(predictions_test, uncertainties_test, hr_labels_test, best_params)
    valid_error_custom, test_error_custom = best_error, test_error
    valid_loss_custom, test_loss_custom = best_loss, test_loss
    custom_result = (test_error_custom, valid_error_custom, test_loss_custom, valid_loss_custom)

    test_error, test_loss = avg_error_fft(predictions_test, uncertainties_test, hr_labels_test)
    valid_error_fft, valid_loss_fft = avg_error_fft(predictions_valid, uncertainties_valid, hr_labels_valid)
    test_error_fft = test_error
    fft_result = (test_error_fft, valid_error_fft, test_loss, valid_loss_fft)

    best_params, best_error, best_loss = autocorrelation_search(predictions_valid, uncertainties_valid, hr_labels_valid)

    test_error_autocorr, test_loss_autocorr = avg_error_autocorrelation(predictions_test, uncertainties_test, hr_labels_test, best_params)
    valid_error_autocorr, valid_loss_autocorr = avg_error_autocorrelation(predictions_valid, uncertainties_valid, hr_labels_valid, best_params)
    autocorr_result = (test_error_autocorr, valid_error_autocorr, test_loss_autocorr, valid_loss_autocorr)

    best_params, best_error, best_loss = autocorrelation_fft_search(predictions_valid, uncertainties_valid, hr_labels_valid)

    test_error_fft_autocorr, test_loss_fft_autocorr = avg_error_fft_autocorrelation(predictions_test, uncertainties_test, hr_labels_test, best_params)
    valid_error_fft_autocorr, valid_loss_fft_autocorr = avg_error_fft_autocorrelation(predictions_valid, uncertainties_valid, hr_labels_valid, best_params)
    fft_result_autocorr = (test_error_fft_autocorr, valid_error_fft_autocorr, test_loss_fft_autocorr, valid_loss_fft_autocorr)

    return [peaks_result, custom_result, fft_result, autocorr_result, fft_result_autocorr]

def get_hr_errors_from_pickle(pickle_path, hr_labels=None):
    hr_predictions = read_from_pickle_0(pickle_path, data_to_load="heart_rates", normalize=False) * 240  # Convert to beats per minute
    uncertainty_predictions = read_from_pickle_0(pickle_path, data_to_load="heart_rates_uncertainty", normalize=False) * 240  # Convert to beats per minute

    if hr_labels is None:
        hr_labels = read_from_pickle_0(pickle_path, data_to_load="heart_rates_labels", normalize=False) * 240  # Convert to beats per minute
        hr_labels = hr_labels.reshape(-1)
    
    hr_predictions = hr_predictions.reshape(-1)
    uncertainty_predictions = uncertainty_predictions.reshape(-1)

    error = np.mean(np.abs(hr_predictions - hr_labels))
    all_losses = neg_log_likelihood_function_numpy(hr_predictions, uncertainty_predictions, hr_labels)
    loss = np.mean(all_losses)
    return error, loss

def evaluate_neural_network(validation_set_path, test_set_path, hr_labels_valid, hr_labels_test):
    valid_error, valid_loss = get_hr_errors_from_pickle(validation_set_path, hr_labels=hr_labels_valid)
    test_error, test_loss = get_hr_errors_from_pickle(test_set_path, hr_labels=hr_labels_test)

    return (test_error, valid_error, test_loss, valid_loss)

def append_to_results(results, name, list_of_results):
    for i in range(len(results)):
        results[i].append([name])

    for i, result in enumerate(list_of_results):
        for j in range(len(result)):
            results[j][-1].append(result[j])

    return results

def print_table(results, csv_path=None):
    headers = ["Preprocessing", "Peaks", "Custom", "FFT", "Autocorrelation", "FFT+Autocorrelation", "DNN"]
    table = []
    for row in results:
        table.append([
            row[0],
            f"{row[1]:.2f}",
            f"{row[2]:.2f}",
            f"{row[3]:.2f}",
            f"{row[4]:.2f}",
            f"{row[5]:.2f}",
            f"{row[6]:.2f}"
        ])
    print()
    print(tabulate(table, headers=headers, tablefmt="latex"))
    print()

    if csv_path:
        df = pd.DataFrame(table, columns=headers)
        df.to_excel(csv_path, index=False)
        print(f"Results exported to {csv_path}")

def generate_path(dataset_name="validation", uncertainty_type="NegLogLikelihood", preprocessing="None"):
    model_name = f"Physnet_HR_Classifier_{uncertainty_type}"
    return get_pickle_path("Uncertainty", dataset_name=dataset_name, model_name=model_name, ppg_preprocessor=preprocessing)
    #return f"runs/Uncertainty/{dataset_name}/VitalVideos_and_UBFC_SizeW72_SizeH72_ClipLength160_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendY5F_Large_boxTrue_Large_size1.5_Dyamic_DetTrue_det_len30_Median_face_boxFalse_DNone_amp10.0/saved_test_outputs/VitalLens_Physnet_HR_Classifier_{uncertainty_type}_{preprocessing}_best_VitalVideos_and_UBFC_outputs.pickle"

if __name__ == "__main__":

    fs = 30  # Sampling frequency in Hz
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    for uncertainty_type in ["NegLogLikelihood", "Quantile"]:
        print(f"Running hyperparameter search for uncertainty type: {uncertainty_type}")

        validation_set_path = generate_path(dataset_name="validation", uncertainty_type=uncertainty_type, preprocessing="None")
        test_set_path = generate_path(dataset_name="test", uncertainty_type=uncertainty_type, preprocessing="None")

        cumsum_validation_set_path = generate_path(dataset_name="validation", uncertainty_type=uncertainty_type, preprocessing="cumsum")
        cumsum_test_set_path = generate_path(dataset_name="test", uncertainty_type=uncertainty_type, preprocessing="cumsum")

        detrend_validation_set_path = generate_path(dataset_name="validation", uncertainty_type=uncertainty_type, preprocessing="detrend")
        detrend_test_set_path = generate_path(dataset_name="test", uncertainty_type=uncertainty_type, preprocessing="detrend")

        butter_validation_set_path = generate_path(dataset_name="validation", uncertainty_type=uncertainty_type, preprocessing="butter")
        butter_test_set_path = generate_path(dataset_name="test", uncertainty_type=uncertainty_type, preprocessing="butter")


        predictions_valid = read_from_pickle_0(validation_set_path, data_to_load="predictions", normalize=True)
        predictions_test = read_from_pickle_0(test_set_path, data_to_load="predictions", normalize=True)

        uncertainties_predictions_valid = read_from_pickle_0(validation_set_path, data_to_load="uncertainties", normalize=False)
        uncertainties_predictions_test = read_from_pickle_0(test_set_path, data_to_load="uncertainties", normalize=False)

        hr_labels_valid = read_from_pickle_0(validation_set_path, data_to_load="heart_rates_labels", normalize=False) * 240  # Convert to beats per minute
        hr_labels_test = read_from_pickle_0(test_set_path, data_to_load="heart_rates_labels", normalize=False) * 240  # Convert to beats per minute

        hr_predictions_valid = read_from_pickle_0(validation_set_path, data_to_load="heart_rates", normalize=False) * 240  # Convert to beats per minute
        hr_predictions_test = read_from_pickle_0(test_set_path, data_to_load="heart_rates", normalize=False) * 240  # Convert to beats per minute

        predictions_valid = predictions_valid.reshape(-1, predictions_valid.shape[-1])
        predictions_test = predictions_test.reshape(-1, predictions_test.shape[-1])

        hr_labels_valid = hr_labels_valid.reshape(-1)
        hr_labels_test = hr_labels_test.reshape(-1)

        hr_predictions_valid = hr_predictions_valid.reshape(-1)
        hr_predictions_test = hr_predictions_test.reshape(-1)

        # Collect errors for LaTeX table
        results = [[] for _ in range(4)]

        uncertainties_valid = np.mean(uncertainties_predictions_valid, axis=-1)
        uncertainties_test = np.mean(uncertainties_predictions_test, axis=-1)

        uncertainties_valid = uncertainties_valid.reshape(-1)
        uncertainties_test = uncertainties_test.reshape(-1)

        # Raw predictions
        result_list = do_tests(predictions_valid, hr_labels_valid, uncertainties_valid, predictions_test, hr_labels_test, uncertainties_test)
        result_list.append(evaluate_neural_network(validation_set_path, test_set_path, hr_labels_valid, hr_labels_test))
        results = append_to_results(results, "Raw", result_list)

        # Cumsum predictions
        predictions_valid_cumsum = np.cumsum(predictions_valid, axis=1)
        predictions_test_cumsum = np.cumsum(predictions_test, axis=1)
        result_list = do_tests(predictions_valid_cumsum, hr_labels_valid, uncertainties_valid, predictions_test_cumsum, hr_labels_test, uncertainties_test)
        result_list.append(evaluate_neural_network(cumsum_validation_set_path, cumsum_test_set_path, hr_labels_valid, hr_labels_test))
        results = append_to_results(results, "Cumsum", result_list)

        # Detrended predictions
        predictions_valid_detrend = _detrend(predictions_valid_cumsum, 100)
        predictions_test_detrend = _detrend(predictions_test_cumsum, 100)
        result_list = do_tests(predictions_valid_detrend, hr_labels_valid, uncertainties_valid, predictions_test_detrend, hr_labels_test, uncertainties_test)
        result_list.append(evaluate_neural_network(detrend_validation_set_path, detrend_test_set_path, hr_labels_valid, hr_labels_test))
        results = append_to_results(results, "Detrend", result_list)

        [b, a] = butter(1, [0.5 / fs * 2, 3.3 / fs * 2], btype='bandpass')
        predictions_valid_butter = filtfilt(b, a, np.double(predictions_valid_detrend))
        predictions_test_butter = filtfilt(b, a, np.double(predictions_test_detrend))
        result_list = do_tests(predictions_valid_butter, hr_labels_valid, uncertainties_valid, predictions_test_butter, hr_labels_test, uncertainties_test)
        result_list.append(evaluate_neural_network(butter_validation_set_path, butter_test_set_path, hr_labels_valid, hr_labels_test))
        results = append_to_results(results, "Butter", result_list)

        # Print LaTeX table for Validation
        print_table(results[0], csv_path=f"csv/hrfilter_{uncertainty_type}_validation.xlsx")
        print_table(results[1], csv_path=f"csv/hrfilter_{uncertainty_type}_test.xlsx")
        #print_table(results[2], csv_path=f"csv/hrfilter_{uncertainty_type}_validation_loss.xlsx")
        #print_table(results[3], csv_path=f"csv/hrfilter_{uncertainty_type}_test_loss.xlsx")
