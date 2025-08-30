import os
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib

import numpy as np
from evaluation.utils import get_interval_predictions, read_from_pickle, conformal_prediction, get_pickle_path
from scipy.stats import norm
from tqdm import tqdm
import matplotlib.pyplot as plt

def plot_conformal_prediction_evaluation(validation_outputs, test_outputs, mode="neg_log_likelihood", data_to_load="rppg", probability_in_interval=np.linspace(0.5, 0.99, 49), filename="test_set_violations.png", title="Conformal Prediction Evaluation"):

    violations_valid_list = []
    violations_test_list = []
    mean_interval_valid_list = []
    mean_interval_test_list = []

    for prob in tqdm(probability_in_interval, desc="Processing probabilities"):
        interval_lower_valid, intervals_upper_valid, violations_valid, interval_lower_test, intervals_upper_test, violations_test = conformal_prediction(validation_outputs, test_outputs, mode, data_to_load, prob)
        violations_valid_list.append(violations_valid)
        violations_test_list.append(violations_test)
        mean_interval_valid = np.mean(intervals_upper_valid - interval_lower_valid)
        mean_interval_test = np.mean(intervals_upper_test - interval_lower_test)
        mean_interval_valid_list.append(mean_interval_valid)
        mean_interval_test_list.append(mean_interval_test)

    violations_valid_list = np.array(violations_valid_list)
    violations_test_list = np.array(violations_test_list)
    mean_interval_valid_list = np.array(mean_interval_valid_list)
    mean_interval_test_list = np.array(mean_interval_test_list)

    ax1 = plt.gca()
    #ax2 = ax1.twinx()

    ax1.plot(probability_in_interval * 100, 100 - violations_test_list, label="Test Coverage")
    ax1.plot(probability_in_interval * 100, 100 - violations_valid_list, label="Validation Coverage")
    ax1.set_ylabel("Actual Coverage (%)")

    #ax2.plot(probability_in_interval * 100, mean_interval_test_list, 'r--', label="Test Interval")
    #ax2.plot(probability_in_interval * 100, mean_interval_valid_list, 'b--', label="Validation Interval")
    #ax2.set_ylabel("Mean Interval Width")

    plt.xlabel("Desired Coverage (%)")
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.savefig(filename)
    plt.close()

def get_ensemble_prediction(ensemble_outputs):
    """
    Get the ensemble predictions by averaging the predictions from multiple models.
    """
    all_predictions = []
    for output in ensemble_outputs:
        preds = read_from_pickle(output, data_to_load=["predictions"])[0]
        all_predictions.append(preds)

    labels = read_from_pickle(ensemble_outputs[0], data_to_load=["labels"])[0]
    mean_pred = np.mean(all_predictions, axis=0)
    std_pred = np.std(all_predictions, axis=0)

    print(type(labels))

    return std_pred, mean_pred, labels

def ensemble_conformal_prediction_evaluation(ensemble_validation_outputs, ensemble_test_outputs, filename="graphics/ensemble_conformal_prediction_evaluation.png", title="Ensemble Conformal Prediction Evaluation"):

    valid_output_tuple = get_ensemble_prediction(ensemble_validation_outputs)
    test_output_tuple = get_ensemble_prediction(ensemble_test_outputs)

    plot_conformal_prediction_evaluation(valid_output_tuple, test_output_tuple, mode="neg_log_likelihood", filename=filename, title=title)

if __name__ == "__main__":
    
    models_all = [
        "Physnet_NegLogLikelihood",
        "Physnet_Quantile",
        "Physnet_HR_Classifier_NegLogLikelihood_None",
        "Physnet_HR_Classifier_NegLogLikelihood_cumsum",
        "Physnet_HR_Classifier_NegLogLikelihood_detrend",
        "Physnet_HR_Classifier_NegLogLikelihood_butter",
        "Physnet_HR_Classifier_Quantile_None",
        "Physnet_HR_Classifier_Quantile_cumsum",
        "Physnet_HR_Classifier_Quantile_detrend",
        "Physnet_HR_Classifier_Quantile_butter"
    ]

    ensemble_models = [
        "EfficientPhys", "PhysFormer", "Physnet", "RythmFormer", "TSCAN"
    ]

    ensemble_validation_outputs = []
    ensemble_test_outputs = []

    for ensemble_model in ensemble_models:
        ensemble_validation_outputs.append(get_pickle_path("DeadAlive", model_name=ensemble_model, dataset_name="validation"))
        ensemble_test_outputs.append(get_pickle_path("DeadAlive", model_name=ensemble_model, dataset_name="test"))

        assert os.path.exists(ensemble_validation_outputs[-1]), f"Validation outputs file {ensemble_validation_outputs[-1]} does not exist."
        assert os.path.exists(ensemble_test_outputs[-1]), f"Test outputs file {ensemble_test_outputs[-1]} does not exist."

    validation_outputs_all = []
    test_outputs_all = []
    titles_all = []
    filenames_all = []
    data_to_load_all = []
    modes_all = []

    for model in models_all:
        validation_outputs_all.append(get_pickle_path("Uncertainty", model_name=model, dataset_name="validation"))
        test_outputs_all.append(get_pickle_path("Uncertainty", model_name=model, dataset_name="test"))

        if "_Quantile_" in model:
            mode = "quantile_regression"
        else:
            mode = "neg_log_likelihood"

        if "_HR_Classifier_" in model:
            data_to_load = "heart_rate"
        else:
            data_to_load = "rppg"

        titles_all.append(f"Conformal Prediction Evaluation - {model}")
        filenames_all.append(f"graphics/conformalize_evaluation_{model}.png")
        modes_all.append(mode)
        data_to_load_all.append(data_to_load)

        assert os.path.exists(validation_outputs_all[-1]), f"Validation outputs file {validation_outputs_all[-1]} does not exist."
        assert os.path.exists(test_outputs_all[-1]), f"Test outputs file {test_outputs_all[-1]} does not exist."


    ensemble_conformal_prediction_evaluation(ensemble_validation_outputs, ensemble_test_outputs)

    for validation_outputs, test_outputs, mode, data_to_load, filename, title in zip(validation_outputs_all, test_outputs_all, modes_all, data_to_load_all, filenames_all, titles_all):

        print(f"Processing {mode} for {data_to_load}...")
        plot_conformal_prediction_evaluation(validation_outputs, test_outputs, mode, data_to_load, filename=filename, title=title)   
