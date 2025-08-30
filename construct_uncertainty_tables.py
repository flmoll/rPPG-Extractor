import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
matplotlib.use("Agg")  # Use a non-interactive backend for matplotlib

import os

import numpy as np
import torch
from evaluation.utils import get_interval_predictions, read_from_pickle, read_from_multiple_pickles, get_pickle_path
from tabulate import tabulate


def get_ensamble_pickles(datasets, stabilizers, models):
    """
    datasets: List of dataset names to get the ensamble pickles for.
    models: List of model names to get the ensamble pickles for.
    """
    ensamble_pickles = []
    for dataset, curr_stabilizer in zip(datasets, stabilizers):
        curr_ensamble_pickles = []
        for model in models:
            curr_ensamble_pickles.append(get_pickle_path("DeadAlive", dataset, model, curr_stabilizer))
            assert os.path.exists(curr_ensamble_pickles[-1]), f"Pickle file {curr_ensamble_pickles[-1]} does not exist!"
        ensamble_pickles.append(curr_ensamble_pickles)
    
    return ensamble_pickles

def get_uncertainty_pickles(datasets, stabilizers, model):

    pickles = []

    for dataset, curr_stabilizer in zip(datasets, stabilizers):
        pickles.append(get_pickle_path("DeadAlive", dataset, model, curr_stabilizer))
        assert os.path.exists(pickles[-1]), f"Pickle file {pickles[-1]} does not exist!"

    return pickles
            

def evaluate_ensamble_experiment_results(ensamble_pickles, experiment_names, probability_in_interval=0.95):
    """
    ensamble_pickles: List of lists of paths to pickles containing predictions and labels from different models. the first dimension contains the different datasets, the second dimension contains the different models.
    """

    assert len(ensamble_pickles) == len(experiment_names), "Number of ensamble pickles must match number of experiment names."

    #std_predictions_list = []
    table_data = []

    for curr_ensamble_pickles, curr_experiment_name, idx in zip(ensamble_pickles, experiment_names, range(len(experiment_names))):

        preds = read_from_multiple_pickles(curr_ensamble_pickles, data_to_load="predictions", read_batched=True)
        labels = read_from_multiple_pickles(curr_ensamble_pickles, data_to_load="labels", read_batched=True)

        preds = (preds - np.mean(preds, axis=-1, keepdims=True)) / np.std(preds, axis=-1, keepdims=True)
        labels = (labels - np.mean(labels, axis=-1, keepdims=True)) / np.std(labels, axis=-1, keepdims=True)

        std_preds = np.std(preds, axis=0)

        table_data.append([
            curr_experiment_name,
            f"{np.mean(std_preds):.4f}",
        ])

    return table_data

def evaluate_uncertainty_experiment_results(result_pickles, experiment_names, modes, probability_in_interval=0.95):
    
    uncertainties_list = []
    preds_list = []
    labels_list = []
    interval_lower_list = []
    interval_upper_list = []
    uncertainties_list = []
    average_interval_widths = []
    median_interval_widths = []

    for results_file, experiment_name, mode in zip(result_pickles, experiment_names, modes):
        print(f"Processing {experiment_name}...")
        print(results_file)

        uncertainties, preds, labels = read_from_pickle(results_file)
        assert preds.shape == labels.shape and preds.shape[0] == uncertainties.shape[0], f"Shape mismatch: preds {preds.shape}, labels {labels.shape}, uncertainties {uncertainties.shape}"

        print(np.mean(uncertainties), np.std(uncertainties))

        interval_lower, interval_upper = get_interval_predictions(uncertainties, preds, mode, probability_in_interval)

        uncertainties_list.append(uncertainties)
        preds_list.append(preds)
        labels_list.append(labels)
        interval_lower_list.append(interval_lower)
        interval_upper_list.append(interval_upper)

        # Compute average interval width
        average_interval_width = np.mean(interval_upper - interval_lower)
        average_interval_widths.append(average_interval_width)

        # Compute median interval width
        median_interval_width = np.median(interval_upper - interval_lower)
        median_interval_widths.append(median_interval_width)


    table_data = []
    for i, experiment_name in enumerate(experiment_names):
        table_data.append([
            experiment_name,
            #f"{average_interval_widths[i]:.4f}",
            f"{median_interval_widths[i]:.4f}"
        ])

    return table_data

if __name__ == "__main__":

    if not hasattr(torch.serialization, "_package_registered"):
        torch.serialization.register_package(0, lambda x: x.device.type, lambda x, _: x.cpu())
        torch.serialization._package_registered = True

    uncertainty_models = ["Physnet_NegLogLikelihood", "Physnet_Quantile"]
    modes = ["neg_log_likelihood", "quantile_regression"]
    normalization_dataset = "validation"

    ensamble_models = ["EfficientPhys", "PhysFormer", "Physnet", "RythmFormer", "TSCAN"]
    datasets = ["validation", "test", "own_videos", "validation_shuffle", "test_shuffle", "dead", "emergency", "emergency"]
    stabilizers = ["YUNET", "YUNET", "YUNET", "YUNET", "YUNET", "YUNET", "YUNET", "MOSSE"]
    results_matrix = np.zeros((len(datasets), len(uncertainty_models) + 1)) # +1 for the ensamble model

    normalization_dataset_index = datasets.index(normalization_dataset)

    ensamble_pickles = get_ensamble_pickles(datasets, stabilizers, ensamble_models)
    experiment_names = [
        "Validation",
        "Test",
        "Own Videos",
        "Validation Shuffled",
        "Test Shuffled",
        "Dead Videos",
        "Emergency Videos",
        "Emergency Videos Stable"
    ]

    ensamble_table_data = evaluate_ensamble_experiment_results(ensamble_pickles, experiment_names)
    results_matrix[:, 0] = np.array([float(row[1]) for row in ensamble_table_data])

    print("Ensamble Table Data:")
    print(tabulate(ensamble_table_data, headers=["Experiment", "STD Predictions"], tablefmt="grid"))

    for curr_model, curr_mode in zip(uncertainty_models, modes):
        print(f"Evaluating uncertainty model: {curr_model}")

        curr_modes = [curr_mode] * len(datasets)

        uncertainty_pickles = get_uncertainty_pickles(datasets, stabilizers, curr_model)
        uncertainty_table = evaluate_uncertainty_experiment_results(uncertainty_pickles, experiment_names, curr_modes)
        results_matrix[:, uncertainty_models.index(curr_model) + 1] = np.array([float(row[1]) for row in uncertainty_table])

        print(f"Uncertainty Table Data for {curr_model}:")
        print(tabulate(uncertainty_table, headers=["Experiment", "Median Interval Width"], tablefmt="grid"))

    results_matrix = results_matrix / results_matrix[normalization_dataset_index] # Normalize the results matrix
    results_matrix = np.round(results_matrix, 2)  # Round the results matrix to 2 decimal places

    # Print the general table with all results
    print("General Table Data:")
    # Add datasets as row names for better readability
    headers = ["Experiment", "Ensamble Model"] + uncertainty_models
    table_with_datasets = []
    for i, row in enumerate(results_matrix):
        table_with_datasets.append([experiment_names[i]] + list(row))
    print(tabulate(table_with_datasets, headers=headers, tablefmt="latex"))

    df = pd.DataFrame(table_with_datasets, columns=headers)
    df.to_excel('csv/uncertainty_table.xlsx', index=False)
