import matplotlib
import pandas as pd
matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib

import os

import numpy as np

from evaluation.utils import read_from_pickle
from evaluation.utils import ask_and_load_results, load_rppg_toolbox_results, save_results, get_mean_SNR_from_results, get_pickle_path, get_results, get_MAE_from_results
import csv
import xlsxwriter
from tabulate import tabulate


def get_metric_from_results(results):
    return get_MAE_from_results(results)
    #return get_mean_SNR_from_results(results)

def generate_performance_table(table_content, stabilizer_names, model_names, xlsx_path='csv/stabilizers.xlsx'):

    headers = ["Model"] + stabilizer_names
    table = []
    for row_idx, row in enumerate(table_content):
        table.append([model_names[row_idx]] + [f"{val:.2f}" for val in row])

    print(tabulate(table, headers=headers, tablefmt="latex"))

    df = pd.DataFrame(table, columns=headers)
    df.to_excel(xlsx_path, index=False)

if __name__ == "__main__":

    datasets = ["test", "validation"]
    stabilizers = ["YUNET", "CORRYU", "OPTYU", "MOSSE"]
    destabilizers = ["None", "DEEPSTAB"]

    model_names = [
        "EfficientPhys",
        "Physnet",
        "TSCAN",
        "RythmFormer",
        "PhysFormer",
    ]

    stabilizer_names = [
        "Median Face Box",
        "Template Matching",
        "Optical Flow",
        "MOSSE"
    ]

    results_matrix = np.zeros((len(datasets), len(destabilizers), len(model_names), len(stabilizers)))

    for dataset_idx, dataset in enumerate(datasets):
        for destabilizer_idx, destabilizer in enumerate(destabilizers):
            for stabilizer_idx, stabilizer in enumerate(stabilizers):
                for model_idx, model in enumerate(model_names):

                    curr_pickle = get_pickle_path("DeepStab", dataset_name=dataset, stabilizer=stabilizer, model_name=model, destabilizer=destabilizer)
                    if not os.path.exists(curr_pickle):
                        print(f"Results for {dataset} dataset with {stabilizer} stabilizer and {destabilizer} destabilizer do not exist.")
                        continue

                    results = get_results(curr_pickle, dataset=dataset)
                    metric = get_metric_from_results(results)
                    results_matrix[dataset_idx, destabilizer_idx, model_idx, stabilizer_idx] = metric

            print(f"Results for {dataset} dataset with {destabilizer} destabilizer:")
            generate_performance_table(results_matrix[dataset_idx, destabilizer_idx], stabilizer_names, model_names, xlsx_path=f'csv/stabilizers_{dataset}_{destabilizer}.xlsx')

