

import os
import numpy as np
import pandas as pd
from evaluation.utils import ask_and_load_results, load_rppg_toolbox_results, save_results, get_mean_SNR_from_results, get_data_loader_name_for_dataset, get_pickle_path, get_results, get_MAE_from_results
from tabulate import tabulate

def get_metric_from_results(results):
    return get_MAE_from_results(results)
    #return get_mean_SNR_from_results(results)


if __name__ == "__main__":

    gt_files_parent_folder = "/mnt/data/vitalVideos"
    vital_videos_gt_peaks_file = "/mnt/data/vitalVideos/00_peaks.json"
    #gt_files_parent_folder = "/mnt/data/ubfc"

    expected_recording_length = 900
    batch_len = 160
    bandpass_lower = 0.6
    bandpass_upper = 3.3
    vital_videos = True

    datasets = ["emergency"]
    
    models = ["EfficientPhys",
              "Physnet",
              "TSCAN",
              "RythmFormer",
              "PhysFormer", 
              "Physnet_Quantile", 
              "Physnet_NegLogLikelihood"]

    destabilizers = ["None", "DEEPSTAB"]
    stabilizers = ["YUNET", "MOSSE"]

    for dataset in datasets:
        
        result_table = []
    
        for model in models:
            curr_results_row = [model]

            for stabilizer in stabilizers:
                for destabilizer in destabilizers:

                    if dataset == "emergency":
                        exp_name = f"Emergency/destab_{destabilizer}"
                        destabilizer = "None"
                    else:
                        exp_name = "Emergency"

                    experiment_path = get_pickle_path(exp_name, dataset, model, stabilizer, destabilizer)
                    print("Looking for results in ", experiment_path)

                    if not os.path.exists(experiment_path):
                        curr_results_row.append(np.nan)
                        continue

                    if dataset == "test":
                        vital_videos_offset = 0.7
                    elif dataset == "validation":
                        vital_videos_offset = 0.86

                    results_experiment = get_results(experiment_path, dataset=dataset)
                    metric_experiment = get_metric_from_results(results_experiment)
                    curr_results_row.append(f"{metric_experiment:.2f}")

            result_table.append(tuple(curr_results_row))

        print("Results for dataset ", dataset)

        headers = ["Model"]

        for stabilizer in stabilizers:
            for destabilizer in destabilizers:
                headers.append(f"{stabilizer} {destabilizer}")

        print(tabulate(result_table, headers, tablefmt="latex"))

        df = pd.DataFrame(result_table, columns=headers)
        df.to_excel(f'csv/emergency_comparison_{dataset}.xlsx', index=False)
