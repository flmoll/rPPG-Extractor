import matplotlib
from tqdm import tqdm
matplotlib.use('Agg')  # Use a non-interactive backend for saving figures

import os
import numpy as np
from evaluation.utils import ask_and_load_results, load_rppg_toolbox_results, save_results, get_mean_SNR_from_results, get_MAE_from_results, get_pickle_path, get_results
import seaborn as sns
import matplotlib.pyplot as plt


def get_metric_from_results(results):
    return get_mean_SNR_from_results(results)

if __name__ == "__main__":

    parent_folder = "runs/Compression"

    train_compression_values = np.array(range(0, 50, 5))
    test_compression_values = np.array(range(0, 55, 5))

    for dataset_name in ["validation", "test"]:
        for destabilizer in ["H264_QP", "H265_QP"]:

            if destabilizer == "H264_QP":
                plot_title = "H.264 Compression Evaluation"
            elif destabilizer == "H265_QP":
                plot_title = "H.265 Compression Evaluation"

            results_matrix = np.zeros((len(train_compression_values), len(test_compression_values)))

            for i in tqdm(train_compression_values):
                for j in test_compression_values:
                    path = get_pickle_path("Compression", dataset_name=dataset_name, destabilizer=destabilizer, destabilizer_amp=i)
                    path = path.replace(f"/{dataset_name}/", f"/{dataset_name}/{str(float(j))}/")

                    if not os.path.exists(path):
                        print(f"Compression {i} -> {j} does not exist")
                        continue

                    results = get_results(path, dataset=dataset_name)
                    results_matrix[int(i // 5), int(j // 5)] = get_metric_from_results(results)

            plt.figure(figsize=(8, 6))
            ax = sns.heatmap(results_matrix, annot=True, fmt=".2f", cmap='viridis',
                            xticklabels=test_compression_values, yticklabels=train_compression_values,
                            cbar_kws={'label': 'SNR (dB)'})
            plt.xlabel('Test Compression Value')
            plt.ylabel('Train Compression Value')
            plt.title(plot_title)
            plt.tight_layout()
            plt.savefig(f"graphics/results_matrix_heatmap_{dataset_name}_{destabilizer}.png")
