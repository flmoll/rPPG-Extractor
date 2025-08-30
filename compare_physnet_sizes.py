

from matplotlib import pyplot as plt
import numpy as np
import tqdm
from evaluation.utils import ask_and_load_results, load_rppg_toolbox_results, save_results, get_mean_SNR_from_results, get_MAE_from_results, get_pickle_path, get_results
from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
import torch
from ptflops import get_model_complexity_info


def get_metric_from_results(results):
    return get_mean_SNR_from_results(results)

if __name__ == "__main__":

    gt_files_parent_folder = "/mnt/data/vitalVideos"
    vital_videos_gt_peaks_file = "/mnt/data/vitalVideos/00_peaks.json"
    #gt_files_parent_folder = "/mnt/data/ubfc"

    expected_recording_length = 900
    batch_len = 160
    bandpass_lower = 0.6
    bandpass_upper = 3.3
    vital_videos = True
    
    for dataset in ["test", "validation"]:
        if dataset == "test":
            vital_videos_offset = 0.7
        elif dataset == "validation":
            vital_videos_offset = 0.86

        sizes = np.arange(20, 110, 10)  # Sizes from 20 to 100 with step 10
        results_vector = []

        for size in tqdm.tqdm(sizes):
            path = get_pickle_path("physnet_in_size", dataset_name=dataset, size=size)
            results = get_results(path, dataset=dataset)
            results_vector.append(get_metric_from_results(results))

        plt.plot(sizes, results_vector, marker='o')
        plt.xlabel('Size')
        plt.ylabel('Mean SNR')
        plt.title('Mean SNR vs Size')
        plt.xticks(sizes)
        plt.grid()
        plt.savefig(f"graphics/physnet_in_size_{dataset}.png")
        plt.close()


    physnet = PhysNet_padding_Encoder_Decoder_MAX(frames=batch_len)

    macs_all = []

    for size in tqdm.tqdm(sizes):
        dummy_input = np.random.rand(1, 3, batch_len, size, size)  # Batch size of B, C, T, W, H
        with torch.cuda.device(0):  # if you have GPU
            macs, params = get_model_complexity_info(physnet, (3, batch_len, size, size), as_strings=False,
                                                    print_per_layer_stat=False, verbose=False)
        macs_all.append(macs / 1e9)  # Convert to Giga MACCs

    plt.plot(sizes, macs_all, marker='o')
    plt.xlabel('Input Size')
    plt.ylabel('Giga MACCs')
    plt.title('Physnet MACCs vs Input Size')
    plt.xticks(sizes)
    plt.grid()
    plt.savefig("graphics/physnet_in_size_macs.png")
    plt.close()
    