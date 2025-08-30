
import os

import numpy as np
from run_pipeline_check import check_paths, get_output_paths
from difflib import get_close_matches
from config_generation.generate_compression import generate_compression_configs
from config_generation.generate_deadalive import generate_deadalive_configs
from config_generation.generate_physnet_in_size import generate_physnet_in_size_configs
from config_generation.generate_uncertainty import generate_uncertainty_configs
from config_generation.generate_deepstab import generate_deepstab_configs
from config_generation.generate_lifeness import generate_lifeness_configs
from config_generation.generate_dataAug import generate_dataAug
from config_generation.generate_facedetection import generate_facedetection_configs
from config_generation.generate_emergency import generate_emergency_configs

config_output_folder = "configs/all_experiment_configs"

def check_test_exists(path_tuples):
    all_outputs = []
    path_tuples = sorted(path_tuples, key=lambda x: os.path.basename(x[0]))
    for path_tuple in path_tuples:
        file, output_path, train_basename, valid_basename, test_basename, model_output_path, config = path_tuple
        all_outputs.append(model_output_path)
        
        if config.TOOLBOX_MODE == 'only_test':
            if not os.path.exists(model_output_path) and not config.INFERENCE.MODEL_PATH in all_outputs:
                print(f"Model output path {model_output_path} does not exist for config {file}")

def check_if_all_preprocessing_exists(preprocessor_path_tuples, other_path_tuples):
    preprocessor_paths = [item[-1].TEST.DATA.CACHED_PATH for item in preprocessor_path_tuples] # the config is the last item in the tuple
    other_paths = [item[-1].TEST.DATA.CACHED_PATH for item in other_path_tuples] # the config is the last item in the tuple
    
    for idx, curr_path in enumerate(other_paths):
        if curr_path not in preprocessor_paths:
            print(f"File: {other_path_tuples[idx][0]} no preprocessing found")
            # Find the closest preprocessing path (by string distance)
            closest = get_close_matches(curr_path, preprocessor_paths, n=1)
            if closest:
                diff_indices = [i for i, (a, b) in enumerate(zip(curr_path, closest[0])) if a != b]
                print(f"Closest preprocessing path for {curr_path}: {closest[0]}")
                print(f"Differences at positions: {diff_indices} length {len(curr_path)}")
            else:
                print(f"No close preprocessing path found for {curr_path}")


if __name__ == "__main__":

    generate_deepstab_configs(config_output_folder)
    generate_deadalive_configs(config_output_folder)
    generate_physnet_in_size_configs(config_output_folder)
    generate_uncertainty_configs(config_output_folder)
    generate_compression_configs(config_output_folder)
    generate_lifeness_configs(config_output_folder)
    generate_dataAug(config_output_folder)
    generate_facedetection_configs(config_output_folder)
    generate_emergency_configs(config_output_folder)

    # Check if test configs are valid
    for curr_folder in os.listdir(config_output_folder):

        curr_folder_full = os.path.join(config_output_folder, curr_folder)
        print(f"Checking configs in {curr_folder_full}")
        
        config_paths = []

        for root, dirs, files in os.walk(curr_folder_full):
            for file in files:
                if file.endswith(".yaml"):
                    config_paths.append(os.path.join(root, file))
                else:
                    print(f"File {file} in {root} is not a YAML file, skipping.")
        
        config_paths_without_preprocessing = [path for path in config_paths if "preprocessing" not in path]
        config_paths_with_preprocessing = [path for path in config_paths if "preprocessing" in path]

        output_paths_without_preprocessing = get_output_paths(config_paths_without_preprocessing)
        output_paths_with_preprocessing = get_output_paths(config_paths_with_preprocessing)

        check_if_all_preprocessing_exists(output_paths_with_preprocessing, output_paths_without_preprocessing)
        check_test_exists(output_paths_without_preprocessing)
        check_paths(output_paths_without_preprocessing, suppress_output_exists_warning=True, suppress_model_not_exists_warning=True)





