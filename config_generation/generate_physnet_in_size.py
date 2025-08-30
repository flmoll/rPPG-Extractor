import os
import numpy as np
from run_pipeline_check import get_output_paths

from config_generation.utils import get_config_template_paths, generate_train_validation_test_triplet, generate_config_from_template, template_config_files, template_config_path

def generate_physnet_in_size_configs(config_output_folder):
    # Physnet in size experiment
    models = ["physnet"]
    stabilization_backend = "YUNET"
    sizes = [20, 30, 40, 50, 60, 70, 80, 90, 100]
    files_normal = get_config_template_paths(models)

    for train, validation, test in zip(*files_normal):
        for size in sizes:
            config_name = f"size_{size}_{train.split('/')[-1]}"

            kwargs = {
                "EXP_NAME": "physnet_in_size",
                "SIZE": size,
            }
            
            train_out_path, _, _ = generate_train_validation_test_triplet(
                train_template=train,
                validation_template=validation,
                test_template=test,
                output_folder=os.path.join(config_output_folder, "physnet_in_size"),
                config_name=config_name,
                **kwargs
            )
            
            preprocessing_config_path = os.path.join(template_config_path, "preprocessing", template_config_files["preprocessing"])
            out_config_path = os.path.join(config_output_folder, "physnet_in_size", "preprocessing", f"0_{size}_preprocessing.yaml")
            generate_config_from_template(
                preprocessing_config_path, out_config_path, 
                EXP_NAME="physnet_in_size",
                SIZE=size,
            )



