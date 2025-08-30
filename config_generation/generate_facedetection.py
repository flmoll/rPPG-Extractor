import os
import numpy as np
from run_pipeline_check import get_output_paths

from config_generation.utils import get_config_template_paths, generate_train_validation_test_triplet, generate_config_from_template, template_config_files, template_config_path

def generate_facedetection_configs(config_output_folder):
    # Deepstab experiment
    models = ["physnet"]
    stabilization_backends = ["Y5F", "HC", "YUNET"]

    train_base_config_files, validation_base_config_files, test_base_config_files = get_config_template_paths(models)

    for stabilization_backend in stabilization_backends:
        for train, validation, test, idx in zip(train_base_config_files, validation_base_config_files, test_base_config_files, range(len(train_base_config_files))):
            name_extension = f"{stabilization_backend}"
            config_name = f"{name_extension}_{train.split('/')[-1]}"

            if models[idx] == "rhythmformer" or models[idx] == "physformer":
                size = 128
            else:
                size = 72

            generate_train_validation_test_triplet(
                train_template=train,
                validation_template=validation,
                test_template=test,
                output_folder=os.path.join(config_output_folder, "facedetection"),
                config_name=config_name,
                EXP_NAME="FaceDetection",
                STABILIZATION_BACKEND=stabilization_backend,
                MODEL_NAME_EXTENSION="",
                SIZE=size,
            )

        preprocessing_config_path = os.path.join(template_config_path, "preprocessing", template_config_files["preprocessing"])
        out_config_path = os.path.join(config_output_folder, "facedetection", "preprocessing", f"0_{name_extension}_72_preprocessing.yaml")
        generate_config_from_template(
            preprocessing_config_path, out_config_path, 
            EXP_NAME="FaceDetection",
            STABILIZATION_BACKEND=stabilization_backend,
            MODEL_NAME_EXTENSION=""
        )