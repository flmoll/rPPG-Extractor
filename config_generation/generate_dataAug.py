import os
import numpy as np
from run_pipeline_check import get_output_paths

from config_generation.utils import get_config_template_paths, generate_train_validation_test_triplet, generate_config_from_template, template_config_files, template_config_path

def generate_dataaug_preprocessing(config_output_folder, template_filename, stabilization_backend="Y5F", destabilizer="None", size=72):
    
    preprocessing_config_path = os.path.join(template_config_path, "preprocessing", template_filename)
    out_config_path = os.path.join(config_output_folder, "dataAug", "preprocessing", f"0_{destabilizer}_{stabilization_backend}_{size}_{template_filename}_preprocessing.yaml")
    
    generate_config_from_template(
        preprocessing_config_path, out_config_path, 
        EXP_NAME="DataAugmentation",
        STABILIZATION_BACKEND=stabilization_backend,
        MODEL_NAME_EXTENSION="",
        DESTAB_BACKEND=destabilizer,
        DESTAB_AMPLITUDE=10.0,
        SIZE=size
    )

def generate_dataAug(config_output_folder):
    """
    Generates data augmentation configurations for the rPPG Toolbox.
    """
    models = ["physnet"]
    destabilizers = ["None", "DEEPSTAB", "RANDOM_AFFINE"]
    stabilization_backends = ["Y5F"]

    
    train_base_config_files, validation_base_config_files, test_base_config_files = get_config_template_paths(models)
    dead_base_config_files, own_videos_base_config_files, emergency_base_config_files = get_config_template_paths(models, train_folder="dead", validation_folder="own_videos", test_folder="emergency")

    for train, validation, test, dead, own_videos, emergency, idx in zip(train_base_config_files, validation_base_config_files, test_base_config_files, dead_base_config_files, own_videos_base_config_files, emergency_base_config_files, range(len(train_base_config_files))):
        for destabilizer in destabilizers:
            for stabilization_backend in stabilization_backends:

                name_extension = f"_{destabilizer}_{stabilization_backend}"
                config_name = f"{destabilizer}_{stabilization_backend}_{train.split('/')[-1]}"

                if models[idx] == "physnet":
                    size = 72
                else:
                    size = 128

                kwargs = {
                    "EXP_NAME": "DataAugmentation",
                    "SIZE": size,
                    "MODEL_NAME_EXTENSION": name_extension,
                    "DESTAB_BACKEND": destabilizer,
                    "DESTAB_AMPLITUDE": 10.0,
                    "STABILIZATION_BACKEND": stabilization_backend,
                }

                train_output_path, _, _ = generate_train_validation_test_triplet(
                    train_template=train,
                    validation_template=validation,
                    test_template=test,
                    output_folder=os.path.join(config_output_folder, "dataAug"),
                    config_name=config_name,
                    **kwargs,
                )

                
                file, output_path, train_basename, valid_basename, test_basename, model_hr_output_path, config = get_output_paths([train_output_path])[0]

                kwargs["MODEL_PATH"] = model_hr_output_path
                kwargs["EXP_NAME"] = "DataAugmentation/dead"
                kwargs["STABILIZATION_BACKEND"] = "Y5F"
                kwargs["DESTAB_BACKEND"] = "None"  # No destabilization for dead videos
                kwargs["DESTAB_AMPLITUDE"] = 10.0

                generate_config_from_template(
                    dead,
                    output_path=os.path.join(config_output_folder, "dataAug", "dead", f"2_{config_name}"),
                    **kwargs,
                )

                kwargs["EXP_NAME"] = "DataAugmentation/own_videos"

                generate_config_from_template(
                    own_videos,
                    output_path=os.path.join(config_output_folder, "dataAug", "own_videos", f"2_{config_name}"),
                    **kwargs,
                )

                kwargs["EXP_NAME"] = "DataAugmentation/emergency"

                generate_config_from_template(
                    emergency,
                    output_path=os.path.join(config_output_folder, "dataAug", "emergency", f"2_{config_name}"),
                    **kwargs,
                )

    for stabilization_backend in stabilization_backends:
        for destabilizer in destabilizers:
            generate_dataaug_preprocessing(config_output_folder, template_config_files["preprocessing"], stabilization_backend, destabilizer, 72)
    
    generate_dataaug_preprocessing(config_output_folder, template_config_files["preprocessing_dead"], size=72)
    generate_dataaug_preprocessing(config_output_folder, template_config_files["preprocessing_own_videos"], size=72)
    generate_dataaug_preprocessing(config_output_folder, template_config_files["preprocessing_emergency"], size=72)