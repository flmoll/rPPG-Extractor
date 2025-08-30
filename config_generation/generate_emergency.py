import os
import numpy as np
from run_pipeline_check import get_output_paths

from config_generation.utils import get_config_template_paths, generate_train_validation_test_triplet, generate_config_from_template, template_config_files, template_config_path

def generate_emergency_preprocessing(config_output_folder, template_filename, stabilization_backend="YUNET", destabilizer="None", size=72):
    
    preprocessing_config_path = os.path.join(template_config_path, "preprocessing", template_filename)
    out_config_path = os.path.join(config_output_folder, "emergency", "preprocessing", f"0_{destabilizer}_{stabilization_backend}_{size}_{template_filename}_preprocessing.yaml")

    generate_config_from_template(
        preprocessing_config_path, out_config_path, 
        EXP_NAME="Emergency",
        STABILIZATION_BACKEND=stabilization_backend,
        MODEL_NAME_EXTENSION="",
        DESTAB_BACKEND=destabilizer,
        DESTAB_AMPLITUDE=10.0,
        SIZE=size
    )

def generate_emergency_config_0(destabilizer="None", stabilization_backend="YUNET", models=[], config_output_folder="config/emergency"):
    train_base_config_files, validation_base_config_files, test_base_config_files = get_config_template_paths(models)
    _, emergency_base_config_files, _ = get_config_template_paths(models, validation_folder="emergency")

    for train, validation, test, emergency, idx in zip(train_base_config_files, validation_base_config_files, test_base_config_files, emergency_base_config_files, range(len(train_base_config_files))):
        name_extension = f"{destabilizer}_{stabilization_backend}"
        config_name = f"{name_extension}_{train.split('/')[-1]}"

        if models[idx] == "rhythmformer" or models[idx] == "physformer":
            size = 128
        else:
            size = 72

        train_out_path, _, _ = generate_train_validation_test_triplet(
            train_template=train,
            validation_template=validation,
            test_template=test,
            output_folder=os.path.join(config_output_folder, "emergency"),
            config_name=config_name,
            EXP_NAME=f"Emergency",
            STABILIZATION_BACKEND=stabilization_backend,
            MODEL_NAME_EXTENSION="",
            DESTAB_BACKEND=destabilizer,
            DESTAB_AMPLITUDE=10.0,
            SIZE=size,
        )

        file, output_path, train_basename, valid_basename, test_basename, model_output_path, config = get_output_paths([train_out_path])[0]

        generate_config_from_template(
            emergency, 
            os.path.join(config_output_folder, "emergency", "emergency", f"2_{destabilizer}_{stabilization_backend}_{models[idx]}_emergency.yaml"), 
            config_name=config_name,
            EXP_NAME=f"Emergency/destab_{destabilizer}/emergency",
            STABILIZATION_BACKEND=stabilization_backend,
            MODEL_NAME_EXTENSION="",
            DESTAB_BACKEND="None",
            DESTAB_AMPLITUDE=10.0,
            SIZE=size,
            MODEL_PATH=model_output_path
        )

    for size in [72, 128]:
        preprocessing_config_path = os.path.join(template_config_path, "preprocessing", template_config_files["preprocessing"])
        out_config_path = os.path.join(config_output_folder, "emergency", "preprocessing", f"0_{name_extension}_{size}_preprocessing.yaml")
        generate_config_from_template(
            preprocessing_config_path, out_config_path, 
            EXP_NAME="Emergency",
            STABILIZATION_BACKEND=stabilization_backend,
            MODEL_NAME_EXTENSION="",
            DESTAB_BACKEND=destabilizer,
            DESTAB_AMPLITUDE=10.0,
            SIZE=size
        )

        preprocessing_config_path = os.path.join(template_config_path, "preprocessing", template_config_files["preprocessing_emergency"])
        out_config_path = os.path.join(config_output_folder, "emergency", "preprocessing", f"0_{name_extension}_emergency_{size}_preprocessing.yaml")
        generate_config_from_template(
            preprocessing_config_path, out_config_path, 
            EXP_NAME="Emergency",
            STABILIZATION_BACKEND=stabilization_backend,
            MODEL_NAME_EXTENSION="",
            DESTAB_BACKEND="None",
            DESTAB_AMPLITUDE=10.0,
            SIZE=size
        )

def generate_emergency_configs(config_output_folder):
    """
    Generates data augmentation configurations for the rPPG Toolbox.
    """
    models = ["efficientphys", "physformer", "physnet", "rhythmformer", "tscan", "physnet_uncertainty", "physnet_quantile"]

    destabilizers = ["None", "DEEPSTAB"]
    stabilizers = ["YUNET", "MOSSE"]

    for destabilizer in destabilizers:
        for stabilizer in stabilizers:
            generate_emergency_config_0(destabilizer=destabilizer, 
                                        stabilization_backend=stabilizer, 
                                        models=models, 
                                        config_output_folder=config_output_folder)


