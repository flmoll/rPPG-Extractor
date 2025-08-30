import os
from run_pipeline_check import get_output_paths

template_config_files = {
    "efficientphys": "VitalVideos_UBFC-rPPG_UBFC-PHYS_EFFICIENTPHYS.yaml",
    "physformer": "VitalVideos_UBFC-rPPG_UBFC-PHYS_PHYSFORMER.yaml",
    "physnet": "VitalVideos_UBFC-rPPG_UBFC-PHYS_PHYSNET_BASIC.yaml",
    "rhythmformer": "VitalVideos_UBFC-rPPG_UBFC-PHYS_RHYTHMFORMER.yaml",
    "tscan": "VitalVideos_UBFC-rPPG_UBFC-PHYS_TSCAN.yaml",
    "physnet_uncertainty": "VitalVideos_UBFC-rPPG_UBFC-PHYS_PHYSNET_UNCERTAINTY.yaml",
    "physnet_quantile": "VitalVideos_UBFC-rPPG_UBFC-PHYS_PHYSNET_QUANTILE.yaml",
    "physnet_lifeness": "VitalVideos_UBFC-rPPG_UBFC-PHYS_PHYSNET_LIFENESS.yaml",
    "hr_classifier_uncertainty": "VitalVideos_UBFC-rPPG_UBFC-PHYS_HR_CLASSIFIER_UNCERTAINTY.yaml",
    "hr_classifier_quantile": "VitalVideos_UBFC-rPPG_UBFC-PHYS_HR_CLASSIFIER_QUANTILE.yaml",
    "preprocessing": "VitalVideos_UBFC-rPPG_UBFC-PHYS_preprocessing.yaml",
    "preprocessing_dead": "Dead_preprocessing.yaml",
    "preprocessing_own_videos": "OwnVideos_preprocessing.yaml",
    "preprocessing_emergency": "Emergency_preprocessing.yaml",
}

default_config = {
    "EXP_NAME": "Experiment",
    "STABILIZATION_BACKEND": "YUNET",
    "MODEL_NAME_EXTENSION": "",
    "DESTAB_BACKEND": "None",
    "DESTAB_AMPLITUDE": 10.0,
    "ADDITIONAL_SIZES": "[]",  # Additional sizes for resizing, if needed
    "SIZE": 72,  # Default size for resizing
    "POSTPROCESS": "None",  # Default post-processing method
    "PPG_INFERENCE_MODEL_PATH": "",  # Path to the PPG inference model
    "DATA_AUG": "[]",  # Default data augmentation method
}

stabilization_backends_with_dd_frequency = {
    "YUNET": 30,
    "Y5F": 30,
    "HC": 30,
    "MOSSE": 1,
    "CORRYU": 1,
    "OPTYU": 1
}

template_config_path = "configs/template_configs"

def generate_config_from_template(template_path, output_path, **kwargs):
    with open(template_path, "r") as template_file:
        template_content = template_file.read()

    for key, value in default_config.items():
        if key not in kwargs:
            kwargs[key] = value

    kwargs["DD_FREQUENCY"] = stabilization_backends_with_dd_frequency[kwargs["STABILIZATION_BACKEND"]]

    # Replace placeholders with actual values
    for key, value in kwargs.items():
        template_content = template_content.replace(f"%%{key}%%", str(value))

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as config_file:
        config_file.write(template_content)

def generate_train_validation_test_triplet(train_template, validation_template, test_template, output_folder, config_name, train_folder_name="train", validation_folder_name="validation", test_folder_name="test", pipeline_index_offset=0, **kwargs):
    """
    Generates train, validation, and test config files from templates.
    """
    train_pipeline_index = pipeline_index_offset + 1
    valid_and_test_pipeline_index = pipeline_index_offset + 2

    train_output_path = os.path.join(output_folder, train_folder_name, f"{train_pipeline_index}_{config_name}")
    validation_output_path = os.path.join(output_folder, validation_folder_name, f"{valid_and_test_pipeline_index}_{config_name}")
    test_output_path = os.path.join(output_folder, test_folder_name, f"{valid_and_test_pipeline_index}_{config_name}")

    exp_name = kwargs["EXP_NAME"]
    kwargs["EXP_NAME"] = exp_name + "/" + train_folder_name

    generate_config_from_template(train_template, train_output_path, **kwargs)

    file, output_path, train_basename, valid_basename, test_basename, model_output_path, config = get_output_paths([train_output_path])[0]
    kwargs['MODEL_PATH'] = model_output_path

    kwargs["EXP_NAME"] = exp_name + "/" + validation_folder_name
    generate_config_from_template(validation_template, validation_output_path, **kwargs)
    kwargs["EXP_NAME"] = exp_name + "/" + test_folder_name
    generate_config_from_template(test_template, test_output_path, **kwargs)

    return train_output_path, validation_output_path, test_output_path

def get_config_template_paths(models, train_folder="train", validation_folder="validation", test_folder="test"):
    
    curr_template_files = [template_config_files[model] for model in models]

    train_base_path = os.path.join(template_config_path, train_folder)
    train_base_config_files = [os.path.join(train_base_path, f) for f in curr_template_files]

    validation_base_path = os.path.join(template_config_path, validation_folder)
    validation_base_config_files = [os.path.join(validation_base_path, f) for f in curr_template_files]

    test_base_path = os.path.join(template_config_path, test_folder)
    test_base_config_files = [os.path.join(test_base_path, f) for f in curr_template_files]

    return train_base_config_files, validation_base_config_files, test_base_config_files
    