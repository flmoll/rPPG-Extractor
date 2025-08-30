import os
import numpy as np
from run_pipeline_check import get_output_paths

from config_generation.utils import get_config_template_paths, generate_train_validation_test_triplet, generate_config_from_template, template_config_files, template_config_path

def generate_uncertainty_configs(config_output_folder):
    # Quantile and uncertainty experiment
    models = ["physnet_uncertainty", "physnet_quantile"]
    models_hr = ["hr_classifier_uncertainty", "hr_classifier_quantile"]
    post_processors = ['None', 'cumsum', 'detrend', 'butter']
    stabilization_backend = "YUNET"
    destabilizer = "None"
    files_normal = get_config_template_paths(models)
    files_hr = get_config_template_paths(models_hr)

    for train, validation, test, train_hr, validation_hr, test_hr in zip(*files_normal, *files_hr):
        config_name = f"{train.split('/')[-1]}"

        kwargs = {
            "EXP_NAME": "Uncertainty",
            "STABILIZATION_BACKEND": stabilization_backend,
            "DESTAB_BACKEND": destabilizer,
            "DESTAB_AMPLITUDE": 10.0,
            "MODEL_NAME_EXTENSION": ""
        }
        
        train_out_path, _, _ = generate_train_validation_test_triplet(
            train_template=train,
            validation_template=validation,
            test_template=test,
            output_folder=os.path.join(config_output_folder, "uncertainty"),
            config_name=config_name,
            **kwargs
        )
        
        file, output_path, train_basename, valid_basename, test_basename, model_output_path, config = get_output_paths([train_out_path])[0]

        kwargs['PPG_INFERENCE_MODEL_PATH'] = model_output_path

        for post_processor in post_processors:
            name_extension = f"{post_processor}"
            config_name = f"{name_extension}_{train_hr.split('/')[-1]}"

            kwargs["MODEL_NAME_EXTENSION"] = f"_{post_processor}"

            generate_train_validation_test_triplet(
                train_template=train_hr,
                validation_template=validation_hr,
                test_template=test_hr,
                output_folder=os.path.join(config_output_folder, "uncertainty"),
                config_name=config_name,
                pipeline_index_offset=1,
                **kwargs,
                POSTPROCESS=post_processor,
            )
        
        preprocessing_config_path = os.path.join(template_config_path, "preprocessing", template_config_files["preprocessing"])
        out_config_path = os.path.join(config_output_folder, "uncertainty", "preprocessing", f"0_preprocessing.yaml")
        generate_config_from_template(
            preprocessing_config_path, out_config_path, 
            EXP_NAME="Uncertainty",
            STABILIZATION_BACKEND=stabilization_backend,
            MODEL_NAME_EXTENSION=name_extension,
            DESTAB_BACKEND=destabilizer,
            DESTAB_AMPLITUDE=10.0
        )
