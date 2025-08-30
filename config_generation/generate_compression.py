import os
import numpy as np
from run_pipeline_check import get_output_paths

from config_generation.utils import get_config_template_paths, generate_train_validation_test_triplet, generate_config_from_template, template_config_files, template_config_path


def generate_compression_configs(config_output_folder):
    # compression Experiment
    models = ["physnet"]
    compression_backends = ["H264_QP", "H265_QP"]
    stabilization_backend = "Y5F"
    q_factors = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50], dtype=float)

    train_base_config_files, validation_base_config_files, test_base_config_files = get_config_template_paths(models)

    for compression_backend in compression_backends:
        for q_factor in q_factors:
            for train, validation, test in zip(train_base_config_files, validation_base_config_files, test_base_config_files):
                name_extension = f"{compression_backend}_{q_factor}"
                config_name = f"{name_extension}_{train.split('/')[-1]}"

                kwargs = {
                    "EXP_NAME": "Compression/train",
                    "STABILIZATION_BACKEND": stabilization_backend,
                    "MODEL_NAME_EXTENSION": "",
                    "DESTAB_BACKEND": compression_backend,
                    "DESTAB_AMPLITUDE": q_factor
                }

                train_out_path = os.path.join(config_output_folder, "compression", "train", f"1_{name_extension}_{train.split('/')[-1]}")
                generate_config_from_template(
                    template_path=train,
                    output_path=train_out_path,
                    **kwargs
                )

                file, output_path, train_basename, valid_basename, test_basename, model_output_path, config = get_output_paths([train_out_path])[0]

                kwargs['MODEL_PATH'] = model_output_path

                for test_q_factor in q_factors:
                    
                    kwargs['DESTAB_AMPLITUDE'] = test_q_factor
                    kwargs["EXP_NAME"] = f"Compression/validation/{q_factor}"
                    validation_out_path = os.path.join(config_output_folder, "compression", "validation", f"2_{name_extension}_{test_q_factor}_{validation.split('/')[-1]}")

                    generate_config_from_template(
                        template_path=validation,
                        output_path=validation_out_path,
                        **kwargs
                    )

                    kwargs["EXP_NAME"] = f"Compression/test/{q_factor}"
                    test_out_path = os.path.join(config_output_folder, "compression", "test", f"2_{name_extension}_{test_q_factor}_{test.split('/')[-1]}")

                    generate_config_from_template(
                        template_path=test,
                        output_path=test_out_path,
                        **kwargs
                    )

            preprocessing_config_path = os.path.join(template_config_path, "preprocessing", template_config_files["preprocessing"])
            out_config_path = os.path.join(config_output_folder, "compression", "preprocessing", f"0_{name_extension}_preprocessing.yaml")
            generate_config_from_template(
                preprocessing_config_path, out_config_path, 
                EXP_NAME="Compression",
                STABILIZATION_BACKEND=stabilization_backend,
                DESTAB_BACKEND=compression_backend,
                Q_FACTOR=q_factor,
                MODEL_NAME_EXTENSION="",
                DESTAB_AMPLITUDE=q_factor
            )