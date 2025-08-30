import os
import numpy as np
from run_pipeline_check import get_output_paths

from config_generation.utils import get_config_template_paths, generate_train_validation_test_triplet, generate_config_from_template, template_config_files, template_config_path

def generate_preprocessing(config_output_folder, template_filename, size, stabilization_backend="YUNET"):
    preprocessing_config_path = os.path.join(template_config_path, "preprocessing", template_filename)
    out_config_path = os.path.join(config_output_folder, "deadalive", "preprocessing", f"0_{size}_{template_filename}_{stabilization_backend}.yaml")
    
    generate_config_from_template(
        preprocessing_config_path, out_config_path, 
        EXP_NAME="DeadAlive",
        SIZE=size,
        STABILIZATION_BACKEND=stabilization_backend
    )

def generate_deadalive_configs(config_output_folder):
    # Inference on Dead and Own Videos
    models = ["efficientphys", "physformer", "physnet", "rhythmformer", "tscan", "physnet_uncertainty", "physnet_quantile"]
    train_templates, dead_templates, own_videos_templates = get_config_template_paths(models, validation_folder="dead", test_folder="own_videos")
    _, validation_templates, test_templates = get_config_template_paths(models)
    _, emergency_templates, _ = get_config_template_paths(models, validation_folder="emergency")
    
    for train, validation, test, idx in zip(train_templates, dead_templates, own_videos_templates, range(len(train_templates))):
        config_name = f"{train.split('/')[-1]}"

        if models[idx] == "rhythmformer" or models[idx] == "physformer":
            size = 128
        else:
            size = 72
        
        train_output_path, _, _ = generate_train_validation_test_triplet(
            train_template=train,
            validation_template=validation,
            test_template=test,
            validation_folder_name="dead",
            test_folder_name="own_videos",
            output_folder=os.path.join(config_output_folder, "deadalive"),
            config_name=config_name,
            EXP_NAME="DeadAlive",
            SIZE=size,
        )

        file, output_path, train_basename, valid_basename, test_basename, model_output_path, config = get_output_paths([train_output_path])[0]

        generate_config_from_template(
            validation_templates[idx],
            output_path=os.path.join(config_output_folder, "deadalive", "validation", f"2_{config_name}"),
            EXP_NAME="DeadAlive/validation",
            SIZE=size,
            MODEL_PATH=model_output_path,
        )

        generate_config_from_template(
            test_templates[idx],
            output_path=os.path.join(config_output_folder, "deadalive", "test", f"2_{config_name}"),
            EXP_NAME="DeadAlive/test",
            SIZE=size,
            MODEL_PATH=model_output_path,
        )

        generate_config_from_template(
            validation_templates[idx],
            output_path=os.path.join(config_output_folder, "deadalive", "validation_shuffle", f"2_{config_name}"),
            EXP_NAME="DeadAlive/validation_shuffle",
            SIZE=size,
            DATA_AUG="[\"RandomFrameShuffle 1.0\"]",
            MODEL_PATH=model_output_path,
        )

        generate_config_from_template(
            test_templates[idx],
            output_path=os.path.join(config_output_folder, "deadalive", "test_shuffle", f"2_{config_name}"),
            EXP_NAME="DeadAlive/test_shuffle",
            SIZE=size,
            DATA_AUG="[\"RandomFrameShuffle 1.0\"]",
            MODEL_PATH=model_output_path,
        )

        generate_config_from_template(
            emergency_templates[idx],
            output_path=os.path.join(config_output_folder, "deadalive", "emergency", f"2_{config_name}"),
            EXP_NAME="DeadAlive/emergency",
            SIZE=size,
            MODEL_PATH=model_output_path,
        )

        generate_config_from_template(
            emergency_templates[idx],
            output_path=os.path.join(config_output_folder, "deadalive", "emergency_mosse", f"2_{config_name}"),
            EXP_NAME="DeadAlive/emergency",
            SIZE=size,
            MODEL_PATH=model_output_path,
            STABILIZATION_BACKEND="MOSSE"
        )

        if models[idx] == "physnet_uncertainty" or models[idx] == "physnet_quantile":
            
            kwargs = {
                "PPG_INFERENCE_MODEL_PATH": model_output_path,
                "EXP_NAME": "DeadAlive",
                "SIZE": size,
            }
            if models[idx] == "physnet_uncertainty":
                curr_hr_model = "hr_classifier_uncertainty"
            else:
                curr_hr_model = "hr_classifier_quantile"
            
            train_hr_templates, dead_hr_templates, own_videos_hr_templates = get_config_template_paths([curr_hr_model], validation_folder="dead", test_folder="own_videos")
            _, validation_hr_templates, test_hr_templates = get_config_template_paths([curr_hr_model])
            _, emergency_hr_templates, _ = get_config_template_paths([curr_hr_model], validation_folder="emergency")
            config_name = f"{train_hr_templates[0].split('/')[-1]}"

            train_hr_output_path, _, _ = generate_train_validation_test_triplet(
                train_template=train_hr_templates[0],
                validation_template=dead_hr_templates[0],
                test_template=own_videos_hr_templates[0],
                validation_folder_name="dead",
                test_folder_name="own_videos",
                pipeline_index_offset=1,
                output_folder=os.path.join(config_output_folder, "deadalive"),
                config_name=config_name,
                **kwargs,
            )
            
            file, output_path, train_basename, valid_basename, test_basename, model_hr_output_path, config = get_output_paths([train_hr_output_path])[0]

            kwargs["MODEL_PATH"] = model_hr_output_path
            kwargs["EXP_NAME"] = "DeadAlive/validation"

            generate_config_from_template(
                validation_hr_templates[0],
                output_path=os.path.join(config_output_folder, "deadalive", "validation", f"3_{config_name}"),
                **kwargs,
            )

            kwargs["EXP_NAME"] = "DeadAlive/test"

            generate_config_from_template(
                test_hr_templates[0],
                output_path=os.path.join(config_output_folder, "deadalive", "test", f"3_{config_name}"),
                **kwargs,
            )

            kwargs["EXP_NAME"] = "DeadAlive/validation_shuffle"

            generate_config_from_template(
                validation_hr_templates[0],
                output_path=os.path.join(config_output_folder, "deadalive", "validation_shuffle", f"3_{config_name}"),
                **kwargs,
                DATA_AUG="[\"RandomFrameShuffle 1.0\"]",
            )

            kwargs["EXP_NAME"] = "DeadAlive/test_shuffle"

            generate_config_from_template(
                test_hr_templates[0],
                output_path=os.path.join(config_output_folder, "deadalive", "test_shuffle", f"3_{config_name}"),
                **kwargs,
                DATA_AUG="[\"RandomFrameShuffle 1.0\"]",
            )

            kwargs["EXP_NAME"] = "DeadAlive/emergency"

            generate_config_from_template(
                emergency_hr_templates[0],
                output_path=os.path.join(config_output_folder, "deadalive", "emergency", f"3_{config_name}"),
                **kwargs,
            )

            kwargs["EXP_NAME"] = "DeadAlive/emergency"

            generate_config_from_template(
                emergency_hr_templates[0],
                output_path=os.path.join(config_output_folder, "deadalive", "emergency_mosse", f"3_{config_name}"),
                **kwargs,
                STABILIZATION_BACKEND="MOSSE"
            )



    generate_preprocessing(config_output_folder, template_config_files["preprocessing"], 72)
    generate_preprocessing(config_output_folder, template_config_files["preprocessing"], 128)
    generate_preprocessing(config_output_folder, template_config_files["preprocessing_dead"], 72)
    generate_preprocessing(config_output_folder, template_config_files["preprocessing_dead"], 128)
    generate_preprocessing(config_output_folder, template_config_files["preprocessing_own_videos"], 72)
    generate_preprocessing(config_output_folder, template_config_files["preprocessing_own_videos"], 128)
    generate_preprocessing(config_output_folder, template_config_files["preprocessing_emergency"], 72)
    generate_preprocessing(config_output_folder, template_config_files["preprocessing_emergency"], 128)
    generate_preprocessing(config_output_folder, template_config_files["preprocessing_emergency"], 72, stabilization_backend="MOSSE")
    generate_preprocessing(config_output_folder, template_config_files["preprocessing_emergency"], 128, stabilization_backend="MOSSE")
