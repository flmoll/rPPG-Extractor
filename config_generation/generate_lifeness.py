import os
import numpy as np
from run_pipeline_check import get_output_paths

from config_generation.utils import get_config_template_paths, generate_train_validation_test_triplet, generate_config_from_template, template_config_files, template_config_path

def generate_lifeness_configs(config_output_folder):
    train_templates, validation_templates, test_templates = get_config_template_paths(["physnet_lifeness"])
    _, dead_templates, own_videos_templates = get_config_template_paths(["physnet_lifeness"], validation_folder="dead", test_folder="own_videos")
    _, emergency_templates, _ = get_config_template_paths(["physnet_lifeness"], validation_folder="emergency")

    for train_template, validation_template, test_template, dead_template, own_videos_template, emergency_template, idx in zip(train_templates, validation_templates, test_templates, dead_templates, own_videos_templates, emergency_templates, range(len(train_templates))):
        config_name = os.path.basename(train_template)

        generate_train_validation_test_triplet(
            train_template=train_template,
            validation_template=validation_template,
            test_template=test_template,
            output_folder=os.path.join(config_output_folder, "lifeness"),
            config_name=config_name,
            EXP_NAME="Lifeness",
        )
        
        generate_train_validation_test_triplet(
            train_template=train_template,
            validation_template=dead_template,
            test_template=own_videos_template,
            validation_folder_name="dead",
            test_folder_name="own_videos",
            output_folder=os.path.join(config_output_folder, "lifeness"),
            config_name=config_name,
            EXP_NAME="Lifeness",
        )
        
        generate_train_validation_test_triplet(
            train_template=train_template,
            validation_template=emergency_template,
            test_template=own_videos_template,
            validation_folder_name="emergency",
            test_folder_name="own_videos",
            output_folder=os.path.join(config_output_folder, "lifeness"),
            config_name=config_name,
            EXP_NAME="Lifeness",
        )
        
        generate_train_validation_test_triplet(
            train_template=train_template,
            validation_template=validation_template,
            test_template=test_template,
            validation_folder_name="validation_shuffle",
            test_folder_name="test_shuffle",
            output_folder=os.path.join(config_output_folder, "lifeness"),
            config_name=config_name,
            EXP_NAME="Lifeness",
            DATA_AUG="[\"RandomFrameShuffle 1.0\"]",
        )

    preprocessing_config_path = os.path.join(template_config_path, "preprocessing", template_config_files["preprocessing"])
    out_config_path = os.path.join(config_output_folder, "lifeness", "preprocessing", f"0_preprocessing.yaml")
    generate_config_from_template(
        template_path=preprocessing_config_path,
        output_path=out_config_path,
        EXP_NAME="Lifeness"
    )

    preprocessing_config_path = os.path.join(template_config_path, "preprocessing", template_config_files["preprocessing_dead"])
    out_config_path = os.path.join(config_output_folder, "lifeness", "preprocessing", f"0_dead_preprocessing.yaml")
    generate_config_from_template(
        template_path=preprocessing_config_path,
        output_path=out_config_path,
        EXP_NAME="Lifeness"
    )

    preprocessing_config_path = os.path.join(template_config_path, "preprocessing", template_config_files["preprocessing_own_videos"])
    out_config_path = os.path.join(config_output_folder, "lifeness", "preprocessing", f"0_own_videos_preprocessing.yaml")
    generate_config_from_template(
        template_path=preprocessing_config_path,
        output_path=out_config_path,
        EXP_NAME="Lifeness"
    )
    
    preprocessing_config_path = os.path.join(template_config_path, "preprocessing", template_config_files["preprocessing_emergency"])
    out_config_path = os.path.join(config_output_folder, "lifeness", "preprocessing", f"0_emergency_preprocessing.yaml")
    generate_config_from_template(
        template_path=preprocessing_config_path,
        output_path=out_config_path,
        EXP_NAME="Lifeness"
    )
