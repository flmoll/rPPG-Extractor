import subprocess
import sys
import os

experiment_configs_dir = "configs/all_experiment_configs"
pipeline_configs_dir = "configs/pipeline"

def prepare_experiment(experiment_name):
    base_path = os.path.join(experiment_configs_dir, experiment_name)

    if not os.path.exists(base_path):
        print(f"Experiment {experiment_name} does not exist in {experiment_configs_dir}.")
        return
    
    print(f"Preparing experiment: {experiment_name}")

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".yaml"):
                config_file = os.path.join(root, file)
                config_file_relative = os.path.relpath(config_file, experiment_configs_dir)
                config_parent_folder = os.path.dirname(config_file_relative)
                file_without_extension = os.path.splitext(file)[0]
                new_filename = f"{file_without_extension}_{config_parent_folder.replace(os.sep, '_')}.yaml"
                pipeline_config_file = os.path.join(pipeline_configs_dir, new_filename)

                # Copy the config file to the pipeline configs directory
                with open(config_file, "rb") as src, open(pipeline_config_file, "wb") as dst:
                    dst.write(src.read())


if __name__ == "__main__":

    experiment_names = os.listdir(experiment_configs_dir)
    experiment_names.append("all")

    if len(sys.argv) < 2 or len(sys.argv) > 3 or sys.argv[1] not in experiment_names:
        print("Usage: python run_experiment.py <experiment_to_run> [output_dir]")
        print("Available experiments:")

        for experiment in experiment_names:
            print(f"- {experiment}")

        sys.exit(1)

    experiment_to_run = sys.argv[1]

    if len(sys.argv) == 3:
        output_dir = sys.argv[2]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        pipeline_configs_dir = output_dir
        
    os.makedirs(pipeline_configs_dir, exist_ok=True)

    if experiment_to_run == "all":
        for experiment in experiment_names:
            if experiment != "all":
                prepare_experiment(experiment)
    else:
        prepare_experiment(experiment_to_run)

    print(f"Experiment {experiment_to_run} setup complete.")



