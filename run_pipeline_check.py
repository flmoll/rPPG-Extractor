
import os
import shutil

import filelock
from config import update_config, _C

pipeline_configs_dir = "configs/pipeline"
pre_pipeline_configs_dir = "configs/pre_pipeline"
lock_file = os.path.join(pipeline_configs_dir, "lockfile.lock")

class argsMock:
    def __init__(self, config_file):
        self.config_file = config_file

def get_output_paths(files_pipelined_all):
    output_paths = []
    for file in files_pipelined_all:
        config = _C.clone()
        args = argsMock(config_file=file)
        update_config(config, args)

        train_basename = os.path.basename(config.TRAIN.DATA.CACHED_PATH)
        valid_basename = os.path.basename(config.VALID.DATA.CACHED_PATH)
        test_basename = os.path.basename(config.TEST.DATA.CACHED_PATH)

        output_dir = config.TEST.OUTPUT_SAVE_DIR
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Filename ID to be used in any output files that get saved
        if config.TOOLBOX_MODE == 'train_and_test':
            filename_id = config.TRAIN.MODEL_FILE_NAME
            model_output_path = os.path.join(config.MODEL.MODEL_DIR, filename_id + "_best.pth")
        elif config.TOOLBOX_MODE == 'only_test':
            model_file_root = config.INFERENCE.MODEL_PATH.split("/")[-1].split(".pth")[0]
            filename_id = model_file_root + "_" + config.TEST.DATA.DATASET
            model_output_path = ""
        else:
            raise ValueError('Metrics.py evaluation only supports train_and_test and only_test!')
        
        output_path = os.path.join(output_dir, filename_id + '_outputs.pickle')

        #output_path = os.path.join(config.LOG.PATH, train_basename, "saved_test_outputs", f"{config.TRAIN.MODEL_FILE_NAME}_outputs.pickle")
        output_paths.append((file, output_path, train_basename, valid_basename, test_basename, model_output_path, config))
    return output_paths

def check_paths(output_paths, suppress_duplicate_warning=False, suppress_output_exists_warning=False, suppress_train_valid_test_warning=False, suppress_model_not_exists_warning=False):

    seen_paths = set()
    out_path_to_file = dict()
    for file, output_path, _, _, _, _, _ in output_paths:
        if output_path in seen_paths and not suppress_duplicate_warning:
            print(f"Duplicate output path detected: {output_path}")
            print(f"File: {file}")
            print(f"Conflicting path: {out_path_to_file[output_path]}")
        seen_paths.add(output_path)
        out_path_to_file[output_path] = file

    for file_path, output_path, train_basename, valid_basename, test_basename, model_output_path, config in output_paths:
        if os.path.exists(output_path) and not suppress_output_exists_warning:
            print(f"Output file {output_path} already exists")

        if config.TOOLBOX_MODE == 'train_and_test':
            if train_basename != valid_basename or train_basename != test_basename or valid_basename != test_basename:
                if not suppress_train_valid_test_warning:
                    print(f"Train, valid, and test configs are not the same in {output_path}")
                    print(f"Train: {train_basename}, Valid: {valid_basename}, Test: {test_basename}")
        else:
            model_path = config.INFERENCE.MODEL_PATH
            if not os.path.exists(model_path) and not suppress_model_not_exists_warning:
                print(f"Model file {model_path} does not exist")



if __name__ == "__main__":

    files_pipelined_all = os.listdir(pipeline_configs_dir)
    files_pipelined_all = sorted(files_pipelined_all)
    files_pipelined_all = [f for f in files_pipelined_all if f.endswith(".yaml")]
    files_pipelined_all = [os.path.join(pipeline_configs_dir, f) for f in files_pipelined_all]

    files_pipelined_pre = os.listdir(pre_pipeline_configs_dir)
    files_pipelined_pre = sorted(files_pipelined_pre)
    files_pipelined_pre = [f for f in files_pipelined_pre if f.endswith(".yaml")]
    files_pipelined_pre = [os.path.join(pre_pipeline_configs_dir, f) for f in files_pipelined_pre]

    files_pipelined_all.extend(files_pipelined_pre)

    output_paths = get_output_paths(files_pipelined_all)
    check_paths(output_paths)

    print("The following files are in the pipeline directory:")
    for file in files_pipelined_all:
        print(file)

    print("Do you want to continue? (y/n)")
    response = input().strip().lower()
    if response == 'y':
        print("Continuing...")
    else:
        print("Exiting...")
        exit(0)

    with filelock.FileLock(lock_file):
        for i, file in enumerate(files_pipelined_all):
            shutil.move(file, os.path.join(pipeline_configs_dir, os.path.basename(file)))