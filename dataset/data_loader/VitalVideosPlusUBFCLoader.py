
import logging
import os
import shutil
from dataset.data_loader.BaseLoader import BaseLoader
from dataset.data_loader.VitalVideosLoader import VitalVideosLoader
from dataset.data_loader.UBFCrPPGLoader import UBFCrPPGLoader
from torch.utils.data import Dataset


"""
Update the cached path for the dataset.
This is necessary since this dataloader loads two datasets
"""
def update_cached_path(config):
    print(config.DATASET)
    EXP_DATA_NAME = "_".join([config.DATASET, "SizeW{0}".format(
    str(config.PREPROCESS.RESIZE.W)), "SizeH{0}".format(str(config.PREPROCESS.RESIZE.H)), "ClipLength{0}".format(
    str(config.PREPROCESS.CHUNK_LENGTH)), "DataType{0}".format("_".join(config.PREPROCESS.DATA_TYPE)),
                                "DataAug{0}".format("_".join(config.PREPROCESS.DATA_AUG)),
                                "LabelType{0}".format(config.PREPROCESS.LABEL_TYPE),
                                "Crop_face{0}".format(config.PREPROCESS.CROP_FACE.DO_CROP_FACE),
                                "Backend{0}".format(config.PREPROCESS.CROP_FACE.BACKEND),
                                "Large_box{0}".format(config.PREPROCESS.CROP_FACE.USE_LARGE_FACE_BOX),
                                "Large_size{0}".format(config.PREPROCESS.CROP_FACE.LARGE_BOX_COEF),
                                "Dyamic_Det{0}".format(config.PREPROCESS.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION),
                                "det_len{0}".format(config.PREPROCESS.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY),
                                "Median_face_box{0}".format(config.PREPROCESS.CROP_FACE.DETECTION.USE_MEDIAN_FACE_BOX),
                                "D{0}".format(config.PREPROCESS.ARTIFICIAL_DESTABILISATION_BACKEND),
                                "amp{0}".format(config.PREPROCESS.ARTIFICIAL_DESTABILISATION_AMPLITUDE)

                                        ])
    CACHED_PATH = os.path.join("/mnt/results/preprocessed_rppgToolbox/", EXP_DATA_NAME)
    return CACHED_PATH
    
class VitalVideosAndUBFCLoader(Dataset):

    """
    This class loads both Vital Videos and UBFC datasets.
    It can be used as a normal dataloader. It will abstract the process and act like UBFC and VitalVideos are merged into one dataset
    """

    def __init__(self, name, data_path, config_data, device=None, logger=None):

        if logger is None:
            logger = logging.getLogger(__name__)

        self.logger = logger
        print(data_path)

        self.ubfc_data_path = data_path + "/ubfc"
        self.vital_data_path = data_path + "/vitalVideos"

        if isinstance(config_data.BEGINS, list) and len(config_data.BEGINS) == 2:
            vital_videos_begin = config_data.BEGINS[0]
            ubfc_begin = config_data.BEGINS[1]
            vital_videos_end = config_data.ENDS[0]
            ubfc_end = config_data.ENDS[1]
        else:
            raise ValueError("The config_data.BEGINS should be a list of tuples.")
        
        vital_videos_config = config_data.clone()
        vital_videos_config.defrost()
        vital_videos_config.DATASET = "VitalVideos"
        vital_videos_config.DATA_PATH = self.vital_data_path
        vital_videos_config.CACHED_PATH = update_cached_path(vital_videos_config)
        vital_videos_config.FILE_LIST_PATH = os.path.join(vital_videos_config.CACHED_PATH, 'DataFileLists')
        vital_videos_config.BEGIN = vital_videos_begin
        vital_videos_config.END = vital_videos_end
        vital_videos_config.freeze()

        ubfc_config = config_data.clone()
        ubfc_config.defrost()
        ubfc_config.DATASET = "UBFC-rPPG"
        ubfc_config.DATA_PATH = self.ubfc_data_path
        ubfc_config.CACHED_PATH = update_cached_path(ubfc_config)
        ubfc_config.FILE_LIST_PATH = os.path.join(ubfc_config.CACHED_PATH, 'DataFileLists')
        ubfc_config.BEGIN = ubfc_begin
        ubfc_config.END = ubfc_end
        ubfc_config.freeze()

        if vital_videos_begin < vital_videos_end:
            self.vital_videos_loader = VitalVideosLoader(name, self.vital_data_path, vital_videos_config, device)
        else:
            self.vital_videos_loader = None
            print("No Vital Videos data to load.")
        if ubfc_begin < ubfc_end:
            self.ubfc_loader = UBFCrPPGLoader(name, self.ubfc_data_path, ubfc_config, device)
        else:
            self.ubfc_loader = None
            print("No UBFC data to load.")


    def __len__(self):
        """Returns the length of the dataset."""
        data_len = 0
        if self.vital_videos_loader is not None:
            data_len += len(self.vital_videos_loader)
        if self.ubfc_loader is not None:
            data_len += len(self.ubfc_loader)
        return data_len

    def __getitem__(self, index):

        if self.vital_videos_loader is None:
            return self.ubfc_loader[index]
        if self.ubfc_loader is None:
            return self.vital_videos_loader[index]

        if index < len(self.vital_videos_loader):
            return self.vital_videos_loader[index]
        else:
            index -= len(self.vital_videos_loader)
            return self.ubfc_loader[index]
