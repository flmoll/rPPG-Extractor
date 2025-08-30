""" The main function of rPPG deep learning pipeline."""
import matplotlib
matplotlib.use("Agg")  # Non-GUI backend, outputs to files instead of screen

import argparse
import random
import time

import numpy as np
import torch
from config import get_config
from dataset import data_loader
import dataset.data_loader.VitalVideosLoader
from neural_methods import trainer
from unsupervised_methods.unsupervised_predictor import unsupervised_predict
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity

RANDOM_SEED = 100
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Create a general generator for use with the validation dataloader,
# the test dataloader, and the unsupervised dataloader
general_generator = torch.Generator()
general_generator.manual_seed(RANDOM_SEED)
# Create a training generator to isolate the train dataloader from
# other dataloaders and better control non-deterministic behavior
train_generator = torch.Generator()
train_generator.manual_seed(RANDOM_SEED)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def add_args(parser):
    """Adds arguments for parser."""
    parser.add_argument('--config_file', required=False,
                        default="configs/train_configs/PURE_PURE_UBFC-rPPG_TSCAN_BASIC.yaml", type=str, help="The name of the model.")
    '''Neural Method Sample YAML LIST:
      SCAMPS_SCAMPS_UBFC-rPPG_TSCAN_BASIC.yaml
      SCAMPS_SCAMPS_UBFC-rPPG_DEEPPHYS_BASIC.yaml
      SCAMPS_SCAMPS_UBFC-rPPG_PHYSNET_BASIC.yaml
      SCAMPS_SCAMPS_PURE_DEEPPHYS_BASIC.yaml
      SCAMPS_SCAMPS_PURE_TSCAN_BASIC.yaml
      SCAMPS_SCAMPS_PURE_PHYSNET_BASIC.yaml
      PURE_PURE_UBFC-rPPG_TSCAN_BASIC.yaml
      PURE_PURE_UBFC-rPPG_DEEPPHYS_BASIC.yaml
      PURE_PURE_UBFC-rPPG_PHYSNET_BASIC.yaml
      PURE_PURE_MMPD_TSCAN_BASIC.yaml
      UBFC-rPPG_UBFC-rPPG_PURE_TSCAN_BASIC.yaml
      UBFC-rPPG_UBFC-rPPG_PURE_DEEPPHYS_BASIC.yaml
      UBFC-rPPG_UBFC-rPPG_PURE_PHYSNET_BASIC.yaml
      MMPD_MMPD_UBFC-rPPG_TSCAN_BASIC.yaml
    Unsupervised Method Sample YAML LIST:
      PURE_UNSUPERVISED.yaml
      UBFC-rPPG_UNSUPERVISED.yaml
    '''
    return parser


def train_and_test(config, data_loader_dict):
    """Trains the model."""
    if config.MODEL.NAME == "Physnet":
        model_trainer = trainer.PhysnetTrainer.PhysnetTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "iBVPNet":
        model_trainer = trainer.iBVPNetTrainer.iBVPNetTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "FactorizePhys":
        model_trainer = trainer.FactorizePhysTrainer.FactorizePhysTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "Tscan":
        model_trainer = trainer.TscanTrainer.TscanTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "EfficientPhys":
        model_trainer = trainer.EfficientPhysTrainer.EfficientPhysTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'DeepPhys':
        model_trainer = trainer.DeepPhysTrainer.DeepPhysTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'BigSmall':
        model_trainer = trainer.BigSmallTrainer.BigSmallTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'PhysFormer':
        model_trainer = trainer.PhysFormerTrainer.PhysFormerTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'PhysMamba':
        model_trainer = trainer.PhysMambaTrainer.PhysMambaTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'RhythmFormer':
        model_trainer = trainer.RhythmFormerTrainer.RhythmFormerTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'PhysNetLifeness':
        model_trainer = trainer.PhysnetLifenessTrainer.PhysnetLifenessTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'PhysNetUncertainty':
        model_trainer = trainer.PhysnetUncertaintyTrainer.PhysnetUncertaintyTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'PhysNetQuantile':
        model_trainer = trainer.PhysnetQuantileTrainer.PhysnetQuantileTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'HRClassifierUncertainty':
        model_trainer = trainer.HRClassifierUncertaintyTrainer.HRClassifierUncertaintyTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'HRClassifierQuantile':
        model_trainer = trainer.HRClassifierQuantileTrainer.HRClassifierQuantileTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == None:
        return
    else:
        raise ValueError(f'Your Model {config.MODEL.NAME} is Not Supported  Yet!')
    model_trainer.train(data_loader_dict)
    model_trainer.test(data_loader_dict)


def test(config, data_loader_dict):
    """Tests the model."""
    if config.MODEL.NAME == "Physnet":
        model_trainer = trainer.PhysnetTrainer.PhysnetTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "iBVPNet":
        model_trainer = trainer.iBVPNetTrainer.iBVPNetTrainer(config, data_loader_dict)    
    elif config.MODEL.NAME == "FactorizePhys":
        model_trainer = trainer.FactorizePhysTrainer.FactorizePhysTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "Tscan":
        model_trainer = trainer.TscanTrainer.TscanTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "EfficientPhys":
        model_trainer = trainer.EfficientPhysTrainer.EfficientPhysTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'DeepPhys':
        model_trainer = trainer.DeepPhysTrainer.DeepPhysTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'BigSmall':
        model_trainer = trainer.BigSmallTrainer.BigSmallTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'PhysFormer':
        model_trainer = trainer.PhysFormerTrainer.PhysFormerTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'PhysMamba':
        model_trainer = trainer.PhysMambaTrainer.PhysMambaTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'RhythmFormer':
        model_trainer = trainer.RhythmFormerTrainer.RhythmFormerTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'PhysNetLifeness':
        model_trainer = trainer.PhysnetLifenessTrainer.PhysnetLifenessTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'PhysNetUncertainty':
        model_trainer = trainer.PhysnetUncertaintyTrainer.PhysnetUncertaintyTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'PhysNetQuantile':
        model_trainer = trainer.PhysnetQuantileTrainer.PhysnetQuantileTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'HRClassifierUncertainty':
        model_trainer = trainer.HRClassifierUncertaintyTrainer.HRClassifierUncertaintyTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'HRClassifierQuantile':
        model_trainer = trainer.HRClassifierQuantileTrainer.HRClassifierQuantileTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == None:
        return
    else:
        raise ValueError(f'Your Model {config.MODEL.NAME} is Not Supported  Yet!')
    model_trainer.test(data_loader_dict)


def unsupervised_method_inference(config, data_loader):
    if not config.UNSUPERVISED.METHOD:
        raise ValueError("Please set unsupervised method in yaml!")
    for unsupervised_method in config.UNSUPERVISED.METHOD:
        if unsupervised_method == "POS":
            unsupervised_predict(config, data_loader, "POS")
        elif unsupervised_method == "CHROM":
            unsupervised_predict(config, data_loader, "CHROM")
        elif unsupervised_method == "ICA":
            unsupervised_predict(config, data_loader, "ICA")
        elif unsupervised_method == "GREEN":
            unsupervised_predict(config, data_loader, "GREEN")
        elif unsupervised_method == "LGI":
            unsupervised_predict(config, data_loader, "LGI")
        elif unsupervised_method == "PBV":
            unsupervised_predict(config, data_loader, "PBV")
        elif unsupervised_method == "OMIT":
            unsupervised_predict(config, data_loader, "OMIT")
        else:
            raise ValueError("Not supported unsupervised method!")

def get_data_loader(config, device="cuda", batch_size=1):
    """Get the data loader by name."""
    # train_loader
    if config.DATA.DATASET == "UBFC-rPPG":
        loader_class = data_loader.UBFCrPPGLoader.UBFCrPPGLoader
    elif config.DATA.DATASET == "PURE":
        loader_class = data_loader.PURELoader.PURELoader
    elif config.DATA.DATASET == "SCAMPS":
        loader_class = data_loader.SCAMPSLoader.SCAMPSLoader
    elif config.DATA.DATASET == "MMPD":
        loader_class = data_loader.MMPDLoader.MMPDLoader
    elif config.DATA.DATASET == "BP4DPlus":
        loader_class = data_loader.BP4DPlusLoader.BP4DPlusLoader
    elif config.DATA.DATASET == "BP4DPlusBigSmall":
        loader_class = data_loader.BP4DPlusBigSmallLoader.BP4DPlusBigSmallLoader
    elif config.DATA.DATASET == "UBFC-PHYS":
        loader_class = data_loader.UBFCPHYSLoader.UBFCPHYSLoader
    elif config.DATA.DATASET == "iBVP":
        loader_class = data_loader.iBVPLoader.iBVPLoader
    elif config.DATA.DATASET == "VitalVideos":
        loader_class = data_loader.VitalVideosLoader.VitalVideosLoader
    elif config.DATA.DATASET == "VitalVideos_and_UBFC":
        loader_class = data_loader.VitalVideosPlusUBFCLoader.VitalVideosAndUBFCLoader
    elif config.DATA.DATASET == "Own_Videos":
        loader_class = data_loader.OwnVideosLoader.OwnVideosLoader
    elif config.DATA.DATASET == "Emergency_Videos":
        loader_class = data_loader.EmergencyVideosLoader.EmergencyVideosLoader
    elif config.DATA.DATASET == "Youtube_Videos":
        loader_class = data_loader.YoutubeVideosLoader.YoutubeVideosLoader
    else:
        raise ValueError("Unsupported dataset! Currently supporting UBFC-rPPG, PURE, MMPD, \
                            SCAMPS, BP4D+ (Normal and BigSmall preprocessing), UBFC-PHYS and iBVP.")

    # Create and initialize the train dataloader given the correct toolbox mode,
    # a supported dataset name, and a valid dataset paths
    if (config.DATA.DATASET and config.DATA.DATA_PATH):

        train_dataset = loader_class(
            name="train",
            data_path=config.DATA.DATA_PATH,
            config_data=config.DATA,
            device=device)
        train_data_loader = DataLoader(
            dataset=train_dataset,
            num_workers=config.NUM_WORKERS,
            batch_size=batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=train_generator, 
            pin_memory=config.PIN_MEMORY,
            persistent_workers=config.PERSISTENT_WORKERS
        )
        return train_data_loader
    else:
        return None

if __name__ == "__main__":
    # parse arguments.
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    parser = trainer.BaseTrainer.BaseTrainer.add_trainer_args(parser)
    parser = data_loader.BaseLoader.BaseLoader.add_data_loader_args(parser)
    args = parser.parse_args()

    # configurations.
    config = get_config(args)

    data_loader_dict = dict() # dictionary of data loaders 
    if config.TOOLBOX_MODE == "train_and_test":
        # train_loader
        if config.TRAIN.DATA.DATASET == "UBFC-rPPG":
            train_loader = data_loader.UBFCrPPGLoader.UBFCrPPGLoader
        elif config.TRAIN.DATA.DATASET == "PURE":
            train_loader = data_loader.PURELoader.PURELoader
        elif config.TRAIN.DATA.DATASET == "SCAMPS":
            train_loader = data_loader.SCAMPSLoader.SCAMPSLoader
        elif config.TRAIN.DATA.DATASET == "MMPD":
            train_loader = data_loader.MMPDLoader.MMPDLoader
        elif config.TRAIN.DATA.DATASET == "BP4DPlus":
            train_loader = data_loader.BP4DPlusLoader.BP4DPlusLoader
        elif config.TRAIN.DATA.DATASET == "BP4DPlusBigSmall":
            train_loader = data_loader.BP4DPlusBigSmallLoader.BP4DPlusBigSmallLoader
        elif config.TRAIN.DATA.DATASET == "UBFC-PHYS":
            train_loader = data_loader.UBFCPHYSLoader.UBFCPHYSLoader
        elif config.TRAIN.DATA.DATASET == "iBVP":
            train_loader = data_loader.iBVPLoader.iBVPLoader
        elif config.TRAIN.DATA.DATASET == "VitalVideos":
            train_loader = data_loader.VitalVideosLoader.VitalVideosLoader
        elif config.TRAIN.DATA.DATASET == "VitalVideos_and_UBFC":
            train_loader = data_loader.VitalVideosPlusUBFCLoader.VitalVideosAndUBFCLoader
        elif config.TRAIN.DATA.DATASET == "Own_Videos":
            train_loader = data_loader.OwnVideosLoader.OwnVideosLoader
        elif config.TRAIN.DATA.DATASET == "Emergency_Videos":
            train_loader = data_loader.EmergencyVideosLoader.EmergencyVideosLoader
        elif config.TRAIN.DATA.DATASET == "Youtube_Videos":
            train_loader = data_loader.YoutubeVideosLoader.YoutubeVideosLoader
        else:
            raise ValueError("Unsupported dataset! Currently supporting UBFC-rPPG, PURE, MMPD, \
                             SCAMPS, BP4D+ (Normal and BigSmall preprocessing), UBFC-PHYS and iBVP.")

        # Create and initialize the train dataloader given the correct toolbox mode,
        # a supported dataset name, and a valid dataset paths
        if (config.TRAIN.DATA.DATASET and config.TRAIN.DATA.DATA_PATH):

            train_data_loader = train_loader(
                name="train",
                data_path=config.TRAIN.DATA.DATA_PATH,
                config_data=config.TRAIN.DATA,
                device=config.DEVICE)
            data_loader_dict['train'] = DataLoader(
                dataset=train_data_loader,
                num_workers=config.TRAIN.NUM_WORKERS,
                batch_size=config.TRAIN.BATCH_SIZE,
                shuffle=True,
                worker_init_fn=seed_worker,
                generator=train_generator, 
                pin_memory=config.TRAIN.PIN_MEMORY,
                persistent_workers=config.TRAIN.PERSISTENT_WORKERS
            )
        else:
            data_loader_dict['train'] = None

        # valid_loader
        if config.VALID.DATA.DATASET == "UBFC-rPPG":
            valid_loader = data_loader.UBFCrPPGLoader.UBFCrPPGLoader
        elif config.VALID.DATA.DATASET == "PURE":
            valid_loader = data_loader.PURELoader.PURELoader
        elif config.VALID.DATA.DATASET == "SCAMPS":
            valid_loader = data_loader.SCAMPSLoader.SCAMPSLoader
        elif config.VALID.DATA.DATASET == "MMPD":
            valid_loader = data_loader.MMPDLoader.MMPDLoader
        elif config.VALID.DATA.DATASET == "BP4DPlus":
            valid_loader = data_loader.BP4DPlusLoader.BP4DPlusLoader
        elif config.VALID.DATA.DATASET == "BP4DPlusBigSmall":
            valid_loader = data_loader.BP4DPlusBigSmallLoader.BP4DPlusBigSmallLoader
        elif config.VALID.DATA.DATASET == "UBFC-PHYS":
            valid_loader = data_loader.UBFCPHYSLoader.UBFCPHYSLoader
        elif config.VALID.DATA.DATASET == "iBVP":
            valid_loader = data_loader.iBVPLoader.iBVPLoader
        elif config.VALID.DATA.DATASET == "VitalVideos":
            valid_loader = data_loader.VitalVideosLoader.VitalVideosLoader
        elif config.VALID.DATA.DATASET == "VitalVideos_and_UBFC":
            valid_loader = data_loader.VitalVideosPlusUBFCLoader.VitalVideosAndUBFCLoader
        elif config.VALID.DATA.DATASET == "Own_Videos":
            valid_loader = data_loader.OwnVideosLoader.OwnVideosLoader
        elif config.VALID.DATA.DATASET == "Emergency_Videos":
            valid_loader = data_loader.EmergencyVideosLoader.EmergencyVideosLoader
        elif config.VALID.DATA.DATASET == "Youtube_Videos":
            valid_loader = data_loader.YoutubeVideosLoader.YoutubeVideosLoader
        elif config.VALID.DATA.DATASET is None and not config.TEST.USE_LAST_EPOCH:
            raise ValueError("Validation dataset not specified despite USE_LAST_EPOCH set to False!")
        else:
            raise ValueError("Unsupported dataset! Currently supporting UBFC-rPPG, PURE, MMPD, \
                             SCAMPS, BP4D+ (Normal and BigSmall preprocessing), UBFC-PHYS and iBVP")
        
        # Create and initialize the valid dataloader given the correct toolbox mode,
        # a supported dataset name, and a valid dataset path
        if (config.VALID.DATA.DATASET and config.VALID.DATA.DATA_PATH and not config.TEST.USE_LAST_EPOCH):
            valid_data = valid_loader(
                name="valid",
                data_path=config.VALID.DATA.DATA_PATH,
                config_data=config.VALID.DATA,
                device=config.DEVICE)
            data_loader_dict["valid"] = DataLoader(
                dataset=valid_data,
                num_workers=config.VALID.NUM_WORKERS,
                batch_size=config.TRAIN.BATCH_SIZE,  # batch size for val is the same as train
                shuffle=False,
                worker_init_fn=seed_worker,
                generator=general_generator, 
                pin_memory=config.VALID.PIN_MEMORY,
                persistent_workers=config.VALID.PERSISTENT_WORKERS
            )
        else:
            data_loader_dict['valid'] = None

    if config.TOOLBOX_MODE == "train_and_test" or config.TOOLBOX_MODE == "only_test":
        # test_loader
        if config.TEST.DATA.DATASET == "UBFC-rPPG":
            test_loader = data_loader.UBFCrPPGLoader.UBFCrPPGLoader
        elif config.TEST.DATA.DATASET == "PURE":
            test_loader = data_loader.PURELoader.PURELoader
        elif config.TEST.DATA.DATASET == "SCAMPS":
            test_loader = data_loader.SCAMPSLoader.SCAMPSLoader
        elif config.TEST.DATA.DATASET == "MMPD":
            test_loader = data_loader.MMPDLoader.MMPDLoader
        elif config.TEST.DATA.DATASET == "BP4DPlus":
            test_loader = data_loader.BP4DPlusLoader.BP4DPlusLoader
        elif config.TEST.DATA.DATASET == "BP4DPlusBigSmall":
            test_loader = data_loader.BP4DPlusBigSmallLoader.BP4DPlusBigSmallLoader
        elif config.TEST.DATA.DATASET == "UBFC-PHYS":
            test_loader = data_loader.UBFCPHYSLoader.UBFCPHYSLoader
        elif config.TEST.DATA.DATASET == "iBVP":
            test_loader = data_loader.iBVPLoader.iBVPLoader
        elif config.TEST.DATA.DATASET == "VitalVideos":
            test_loader = data_loader.VitalVideosLoader.VitalVideosLoader
        elif config.TEST.DATA.DATASET == "VitalVideos_and_UBFC":
            test_loader = data_loader.VitalVideosPlusUBFCLoader.VitalVideosAndUBFCLoader
        elif config.TEST.DATA.DATASET == "Own_Videos":
            test_loader = data_loader.OwnVideosLoader.OwnVideosLoader
        elif config.TEST.DATA.DATASET == "Emergency_Videos":
            test_loader = data_loader.EmergencyVideosLoader.EmergencyVideosLoader
        elif config.TEST.DATA.DATASET == "Youtube_Videos":
            test_loader = data_loader.YoutubeVideosLoader.YoutubeVideosLoader
        else:
            raise ValueError("Unsupported dataset! Currently supporting UBFC-rPPG, PURE, MMPD, \
                             SCAMPS, BP4D+ (Normal and BigSmall preprocessing), UBFC-PHYS and iBVP.")
        
        if config.TOOLBOX_MODE == "train_and_test" and config.TEST.USE_LAST_EPOCH:
            print("Testing uses last epoch, validation dataset is not required.", end='\n\n')   

        # Create and initialize the test dataloader given the correct toolbox mode,
        # a supported dataset name, and a valid dataset path
        if config.TEST.DATA.DATASET and config.TEST.DATA.DATA_PATH:
            test_data = test_loader(
                name="test",
                data_path=config.TEST.DATA.DATA_PATH,
                config_data=config.TEST.DATA,
                device=config.DEVICE)
            data_loader_dict["test"] = DataLoader(
                dataset=test_data,
                num_workers=config.TEST.NUM_WORKERS,
                batch_size=config.INFERENCE.BATCH_SIZE,
                shuffle=False,
                worker_init_fn=seed_worker,
                generator=general_generator, 
                pin_memory=config.TEST.PIN_MEMORY,
                persistent_workers=config.TEST.PERSISTENT_WORKERS
            )
        else:
            data_loader_dict['test'] = None

    elif config.TOOLBOX_MODE == "unsupervised_method":
        # unsupervised method dataloader
        if config.UNSUPERVISED.DATA.DATASET == "UBFC-rPPG":
            unsupervised_loader = data_loader.UBFCrPPGLoader.UBFCrPPGLoader
        elif config.UNSUPERVISED.DATA.DATASET == "PURE":
            unsupervised_loader = data_loader.PURELoader.PURELoader
        elif config.UNSUPERVISED.DATA.DATASET == "SCAMPS":
            unsupervised_loader = data_loader.SCAMPSLoader.SCAMPSLoader
        elif config.UNSUPERVISED.DATA.DATASET == "MMPD":
            unsupervised_loader = data_loader.MMPDLoader.MMPDLoader
        elif config.UNSUPERVISED.DATA.DATASET == "BP4DPlus":
            unsupervised_loader = data_loader.BP4DPlusLoader.BP4DPlusLoader
        elif config.UNSUPERVISED.DATA.DATASET == "UBFC-PHYS":
            unsupervised_loader = data_loader.UBFCPHYSLoader.UBFCPHYSLoader
        elif config.UNSUPERVISED.DATA.DATASET == "iBVP":
            unsupervised_loader = data_loader.iBVPLoader.iBVPLoader
        elif config.UNSUPERVISED.DATA.DATASET == "VitalVideos":
            unsupervised_loader = data_loader.VitalVideosLoader.VitalVideosLoader
        elif config.UNSUPERVISED.DATA.DATASET == "VitalVideos_and_UBFC":
            unsupervised_loader = data_loader.VitalVideosPlusUBFCLoader.VitalVideosAndUBFCLoader
        elif config.UNSUPERVISED.DATA.DATASET == "Own_Videos":
            unsupervised_loader = data_loader.OwnVideosLoader.OwnVideosLoader
        elif config.UNSUPERVISED.DATA.DATASET == "Emergency_Videos":
            unsupervised_loader = data_loader.EmergencyVideosLoader.EmergencyVideosLoader
        elif config.UNSUPERVISED.DATA.DATASET == "Youtube_Videos":
            unsupervised_loader = data_loader.YoutubeVideosLoader.YoutubeVideosLoader
        else:
            raise ValueError("Unsupported dataset! Currently supporting UBFC-rPPG, PURE, MMPD, \
                             SCAMPS, BP4D+, UBFC-PHYS and iBVP.")
        
        unsupervised_data = unsupervised_loader(
            name="unsupervised",
            data_path=config.UNSUPERVISED.DATA.DATA_PATH,
            config_data=config.UNSUPERVISED.DATA,
            device=config.DEVICE)
        data_loader_dict["unsupervised"] = DataLoader(
            dataset=unsupervised_data,
            num_workers=4,
            batch_size=1,
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=general_generator, 
            pin_memory=True,
            persistent_workers=True
        )

    else:
        raise ValueError("Unsupported toolbox_mode! Currently support train_and_test or only_test or unsupervised_method.")
    
    torch.autograd.set_detect_anomaly(True)
    #activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    #with profile(activities=activities) as prof:
    if config.TOOLBOX_MODE == "train_and_test":
        train_and_test(config, data_loader_dict)
    elif config.TOOLBOX_MODE == "only_test":
        test(config, data_loader_dict)
    elif config.TOOLBOX_MODE == "unsupervised_method":
        unsupervised_method_inference(config, data_loader_dict)
    else:
        print("TOOLBOX_MODE only support train_and_test or only_test !", end='\n\n')


    #prof.export_chrome_trace("trace.json")
