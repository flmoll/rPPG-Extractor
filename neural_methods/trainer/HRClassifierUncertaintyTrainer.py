
import os
import shutil

import numpy as np
import torch
import torch.optim as optim
from neural_methods.trainer.BaseTrainer import BaseTrainer
from tqdm import tqdm

from neural_methods.model.HRClassifierUncertainty import HRClassifierUncertainty
from neural_methods.loss.HRClassifierUncertaintyLoss import HRClassifierUncertaintyLoss


class HRClassifierUncertaintyTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.base_len = self.num_of_gpu
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0

        if config.TOOLBOX_MODE == "train_and_test":
            
            self.model = HRClassifierUncertainty(
                in_size=[2, config.TRAIN.DATA.PPG_INFERENCE_MODEL.PHYSNET.FRAME_NUM]
            ).to(self.device)  # [3, T, 128,128]

            self.num_train_batches = len(data_loader["train"])
            self.loss_model = HRClassifierUncertaintyLoss()
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=config.TRAIN.LR)
            # See more details on the OneCycleLR scheduler here: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)
        elif config.TOOLBOX_MODE == "only_test":
            self.model = HRClassifierUncertainty(
                in_size=[2, config.TEST.DATA.PPG_INFERENCE_MODEL.PHYSNET.FRAME_NUM]
            ).to(self.device)  # [3, T, 128,128]
        else:
            raise ValueError("PhysNet trainer initialized in incorrect toolbox mode!")

    def train(self, data_loader):
        """Training routine for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        mean_training_losses = []
        mean_valid_losses = []
        lrs = []
        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            running_loss = 0.0
            train_loss = []
            self.model.train()
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                data_dict = batch[0]
                rPPG, uncertainty, gt_hr = data_dict["rppg"].to(self.device), data_dict["uncertainty"].to(self.device), data_dict["gt_hr"].to(self.device)
                label = data_dict["label"].to(self.device)
                gt_hr = gt_hr / 240     # We normalize the heart rate such that the network does not need to output very high values. 240 is chosen arbitrarily
                rPPG = (rPPG - torch.mean(rPPG, dim=-1, keepdim=True)) / torch.std(rPPG, dim=-1, keepdim=True)  # Normalize rPPG but keep uncertainty as it is
                
                model_input = torch.stack([rPPG, uncertainty], dim=1)  # Stack rPPG and uncertainty along the channel dimension
                model_input = model_input.detach()
                hr_pred, hr_uncertainty = self.model(model_input)

                loss = self.loss_model(hr_pred, hr_uncertainty, gt_hr)
                loss.backward()
                running_loss += loss.item()
                if idx % 100 == 99:  # print every 100 mini-batches
                    print(
                        f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                train_loss.append(loss.item())

                # Append the current learning rate to the list
                lrs.append(self.scheduler.get_last_lr())

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                tbar.set_postfix(loss=loss.item())

            # Append the mean training loss for the epoch
            mean_training_losses.append(np.mean(train_loss))

            self.save_model(epoch)
            if not self.config.TEST.USE_LAST_EPOCH: 
                valid_loss = self.valid(data_loader)
                mean_valid_losses.append(valid_loss)
                print('validation loss: ', valid_loss)
                if self.min_valid_loss is None:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
                elif (valid_loss < self.min_valid_loss):
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
                    
            
        if not self.config.TEST.USE_LAST_EPOCH: 
            print("best trained epoch: {}, min_val_loss: {}".format(
                self.best_epoch, self.min_valid_loss))
        if self.config.TRAIN.PLOT_LOSSES_AND_LR:
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, self.config)

        shutil.copyfile(
            os.path.join(self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth'),
            os.path.join(self.model_dir, self.model_file_name + '_best.pth')
        )
        
    def valid(self, data_loader):
        """ Runs the model on valid sets."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print('')
        print(" ====Validing===")
        valid_loss = []
        abs_errors_hr = []
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")

                data_dict = valid_batch[0]
                rPPG, uncertainty, gt_hr = data_dict["rppg"].to(self.device), data_dict["uncertainty"].to(self.device), data_dict["gt_hr"].to(self.device)
                gt_hr = gt_hr / 240

                rPPG = (rPPG - torch.mean(rPPG, dim=-1, keepdim=True)) / torch.std(rPPG, dim=-1, keepdim=True)  # Normalize rPPG
                
                model_input = torch.stack([rPPG, uncertainty], dim=1)  # Stack rPPG and uncertainty along the channel dimension
                model_input = model_input.detach()
                hr_pred, hr_uncertainty = self.model(model_input)

                abs_errors_hr.extend(torch.abs(240 * (hr_pred - gt_hr)).cpu().numpy().tolist())
                loss_ecg = self.loss_model(hr_pred, hr_uncertainty, gt_hr)
                valid_loss.append(loss_ecg.item())
                valid_step += 1
                vbar.set_postfix(loss=loss_ecg.item())
            valid_loss = np.asarray(valid_loss)
            print("Absolute errors in heart rate: ", np.mean(abs_errors_hr))
        return np.mean(valid_loss)

    def test(self, data_loader):
        """ Runs the model on test sets."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        
        print('')
        print("===Testing===")
        predictions = dict()
        uncertainties = dict()
        heart_rates_dict = dict()
        heart_rates_uncertainty_dict = dict()
        heart_rates_labels = dict()
        labels = dict()

        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")
            print(self.config.INFERENCE.MODEL_PATH)
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(
                self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                self.model.load_state_dict(torch.load(last_epoch_model_path))
            else:
                best_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                self.model.load_state_dict(torch.load(best_model_path))

        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        print("Running model evaluation on the testing dataset!")
        with torch.no_grad():
            abs_errors_hr = []
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80)):
                data_dict = test_batch[0]
                rPPG, uncertainty, gt_hr, bvp_gt = data_dict["rppg"].to(self.device), data_dict["uncertainty"].to(self.device), data_dict["gt_hr"].to(self.device), data_dict["label"].to(self.device)
                rPPG = (rPPG - torch.mean(rPPG, dim=-1, keepdim=True)) / torch.std(rPPG, dim=-1, keepdim=True)  # Normalize rPPG
                label = (bvp_gt - torch.mean(bvp_gt, dim=-1, keepdim=True)) / torch.std(bvp_gt, dim=-1, keepdim=True)  # Normalize BVP
                gt_hr = gt_hr / 240

                batch_size = rPPG.shape[0]

                model_input = torch.stack([rPPG, uncertainty], dim=1)  # Stack rPPG and uncertainty along the channel dimension
                model_input = model_input.detach()
                heart_rates, heart_rates_uncertainty = self.model(model_input)

                if rPPG.shape[-1] > label.shape[-1]:
                    rPPG = rPPG[:, :label.shape[-1]]

                pred_ppg_test = rPPG
                pred_uncertainty_test = uncertainty
                abs_errors_hr.extend(torch.abs(240 * (heart_rates - gt_hr)).cpu().numpy().tolist())

                if self.config.TEST.OUTPUT_SAVE_DIR:
                    label = label.cpu()
                    pred_ppg_test = pred_ppg_test.cpu()
                    pred_uncertainty_test = pred_uncertainty_test.cpu()
                    heart_rates = heart_rates.cpu()
                    heart_rates_uncertainty = heart_rates_uncertainty.cpu()
                    gt_hr = gt_hr.cpu()

                for idx in range(batch_size):
                    subj_index = test_batch[1][idx]
                    sort_index = int(test_batch[2][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        uncertainties[subj_index] = dict()
                        heart_rates_dict[subj_index] = dict()
                        heart_rates_uncertainty_dict[subj_index] = dict()
                        heart_rates_labels[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx]
                    uncertainties[subj_index][sort_index] = pred_uncertainty_test[idx]
                    heart_rates_dict[subj_index][sort_index] = heart_rates[idx]
                    heart_rates_uncertainty_dict[subj_index][sort_index] = heart_rates_uncertainty[idx]
                    heart_rates_labels[subj_index][sort_index] = gt_hr[idx]
                    labels[subj_index][sort_index] = label[idx]

        print("Absolute errors in heart rate: ", np.mean(abs_errors_hr))
        if self.config.TEST.OUTPUT_SAVE_DIR: # saving test outputs
            self.save_test_outputs(predictions, labels, self.config, uncertainties=uncertainties, heart_rates=heart_rates_dict, heart_rates_uncertainty=heart_rates_uncertainty_dict, heart_rates_labels=heart_rates_labels)

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)
