import cv2
import matplotlib
from matplotlib import pyplot as plt
import torch
matplotlib.use('Agg')

import base64
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import numpy as np
import zstandard as zstd
import lz4.frame
import blosc

from neural_methods.model.EfficientPhys import EfficientPhys as Efficientphys
from neural_methods.model.DeepPhys import DeepPhys as Deepphys
from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX as Physnet
from neural_methods.model.RhythmFormer import RhythmFormer as Rhythmformer
from neural_methods.model.PhysFormer import ViT_ST_ST_Compact3_TDC_gra_sharp as Physformer
from neural_methods.model.TS_CAN import TSCAN as Tscan
from neural_methods.model.PhysNetUncertainty import PhysNet_Uncertainty
from neural_methods.model.PhysNetQuantile import PhysNet_Quantile
from neural_methods.model.HRClassifierUncertainty import HRClassifierUncertainty
from neural_methods.model.HRClassifierQuantile import HRClassifierQuantile

import http_utils as utils

import sys
import scipy

from evaluation.utils import conformal_prediction, calculate_conformalized_intervals, get_quantile
from evaluation.utils import read_from_pickle, get_interval_predictions
from evaluation.heart_rate_filter import HeartRateFilter, AutocorrelationFilter
from evaluation.utils import postprocess_rppg


# replace these with your pretrained models
efficientphys_pretrained_path = "runs/DeepStab/train/VitalVideos_and_UBFC_SizeW72_SizeH72_ClipLength160_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendCORRYU_Large_boxTrue_Large_size1.5_Dyamic_DetTrue_det_len1_Median_face_boxFalse_DNone_amp10.0/PreTrainedModels/VitalLens_EfficientPhys_best.pth"
physnet_pretrained_path = "runs/DeepStab/train/VitalVideos_and_UBFC_SizeW72_SizeH72_ClipLength160_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendCORRYU_Large_boxTrue_Large_size1.5_Dyamic_DetTrue_det_len1_Median_face_boxFalse_DNone_amp10.0/PreTrainedModels/VitalLens_Physnet_best.pth"
tscan_pretrained_path = "runs/DeepStab/train/VitalVideos_and_UBFC_SizeW72_SizeH72_ClipLength160_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendCORRYU_Large_boxTrue_Large_size1.5_Dyamic_DetTrue_det_len1_Median_face_boxFalse_DNone_amp10.0/PreTrainedModels/VitalLens_TSCAN_best.pth"
physformer_pretrained_path = "runs/DeepStab/train/VitalVideos_and_UBFC_SizeW128_SizeH128_ClipLength160_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendCORRYU_Large_boxTrue_Large_size1.5_Dyamic_DetTrue_det_len1_Median_face_boxFalse_DNone_amp10.0/PreTrainedModels/VitalLens_PhysFormer_best.pth"
rythmformer_pretrained_path = "runs/DeepStab/train/VitalVideos_and_UBFC_SizeW128_SizeH128_ClipLength160_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendCORRYU_Large_boxTrue_Large_size1.5_Dyamic_DetTrue_det_len1_Median_face_boxFalse_DNone_amp10.0/PreTrainedModels/VitalLens_RythmFormer_best.pth"
loglikelihood_pretrained_path = "runs/Uncertainty/train/VitalVideos_and_UBFC_SizeW72_SizeH72_ClipLength160_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendY5F_Large_boxTrue_Large_size1.5_Dyamic_DetTrue_det_len30_Median_face_boxFalse_DNone_amp10.0/PreTrainedModels/VitalLens_Physnet_NegLogLikelihood_best.pth"
quantile_regression_pretrained_path = "runs/Uncertainty/train/VitalVideos_and_UBFC_SizeW72_SizeH72_ClipLength160_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendY5F_Large_boxTrue_Large_size1.5_Dyamic_DetTrue_det_len30_Median_face_boxFalse_DNone_amp10.0/PreTrainedModels/VitalLens_Physnet_Quantile_best.pth"

hr_dnn_loglikelihood_cumsum_pretrained_path = "runs/Uncertainty/train/VitalVideos_and_UBFC_SizeW72_SizeH72_ClipLength160_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendY5F_Large_boxTrue_Large_size1.5_Dyamic_DetTrue_det_len30_Median_face_boxFalse_DNone_amp10.0/PreTrainedModels/VitalLens_Physnet_HR_Classifier_NegLogLikelihood_cumsum_best.pth"
hr_dnn_quantile_cumsum_pretrained_path = "runs/Uncertainty/train/VitalVideos_and_UBFC_SizeW72_SizeH72_ClipLength160_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendY5F_Large_boxTrue_Large_size1.5_Dyamic_DetTrue_det_len30_Median_face_boxFalse_DNone_amp10.0/PreTrainedModels/VitalLens_Physnet_HR_Classifier_Quantile_cumsum_best.pth"

validation_outputs_loglikelihood = "runs/Uncertainty/validation/VitalVideos_and_UBFC_SizeW72_SizeH72_ClipLength160_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendY5F_Large_boxTrue_Large_size1.5_Dyamic_DetTrue_det_len30_Median_face_boxFalse_DNone_amp10.0/saved_test_outputs/VitalLens_Physnet_HR_Classifier_NegLogLikelihood_cumsum_best_VitalVideos_and_UBFC_outputs.pickle"
test_outputs_loglikelihood = "runs/Uncertainty/test/VitalVideos_and_UBFC_SizeW72_SizeH72_ClipLength160_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendY5F_Large_boxTrue_Large_size1.5_Dyamic_DetTrue_det_len30_Median_face_boxFalse_DNone_amp10.0/saved_test_outputs/VitalLens_Physnet_HR_Classifier_NegLogLikelihood_cumsum_best_VitalVideos_and_UBFC_outputs.pickle"

validation_outputs_quantile = "runs/Uncertainty/validation/VitalVideos_and_UBFC_SizeW72_SizeH72_ClipLength160_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendY5F_Large_boxTrue_Large_size1.5_Dyamic_DetTrue_det_len30_Median_face_boxFalse_DNone_amp10.0/saved_test_outputs/VitalLens_Physnet_HR_Classifier_Quantile_cumsum_best_VitalVideos_and_UBFC_outputs.pickle"
test_outputs_quantile = "runs/Uncertainty/test/VitalVideos_and_UBFC_SizeW72_SizeH72_ClipLength160_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendY5F_Large_boxTrue_Large_size1.5_Dyamic_DetTrue_det_len30_Median_face_boxFalse_DNone_amp10.0/saved_test_outputs/VitalLens_Physnet_HR_Classifier_Quantile_cumsum_best_VitalVideos_and_UBFC_outputs.pickle"




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
chunk_len = 160  # Length of the video chunk

# Load models
efficientphys = Efficientphys(frame_depth=10, img_size=72).to(device)
efficientphys = torch.nn.DataParallel(efficientphys, device_ids=[0, 1])
efficientphys.load_state_dict(torch.load(efficientphys_pretrained_path, map_location=device))
efficientphys.eval()

physnet = Physnet(frames=chunk_len)
physnet.load_state_dict(torch.load(physnet_pretrained_path, map_location=device))
physnet = physnet.to(device)
physnet.eval()

tscan = Tscan(frame_depth=chunk_len, img_size=72)
tscan = torch.nn.DataParallel(tscan, device_ids=[0, 1])
tscan.load_state_dict(torch.load(tscan_pretrained_path, map_location=device))
tscan.eval()

rythmformer = Rhythmformer()
rythmformer = torch.nn.DataParallel(rythmformer, device_ids=[0, 1])
rythmformer.load_state_dict(torch.load(rythmformer_pretrained_path, map_location=device))
rythmformer.eval()

physformer = Physformer(image_size=(chunk_len, 128, 128), 
                patches=(4,) * 3, dim=96, ff_dim=144, num_heads=4, num_layers=12, 
                dropout_rate=0.1, theta=0.7)
physformer = torch.nn.DataParallel(physformer, device_ids=[0, 1])
physformer.load_state_dict(torch.load(physformer_pretrained_path, map_location=device))
physformer.eval()

loglikelihood_model = PhysNet_Uncertainty(frames=chunk_len)
loglikelihood_model.load_state_dict(torch.load(loglikelihood_pretrained_path, map_location=device))
loglikelihood_model = loglikelihood_model.to(device)
loglikelihood_model.eval()

quantile_regression_model = PhysNet_Quantile(frames=chunk_len)
quantile_regression_model.load_state_dict(torch.load(quantile_regression_pretrained_path, map_location=device))
quantile_regression_model = quantile_regression_model.to(device)
quantile_regression_model.eval()

hr_dnn_loglikelihood_model = HRClassifierUncertainty()
hr_dnn_loglikelihood_model.load_state_dict(torch.load(hr_dnn_loglikelihood_cumsum_pretrained_path, map_location=device))
hr_dnn_loglikelihood_model = hr_dnn_loglikelihood_model.to(device)
hr_dnn_loglikelihood_model.eval()

hr_dnn_quantile_model = HRClassifierQuantile()
hr_dnn_quantile_model.load_state_dict(torch.load(hr_dnn_quantile_cumsum_pretrained_path, map_location=device))
hr_dnn_quantile_model = hr_dnn_quantile_model.to(device)
hr_dnn_quantile_model.eval()


uncertainties_valid_loglikelihood, preds_valid_loglikelihood, labels_valid_loglikelihood = read_from_pickle(validation_outputs_loglikelihood)
uncertainties_valid_quantile, preds_valid_quantile, labels_valid_quantile = read_from_pickle(validation_outputs_quantile)

uncertainties_test_loglikelihood, preds_test_loglikelihood, labels_test_loglikelihood = read_from_pickle(test_outputs_loglikelihood)
uncertainties_test_quantile, preds_test_quantile, labels_test_quantile = read_from_pickle(test_outputs_quantile)


hr_uncertainties_valid_loglikelihood, hr_preds_valid_loglikelihood, hr_labels_valid_loglikelihood = read_from_pickle(validation_outputs_loglikelihood, data_to_load="heart_rate")
hr_uncertainties_valid_quantile, hr_preds_valid_quantile, hr_labels_valid_quantile = read_from_pickle(validation_outputs_quantile, data_to_load="heart_rate")

hr_uncertainties_test_loglikelihood, hr_preds_test_loglikelihood, hr_labels_test_loglikelihood = read_from_pickle(test_outputs_loglikelihood, data_to_load="heart_rate")
hr_uncertainties_test_quantile, hr_preds_test_quantile, hr_labels_test_quantile = read_from_pickle(test_outputs_quantile, data_to_load="heart_rate")


def decompress(data: bytes, format: str, dtype: np.dtype, shape: tuple) -> bytes:
    if format == "zstd":
        raw = zstd.decompress(data)
    elif format == "lz4":
        raw = lz4.frame.decompress(data)
    elif format == "blosc":
        raw = blosc.decompress(data)
    elif format == "raw":
        raw = data
    else:
        raise ValueError(f"Unsupported format: {format}")

    raw = np.frombuffer(raw, dtype=dtype).reshape(shape)
    return np.array(raw, copy=True)

def get_model_format(model: str) -> str:
    if model.lower() in ["efficientphys", "rythmformer"]:
        return "standardized"
    else:
        return "diffNorm"
    
def get_format_pipeline_index(format: str) -> int:
    if format == "raw":
        return 0
    elif format == "facecropped":
        return 1
    elif format == "resized":
        return 2
    elif format == "standardized" or format == "diffNorm":
        return 3
    else:
        raise ValueError(f"Unsupported format: {format}")

def preprocess_video(video: np.ndarray, in_format: str, out_format: str, image_size: tuple) -> np.ndarray:
    
    in_format_index = get_format_pipeline_index(in_format)
    out_format_index = get_format_pipeline_index(out_format)

    if in_format_index <= 0 and out_format_index >= 1:
        # Resize if necessary
        print("Face cropping video data")
        video = utils.face_detection(video)

        
    if in_format_index <= 1 and out_format_index >= 2:
        # Resize if necessary
        print("Resizing video data")
        video = utils.resize_video(video, image_size=image_size)

        
    if in_format_index <= 2 and out_format_index >= 3:
        if out_format == "standardized":
            # Standardize the video
            print("Standardizing video data")
            video = utils.standardized_data(video)
        elif out_format == "diffNorm":
            # Apply diff normalization
            print("Applying diff normalization to video data")
            video = utils.diff_normalize_data(video)

    return video

def conformalize_predictions(preds: np.ndarray, uncertainties: np.ndarray, flavour: str, probability_in_interval: float = 0.9) -> tuple:

    interval_lower_valid, interval_upper_valid = get_interval_predictions(uncertainties_valid_quantile, preds_valid_quantile, flavour, probability_in_interval)
    interval_lower_query, interval_upper_query = get_interval_predictions(uncertainties, preds, flavour, probability_in_interval)

    quantile = get_quantile(interval_lower_valid, interval_upper_valid, labels_valid_loglikelihood, alpha=(1 - probability_in_interval))
    interval_lower_valid, intervals_upper_valid = calculate_conformalized_intervals(interval_lower_query, interval_upper_query, quantile)
    
    print(np.mean(interval_lower_valid - interval_lower_valid))

    rppg = (interval_lower_valid + intervals_upper_valid) / 2  # Average of lower and upper bounds
    uncertainties = (intervals_upper_valid - interval_lower_valid) / 2  # Half the width of the interval
    
    return rppg, uncertainties

def conformalize_hr_predictions(preds: np.ndarray, uncertainties: np.ndarray, flavour: str, probability_in_interval: float = 0.9) -> tuple:
    
    #preds = 60 / preds  # Convert to BPM
    #uncertainties = 60 / uncertainties  # Convert to BPM

    interval_lower_valid, interval_upper_valid = get_interval_predictions(hr_uncertainties_valid_quantile, hr_preds_valid_quantile, flavour, probability_in_interval)
    interval_lower_query, interval_upper_query = get_interval_predictions(uncertainties, preds, flavour, probability_in_interval)
    
    quantile = get_quantile(interval_lower_valid, interval_upper_valid, hr_labels_valid_loglikelihood, alpha=(1 - probability_in_interval))
    interval_lower_valid, intervals_upper_valid = calculate_conformalized_intervals(interval_lower_query, interval_upper_query, quantile)
    
    hr = (interval_lower_valid + intervals_upper_valid) / 2  # Average of lower and upper bounds
    uncertainty = (intervals_upper_valid - interval_lower_valid) / 2  # Half the width of the interval
    return hr, uncertainty

def normalize_tensor(tensor: torch.Tensor) -> np.ndarray:
    tensor = (tensor - torch.mean(tensor)) / torch.std(tensor)
    return tensor

def execute_model(model: str, video: np.ndarray, probability_in_interval: float = 0.9) -> np.ndarray:

    print(f"Executing model: {model} with video shape: {video.shape}")

    video = utils.format_video_tensor_for_network(video, model_name=model)
    video = video.to(device)

    if model.lower() == "efficientphys":
        N, D, C, H, W = video.shape
        video = video.view(N * D, C, H, W)
        video = video[:(N * D) // 10 * 10]
        # Add one more frame for EfficientPhys since it does torch.diff for the input
        last_frame = torch.unsqueeze(video[-1, :, :, :], 0).repeat(1, 1, 1, 1)
        video = torch.cat((video, last_frame), 0)
        return efficientphys(video), None
    elif model.lower() == "physnet":
        rPPG, x_visual, x_visual3232, x_visual1616 = physnet(video)
        return normalize_tensor(rPPG).cpu().detach().numpy(), None
    elif model.lower() == "loglikelihood":
        rPPG, uncertainty = loglikelihood_model(video)
        rPPG = normalize_tensor(rPPG).cpu().detach().numpy()
        uncertainty = uncertainty.cpu().detach().numpy()
        return conformalize_predictions(rPPG, uncertainty, "neg_log_likelihood", probability_in_interval)
    elif model.lower() == "quantile":
        lower, upper = quantile_regression_model(video)
        lower = normalize_tensor(lower).cpu().detach().numpy()
        upper = normalize_tensor(upper).cpu().detach().numpy()
        rppg = (lower + upper) / 2  # Average of lower and upper bounds
        uncertainties = (upper - lower) / 2
        return conformalize_predictions(rppg, uncertainties, "quantile_regression", probability_in_interval)
    elif model.lower() == "tscan":
        return tscan(video), None
    elif model.lower() == "physformer":
        return physformer(video), None
    elif model.lower() == "rythmformer":
        return rythmformer(video), None
    else:
        raise ValueError(f"Unsupported model: {model}")

def heart_rate_from_rppg(rppg: np.ndarray, uncertainties: np.ndarray, hr_extractor, fps: int = 30, probability_in_interval: float = 0.9) -> float:

    if hr_extractor in ["dnn_quantile", "dnn_loglikelihood"]:
        stacked_vector = np.stack([rppg, uncertainties], axis=1)
        stacked_vector = torch.tensor(stacked_vector, device=device, dtype=torch.float32)

    if hr_extractor == "fft":
        fft = np.fft.fft(rppg)
        freqs = np.fft.fftfreq(rppg.shape[-1], d=1/fps)
        argmax = np.argmax(np.abs(fft))
        # Find the peak frequency in the FFT
        peak_freq = freqs[argmax]
        # Convert frequency to heart rate in BPM
        hr = peak_freq * 60  # Convert to beats per minute (BPM)
    elif hr_extractor == "peaks":
        hr_all = []
        for i in range(rppg.shape[0]):
            peaks = scipy.signal.find_peaks(rppg[i, :])[0]
            if len(peaks) < 2:
                raise ValueError("Not enough peaks found to calculate heart rate")
            peak_intervals = np.diff(peaks) / fps  # Convert peak intervals to seconds
            if len(peak_intervals) == 0:
                raise ValueError("No peak intervals found to calculate heart rate")
            hr = 60 / np.mean(peak_intervals)  # Convert to beats per minute (BPM)
            hr_all.append(hr)
        hr = np.array(hr_all)
    elif hr_extractor == "autocorrelation":
        filter = AutocorrelationFilter(sampling_rate=fps)
        hr = np.array([filter.apply(rppg[i, :]) for i in range(rppg.shape[0])])
    elif hr_extractor == "dnn_quantile":
        lower, upper = hr_dnn_quantile_model(stacked_vector)
        lower = lower.cpu().detach().numpy()
        upper = upper.cpu().detach().numpy()
        hr = (lower + upper) / 2  # Average of lower and upper bounds
        uncertainties = (upper - lower) / 2  # Half the width of the interval
        return conformalize_hr_predictions(hr, uncertainties, "quantile_regression", probability_in_interval=probability_in_interval)
    elif hr_extractor == "dnn_loglikelihood":
        hr, uncertainty = hr_dnn_loglikelihood_model(stacked_vector)
        hr = hr.cpu().detach().numpy()
        uncertainty = uncertainty.cpu().detach().numpy()
        return conformalize_hr_predictions(hr, uncertainty, "neg_log_likelihood", probability_in_interval=probability_in_interval)
    elif hr_extractor == "none":
        return None, None
    else:
        raise ValueError(f"Unsupported heart rate extractor: {hr_extractor}")

    if uncertainties is not None:
        hr_uncertainty = np.mean(uncertainties)
    else:
        hr_uncertainty = None

    return hr, hr_uncertainty


def save_video(video: np.ndarray, output_path: str = "processed_video.mp4", fps: int = 30):

    # Save the processed video tensor as an mp4 file using OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width = video.shape[1], video.shape[2]
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(np.min(video), np.max(video), "Video min and max before normalization")

    video_normalized = (video - np.min(video)) / (np.max(video) - np.min(video)) * 255  # Normalize to [0, 255]

    for frame in video_normalized:
        # Convert from (C, H, W) to (H, W, C) and scale to uint8
        frame_img = np.clip(frame, 0, 255).astype(np.uint8)
        frame_img = cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
        out.write(frame_img)

    out.release()
    print(f"Saved processed video to {output_path}")

app = FastAPI()

@app.post("/upload_tensor")
async def upload_tensor(
    tensor: UploadFile = File(...),
    compression: str = Form(...),  # "zstd", "lz4", or "blosc"
    format: str = Form(...),  # "raw", "resized", "standardized", or "diffNorm"
    shape: str = Form(...),   # e.g., "100,64,64,3"
    dtype: str = Form(...),    # e.g., "uint8"
    endianness: str = Form(...),  # "little" or "big"
    model: str = Form(...),  # e.g., "physnet", "rythmformer", etc.
    postprocessor: str = Form(...),  # e.g., "none", "cumsum", "cumsum_detrend", "cumsum_detrend_butter"
    hr_extractor: str = Form(...),  # e.g. "fft", "peaks", "autocorrelation"
    confidence: float = Form(...),  # Confidence level for conformal prediction
    fps: int = Form(...)
):
    try:
        data = await tensor.read()
        shape_tuple = tuple(map(int, shape.split(",")))
        dtype_np = np.dtype(dtype)
        model = model.lower()
        required_format = get_model_format(model)
        required_image_size = utils.get_model_image_size(model)

        if endianness == "little":
            dtype_np = dtype_np.newbyteorder('<')
        elif endianness == "big":
            dtype_np = dtype_np.newbyteorder('>')
        else:
            raise ValueError(f"Unsupported endianness: {endianness}")
        
        print(f"Received tensor with shape: {shape_tuple}, dtype: {dtype_np}, compression: {compression}, format: {format}, model: {model}, required_format: {required_format}, required_image_size: {required_image_size}")

        video = decompress(data, 
                            compression, 
                            dtype_np, 
                            shape_tuple)
        
        if endianness != sys.byteorder:
            # Convert to the system's byte order if necessary
            video = video.astype(dtype_np.newbyteorder(sys.byteorder))

        save_video(video, output_path="original_video.mp4")

        video = preprocess_video(video, 
                                    in_format=format, 
                                    out_format=required_format, 
                                    image_size=required_image_size)
        
        save_video(video, output_path="processed_video.mp4")

        rppg, uncertainty = execute_model(model, video, probability_in_interval=confidence)
        rppg = postprocess_rppg(rppg, postprocessor)

        plt.plot(rppg[0, :], label='RPPG Signal')

        if uncertainty is not None:
            plt.fill_between(np.arange(rppg.shape[1]), 
                            rppg[0, :] - uncertainty[0, :], 
                            rppg[0, :] + uncertainty[0, :], 
                            color='gray', alpha=0.5, label='Uncertainty Interval')

        plt.title(f"RPPG Signal from {model.upper()}")
        plt.savefig("rppg_signal.png")
        plt.close()

        shape_to_send = rppg.shape

        hr, hr_uncertainty = heart_rate_from_rppg(rppg, uncertainty, hr_extractor, fps=fps, probability_in_interval=confidence)

        if isinstance(hr, np.ndarray):
            hr = hr.squeeze().tolist()
        if isinstance(hr_uncertainty, np.ndarray):
            hr_uncertainty = hr_uncertainty.squeeze().tolist()

        # Encode to base64
        if rppg.ndim == 3:
            base64_rppg = utils.encode_tensor_to_base64(rppg[:, 0, :])
            base64_uncertainty = utils.encode_tensor_to_base64(rppg[:, 1, :])
            shape_to_send = (rppg.shape[0], rppg.shape[2])
        else:
            base64_rppg = utils.encode_tensor_to_base64(rppg)
            base64_uncertainty = None

        return JSONResponse({
            "rppg": base64_rppg,
            "rppg_uncertainty": base64_uncertainty,
            "hr": hr,
            "hr_uncertainty": hr_uncertainty,
            "shape": shape_to_send,
            "dtype": str(rppg.dtype),
            "endianness": sys.byteorder,
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})