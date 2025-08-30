from abc import ABC, abstractmethod
import random

import cv2
from matplotlib import pyplot as plt
import pandas as pd
import scipy
import torch
import numpy as np
from torchvision.transforms import functional as F

import sys
path_before = sys.path
sys.path.append("/workspaces/src/")

from dataset.artificial_destabilisation.destabilisers import H264CompressionDestabiliser
import os


def diff_normalize_data(data):
    """Calculate discrete difference in video data along the time-axis and nornamize by its standard deviation."""
    n, h, w, c = data.shape
    diffnormalized_len = n - 1
    diffnormalized_data = np.zeros((diffnormalized_len, h, w, c), dtype=np.float32)
    diffnormalized_data_padding = np.zeros((1, h, w, c), dtype=np.float32)
    for j in range(diffnormalized_len):
        diffnormalized_data[j, :, :, :] = (data[j + 1, :, :, :] - data[j, :, :, :]) / (
                data[j + 1, :, :, :] + data[j, :, :, :] + 1e-7)
    diffnormalized_data = diffnormalized_data / np.std(diffnormalized_data)
    diffnormalized_data = np.append(diffnormalized_data, diffnormalized_data_padding, axis=0)
    diffnormalized_data[np.isnan(diffnormalized_data)] = 0
    return diffnormalized_data

class VideoTransform(ABC):
    """
    Abstract base class for video transformations.
    All video transformation classes should inherit from this class
    and implement the `apply` method.
    """

    @abstractmethod
    def apply(self, video_tensor, bvp_tensor):
        """
        Apply the transformation to the given video.

        Args:
            video: The input video data to be transformed. Format (T, W, H, C)

        Returns:
            The transformed video data.
        """
        pass

class VideoTransformFactory:
    """
    Factory class to create video transformations.
    """

    @staticmethod
    def get_function_from_name(function_name):
        """
        Get the function from the name.

        Args:
            function_name (str): The name of the function.

        Returns:
            The function corresponding to the name.
        """
        if function_name == "brightness":
            return F.adjust_brightness
        elif function_name == "contrast":
            return F.adjust_contrast
        elif function_name == "color":
            return F.adjust_color
        elif function_name == "sharpness":
            return F.adjust_sharpness
        elif function_name == "hue":
            return F.adjust_hue
        elif function_name == "saturation":
            return F.adjust_saturation
        else:
            raise ValueError(f"Unknown transform: {function_name}")

    @staticmethod
    def create_transform(parameters, device="cuda:0", type_to_be_used="DiffNormalized"):
        params = parameters.split(" ")
        transform_type = params[0]

        if transform_type == "Torchvision":
            function_name = params[1]
            function = VideoTransformFactory.get_function_from_name(function_name)
            min = float(params[2]) if len(params) > 2 else 0.5
            max = float(params[3]) if len(params) > 3 else 1.5
            return TorchvisionVideoTransform(function, min, max, device, type_to_be_used)
        elif transform_type == "RandomAffine":
            degrees = float(params[1]) if len(params) > 1 else 30.0
            translate = (float(params[2]), float(params[3])) if len(params) > 3 else (0.1, 0.1)
            scale = (float(params[4]), float(params[5])) if len(params) > 5 else (0.8, 1.2)
            shear = (float(params[6]), float(params[7])) if len(params) > 7 else (0, 0)
            return RandomAffineVideoTransform(degrees, translate, scale, shear, type_to_be_used)
        elif transform_type == "RandomSampling":
            min_interp_factor = float(params[1]) if len(params) > 1 else 0.5
            max_interp_factor = float(params[2]) if len(params) > 2 else 2.0
            min_hr = int(params[3]) if len(params) > 3 else 45
            max_hr = int(params[4]) if len(params) > 4 else 120
            fs = int(params[5]) if len(params) > 5 else 30
            crop_length = int(params[6]) if len(params) > 6 else 160
            return RandomSamplingVideoTransform(min_interp_factor, max_interp_factor, min_hr, max_hr, fs, crop_length, type_to_be_used)
        elif transform_type == "RandomFrameShuffle":
            shuffle_rate = float(params[1]) if len(params) > 1 else 0.05
            return RandomFrameShuffleTransform(shuffle_rate, type_to_be_used)
        elif transform_type == "RandomBlack":
            black_size = tuple(map(int, params[1:])) if len(params) > 1 else (20, 20, 10)
            return RandomBlackTransform(black_size, type_to_be_used)
        elif transform_type == "RandomVideoCompression":
            compression_factor_min = float(params[1]) if len(params) > 1 else 0.0
            compression_factor_max = float(params[2]) if len(params) > 2 else 30.0
            recreate_diff_frames = params[3].lower() == 'true' if len(params) > 3 else True
            vcodec = params[4] if len(params) > 4 else 'libx264'
            preset = params[5] if len(params) > 5 else 'slow'
            use_constant_qp = params[6].lower() == 'true' if len(params) > 6 else False
            return RandomVideoCompressionTransform(compression_factor_min, compression_factor_max, recreate_diff_frames, vcodec, preset, use_constant_qp, type_to_be_used)
        elif transform_type == "RandomVideoGaussianMask":
            mask_size_x_min = int(params[1]) if len(params) > 1 else 20
            mask_size_x_max = int(params[2]) if len(params) > 2 else 20
            mask_size_y_min = int(params[3]) if len(params) > 3 else 20
            mask_size_y_max = int(params[4]) if len(params) > 4 else 20
            type_to_be_used = params[5] if len(params) > 5 else "DiffNormalized"
            return RandomVideoGaussianMask(mask_std_x=(mask_size_x_min, mask_size_x_max), mask_std_y=(mask_size_y_min, mask_size_y_max), type_to_be_used=type_to_be_used)
        elif transform_type == "FloatPrecisionDownsampling":
            num_float_bits = int(params[1]) if len(params) > 1 else 32
            return FloatPrecisionDownsamplingTransform(num_float_bits, type_to_be_used)
        elif transform_type == "FFT":
            mode = int(params[1]) if len(params) > 1 else 0
            log_scale = params[2].lower() == 'true' if len(params) > 2 else False
            return FFTVideoTransform(type_to_be_used, mode, log_scale)
        else:
            raise ValueError(f"Unknown transform type: {transform_type}")

class TorchvisionVideoTransform(VideoTransform):
    """
    Adjust the brightness of the video.
    """

    def __init__(self, adjust_function, min_value=0.5, max_value=1.5, device="cuda:0", type_to_be_used="DiffNormalized"):
        self.adjust_function = adjust_function
        self.min_value = min_value
        self.max_value = max_value
        self.device = device
        self.type_to_be_used = type_to_be_used

    def apply(self, video_tensor, bvp_tensor):
        brightness = random.uniform(self.min_value, self.max_value)

        T, W, H, C = video_tensor.shape
        video_tensor = video_tensor.permute(0, 3, 2, 1)  # (T, C, H, W)

        # Apply brightness adjustment
        if C == 6:
            video_tensor[:, 3:, :, :] = self.adjust_function(video_tensor[:, 3:, :, :], brightness)
            video_tensor[:, :3, :, :] = self.adjust_function(video_tensor[:, :3, :, :], brightness)
        else:
            video_tensor = self.adjust_function(video_tensor, brightness)

        # Convert back to original format
        video_tensor = video_tensor.permute(0, 3, 2, 1)  # (T, W, H, C)

        info = dict()
        info["factor"] = brightness

        return video_tensor, bvp_tensor, info
    
class RandomVideoGaussianMask(VideoTransform):
    """
    Randomly apply a Gaussian mask to the video frames.
    """

    def __init__(self, mask_std_x=(20, 20), mask_std_y=(20, 20), type_to_be_used="DiffNormalized"):
        self.mask_std_x = mask_std_x
        self.mask_std_y = mask_std_y
        self.type_to_be_used = type_to_be_used

    def apply(self, video_tensor, bvp_tensor):
        T, W, H, C = video_tensor.shape

        std_x = random.uniform(self.mask_std_x[0], self.mask_std_x[1])
        std_y = random.uniform(self.mask_std_y[0], self.mask_std_y[1])

        # Create a Gaussian mask
        x = np.linspace(-W // 2, W // 2, W)
        y = np.linspace(-H // 2, H // 2, H)
        x, y = np.meshgrid(x, y)
        mask = np.exp(-(x**2 / (2 * std_x**2) + y**2 / (2 * std_y**2)))
        mask = mask.astype(np.float32)
        mask = torch.tensor(mask, dtype=torch.float32, device=video_tensor.device)

        #print("mask stats:", mask.min().item(), mask.max().item(), mask.mean().item(), mask.std().item())

        mask = mask.unsqueeze(0).unsqueeze(-1)  # Add batch and channel dimensions (1, H, W, 1)

        #print(f"Mask shape: {mask.shape}, std_x: {std_x}, std_y: {std_y}, video_tensor shape: {video_tensor.shape}")
        video_tensor = video_tensor * mask  # Apply the mask to all frames

        info = dict()
        info["mask_std_x"] = std_x
        info["mask_std_y"] = std_y

        return video_tensor, bvp_tensor, info
    
class RandomAffineVideoTransform(VideoTransform):
    """
    Randomly apply affine transformations to the video frames.
    """

    def __init__(self, degrees=30.0, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=(0, 0), type_to_be_used="DiffNormalized"):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.type_to_be_used = type_to_be_used

    def apply(self, video_tensor, bvp_tensor):
        T, W, H, C = video_tensor.shape

        # Randomly select affine parameters
        angle = random.uniform(-self.degrees, self.degrees)
        translate_x = random.uniform(-self.translate[0] * W, self.translate[0] * W)
        translate_y = random.uniform(-self.translate[1] * H, self.translate[1] * H)
        scale = random.uniform(self.scale[0], self.scale[1])
        shear = random.uniform(-self.shear[0], self.shear[0])
        #shear_y = random.uniform(-self.shear[1], self.shear[1])

        # Apply the affine transformation to each frame
        transformed_frames = []
        for frame in video_tensor:

            frame = frame.permute(2, 0, 1)  # (C, H, W)

            transformed_frame = F.affine(
                frame,
                angle=angle,
                translate=(translate_x, translate_y),
                scale=scale,
                shear=shear,
                interpolation=F.InterpolationMode.BILINEAR,
                fill=0  # Fill with black
            )

            transformed_frame = transformed_frame.permute(1, 2, 0)  # (H, W, C)

            transformed_frames.append(transformed_frame)

        transformed_frames = torch.stack(transformed_frames)
        
        info = dict()
        info["angle"] = angle
        info["translate"] = (translate_x, translate_y)
        info["scale"] = scale
        info["shear"] = shear

        return transformed_frames.float(), bvp_tensor, info

class RandomSamplingVideoTransform(VideoTransform):
    """
    Randomly interpolate the video frames.
    """

    def __init__(self, min_interp_factor=0.5, max_interp_factor=2.0, min_hr=45, max_hr=120, fs=30, crop_length=160, type_to_be_used="DiffNormalized"):
        """
        Initialize the RandomInterpolation transform.

        Args:
            num_frames (int): The number of frames to interpolate.
        """
        self.min_interp_factor = min_interp_factor
        self.max_interp_factor = max_interp_factor
        self.min_hr = min_hr
        self.max_hr = max_hr
        self.fs = fs
        self.crop_length = crop_length
        self.type_to_be_used = type_to_be_used

    def apply(self, video_tensor, bvp_tensor):

        label_ppg = (bvp_tensor - torch.mean(bvp_tensor)) / torch.std(bvp_tensor)
        peaks_gt = scipy.signal.find_peaks(label_ppg, height=None, distance=15, prominence=1.5)[0]
        peaks_gt = torch.tensor(peaks_gt, dtype=torch.float32, device=bvp_tensor.device)
        
        gt_hr = (self.fs / torch.mean(torch.diff(peaks_gt))) * 60
        min_factor = max(self.min_hr / gt_hr, self.min_interp_factor)
        max_factor = min(self.max_hr / gt_hr, self.max_interp_factor)

        # Randomly select interpolation factor
        interp_factor = random.uniform(min_factor, max_factor)

        num_frames = bvp_tensor.shape[0]
        num_interp_frames = int(num_frames * interp_factor)

        T, W, H, C = video_tensor.shape

        video_tensor = video_tensor.permute(3, 0, 2, 1)  # (C, T, H, W)
        video_tensor = video_tensor.unsqueeze(0)  # Add batch dimension (B, C, T, H, W)

        video_interp = torch.nn.functional.interpolate(
            video_tensor,
            size=(num_interp_frames, H, W),
            mode='trilinear',
            align_corners=True
        )

        bvp_tensor = torch.reshape(bvp_tensor, (1, 1, num_frames))  # Reshape to (B, C, T)

        bvp_tensor = torch.nn.functional.interpolate(
            bvp_tensor,
            size=num_interp_frames,
            mode='linear',
            align_corners=True
        )

        bvp_tensor = torch.reshape(bvp_tensor, (num_interp_frames, ))  # Reshape to (T_new)
        bvp_tensor = bvp_tensor[:self.crop_length]

        video_interp = video_interp.squeeze(0)
        video_interp = video_interp.permute(1, 3, 2, 0) # (T_new, H, W, C)
        video_interp = video_interp[:self.crop_length]

        info = dict()
        info["interp_factor"] = interp_factor

        # Use or return video_interp
        return video_interp, bvp_tensor, info


class RandomFrameShuffleTransform(VideoTransform):
    """
    Randomly shuffle the frames of the video.
    """

    def __init__(self, shuffle_rate=0.05, type_to_be_used="DiffNormalized"):
        self.shuffle_rate = shuffle_rate
        self.type_to_be_used = type_to_be_used

    def apply(self, video_tensor, bvp_tensor):
        T, W, H, C = video_tensor.shape

        info = dict()
        
        if random.random() < self.shuffle_rate:
            # Shuffle the frames
            indices = torch.randperm(T)
            video_tensor = video_tensor[indices]
            bvp_tensor = bvp_tensor[indices]
            info["shuffled"] = True
            return video_tensor, bvp_tensor, info
        else:
            # No shuffling
            info["shuffled"] = False
            return video_tensor, bvp_tensor, info
        
class RandomBlackTransform(VideoTransform):
    """
    Randomly crop the video frames.
    """

    def __init__(self, black_size=(20, 20, 10), type_to_be_used="DiffNormalized"):
        self.black_size = black_size
        self.type_to_be_used = type_to_be_used

    def apply(self, video_tensor, bvp_tensor):
        T, W, H, C = video_tensor.shape
        crop_h, crop_w, crop_d = self.black_size

        # Randomly select the top-left corner of the crop
        x_start = random.randint(0, W)
        y_start = random.randint(0, H)
        z_start = random.randint(0, T)
        x_end = random.randint(x_start, min(x_start + crop_w, W))
        y_end = random.randint(y_start, min(y_start + crop_h, H))
        z_end = random.randint(z_start, min(z_start + crop_d, T))

        info = dict()
        info["black_size"] = (x_start, y_start, z_start, x_end, y_end, z_end)

        video_tensor[z_start:z_end, x_start:x_end, y_start:y_end, :] = 0

        return video_tensor, bvp_tensor, info
    
class RandomVideoCompressionTransform(VideoTransform):
    """
    Randomly compress the video frames.
    """

    def __init__(self, compression_factor_min=0.0, compression_factor_max=30.0, recreate_diff_frames=True, vcodec='libx264', preset='slow', use_constant_qp=False, type_to_be_used="DiffNormalized"):
        self.compression_factor_min = compression_factor_min
        self.compression_factor_max = compression_factor_max
        self.recreate_diff_frames = recreate_diff_frames
        self.vcodec = vcodec
        self.preset = preset
        self.use_constant_qp = use_constant_qp
        self.type_to_be_used = type_to_be_used


    def apply(self, video_tensor, bvp_tensor):
        T, W, H, C = video_tensor.shape
        device = video_tensor.device

        # Randomly select the compression factor
        compression_factor = random.uniform(self.compression_factor_min, self.compression_factor_max)
        compression_factor = int(compression_factor)

        info = dict()
        info["compression_factor"] = compression_factor
        #print("Compression factor:", compression_factor)

        compressor = H264CompressionDestabiliser(crf=compression_factor, preset=self.preset, vcodec=self.vcodec, use_constant_qp=self.use_constant_qp)

        video_tensor = video_tensor.cpu().numpy()

        if C == 6:
            #video_tensor[:, :, :, 3:] = compressor.destabilise(video_tensor[:, :, :, 3:])

            if self.recreate_diff_frames:
                video_tensor[:, :, :, :3] = diff_normalize_data(video_tensor[:, :, :, 3:])
            else:
                video_tensor[:, :, :, :3] = compressor.destabilise(video_tensor[:, :, :, :3])
        else:
            video_tensor = compressor.destabilise(video_tensor)

        video_tensor = torch.from_numpy(video_tensor).float().to(device)

        return video_tensor, bvp_tensor, info
    

class FFTVideoTransform(VideoTransform):
    def __init__(self, type_to_be_used="DiffNormalized", mode=0, log_scale=False):
        self.type_to_be_used = type_to_be_used
        self.mode = mode    # mode = 0: act on all channels, mode = 1: act on RGB channels, mode = 2: act on diff channels
        self.log_scale = log_scale

    def apply(self, video_tensor, bvp_tensor):

        T, W, H, C = video_tensor.shape
        info = dict()
        
        if C == 6:
            if self.mode == 1:
                axes_to_transform = range(3, 6)
            elif self.mode == 2:
                axes_to_transform = range(0, 3)
            else:
                axes_to_transform = range(0, 6)
        elif C == 3:
            axes_to_transform = range(0, 3)  # Transform along all axes (T, W, H)

        video_tensor[:, :, :, axes_to_transform] = torch.abs(torch.fft.fft(video_tensor[:, :, :, axes_to_transform], dim=0))
        if self.log_scale:
            video_tensor[:, :, :, axes_to_transform] = torch.log(video_tensor[:, :, :, axes_to_transform] + 1e-7)

        return video_tensor, bvp_tensor, info


class FloatPrecisionDownsamplingTransform(VideoTransform):
    """
    Downsample the video frames to float precision.
    """

    def __init__(self, num_float_bits=32, type_to_be_used="DiffNormalized"):
        self.num_float_bits = num_float_bits
        self.type_to_be_used = type_to_be_used

    def apply(self, video_tensor, bvp_tensor):
        T, W, H, C = video_tensor.shape
        dtype = video_tensor.dtype
        info = dict()

        if self.num_float_bits == 32:
            # Convert to float32
            video_tensor = video_tensor.to(dtype=torch.float32)
        elif self.num_float_bits == 16:
            # Convert to float16
            video_tensor = video_tensor.to(dtype=torch.float16)
        elif self.num_float_bits == 8:
            # Convert to float8
            video_tensor = (video_tensor * 255.0).to(dtype=torch.uint8)
            video_tensor = video_tensor.to(dtype=torch.float64) / 255.0
        else:
            raise ValueError("Unsupported number of float bits. Use 8, 16 or 32.")

        video_tensor = video_tensor.to(dtype)
        return video_tensor, bvp_tensor, info