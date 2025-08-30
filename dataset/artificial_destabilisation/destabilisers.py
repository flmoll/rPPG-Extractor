from abc import ABC, abstractmethod
import os
import subprocess
import io

import cv2
from matplotlib import pyplot as plt
import numpy as np
import scipy
import ffmpeg
import torch
from torchvision.transforms import functional as F
from tqdm import tqdm
import gc

def relu(x):
    """
    Apply the ReLU function to the input.

    :param x: Input value.
    :return: The ReLU output.
    """
    return max(0, x)

class DestabilisationBackend(ABC):
    """
    Abstract base class for destabilisation backends.
    """

    @abstractmethod
    def generate_noise_sequence(self, length: int) -> np.ndarray:
        """
        Generate a sequence of noise.

        :param length: The length of the noise sequence.
        :return: A list containing the generated noise sequence.
        """
        pass

    def destabilise(self, frames: np.ndarray, keep_frame_dims=True) -> np.ndarray:
        """
        Perform destabilisation on the given frames.

        :param frames: A numpy array containing the frames to be destabilised.
        :return: A numpy array containing the destabilised frames.
        """
        noise_sequence = self.generate_noise_sequence(len(frames))
        noise_sequence = noise_sequence.astype(np.int32)

        if keep_frame_dims:
            for i in range(len(frames)):
                curr_frame = frames[i]

                box1 = np.array([relu(noise_sequence[i, 0]), 
                                 relu(noise_sequence[i, 1]), 
                                 curr_frame.shape[0] - relu(-noise_sequence[i, 0]), 
                                 curr_frame.shape[1] - relu(-noise_sequence[i, 1])])
                
                box2 = np.array([relu(-noise_sequence[i, 0]),
                                 relu(-noise_sequence[i, 1]), 
                                 curr_frame.shape[0] - relu(noise_sequence[i, 0]), 
                                 curr_frame.shape[1] - relu(noise_sequence[i, 1])])

                if keep_frame_dims:
                    new_frame = np.zeros_like(curr_frame)
                    new_frame[box2[0]:box2[2], box2[1]:box2[3], :] = curr_frame[box1[0]:box1[2], box1[1]:box1[3]]
                    frames[i] = new_frame
                else:
                    if i == 0:
                        w = curr_frame.shape[0] - np.max(noise_sequence[:, 0])
                        h = curr_frame.shape[1] - np.max(noise_sequence[:, 1])
                        destabilised_frames = np.zeros((len(frames), w, h, 3), dtype=frames.dtype)
                    
                    destabilised_frames[i] = curr_frame[box1[0]:box1[2], box1[1]:box1[3], :]

            if keep_frame_dims:
                return frames
            else:
                return destabilised_frames

class GaussianNoiseDestabilizer(DestabilisationBackend):
    """
    Implements a Gaussian Noise Destabiliser backend.

    This class generates unfiltered Gaussian noise to simulate random motion
    or jitter in a sequence of video frames. The noise can be applied to
    destabilise frames in the x and y directions independently.

    Attributes:
        noise_amplitude (float): Standard deviation (amplitude) of the Gaussian noise.

    Methods:
        generate_noise_sequence(length):
            Generates a sequence of 2D Gaussian noise vectors of the specified length.
    """

    def __init__(self, noise_amplitude: float):
        """
        Initialize the Unfiltered Noise Destabiliser.

        :param noise_frequency: The frequency of the noise.
        :param noise_amplitude: The amplitude of the noise.
        """
        self.noise_amplitude = noise_amplitude

    def generate_noise_sequence(self, length: int) -> np.ndarray:
        """
        Generate a sequence of noise.

        :param length: The length of the noise sequence.
        :return: A numpy array containing the generated noise sequence.
        """
        t = np.linspace(0, length, num=length)
        noise = self.noise_amplitude * np.random.randn(length, 2)
        return noise
    
class RandomAffineDestabilizer(DestabilisationBackend):
    """
    Implements a Random Affine Destabilisation backend.

    This class applies random affine transformations (rotation, translation, scaling, and shear)
    to a sequence of video frames to simulate camera instability or motion. The transformations
    are applied independently to each frame.

    Attributes:
        angle (float): Maximum rotation angle in degrees.
        translate (tuple of float): Maximum translation in x and y directions as a fraction of frame dimensions.
        scale (tuple of float): Range of scaling factors (min_scale, max_scale).
        shear (tuple of float): Range of shear factors in x and y directions (min_shear, max_shear).

    Methods:
        destabilise(frames, keep_frame_dims=True):
            Applies random affine transformations to each frame in the input array.
            Returns the transformed frames as a NumPy array.
        
        generate_noise_sequence(length):
            Raises NotImplementedError, as Random Affine does not generate an explicit noise sequence.
    """

    def __init__(self, angle: float = 1.0, translate: tuple = (0.02, 0.02), scale: tuple = (0.98, 1.02), shear: tuple = (-1, 1)):
        """
        Initialize the Random Affine Destabiliser.

        :param angle: The maximum rotation angle in degrees.
        :param translate: The maximum translation in x and y directions as a fraction of the frame size.
        :param scale: The scaling factor range.
        :param shear: The shear factor range.
        """
        self.angle = angle
        self.translate = translate
        self.scale = scale
        self.shear = shear

    def destabilise(self, frames, keep_frame_dims=True):
        """
        Perform destabilisation on the given frames.

        :param frames: A numpy array containing the frames to be destabilised.
        :param keep_frame_dims: Whether to keep the original frame dimensions.
        :return: A numpy array containing the destabilised frames.
        """

        # Apply the affine transformation to each frame
        for frame, idx in zip(frames, range(len(frames))):

            angle = np.random.uniform(-self.angle, self.angle)
            translate_x = np.random.uniform(-self.translate[0] * frames.shape[2], self.translate[0] * frames.shape[2])
            translate_y = np.random.uniform(-self.translate[1] * frames.shape[1], self.translate[1] * frames.shape[1])
            scale = np.random.uniform(self.scale[0], self.scale[1])
            shear_x = np.random.uniform(-self.shear[0], self.shear[0])
            shear_y = np.random.uniform(-self.shear[1], self.shear[1])

            curr_frame = torch.tensor(frame).permute(2, 0, 1)  # Change from (H, W, C) to (C, H, W)
            transformed_frame = F.affine(
                curr_frame,
                angle=angle,
                translate=(translate_x, translate_y),
                scale=scale,
                shear=(shear_x, shear_y),
                interpolation=F.InterpolationMode.BILINEAR,
                fill=0  # Fill with black
            )

            frames[idx] = transformed_frame.permute(1, 2, 0).numpy()  # Change back to (H, W, C)

        return frames

    def generate_noise_sequence(self, length):
        raise NotImplementedError("Random Affine does not generate a noise sequence.")

class DeepstabDestabilizer(DestabilisationBackend):
    """
    Implements the Deepstab destabilisation backend.

    This class introduces realistic camera jitter and motion noise to a sequence of frames
    using precomputed noise sequences derived from the Deepstab dataset. It is useful
    for simulating real-world camera instabilities in video-based model testing.

    Attributes:
        noise_amplitude (float): Scaling factor for the magnitude of the jitter/noise.
        random (bool): If True, randomly selects a noise sequence; otherwise, cycles sequentially.
        counter (int): Internal counter to keep track of which noise sequence is currently used.
        jitter_tensors (List[np.ndarray]): List of preprocessed jitter sequences loaded from files.

    Methods:
        _read_deepstab(deepstab_preprocess_folder: str):
            Loads and preprocesses jitter sequences from the given folder. Each `.npy` file
            is high-pass filtered and normalized to produce standardized jitter.

        generate_noise_sequence(length: int) -> np.ndarray:
            Returns a noise sequence of the requested length by concatenating segments
            from the preloaded jitter tensors, scaled by `noise_amplitude`.
    """

    def __init__(self, deepstab_preprocess_folder: str, noise_amplitude: float = 10.0, random=True):
        """
        Initialize the Deepstab Destabiliser.

        :param noise_frequency: The frequency of the noise.
        :param noise_amplitude: The amplitude of the noise.
        """
        self.noise_amplitude = noise_amplitude
        self.random = random
        self.counter = 0
        
        self._read_deepstab(deepstab_preprocess_folder)

    def _read_deepstab(self, deepstab_preprocess_folder: str):
        """
        Read the Deepstab file and return the noise sequence.

        :param deepstab_path_processed: The path to the processed Deepstab file.
        :return: A numpy array containing the noise sequence.
        """
        self.jitter_tensors = []

        for file in os.listdir(deepstab_preprocess_folder):
            if file.endswith(".npy"):
                in_file = os.path.join(deepstab_preprocess_folder, file)
                print(f"Processing file: {in_file}")
                jitter_tensor = np.load(in_file)
                jitter_tensor = jitter_tensor.astype(np.int32)

                highpass = scipy.signal.butter(4, 0.1, btype='highpass', analog=False, output='ba', fs=30)
                jitter_tensor[:, 0] = scipy.signal.filtfilt(highpass[0], highpass[1], jitter_tensor[:, 0])
                jitter_tensor[:, 1] = scipy.signal.filtfilt(highpass[0], highpass[1], jitter_tensor[:, 1])
                
                jitter_tensor = jitter_tensor / np.std(jitter_tensor, axis=0)

                self.jitter_tensors.append(jitter_tensor)


    def generate_noise_sequence(self, length: int) -> np.ndarray:
        """
        Generate a sequence of noise.

        :param length: The length of the noise sequence.
        :return: A numpy array containing the generated noise sequence.
        """
        result_tensor = []
        current_result_len = 0

        while current_result_len < length:
            if self.random:
                self.counter = np.random.randint(0, len(self.jitter_tensors))
            else:
                self.counter = (self.counter + 1) % len(self.jitter_tensors)

            jitter_tensor = self.jitter_tensors[self.counter].copy()
            jitter_tensor = jitter_tensor[:(length - current_result_len)]
            jitter_tensor = jitter_tensor * self.noise_amplitude

            result_tensor.append(jitter_tensor)
            current_result_len += len(jitter_tensor)

        result_tensor = np.concatenate(result_tensor, axis=0)
        assert(result_tensor.shape[0] == length)
        return result_tensor

class H264CompressionDestabiliser(DestabilisationBackend):
    """
    Destabilises video frames by applying H264 compression and decompression.

    This backend uses ffmpeg to compress video frames in-memory using H264 (libx264) 
    and then decompresses them back to raw frames. This process introduces compression 
    artifacts that act as destabilisation/noise. Useful for testing robustness of video 
    processing models to compression artifacts.

    Attributes:
        crf (float): Constant Rate Factor controlling compression quality (lower = higher quality).
        temp_file (str): Temporary file path (not used directly here but for compatibility).
        preset (str): ffmpeg compression preset (e.g., 'slow', 'medium', 'fast').
        vcodec (str): Video codec used for compression (default 'libx264').
        use_constant_qp (bool): If True, uses constant quantization parameter instead of CRF.

    Methods:
        generate_noise_sequence(length):
            Raises NotImplementedError. H264 compression does not generate a standalone noise sequence.

        destabilise(frames, keep_frame_dims=True, fps=30, pix_fmt='bgr24'):
            Compresses and decompresses a sequence of frames to introduce compression artifacts.
            
            Args:
                frames (np.ndarray): Array of shape (T, H, W, C) containing video frames.
                keep_frame_dims (bool): If True, output frames keep the same dimensions as input.
                fps (int): Frames per second for compression.
                pix_fmt (str): Pixel format for ffmpeg ('bgr24' by default).

            Returns:
                np.ndarray: Destabilised frames with same shape as input.
    """

    def __init__(self, crf: float = 23, temp_file: str = 'temp_compressed.mp4', preset='slow', vcodec='libx264', use_constant_qp=False):
        """
        Initialize the H264 Compression Destabiliser.

        :param crf: The Constant Rate Factor for the H264 compression.
        """
        self.crf = crf
        self.temp_file = temp_file
        self.preset = preset
        self.vcodec = vcodec
        self.use_constant_qp = use_constant_qp

    def generate_noise_sequence(self, length):
        raise NotImplementedError("H264 Compression does not generate a noise sequence.")

    def destabilise(self, frames, keep_frame_dims=True, fps=30, pix_fmt='bgr24'):
        
        height, width, _ = frames[0].shape
        dtype = frames.dtype

        if dtype == np.float32 or dtype == np.float64:
            frames = (frames * 255).astype(np.uint8)

        if self.use_constant_qp:
            quantization_arg = '-qp'
        else:
            quantization_arg = '-crf'

        # Step 1: Compress to memory (using libx264, output .mkv)
        compressor = subprocess.Popen(
            [
                'ffmpeg',
                '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-pix_fmt', pix_fmt,
                '-s', f'{width}x{height}',
                '-r', str(fps),
                '-threads', '4',
                '-i', 'pipe:0',
                '-an',
                '-vcodec', self.vcodec,
                '-preset', self.preset,
                quantization_arg, str(self.crf),
                '-f', 'matroska',
                'pipe:1'
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        compressed_bytes, stderr1 = compressor.communicate(input=frames.tobytes())

        if compressor.returncode != 0:
            raise RuntimeError(f"Compression failed:\n{stderr1.decode()}")
        #print("Compression done. Compressed size:", len(compressed_bytes), "; compression ratio:", frames.nbytes / len(compressed_bytes))

        # Step 2: Decompress from memory back to raw frames
        decompressor = subprocess.Popen(
            [
                'ffmpeg',
                '-vsync', '0',
                '-i', 'pipe:0',
                '-f', 'rawvideo',
                '-pix_fmt', pix_fmt,
                'pipe:1'
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        decompressed_raw, err = decompressor.communicate(input=compressed_bytes)
        if decompressor.returncode != 0:
            raise RuntimeError(f"Decompression failed:\n{err.decode()}")

        # Step 3: Convert to NumPy array
        frames = np.frombuffer(decompressed_raw, dtype=np.uint8)
        frames = frames.reshape((-1, height, width, 3))
        
        if dtype == np.float32 or dtype == np.float64:
            frames = frames.astype(dtype) / 255.0

        return frames
    