import base64
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import zstandard as zstd
import requests
from io import BytesIO

import http_utils as utils
import cv2

from evaluation.utils import calculate_gt_hr
from dataset.data_loader.compression.preprocessingCompression import NPZCompressor

patient_number = 0
batch_number = 1
preprocessed_path = ""  # Add one of your preprocessings here

server_ip = "localhost"  # Replace with your server IP
server_port = 8000  # Replace with your server port




compressor = NPZCompressor()

tensor = compressor.load_compressed_data(os.path.join(preprocessed_path, f"{patient_number}_input{batch_number}.npy.compressed.npz"))
tensor = tensor[:, :, :, :3]    # use only diff channels

gt_label = compressor.load_compressed_data(os.path.join(preprocessed_path, f"{patient_number}_label{batch_number}.npy.compressed.npz"))
gt_label = gt_label.reshape(1, -1)  # Ensure gt_label is a 1D array

gt_hr = calculate_gt_hr(gt_label)

tensor = tensor.astype('>f4')  # Ensure tensor is in float32 little-endian format

# Compress tensor in memory
compressed = zstd.compress(tensor.tobytes())

# Prepare multipart form data
files = {
    "tensor": utils.get_octetstream_tuple(compressed)
}

data = {
    "compression": "zstd",
    "model": "loglikelihood",  # Example model name loglikelihood, physnet, quantile
    "format": "diffNorm",
    "shape": ",".join(map(str, tensor.shape)),
    "dtype": str(tensor.dtype),
    "endianness": utils.numpy_byteorder_to_endianness(tensor.dtype.byteorder),
    "postprocessor": "cumsum",  # Example postprocessor
    "hr_extractor": "autocorrelation",  # Example heart rate extractor
    "confidence": 0.95,  # Example confidence level
    "fps": 30,  # Example frames per second
}

# Send POST request
response = requests.post(f"http://{server_ip}:{server_port}/upload_tensor", files=files, data=data)
rppg_base64 = response.json().get("rppg")
rppg_uncertainty_base64 = response.json().get("rppg_uncertainty")

heart_rate = response.json().get("hr")
heart_rate_uncertainty = response.json().get("hr_uncertainty")
shape = response.json().get("shape")
dtype = response.json().get("dtype")
endianness = response.json().get("endianness")

dtype = np.dtype(dtype)

if endianness == 'little':
    dtype = dtype.newbyteorder('<')  # Little-endian
elif endianness == 'big':
    dtype = dtype.newbyteorder('>')

print(f"Received tensor shape: {shape}, dtype: {dtype}, endianness: {endianness}")
print(f"Heart rate: {heart_rate}; Uncertainty: {heart_rate_uncertainty}; Ground Truth HR: {gt_hr}")

# Decode the base64 tensor
decoded_rppg = utils.decode_tensor_from_base64(rppg_base64, dtype, shape)

if rppg_uncertainty_base64 is None:
    decoded_rppg_uncertainty = None
else:
    decoded_rppg_uncertainty = utils.decode_tensor_from_base64(rppg_uncertainty_base64, dtype, shape)

plt.plot(gt_label, label='Ground Truth')

if decoded_rppg_uncertainty is None:
    plt.plot(decoded_rppg[0, :], label='Predicted Signal')
else:
    plt.fill_between(np.arange(decoded_rppg.shape[-1]), decoded_rppg[0, :], decoded_rppg_uncertainty[0, :], color='lightgray', alpha=0.5, label='Uncertainty Interval')

plt.xlabel('Frame')
plt.ylabel('Signal Value')
plt.title('Ground Truth vs Predicted Signal')
plt.legend()

plt.savefig('client_output_plot.png')