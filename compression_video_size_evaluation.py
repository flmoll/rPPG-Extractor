import os
import subprocess

from matplotlib import pyplot as plt
import numpy as np


def compress_video_with_h264(in_file_name, out_file_name, use_constant_qp=True, crf=0, preset="slow", vcodec="libx264", pix_fmt="yuv420p", width=1920, height=1080, fps=30):

    if use_constant_qp:
        quantization_arg = '-qp'
    else:
        quantization_arg = '-crf'

    compressor = subprocess.Popen(
        [
            'ffmpeg',
            '-y',
            '-i', in_file_name,
            '-vcodec', vcodec,
            '-preset', preset,
            quantization_arg, str(crf),
            '-pix_fmt', pix_fmt,
            '-s', f'{width}x{height}',
            '-r', str(fps),
            out_file_name
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    stdout, stderr = compressor.communicate()
    if compressor.returncode != 0:
        print(f"Error compressing video: {stderr.decode('utf-8')}")
    else:
        print(f"Video compressed successfully: {out_file_name}")

if __name__ == "__main__":

    crf_values = range(0, 55, 5)  # H.264 CRF values from 0 to 50
    video_in_path = "/mnt/data/vitalVideos/3cf596e2bcc34862abc89bd2eca4a985_1.mp4"  # Specify the input video path
    out_path = "/mnt/results/compressed_videos"  # Specify the output directory

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    in_video_size = os.path.getsize(video_in_path)
    print(f"Original video size: {in_video_size / (1024 * 1024):.2f} MB")

    compressed_sizes = []

    for i, compressor in enumerate(["libx264", "libx265"]):

        compressed_sizes.append([])

        for crf in crf_values:
            print(f"Testing {compressor} compression with CRF {crf}")

            video_file = f"{video_in_path.split('/')[-1].split('.')[0]}_{compressor}_crf_{crf}.mp4"
            video_out_path = f"{out_path}/{video_file}"

            if os.path.exists(video_out_path):
                print(f"Skipping existing file: {video_out_path}")
            else:
                compress_video_with_h264(video_in_path, video_out_path, use_constant_qp=False, crf=crf, vcodec=compressor)

            compressed_size = os.path.getsize(video_out_path)
            print(f"Compressed video size (CRF {crf}): {compressed_size / (1024 * 1024):.2f} MB")

            size_reduction = (in_video_size - compressed_size) / in_video_size * 100
            print(f"Size reduction: {size_reduction:.2f}%\n")

            compressed_sizes[i].append(compressed_size / (1024 * 1024))

    print("Compression completed for all CRF values.")
    print(f"Compressed sizes: {compressed_sizes}")

    compressed_sizes = np.array(compressed_sizes)
    compressed_sizes = compressed_sizes / compressed_sizes.max(axis=1, keepdims=True)  # Normalize sizes for better visualization

    for i, compressor in enumerate(["libx264", "libx265"]):
        plt.plot(crf_values, compressed_sizes[i], label=compressor)
        plt.yscale('log')

    plt.title('Video Compression Size vs CRF Values')
    plt.xlabel('CRF Value')
    plt.ylabel('Compressed Video Size (MB)')
    plt.grid()
    plt.legend()
    plt.savefig(f"graphics/compression_size_vs_crf.png")