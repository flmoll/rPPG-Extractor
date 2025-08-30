

import os
import time
import matplotlib
matplotlib.use('Agg')

from dataset.data_loader.compression.preprocessingCompression import NPZCompressor
import argparse
from tqdm import tqdm
import multiprocessing as mp

def parse_arguments():
    parser = argparse.ArgumentParser(description="Compression script arguments")
    parser.add_argument("--input", type=str, required=True, help="Path to the input file")
    parser.add_argument("--workers", type=int, required=False, help="The number of workers to use for data loading")
    parser.add_argument("--compress", action='store_true', help="Compress the input folder (default is decompress)")
    parser.add_argument("--subfolders", action='store_true', help="Compress the input folder (default is decompress)")
    return parser.parse_args()

def compress_data(input_folder, compress):

    compressor = NPZCompressor()
    print(f"Processing folder: {input_folder}")

    if compress:
        print("Compressing data...")
        compressor.compress(input_folder)
    else:
        print("Decompressing data...")
        compressor.decompress(input_folder)

def main():

    args = parse_arguments()
    input_folder = args.input
    compress = args.compress if args.compress is not None else False
    subfolders = args.subfolders if args.subfolders is not None else False
    num_workers = args.workers if args.workers is not None else 1

    print("Input folder:", input_folder)

    if not os.path.isdir(input_folder):
        raise ValueError(f"Input path {input_folder} is not a directory.")

    if subfolders:
        folders = []
        for folder in os.listdir(input_folder):
            folder_path = os.path.join(input_folder, folder)
            if os.path.isdir(folder_path):
                folders.append(folder_path)
    else:
        folders = [input_folder]

    workers = []
    compressor = NPZCompressor()

    for folder in tqdm(folders, desc="Processing folders"):
        print(f"Processing folder: {input_folder}")

        if compress:
            print("Compressing data...")
            compressor.compress(input_folder)
        else:
            print("Decompressing data...")
            compressor.decompress(input_folder)

if __name__ == "__main__":
    main()
