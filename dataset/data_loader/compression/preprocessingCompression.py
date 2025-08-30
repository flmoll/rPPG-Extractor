from abc import ABC, abstractmethod
import os
import shutil
import time

import numpy as np
import tqdm

from dataset.data_loader import BaseLoader

import multiprocessing as mp

class PreprocessingCompressorFactory:
    """
    Factory class for creating instances of PreprocessingCompressor.
    """

    @staticmethod
    def create_compressor(compressor_type):
        if compressor_type == 'DIFF':
            return DiffRemoveCompressor()
        elif compressor_type == 'NPZ':
            return NPZCompressor()
        elif compressor_type == 'None' or compressor_type is None:
            return None
        else:
            raise ValueError(f"Unknown compressor type: {compressor_type}")

class NotEnoughDiskSpaceError(Exception):
    """
    Exception raised when there is not enough disk space to compress or decompress files.
    """
    def __init__(self, message="Not enough disk space to perform the operation."):
        self.message = message
        super().__init__(self.message)

class PreprocessingCompressor(ABC):
    """
    Abstract base class for preprocessing compressors.
    Defines the interface for compression and decompression methods.
    """
    def compress(self, data_path, out_path=None, delete_old=True, num_workers=4):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path {data_path} does not exist.")
        
        if not os.path.isdir(data_path):
            raise ValueError(f"Data path {data_path} is not a directory.")
        
        if out_path is not None:
            
            if not os.path.exists(out_path):
                os.makedirs(out_path)

            total, used, free = shutil.disk_usage(out_path)
            if free < 10 * 1024 * 1024:  # less than 10 MB
                raise NotEnoughDiskSpaceError()

        results = []
        
        with mp.Pool(processes=num_workers) as pool:
            for file in tqdm.tqdm(os.listdir(data_path), desc="Compressing files", unit="file"):
                file_path = os.path.join(data_path, file)
                if self.is_decompressed_file(file_path):
                    # Compress the .npy file
                    compressed_file_path = self.get_compressed_data_path(file_path)

                    if out_path is not None:
                        compressed_file_path = os.path.join(out_path, os.path.basename(compressed_file_path))

                    if not os.path.exists(compressed_file_path):
                        #self._compress(file_path, compressed_file_path)
                        result = pool.apply_async(self._compress, args=(file_path, compressed_file_path))
                        results.append((result, file_path))


            for result in tqdm.tqdm(results, desc="Waiting for compression", unit="file"):
                result[0].wait()

                if delete_old:
                    os.remove(result[1])

    def decompress(self, compressed_data_path, out_path=None, delete_old=True, num_workers=4):
        if not os.path.exists(compressed_data_path):
            raise FileNotFoundError(f"Data path {compressed_data_path} does not exist.")
        
        if not os.path.isdir(compressed_data_path):
            raise ValueError(f"Data path {compressed_data_path} is not a directory.")
        
        if out_path is not None:

            if not os.path.exists(out_path):
                os.makedirs(out_path)

        results = []
        
        with mp.Pool(processes=num_workers) as pool:
            for file in tqdm.tqdm(os.listdir(compressed_data_path), desc="Decompressing files", unit="file"):
                file_path = os.path.join(compressed_data_path, file)
                if self.is_compressed_file(file_path):
                    # Compress the .npy file
                    decompressed_file_path = self.get_decompressed_data_path(file_path)
                    
                    if out_path is not None:
                        decompressed_file_path = os.path.join(out_path, os.path.basename(decompressed_file_path))

                    if not os.path.exists(decompressed_file_path):
                        #self._decompress(file_path, decompressed_file_path)
                        result = pool.apply_async(self._decompress, args=(file_path, decompressed_file_path))
                        results.append((result, file_path))


            for result in tqdm.tqdm(results, desc="Waiting for decompression", unit="file"):
                result[0].wait()
                
                if result[0].get() == -1:
                    raise NotEnoughDiskSpaceError()

                if delete_old:
                    os.remove(result[1])

    def get_compressed_files_in_folder(self, folder_path):
        result_files = []

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                if self.is_compressed_file(file_path):
                    result_files.append(file_path)

        return result_files
    
    def _compress(self, data_path, compressed_data_path):
        
        out_folder = os.path.dirname(compressed_data_path)
        total, used, free = shutil.disk_usage(out_folder)
        if free < 1 * 1024 * 1024 * 1024:  # less than 1GB
            return -1
        
        npy_tensor = np.load(data_path)
        self.save_compressed_data(npy_tensor, compressed_data_path)
        return 0

    def _decompress(self, compressed_data_path, data_path):
        
        out_folder = os.path.dirname(data_path)
        total, used, free = shutil.disk_usage(out_folder)
        if free < 1 * 1024 * 1024 * 1024:  # less than 1GB
            return -1
        
        npy_tensor = self.load_compressed_data(compressed_data_path)
        np.save(data_path, npy_tensor)
        return 0

    @abstractmethod
    def is_compressed_file(self, file_path):
        """
        Check if the file is a compressed file.

        Args:
            file_path: The path to the file.

        Returns:
            True if the file is compressed, False otherwise.
        """
        pass

    @abstractmethod
    def is_decompressed_file(self, file_path):
        """
        Check if the file is a decompressed file.

        Args:
            file_path: The path to the file.

        Returns:
            True if the file is decompressed, False otherwise.
        """
        pass

    @abstractmethod
    def get_compressed_data_path(self, data_path):
        pass

    @abstractmethod
    def get_decompressed_data_path(self, compressed_data_path):
        pass

    @abstractmethod
    def load_compressed_data(self, compressed_data_path):
        """
        Load the compressed data.

        Args:
            compressed_data_path: The path to the compressed data.

        Returns:
            The loaded data.
        """
        pass

    def save_compressed_data(self, data, compressed_data_path):
        """
        Save the compressed data.

        Args:
            data: The data to be saved.
            compressed_data_path: The path to save the compressed data.

        Returns:
            None
        """
        pass

class DiffRemoveCompressor(PreprocessingCompressor):
    """
    Compressor for .npy files.
    """
    def load_compressed_data(self, compressed_data_path):
        tensor = np.load(compressed_data_path)

        tensor_diff = BaseLoader.diff_normalize_data(tensor)
        return np.concatenate((tensor, tensor_diff), axis=3)
    
    def save_compressed_data(self, npy_tensor, compressed_data_path):
        tensor_without_diff = npy_tensor[:, :, :, 3:]
        np.save(compressed_data_path, tensor_without_diff)

class NPZCompressor(PreprocessingCompressor):
    """
    Compressor for .npz files.
    """
    def is_compressed_file(self, file_path):
        return file_path.endswith('.compressed.npz')
    
    def is_decompressed_file(self, file_path):
        return file_path.endswith('.npy')

    def load_compressed_data(self, compressed_data_path):
        mapping = np.load(compressed_data_path, allow_pickle=True)
        return mapping['array1']
    
    def save_compressed_data(self, npy_tensor, compressed_data_path):
        np.savez_compressed(compressed_data_path, array1=npy_tensor)

    def get_compressed_data_path(self, data_path):
        return data_path + ".compressed.npz"
    
    def get_decompressed_data_path(self, compressed_data_path):
        return compressed_data_path.replace(".compressed.npz", "")
