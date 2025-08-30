
from matplotlib import pyplot as plt
import numpy as np
import scipy

class AutocorrelationFFTFilter:
    def __init__(self, sampling_rate=30, height=1.0, prominence=1.0, min_freq=0.5, max_freq=3.0):
        self.sampling_rate = sampling_rate
        self.height = height
        self.prominence = prominence
        self.min_freq = min_freq
        self.max_freq = max_freq

    def apply(self, ppg_signal):
        ppg_signal = (ppg_signal - np.mean(ppg_signal)) / np.std(ppg_signal)

        # Apply FFT to the PPG signal
        fft_result = np.abs(np.fft.rfft(ppg_signal)) / len(ppg_signal)
        #freq_axis = np.fft.rfftfreq(len(ppg_signal), d=1/self.sampling_rate)
        # Filter frequencies based on min and max frequency
        #freq_mask = (freq_axis >= self.min_freq) & (freq_axis <= self.max_freq)
        #fft_result[freq_mask == False] = 0

        freq_resolution = self.sampling_rate / len(ppg_signal)
        min_lag = int(self.min_freq / freq_resolution)
        max_lag = int(self.max_freq / freq_resolution)

        fft_autocorr = np.correlate(fft_result, fft_result, mode='full')
        fft_autocorr = fft_autocorr[len(fft_autocorr)//2:]

        peaks, _ = scipy.signal.find_peaks(fft_autocorr, height=self.height, prominence=self.prominence)
        peaks = peaks[(peaks >= min_lag) & (peaks <= max_lag)]

        """plt.plot(np.arange(len(fft_autocorr)) * freq_resolution, fft_autocorr)
        plt.scatter(peaks * freq_resolution, fft_autocorr[peaks], color='red', label='Detected Peaks')
        plt.axvline(min_lag * freq_resolution, color='green', linestyle='--', label='Min Lag')
        plt.axvline(max_lag * freq_resolution, color='orange', linestyle='--', label='Max Lag')
        plt.legend()
        plt.title("FFT Autocorrelation")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.xlim(0, self.sampling_rate / 2)
        plt.grid()
        plt.savefig("fft_autocorr.png")
        plt.close()
        input("Press Enter to continue...")"""

        if len(peaks) == 0:
            return 0
        
        best_peak = peaks[0]
        # Calculate heart rate from the peak
        heart_rate = best_peak * freq_resolution * 60
        return heart_rate

class AutocorrelationFilter:
    def __init__(self, sampling_rate=30, height=1.0, prominence=1.0, min_freq=0.5, max_freq=3):
        self.sampling_rate = sampling_rate
        self.height = height
        self.prominence = prominence
        self.min_freq = min_freq
        self.max_freq = max_freq

        self.min_lag = int(self.sampling_rate / self.max_freq)
        self.max_lag = int(self.sampling_rate / self.min_freq)

    def apply(self, ppg_signal):
        """
        Calculate heart rate from PPG signal using the Autocorrelation filter.

        Parameters:
        ppg_signal (list): The PPG signal data.

        Returns:
        float: Estimated heart rate in beats per minute (BPM).
        """
        ppg_signal = (ppg_signal - np.mean(ppg_signal)) / np.std(ppg_signal)
        
        # Calculate autocorrelation
        autocorr = np.correlate(ppg_signal, ppg_signal, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Keep only the positive lags

        # Find the first peak in the autocorrelation
        peaks, _ = scipy.signal.find_peaks(autocorr, height=self.height, prominence=self.prominence)

        # Filter peaks based on frequency range
        peaks = peaks[(peaks >= self.min_lag) & (peaks <= self.max_lag)]

        if len(peaks) == 0:
            return 0
        
        best_peak = peaks[0]

        # Calculate heart rate from the peak
        heart_rate = self.sampling_rate / best_peak * 60
        return heart_rate

class HeartRateFilter:
    """
    Moll Filter for heart rate estimation from PPG signal.
    """

    def __init__(self, sampling_rate=30, f_res=0.005, num_iterations=2, blur_sigma=0.05, peaks_prior_sigma=0.75, window_size=300, peaks_prominence=1.5, peaks_height=1.0, min_freq=0.5, max_freq=3):
        
        self.sampling_rate = sampling_rate
        self.f_res = f_res
        self.num_iterations = num_iterations
        self.blur_sigma = blur_sigma
        self.peaks_prior_sigma = peaks_prior_sigma
        self.window_size = window_size
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.peaks_prominence = peaks_prominence
        self.peaks_height = peaks_height
        
        self.gaussian_kernel_x = np.arange(-3*self.blur_sigma, 3*self.blur_sigma, self.f_res)
        self.gaussian_kernel = np.exp(-0.5 * (self.gaussian_kernel_x / (self.blur_sigma)) ** 2)

        

    def apply(self, ppg_signal):
        """
        Calculate heart rate from PPG signal using the Moll filter with windowing.

        Parameters:
        ppg_signal (list): The PPG signal data.
        sampling_rate (int): The sampling rate of the PPG signal.
        f_res (float): Frequency resolution for interpolation.
        num_iterations (int): Number of scales for the Moll filter.
        blur_sigma (float): Standard deviation for Gaussian blur.
        window_size (int): Size of the moving window.

        Returns:
        list: Estimated heart rates in beats per minute (BPM) for each window.
        """
        
        heart_rates = []

        window_size = min(len(ppg_signal), self.window_size)
        
        for i in range(0, len(ppg_signal) - window_size + 1, window_size):
            windowed_signal = ppg_signal[i:i + window_size]
            hr = self._get_heart_rate(windowed_signal)
            heart_rates.append(hr)
        
        return heart_rates
    
    def _get_peaks_prior(self, ppg_signal, frequencies):
        peaks, properties = scipy.signal.find_peaks(ppg_signal, prominence=self.peaks_prominence, height=self.peaks_height)

        frequency = np.median(np.diff(peaks)) / self.sampling_rate 
        frequency = len(peaks) / (len(ppg_signal) / self.sampling_rate)

        prior = (1/(2*np.pi*(self.peaks_prior_sigma ** 2))) * np.exp(-0.5 * ((frequencies - frequency) / self.peaks_prior_sigma) ** 2)
        return prior
    
    def _get_heart_rate(self, ppg_signal):
        """
        Calculate heart rate from PPG signal using the Moll filter.

        Parameters:
        ppg_signal (list): The PPG signal data.
        sampling_rate (int): The sampling rate of the PPG signal.
        f_res (float): Frequency resolution for interpolation.
        num_iterations (int): Number of scales for the Moll filter.
        blur_sigma (float): Standard deviation for Gaussian blur.

        Returns:
        float: Estimated heart rate in beats per minute (BPM).
        """
        ppg_signal = (ppg_signal - np.mean(ppg_signal)) / np.std(ppg_signal)
        
        fft = np.fft.fft(ppg_signal)
        freq = np.fft.fftfreq(len(ppg_signal), d=1/self.sampling_rate)

        idx = np.argsort(freq)
        freq = freq[idx]
        fft = fft[idx]

        """plt.plot(freq, np.abs(fft))
        plt.savefig("fft.png")
        plt.close()"""

        freq_interp = np.arange(0, self.sampling_rate/2, self.f_res)
        fft_interp = np.interp(freq_interp, freq, np.abs(fft))

        """plt.plot(gaussian_kernel_x, gaussian_kernel)
        plt.savefig("gaussian_kernel.png")
        plt.close()"""

        fft_interp = np.convolve(fft_interp, self.gaussian_kernel, mode='same')
        fft_sum = np.ones(len(freq_interp))
        fft_sum[freq_interp < self.min_freq] = 0
        fft_sum[freq_interp > self.max_freq] = 0

        peaks_prior = self._get_peaks_prior(ppg_signal, freq_interp)

        """plt.plot(freq_interp, peaks_prior)
        plt.savefig("prior.png")
        plt.close()"""

        for j in range(self.num_iterations):
            scale = 1.0 / (j + 1)
            fft_scaled = np.interp(freq_interp, freq_interp*scale, np.abs(fft_interp))
            fft_sum *= fft_scaled

            """plt.plot(freq_interp, fft_sum, label=f'Scale {j+1}')
            plt.savefig("fft_plot.png")
            plt.close()
            input("Press Enter to continue...")"""


        fft_sum *= peaks_prior
        """plt.plot(freq_interp, fft_sum)
        plt.savefig("fft_sum.png")
        plt.close()
        #input("Press Enter to continue...")"""

        hr_est = np.argmax(fft_sum) * self.f_res
        return hr_est

    