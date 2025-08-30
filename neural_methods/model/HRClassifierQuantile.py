from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvBlock(nn.Module):
    """
    A 1D residual convolutional block.

    This block applies a 1D convolution followed by batch normalization and ReLU activation,
    with a residual (skip) connection. The skip connection ensures that the input can be
    added directly to the output, allowing gradient flow and enabling deeper networks.

    Attributes:
        conv (nn.Sequential): Convolutional block with Conv1d -> BatchNorm1d -> ReLU.
        shortcut (nn.Module): Identity mapping or 1x1 convolution if input and output channels differ.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.

    Forward:
        x (torch.Tensor): Input tensor of shape [batch_size, in_channels, T].
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, out_channels, T].
    """
    
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)

class HRClassifierQuantile(nn.Module):
    """
    A 1D residual convolutional neural network for estimating heart rate (HR) with quantile bounds from rPPG signals.

    The network processes the rPPG signal along with features derived from the FFT magnitude, FFT phase, and
    autocorrelation of the rPPG signal. It outputs lower and upper bounds of the estimated heart rate.

    Architecture:
        - ResidualConvBlock: Four 1D residual convolutional blocks with increasing channels (128, 256, 512, 1024)
        - MaxPool1d: Downsampling after each residual block
        - Fully connected layers: Three layers (1024 -> 256 -> 2) for HR quantile prediction
        - FFT embedding: Linear layers to embed FFT magnitude and phase features
        - Autocorrelation: Autocorrelation of the rPPG signal as an additional feature
        - Concatenation: Combines rPPG, FFT, and autocorrelation features before residual blocks

    Input:
        x (torch.Tensor): Tensor of shape [batch_size, 2, T] where:
            - channel 0: rPPG signal
            - channel 1: auxiliary signal (e.g., uncertainty)

    Output:
        Tuple[torch.Tensor, torch.Tensor]:
            - lower: Lower quantile estimate of heart rate [batch_size]
            - upper: Upper quantile estimate of heart rate [batch_size]

    Notes:
        - The FFT magnitude and phase are linearly embedded to match the temporal dimension of the input.
        - Autocorrelation is computed using 1D convolution of the signal with itself.
        - The residual blocks allow the network to learn hierarchical features efficiently.
    """

    def __init__(self, in_size=[2, 160]):
        super(HRClassifierQuantile, self).__init__()

        self.block1 = ResidualConvBlock(in_size[0]+3, 128, kernel_size=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # Downsample to half length

        self.block2 = ResidualConvBlock(128, 256, kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # Downsample to quarter length

        self.block3 = ResidualConvBlock(256, 512, kernel_size=3)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)  # Downsample to eighth length

        self.block4 = ResidualConvBlock(512, 1024, kernel_size=3)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)  # Downsample to 16-th length

        self.linear1 = nn.Linear(1024 * (in_size[1] // 16), 1024)  # Final linear layer to output heart rate and uncertainty
        self.activation1 = nn.ReLU(inplace=True)  # Activation function after the first linear layer
        self.linear2 = nn.Linear(1024, 256)  # Final linear layer to output heart rate and uncertainty
        self.activation2 = nn.ReLU(inplace=True)  # Activation function after the second linear layer
        self.linear3 = nn.Linear(256, 2)  # Final linear layer to output heart rate and uncertainty
        self.activation3 = nn.Identity()  # Sigmoid activation for the final output

        self.fft_mag_embedding = nn.Linear(in_size[1] // 2, in_size[1])  # Linear layer to embed FFT magnitude
        self.fft_phase_embedding = nn.Linear(in_size[1] // 2, in_size[1])  # Linear layer to embed FFT phase

    def forward(self, x):
        x = x.reshape(x.shape[0], 2, -1)  # Reshape to [batch_size, 2, T]

        rppg_fft = torch.fft.rfft(x[:, 0, :], dim=-1)[:, 1:]  # FFT on the first channel (rPPG)
        rppg_fft_abs = torch.abs(rppg_fft)  # Get the magnitude
        rppg_fft_phase = torch.angle(rppg_fft)  # Get the phase
        #rppg_fft = F.interpolate(rppg_fft_abs.unsqueeze(1), size=x.shape[-1], mode='linear', align_corners=False).squeeze(1)  # Resample to original length
        #rppg_phase = F.interpolate(rppg_fft_phase.unsqueeze(1), size=x.shape[-1], mode='linear', align_corners=False).squeeze(1)  # Resample phase
        rppg_fft = self.fft_mag_embedding(rppg_fft_abs)  # Embed FFT magnitude
        rppg_phase = self.fft_phase_embedding(rppg_fft_phase)  # Embed FFT phase
        #x = torch.stack((rppg_fft, rppg_phase), dim=1)  # Stack FFT magnitude, phase, and uncertainty

        autocorr_input = x[:, 0, :].unsqueeze(0)  # Use the first channel for autocorrelation
        autocorr_input = F.pad(autocorr_input, (0, x.shape[-1] - 1))  # Pad (left=0, right=time-1)
        autocorr_kernel = x[:, 0, :].unsqueeze(1)  # Reshape for convolution
        autocorrelation = torch.nn.functional.conv1d(autocorr_input, autocorr_kernel, padding=0, stride=1, groups=x.shape[0])  # Compute autocorrelation
        autocorrelation = autocorrelation.reshape(x.shape[0], 1, x.shape[-1])  # Reshape to [batch_size, 1, T]

        x = torch.cat((rppg_fft.unsqueeze(1), rppg_phase.unsqueeze(1), autocorrelation, x), dim=1)  # Concatenate FFT magnitude, phase, and uncertainty

        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = self.pool3(x)
        x = self.block4(x)
        x = self.pool4(x)

        x = x.reshape(x.shape[0], -1)  # Flatten the output to [batch_size, features]

        x = self.linear1(x)  # Apply the final linear layer
        x = self.activation1(x)  # Apply the activation function
        x = self.linear2(x)  # Apply the second linear layer
        x = self.activation2(x)  # Apply the second activation function
        x = self.linear3(x)  # Apply the final linear layer
        x = self.activation3(x)  # Apply the final activation function

        lower = x[:, 0]
        upper = x[:, 1]

        return lower, upper  # Return lower and upper bounds
