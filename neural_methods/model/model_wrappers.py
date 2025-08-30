
import torch
from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from neural_methods.model.PhysNetLifeness import PhysNet_Lifeness
from neural_methods.model.PhysNetQuantile import PhysNet_Quantile
from neural_methods.model.PhysNetUncertainty import PhysNet_Uncertainty

class PhysNetUncertainty_Wrapper(torch.nn.Module):
    def __init__(self, model=None, frames=128, in_channels=3, device='cpu'):
        super(PhysNetUncertainty_Wrapper, self).__init__()

        if model is None:
            model = PhysNet_Uncertainty(frames=frames, in_channels=in_channels).to(device)
            # Load pre-trained weights if available
            # model.load_state_dict(torch.load('path_to_weights.pth', map_location='cpu'))

        self.quant = torch.quantization.QuantStub()
        self.model = model
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        rPPG, uncertainty = self.model(x)
        rPPG = self.dequant(rPPG)
        return rPPG, uncertainty
    
class PhysNet_Wrapper(torch.nn.Module):
    def __init__(self, model=None, frames=128, in_channels=3, device='cpu'):
        super(PhysNet_Wrapper, self).__init__()

        if model is None:
            model = PhysNet_padding_Encoder_Decoder_MAX(frames=frames, in_channels=in_channels).to(device)
            # Load pre-trained weights if available
            # model.load_state_dict(torch.load('path_to_weights.pth', map_location='cpu'))

        self.quant = torch.quantization.QuantStub()
        self.model = model
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        rPPG, x_visual, x_visual3232, x_visual1616 = self.model(x)
        rPPG = self.dequant(rPPG)
        return rPPG