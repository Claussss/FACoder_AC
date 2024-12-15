import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from torch.utils.data import DataLoader
import os
import json
from tqdm import tqdm
import torchaudio
from torch.utils.data import Dataset
from einops import rearrange


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        logits = self.fc2(x)
        return logits

class MLPModuleList(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=20):
        super(MLPModuleList, self).__init__()
        self.layers = nn.ModuleList()
        current_dim = input_dim

        for i in range(num_layers):
            # Vary the hidden dimension for each layer
            hidden_dim = current_dim + (i * 2)  # increase by 2 for each layer
            self.layers.append(SimpleMLP(current_dim, hidden_dim, output_dim))
            current_dim = output_dim

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    

class LocalConvModel(nn.Module):
    def __init__(self, input_channels=8, hidden_channels=16, kernel_size=3, num_layers=2):
        super(LocalConvModel, self).__init__()

        layers = []
        in_ch = input_channels
        for i in range(num_layers):
            layers.append(nn.Conv1d(in_ch, hidden_channels, kernel_size, padding=kernel_size//2))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.BatchNorm1d(hidden_channels))
            in_ch = hidden_channels

        # Final projection back to input dimension if needed
        layers.append(nn.Conv1d(hidden_channels, input_channels, kernel_size=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, input_channels, 247)
        return self.model(x)  # (B, input_channels, 247)

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super(TCNBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                               padding=(kernel_size-1)*dilation//2,
                               dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                               padding=(kernel_size-1)*dilation//2,
                               dilation=dilation)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.norm1 = nn.BatchNorm1d(out_ch)
        self.norm2 = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)
        out += residual
        out = F.relu(out)
        return out

class TCNModel(nn.Module):
    def __init__(self, input_channels=8, hidden_channels=16, num_blocks=2, kernel_size=3):
        super(TCNModel, self).__init__()
        layers = []
        in_ch = input_channels
        dilation = 1
        for _ in range(num_blocks):
            layers.append(TCNBlock(in_ch, hidden_channels, kernel_size, dilation))
            in_ch = hidden_channels
            dilation *= 2  # increment dilation to increase receptive field

        # Final projection back to input dimension if desired
        layers.append(nn.Conv1d(hidden_channels, input_channels, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, input_channels, 247)
        return self.model(x) # (B, input_channels, 247)
    
class ACModel(nn.Module):
    def __init__(self, quantizer1_layer, conversion_submodule, d_model=8):
        super().__init__()

        quantizer1_layer.eval() 
        # Freezes in_proj, out_proj, and codebook layers
        for param in quantizer1_layer.parameters():
          param.requires_grad = False  

        # Main layers for downsample and upsample methods
        self.in_proj1 = quantizer1_layer.in_proj
        self.out_proj1 = quantizer1_layer.out_proj

        self.codebook1 = quantizer1_layer.codebook 
        self.codebook1_norm = F.normalize(self.codebook1.weight)  # (1024, D)
        
        self.d_model = d_model
        assert d_model in [8, 256], "d_model can only be 8 or 256"
        self.conversion_submodule = conversion_submodule
        # Check whether submodule outputs the same dimension as the input by passing a dummy input
        # 247 is time axis, the submodule should handle any length of the time axis
        dummy_input = torch.randn(1, d_model, 247)
        dummy_output = self.conversion_submodule(dummy_input)
        assert dummy_output.shape == dummy_input.shape, "Conversion submodule output shape does not match input shape"

    def forward(self, h_input):
        "Returns whether e_c or z_c1 depending on the d_model"
        if self.d_model == 8:
            e_c = self.downsample(h_input)
            e_c = rearrange(e_c, "b t d -> b d t")
            e_c = self.conversion_submodule(e_c)
            # returns (B, d_model, 247)
            # representation before normalization, quantization, upsampling
            return e_c
        elif self.d_model == 256:
            with torch.no_grad():
                z_c1 = self.get_z_c1_from_h(h_input)
            z_c1 = self.conversion_submodule(z_c1)
            # returns (B, d_model, 247)
            return z_c1
        
    def downsample(self, input):
      input = rearrange(input, "b d t -> b t d")
      z_c1_contin = self.in_proj1(input)
      return z_c1_contin

    def upsample(self, c1_emb):
        c1_proj = self.out_proj1(c1_emb)  # (B, T, 256)
        c1_proj = rearrange(c1_proj, "b t d -> b d t")  # (B, 256, T)
        return c1_proj

    def quantize(self, e_c, temperature=1.0):
        logits_c1 = torch.matmul(e_c, self.codebook1_norm.T)

        # Compute soft probabilities
        probs = F.softmax(logits_c1 / temperature, dim=-1)

        # Create a one-hot vector by argmaxing the probs (no gradient here)
        with torch.no_grad():
            indices_c1 = probs.argmax(dim=-1)
            one_hot = torch.zeros_like(probs).scatter_(-1, indices_c1.unsqueeze(-1), 1.0)

        # Straight-through: Forward uses one_hot, backward uses probs
        # (one_hot - probs).detach() + probs is a standard STE trick
        final_probs = (one_hot - probs).detach() + probs

        # Quantize
        c1_emb = final_probs @ self.codebook1.weight
        return c1_emb

    def get_z_c1_from_h(self, h_input):
        e_c = self.downsample(h_input)
        e_c = F.normalize(e_c, dim=-1, eps=1e-8)
        q_c = self.quantize(e_c)
        z_c1 = self.upsample(q_c)
        return z_c1
    
    def inference(self, h_input):
        "Always returns z_c1"
        if self.d_model==256:
            z_c1 = self.forward(h_input)
        elif self.d_model==8:
            e_c = self.forward(h_input)
            e_c = rearrange(e_c, "b d t -> b t d")
            e_c = F.normalize(e_c, dim=-1, eps=1e-8)
            q_c = self.quantize(e_c)
            z_c1 = self.upsample(q_c)
            
        return z_c1