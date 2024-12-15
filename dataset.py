import os
import torch
from torch.utils.data import Dataset
import json
from tqdm import tqdm
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader


class AudioDataset(Dataset):
    def __init__(self, base_path, sr=16000, max_length=3.4, preload=True, path_to_selected_files=''):
        """
        Args:
            base_path (str): Base path containing 'synthetic_wav' and 'wav' directories.
            sr (int): Target sample rate for all audio files.
            max_length (float): Maximum duration (in seconds) for audio files.
            preload (bool): Whether to preload all audio data into memory.
            path_to_selected_files (str): Path to a JSON file containing a list of selected file names. Needed to choose a subset
        """
        self.base_path = base_path
        self.sr = sr
        self.max_samples = int(max_length * sr)
        self.synthetic_path = os.path.join(base_path, "syntetic_wav")
        self.wav_path = os.path.join(base_path, "wav")

        if path_to_selected_files:
          with open(path_to_selected_files, "r") as f:
            clean_files = json.load(f)

        self.synthetic_files = [
            f for f in os.listdir(self.synthetic_path) if f.endswith(".wav")
        ]

        if path_to_selected_files:
          self.synthetic_files = [f for f in self.synthetic_files if f in clean_files]

        self.synthetic_files.sort()

        # Preload all audio data into memory
        self.synthetic_audio = []
        self.original_audio = []

        if preload:
          for synthetic_file in tqdm(self.synthetic_files):
              synthetic_filepath = os.path.join(self.synthetic_path, synthetic_file)
              wav_filepath = os.path.join(self.wav_path, synthetic_file)

              # Load synthetic wav. They are all already trimmed/padded.
              synthetic_waveform, _ = torchaudio.load(synthetic_filepath)

              # Load original wav and resample and trim/pad if needed
              wav_waveform, wav_sr = torchaudio.load(wav_filepath)
              if wav_sr != self.sr:
                  resample = T.Resample(orig_freq=wav_sr, new_freq=self.sr)
                  wav_waveform = resample(wav_waveform)
              wav_waveform = self._pad_or_trim(wav_waveform)

              self.synthetic_audio.append(synthetic_waveform)
              self.original_audio.append(wav_waveform)

    def __len__(self):
        """Return the number of synthetic wav files."""
        return len(self.synthetic_files)

    def __getitem__(self, idx):
        """Return preloaded synthetic (z_c_groundtruth) and original (z_c) audio."""
        return self.synthetic_audio[idx], self.original_audio[idx]

    def _pad_or_trim(self, waveform):
        """Pad or trim waveform to ensure it is max_samples long."""
        if waveform.size(1) > self.max_samples:
            # Trim to max length
            return waveform[:, :self.max_samples]
        elif waveform.size(1) < self.max_samples:
            # Pad with zeros
            padding = self.max_samples - waveform.size(1)
            return torch.cat([waveform, torch.zeros((waveform.size(0), padding))], dim=1)
        return waveform
    

def precompute_latents_in_batches(audio_dataset, fa_encoder, fa_decoder, batch_size=16, device='cuda', output_path='.\latent_data.pth'):
    """
    Precompute latent variables in batches using a DataLoader.

    Args:
        audio_dataset: Instance of AudioDataset.
        fa_encoder: Encoder module to compute initial latent representation.
        fa_decoder: Decoder containing the quantizer.
        batch_size: Batch size for processing.
        device: Device to run computations ('cuda' or 'cpu').

    Returns:
        precomputed_data: List of tuples (input_latent, target_latent).
    """
    # Create DataLoader for batch processing
    dataloader = DataLoader(audio_dataset, batch_size=batch_size, shuffle=False)

    precomputed_data = []

    # Ensure modules are in evaluation mode
    fa_encoder.eval()
    fa_decoder.quantizer[1].eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Precomputing latents"):
            synthetic_audio, original_audio = batch  # Synthetic (target), then original (input)
            synthetic_audio = synthetic_audio.to(device)
            original_audio = original_audio.to(device)

            h_input = fa_encoder(original_audio)
            z_c_input, input_q, _, _ = fa_decoder.quantizer[1](h_input)
            input_q_permuted = input_q.permute(1,0,2)

            h_target = fa_encoder(synthetic_audio)
            z_c_target, target_q, _, _ = fa_decoder.quantizer[1](h_target)
            target_q_permuted = target_q.permute(1,0,2)

            for i in range(h_input.shape[0]):
                precomputed_data.append((h_input[i].cpu(), h_target[i].cpu(), 
                                         input_q_permuted[i].cpu(), target_q_permuted[i].cpu()))

    # Save the precomputed data for reuse
    torch.save(precomputed_data, output_path)
    return precomputed_data


class LatentDataset(Dataset):
    def __init__(self, path_to_precomputed_latent_dataset):
        # a list of tuples: ((h_input, z_c_synth, input_indx, target_indx), ...)
        self.precomputed_data = torch.load(path_to_precomputed_latent_dataset)  

    def __len__(self):
        return len(self.precomputed_data)

    def __getitem__(self, idx):
        h_input = self.precomputed_data[idx][0]
        h_target = self.precomputed_data[idx][1]
        # shape: (2, T), first row c1, second row c2
        indices_input = self.precomputed_data[idx][2] 
        indices_target = self.precomputed_data[idx][3]
        return h_input, h_target, indices_input, indices_target