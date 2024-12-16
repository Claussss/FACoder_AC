import matplotlib.pyplot as plt
import torch
import librosa

def get_total_params(model):
  total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  return total_params

def plot_waveforms_aligned(groundtruth_waveform, fixed_waveform):

  time_audio_norm = range(len(groundtruth_waveform))
  time_waveform = range(len(fixed_waveform))


  plt.figure(figsize=(10, 4))
  plt.plot(time_audio_norm, groundtruth_waveform, label="Groundtruth", alpha=0.7)
  plt.plot(time_waveform, fixed_waveform, label="Fixed Accent", alpha=0.7)
  plt.xlabel("Sample Index")
  plt.ylabel("Amplitude")
  plt.title("Waveforms: Groundtruth vs Fixed Accent")
  plt.legend()
  plt.grid()
  plt.show()

def plot_losses(train_losses, valid_losses, validation_interval):
    """
    Plots the training and validation losses over epochs.

    Args:
        train_losses (list): List of average training losses.
        valid_losses (list): List of average validation losses.
        validation_interval (int): Interval at which validation was performed.
    """
    # Create an array of epoch numbers for training losses
    epochs_train = list(range(1, len(train_losses) + 1))

    # Create an array of epoch numbers for validation losses
    epochs_valid = [i for i in range(validation_interval, len(train_losses) + 1, validation_interval)]

    plt.figure(figsize=(10, 6))

    # Plot training losses
    plt.plot(epochs_train, train_losses, label='Training Loss', color='blue', linestyle='-', marker='o')

    # Plot validation losses
    plt.plot(epochs_valid, valid_losses, label='Validation Loss', color='red', linestyle='-', marker='x')

    # Adding labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()

    # Adding grid for better readability
    plt.grid(True)

    # Show the plot
    plt.show()

def load_prepare_audio(audio_path, sr=16000, target_lenght=3.4, tensor=True):
  target_lenght_samples = int(target_lenght * sr)
  audio, _ = librosa.load(audio_path, sr=sr)
  audio =  librosa.util.fix_length(audio, size=target_lenght_samples)
  if tensor:
    audio = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)
  return audio

def unprepare_audio(audio_tensor):
  return audio_tensor.squeeze(0).squeeze(0).cpu().numpy()
