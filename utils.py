import matplotlib.pyplot as plt

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

