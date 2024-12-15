from torchaudio.transforms import MelSpectrogram
import torch
import torch.nn.functional as F

LAMBDA_WAV = 100

def reconstruction_loss(x, G_x, eps=1e-7):
    # NOTE (lsx): hard-coded now
    L = LAMBDA_WAV * F.mse_loss(x, G_x)  # wav L1 loss
    # loss_sisnr = sisnr_loss(G_x, x) #
    # L += 0.01*loss_sisnr
    # 2^6=64 -> 2^10=1024
    # NOTE (lsx): add 2^11
    for i in range(6, 12):
        # for i in range(5, 12): # Encodec setting
        s = 2**i
        melspec = MelSpectrogram(
            sample_rate=16000,
            n_fft=max(s, 512),
            win_length=s,
            hop_length=s // 4,
            n_mels=64,
            wkwargs={"device": G_x.device}).to(G_x.device)
        S_x = melspec(x)
        S_G_x = melspec(G_x)
        l1_loss = (S_x - S_G_x).abs().mean()
        l2_loss = (((torch.log(S_x.abs() + eps) - torch.log(S_G_x.abs() + eps))**2).mean(dim=-2)**0.5).mean()

        alpha = (s / 2) ** 0.5
        L += (l1_loss + alpha * l2_loss)
    return L