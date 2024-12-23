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


def identity_like_error(model):
    """
    Check how close convolutional layers in the model are to an identity-like transformation.
    Prints out a measure of deviation from identity for each Conv1d layer.
    """
    for i, layer in enumerate(model.modules()):
        if isinstance(layer, nn.Conv1d):
            weight = layer.weight.data  # shape: [out_channels, in_channels, kernel_size]
            out_ch, in_ch, k = weight.shape

            # Construct an ideal identity-like weight
            # For identity in a Conv1d:
            # - If kernel_size > 1, we expect the center tap to be 1 for out_ch == in_ch, and 0 elsewhere.
            # - If kernel_size == 1, we expect something like a linear identity (diagonal matrix) if out_ch == in_ch.

            ideal_weight = torch.zeros_like(weight)

            if out_ch == in_ch:
                # For kernel_size > 1:
                # Put 1 at the center tap of the kernel for the matching in/out channel pair
                if k > 1:
                    center = k // 2
                    for c in range(out_ch):
                        ideal_weight[c, c, center] = 1.0
                else:
                    # kernel_size == 1, we want a diagonal identity mapping
                    # W[c,c,0] = 1
                    for c in range(out_ch):
                        ideal_weight[c, c, 0] = 1.0
            else:
                # If out_ch != in_ch, there's no straightforward identity mapping.
                # The ideal in this scenario could be zero (doing nothing),
                # or you can skip measuring this if an identity doesn't conceptually apply.
                # Here we'll just consider the zero matrix as the "ideal" since no direct identity is possible.
                # You may also choose to skip printing a metric in this case.
                pass

            # Compute a mean squared error (MSE) with the ideal pattern
            mse = torch.mean((weight - ideal_weight) ** 2).item()

            print(f"Layer {i} ({layer}):")
            print(f"  Weight shape: {weight.shape}")
            print(f"  Identity-like MSE: {mse:.6f}\n")