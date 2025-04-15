# import torch
# from torch import nn
# from torchaudio.transforms import MelSpectrogram

# class WavToMel(nn.Module):
#     def __init__(
#         self,
#         config: dict,
#         noise_scale: float = 1e-4
#     ):
#         super().__init__()
#         self.noise_scale = noise_scale

#         self.mel_sampler = MelSpectrogram(
#             sample_rate=config["preprocessing"]["sampling_rate"],
#             win_length=int(config["preprocessing"]["sampling_rate"] * config["preprocessing"]["win_time"]),
#             hop_length=int(config["preprocessing"]["sampling_rate"] * config["preprocessing"]["stride_time"]),
#             n_fft=config["preprocessing"]["n_fft"],
#             n_mels=config["preprocessing"]["mel_filter_size"]
#         )

#     def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#         """
#         Args:
#             inputs (torch.Tensor): shape `(B, T)`
#             input_lengths (torch.Tensor): shape `(B)`

#         Returns:
#             Tuple:
#               - log-mel features (B, T, D)
#               - adjusted input_lengths (B)
#         """
#         # Add noise for log scaling
#         noise = torch.randn_like(inputs) * self.noise_scale
#         inputs = inputs + noise

#         mel_feature = self.mel_sampler(inputs)
#         log_mel_feature = mel_feature.clamp(min=1e-5).log2()

#         # Transpose to (B, T, D)
#         log_mel_feature = log_mel_feature.transpose(-1, -2)

#         # Update lengths after subsampling
#         input_lengths = input_lengths // (inputs.size(-1) / log_mel_feature.size(-2))

#         return log_mel_feature, input_lengths

import torch
from torch import nn
from torchaudio.transforms import MelSpectrogram

class WavToMel(nn.Module):
    def __init__(
        self,
        config,
        noise_scale: float = 1e-4
    ):
        super().__init__()
        self.noise_scale = noise_scale

        #preprocessing = config["preprocessing"]

        self.mel_sampler = MelSpectrogram(
            sample_rate=config["preprocessing"]["sampling_rate"],
            win_length=int(config["preprocessing"]["sampling_rate"] * config["preprocessing"]["win_time"]),
            hop_length=int(config["preprocessing"]["sampling_rate"] * config["preprocessing"]["stride_time"]),
            n_fft=config["preprocessing"]["n_fft"],
            n_mels=config["preprocessing"]["mel_filter_size"]
        )

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs (torch.Tensor): shape `(B, T)`
            input_lengths (torch.Tensor): shape `(B)`

        Returns:
            Tuple:
              - log-mel features (B, T, D)
              - adjusted input_lengths (B)
        """
        noise = torch.randn_like(inputs) * self.noise_scale
        inputs = inputs + noise

        mel_feature = self.mel_sampler(inputs)
        log_mel_feature = mel_feature.clamp(min=1e-5).log2()

        # Transpose to (B, T, D)
        log_mel_feature = log_mel_feature.transpose(-1, -2)

        # Update lengths after subsampling
        input_lengths = input_lengths // (inputs.size(-1) / log_mel_feature.size(-2))

        return log_mel_feature, input_lengths

