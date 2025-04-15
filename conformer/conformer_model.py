# import torch
# from torch import nn

# #from conformer.config import Config
# from conformer.conformer_encoder import ConformerEncoder
# from conformer.conformer_subsampling import ConvSubsampling

import torch
from torch import nn

from conformer.conformer_encoder import ConformerEncoder
from conformer.conformer_subsampling import ConvSubsampling

class ConformerModel(nn.Module):
    def __init__(self, model_config: dict):
        super().__init__()

        self.subsampling_conv = ConvSubsampling(model_config)

        # Just pass model config to encoder
        self.encoder = ConformerEncoder(model_config)

    def forward(self, input_values: torch.Tensor, input_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        # Subsampling step
        hidden_states, input_lengths = self.subsampling_conv(input_values, input_lengths)
        print("✅ After ConvSubsampling:", hidden_states.shape if hidden_states is not None else "None")

        batch_size, length, _ = hidden_states.size()
        range_tensor = torch.arange(length).unsqueeze(0).repeat(batch_size, 1).to(hidden_states.device)
        attention_mask = (range_tensor < input_lengths.unsqueeze(1)).to(hidden_states.device)

        # Encoder step
        hidden_states = self.encoder(hidden_states, attention_mask)
        print("✅ After ConformerEncoder:", hidden_states.shape if hidden_states is not None else "None")


        # hidden_states, input_lengths = self.subsampling_conv(input_values, input_lengths)

        # batch_size, length, _ = hidden_states.size()
        # range_tensor = torch.arange(length).unsqueeze(0).repeat(batch_size, 1).to(hidden_states.device)
        # attention_mask = (range_tensor < input_lengths.unsqueeze(1)).to(hidden_states.device)

        # hidden_states = self.encoder(hidden_states, attention_mask)

        return hidden_states, input_lengths

# import torch
# from torch import nn

# from conformer.conformer_encoder import ConformerEncoder
# from conformer.conformer_subsampling import ConvSubsampling


# class ConformerModel(nn.Module):
#     def __init__(self, config: dict):
#         super().__init__()
#         #self.subsampling_conv = ConvSubsampling(config["model"])
#         self.subsampling_conv = ConvSubsampling(config)
        
#         # Pass both model and preprocessing config to encoder
#         encoder_config = {
#             "model": config["model"],
#             "preprocessing": config["preprocessing"]
#         }
#         self.encoder = ConformerEncoder(encoder_config)

#     def forward(self, input_values: torch.Tensor, input_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#         hidden_states, input_lengths = self.subsampling_conv(input_values, input_lengths)

#         batch_size, length, _ = hidden_states.size()
#         range_tensor = torch.arange(length).unsqueeze(0).repeat(batch_size, 1).to(hidden_states.device)
#         attention_mask = (range_tensor < input_lengths.unsqueeze(1)).to(hidden_states.device)

#         hidden_states = self.encoder(hidden_states, attention_mask)

#         return hidden_states, input_lengths
