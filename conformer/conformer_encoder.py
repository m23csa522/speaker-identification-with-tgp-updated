# import torch
# from torch import nn
# from typing import Optional

# from conformer.conformer_block import ConformerBlock
# from conformer.config import Config
# from conformer.self_attention import PositionalEncoder


# class ConformerEncoder(nn.Module):
#     def __init__(self, config: Config):
#         super().__init__()

#         self.linear = nn.Linear(config['hidden_size'] // 4 * (((config['mel_filter_size'] - 1) // 2 - 1) // 2), config['hidden_size'])
#         self.dropout = nn.Dropout(p=0.1)
#         self.positional_encoder = PositionalEncoder(config)
#         self.layers = nn.ModuleList([ConformerBlock(config) for _ in range(config['num_hidden_layers'])])

#     def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
#         """
#         Args:
#             hidden_states (torch.Tensor): with shape `(B, L, D)`
#             attention_mask (torch.Tensor): with shape `(B, L)`

#         Returns:
#             torch.Tensor with shape `(B, L, D)`
#         """
#         hidden_states = self.linear(hidden_states)
#         hidden_states = self.dropout(hidden_states)

#         position_embeddings = self.positional_encoder(hidden_states)

#         for layer in self.layers:
#             hidden_states = layer(
#                 hidden_states,
#                 attention_mask=attention_mask,
#                 position_embeddings=position_embeddings
#             )

#         return hidden_states
import torch
from torch import nn
from typing import Optional

from conformer.conformer_block import ConformerBlock
from conformer.self_attention import PositionalEncoder


class ConformerEncoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        model_config = config
        print(model_config)
        #model_config = config["model"]
        #preproc_config = config["preprocessing"]

        # Compute input dimension after ConvSubsampling
        subsampled_dim = model_config['hidden_size'] // 4 * (((model_config['mel_filter_size'] - 1) // 2 - 1) // 2)

        self.linear = nn.Linear(subsampled_dim, model_config['hidden_size'])
        self.dropout = nn.Dropout(p=0.1)
        self.positional_encoder = PositionalEncoder(model_config)
        self.layers = nn.ModuleList([
            ConformerBlock(model_config) for _ in range(model_config['num_layers'])
        ])

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = self.linear(hidden_states)
        hidden_states = self.dropout(hidden_states)

        position_embeddings = self.positional_encoder(hidden_states)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings
            )

        return hidden_states
