from typing import List, Dict, Tuple
import torch
import torch.nn as nn

from RL.CURL.utils import tie_weights


# (W - F + 2P) / S + 1
# out_fig size for 84 * 84 fig
OUT_DIM = {2: 39, 4: 35, 6: 31}
# out_fig size for 64 * 64 fig
OUT_DIM_64 = {2: 29, 4: 25, 6: 21}


class PixelEncoder(nn.Module):
    def __init__(
        self,
        obs_shape: List[int],
        feature_dim: int,
        num_layers: int = 2,
        num_filters: int = 32,
        output_logits: bool = False
    ) -> None:
        super().__init__()
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.output_logits = output_logits

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, kernel_size=3, stride=2)]
        )
        for _ in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, 1))
        
        out_fig_size = OUT_DIM[num_layers] if obs_shape[-1] == 84 else OUT_DIM_64[num_layers]

        self.fc = nn.Linear(out_fig_size ** 2 * num_filters, feature_dim)
        self.ln = nn.LayerNorm(feature_dim)

        self.outputs= dict()

    def encode(self, obs: torch.tensor) -> torch.tensor:
        obs = obs / 255.
        self.outputs['obs'] = obs
        conv = obs
        for i in range(self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs[f'conv_{i+1}'] = conv
        
        h = conv.view(conv.size(0), -1)
        return h

    def __forward__(self, obs: torch.tensor, detach_conv: bool = False) -> torch.tensor:
        h = self.encode(obs)
        if detach_conv:
            h = h.detach()

        h_fc = self.fc(h)
        h_norm = self.ln(h_fc)
        self.outputs['fc'], self.outputs['ln'] = h_fc, h_norm

        if self.output_logits:
            out = h_norm
        else:
            out = torch.tanh(h_norm)
            self.outputs['tanh'] = out

        return out

    def tie_conv_to_encoder(self, src_encoder) -> None:
        for i in range(self.num_layers):
            tie_weights(trg = src_encoder.convs[i], src = self.convs[i])