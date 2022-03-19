from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from RL.CURL.encoder import PixelEncoder


class CURL_model(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        encoder: PixelEncoder,
        encoder_trg: PixelEncoder,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.encoder_trg = encoder_trg
        self.W = nn.Parameter(torch.rand((feature_dim, feature_dim)))

    def compute_logits(self, z_a: torch.tensor, z_pos: torch.tensor) -> torch.tensor:
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        Wz = torch.matmul(self.W, z_pos.T)
        logits = torch.matmul(z_a, Wz)
        logits -= torch.max(logits, 1)[0][:, None]
        return logits