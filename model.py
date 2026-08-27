# model.py

import torch
import torch.nn as nn

from env import ACTION_DIM


class TinyWorldModelMLP(nn.Module):
    def __init__(
        self,
        img_size: int = 32,
        channels: int = 1,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.img_size = img_size
        self.channels = channels
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.frame_dim = channels * img_size * img_size

        self.action_embed = nn.Embedding(action_dim, hidden_dim)
        
        self.net = nn.Sequential(
            nn.Linear(self.frame_dim+hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.frame_dim)
        )


    def forward(self, frame: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        B, channels, img_size_x, img_size_y = frame.shape
        
        frame_reshaped = frame.flatten(1)
        action_embd = self.action_embed(action) ## B * dim . dim * hidden_dim -> B * hidden_dim
        res = torch.cat([frame_reshaped, action_embd], dim=1)
        out = self.net(res)

        out_reshaped = torch.reshape(out, (B, channels, img_size_x, img_size_y))
        return out_reshaped
    