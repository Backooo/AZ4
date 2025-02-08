from typing import Tuple
import torch
import torch.nn as nn

class AlphaZeroNet(nn.Module):
    def __init__(self, board_shape, num_actions):
        super(AlphaZeroNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(board_shape[0] * board_shape[1] * 2, num_actions),
            nn.Softmax(dim=1)
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(board_shape[0] * board_shape[1], 1),
            nn.Tanh()
        )

    def forward(self, board: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass. Input shape is (B, 1, H, W) where B is batch size, H is height, W is width.

        Args:
            board (torch.Tensor): Board state tensor

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Policy and value tensors
        """
        x = board.float()
        x = self.conv(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value
