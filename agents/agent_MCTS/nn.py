import torch
import torch.nn as nn
import torch.nn.functional as F

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
            nn.Linear(board_shape[0] * board_shape[1] * 2, num_actions),
            nn.Softmax(dim=1)
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_shape[0] * board_shape[1], 1),
            nn.Tanh()
        )

    def forward(self, board):
        x = board.unsqueeze(1).float()  # Add channel dimension
        x = self.conv(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value
