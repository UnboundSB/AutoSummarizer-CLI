# File: summerizer/model.py

import torch
import torch.nn as nn

class SummaryANN(nn.Module):
    """
    A simple feedforward neural network for text summarization.
    Input: Encoded document vector
    Output: Encoded summary vector
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(SummaryANN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)
