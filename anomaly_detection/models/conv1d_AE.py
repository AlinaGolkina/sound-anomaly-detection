import torch
from torch import nn


class Conv1dAE(nn.Module):
    """
    Parameters:
    -----------
    input_size:
        length of amplitude array x

    return:
        x from decoder
        low dimensional feature representation from encoder

    reference:
        https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/conv_autoencoder.py
    """

    def __init__(self, input_size):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 8, 3, stride=1, padding=1),  # output_size = input_size
            nn.MaxPool1d(4, 4),  # output_size = input_size / 4
            nn.ReLU(True),
            nn.Conv1d(8, 16, 3, stride=1, padding=1),  # output_size = input_size / 4
            nn.MaxPool1d(4, 4),  # output_size = input_size / 16
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            nn.Conv1d(16, 32, 3, stride=1, padding=1),  # output_size = input_size /16
            nn.MaxPool1d(4, 4),  # output_size = input_size / 64
            nn.ReLU(True),
            nn.Flatten(start_dim=1),
            nn.Linear(int((input_size / 64) * 32), 128),
            nn.ReLU(True),
            nn.Linear(128, 100),
        )
        self.decoder = nn.Sequential(
            nn.Linear(100, 128),
            nn.ReLU(True),
            nn.Linear(128, int(input_size / 64) * 32),
            nn.ReLU(True),
            nn.Unflatten(dim=1, unflattened_size=(32, int(input_size / 64))),
            nn.ConvTranspose1d(32, 16, 3, stride=4, padding=0, output_padding=1),
            nn.BatchNorm1d(16),
            nn.ConvTranspose1d(16, 8, 3, stride=4, padding=0, output_padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(True),
            nn.ConvTranspose1d(8, 1, 3, stride=4, padding=0, output_padding=1),
        )

    def forward(self, x):
        encoder = self.encoder(x)
        x = self.decoder(encoder)
        return x, encoder
