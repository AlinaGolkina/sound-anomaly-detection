import torch
from torch import nn


class FCAE(nn.Module):
    """
    Parameters:
    -----------
    input_size:
        length of amplitude array x

    return:
        x from decoder
        low dimensional feature representation from encoder

    """

    def __init__(self, input_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 16),
            nn.LeakyReLU(0.2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, input_size),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        encoder = self.encoder(x)
        x = self.decoder(encoder)
        return x, encoder
