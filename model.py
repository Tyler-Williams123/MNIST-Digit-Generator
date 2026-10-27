import torch.nn as nn
import torch

class NoisePredictor(nn.Module):
    def __init__(self):
        super().__init__()

        self.label_embedding = nn.Embedding(10, 64) # embedding layer for labels 0-9

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        