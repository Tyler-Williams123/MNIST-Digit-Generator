import torch.nn as nn
import torch

class NoisePredictor(nn.Module):
    def __init__(self, maxTimeSteps=1000):
        super().__init__()

        self.labelEmbedding = nn.Embedding(10, 64) # embedding for labels 0-9
        self.timeEmbedding = nn.Embedding(maxTimeSteps, 64) # embedding for time steps

        self.nonlinear = nn.ReLU()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t, labels):
        label_emb = self.labelEmbedding(labels) # get label embeddings
        time_emb = self.timeEmbedding(t) # get time embeddings

        label_emb = label_emb.view(-1, 64, 1, 1) # reshape embeddings to match xs dimensions
        time_emb = time_emb.view(-1, 64, 1, 1) # reshape embeddings to match times dimensions

        x = self.nonlinear(self.conv1(x)) # trnasforms shape to match context embeddings

        x = x + label_emb + time_emb # add embeddings to input image

        x = self.nonlinear(self.conv2(x))
        x = self.nonlinear(self.conv3(x))
        x = self.conv4(x)

        return x