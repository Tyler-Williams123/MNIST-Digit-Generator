import torch.nn as nn
import torch

class NoisePredictor(nn.Module):
    def __init__(self, maxTimeSteps=1000):
        super().__init__()

        self.labelEmbedding = nn.Embedding(10, 128) # embedding for labels 0-9
        self.timeEmbedding = nn.Embedding(maxTimeSteps, 128) # embedding for time steps

        self.nonlinear = nn.ReLU()

        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t, labels):
        label_emb = self.labelEmbedding(labels) # get label embeddings
        time_emb = self.timeEmbedding(t) # get time embeddings

        label_emb = label_emb.view(-1, 128, 1, 1) # reshape embeddings to match xs dimensions
        time_emb = time_emb.view(-1, 128, 1, 1) # reshape embeddings to match times dimensions

        conditional_emb = label_emb + time_emb

        x = self.conv1(x) # transforms shape to match context embeddings
        x = self.nonlinear(x + conditional_emb) # add embeddings to input image

        x = self.conv2(x)
        x = self.nonlinear(x + conditional_emb) # add embeddings to input image

        x = self.conv3(x)
        x = self.nonlinear(x + conditional_emb) # add embeddings to input image

        x = self.conv4(x)

        return x