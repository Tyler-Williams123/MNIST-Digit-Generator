import torch.nn as nn
import torch

#for now im keeping 8 group norm and adding embeds at the beginning but i may come back later
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, group_norm_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm = nn.GroupNorm(group_norm_groups, out_channels)
        self.nonlinear = nn.SiLU()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm(x)
        x = self.nonlinear(x)
        x = self.conv2(x)
        x = self.norm(x)
        return x + residual

class NoisePredictor(nn.Module):
    def __init__(self, maxTimeSteps=1000):
        super().__init__()

        self.labelEmbedding = nn.Embedding(10, 64) # embedding for labels 0-9
        self.timeEmbedding = nn.Embedding(maxTimeSteps, 64) # embedding for time steps

        self.emb_proj64 = nn.Linear(128, 64)
        self.emb_proj128 = nn.Linear(126, 128)
        self.emb_proj256 = nn.Linear(128, 256)

        self.nonlinear = nn.SiLU()
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Res1 = ResidualBlock(1, 64)
        self.Res2 = ResidualBlock(64, 128)
        self.Res3 = ResidualBlock(128, 256)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t, labels):
        label_emb = self.labelEmbedding(labels) # get label embeddings
        time_emb = self.timeEmbedding(t) # get time embeddings

        label_emb = label_emb.view(-1, 128, 1, 1) # reshape embeddings to match xs dimensions
        time_emb = time_emb.view(-1, 128, 1, 1) # reshape embeddings to match times dimensions

        conditional_emb = torch.cat((label_emb, time_emb), dim=1)
        risidual = x

        #downsampling
        x = self.conv1(x)
        x = self.nonlinear(x + self.emb_proj64(conditional_emb)) # add embeddings to input image
        x = self.maxPool(x)

        x = self.conv2(x)
        x = self.nonlinear(x + self.emb_proj128(conditional_emb))
        x = self.maxPool(x)

        x = self.conv3(x)
        x = self.nonlinear(x + self.emb_proj256(conditional_emb))

        #upsampling
        x = self.conv4(x)
        x = self.nonlinear(x)

        return x + risidual