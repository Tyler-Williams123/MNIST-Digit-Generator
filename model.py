import torch.nn as nn
import torch

#using post activation residual blocks because i dont want to customize the first layer.
#for now im keeping 8 group norm and adding embeds at the beginning but i may come back later
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, group_norm_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm = nn.GroupNorm(group_norm_groups, out_channels)
        self.nonlinear = nn.SiLU()
        
        if in_channels != out_channels:
            self.identityConv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        else:
            self.identityConv = nn.Identity()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm(x)
        x = self.nonlinear(x)

        x = self.conv2(x)
        x = self.norm(x)

        residual = self.identityConv(residual)
        return x + residual

class upsample(nn.Module): #learn what ConvTranspose2d does and how it works.
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class NoisePredictor(nn.Module):
    def __init__(self, maxTimeSteps=1000):
        super().__init__()

        self.labelEmbedding = nn.Embedding(10, 64) # embedding for labels 0-9
        self.timeEmbedding = nn.Embedding(maxTimeSteps, 64) # embedding for time steps

        self.emb_proj1 = nn.Linear(128, 1)
        self.emb_proj64 = nn.Linear(128, 64)
        self.emb_proj128 = nn.Linear(128, 128)
        self.emb_proj256 = nn.Linear(128, 256)

        self.nonlinear = nn.SiLU()
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        #downsampling
        self.Res1 = ResidualBlock(1, 64)
        self.Res2 = ResidualBlock(64, 128)
        
        #bottleneck
        self.Res3 = ResidualBlock(128, 256)
        self.Res4 = ResidualBlock(256, 128)

        #upsampling
        self.up1 = upsample(128, 128)
        self.Res5 = ResidualBlock(256, 64)
        self.up2 = upsample(64, 64)
        self.Res6 = ResidualBlock(128, 1, group_norm_groups=1) #final layer has 1 channel so no need for group norm

    def forward(self, x, t, labels):
        label_emb = self.labelEmbedding(labels) # get label embeddings
        time_emb = self.timeEmbedding(t) # get time embeddings

        conditional_emb = torch.cat((label_emb, time_emb), dim=1)

        #encoding
        x = x + self.emb_proj1(conditional_emb)[:, :, None, None]
        x = self.Res1(x)
        x = self.nonlinear(x)
        skip1 = x

        x = self.maxPool(x) #downsample

        x = x + self.emb_proj64(conditional_emb)[:, :, None, None] #add conditional embedding
        x = self.Res2(x)
        x = self.nonlinear(x)
        skip2 = x

        x = self.maxPool(x)

        #bottleneck
        x = x + self.emb_proj128(conditional_emb)[:, :, None, None]
        x = self.Res3(x)
        x = self.nonlinear(x)
       
        x = x + self.emb_proj256(conditional_emb)[:, :, None, None]
        x = self.Res4(x)
        x = self.nonlinear(x)

        #upsampling
        x = self.up1(x)
        x = x + self.emb_proj128(conditional_emb)[:, :, None, None]
        x = torch.cat((x, skip2), dim=1) #skip connection
        x = self.Res5(x)
        x = self.nonlinear(x)

        x = self.up2(x)
        x = x + self.emb_proj64(conditional_emb)[:, :, None, None]
        x = torch.cat((x, skip1), dim=1)
        x = self.Res6(x)

        return x