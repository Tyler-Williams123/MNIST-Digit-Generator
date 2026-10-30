import torch
import diffusion
from model import NoisePredictor
from matplotlib import pyplot as plt

def sample_ddim(model, labels, steps=50, device="cpu"):
    model.eval()

    labels = labels.to(device)
    batch_size = len(labels)

    x = torch.randn(batch_size, 1, 28, 28, device=device)

    alpha_bar = diffusion.alpha_bar.to(device) # alpha bar is cumulitave product of alpha

    timesteps = torch.linspace(diffusion.time - 1, 0, steps).round().long().tolist() # linear space creates a list of timesteps from diffusion.time - 1 to 0 with 'steps' number of points

    for i, time in enumerate(timesteps):
        t = torch.full((batch_size,), time, dtype=torch.long, device=device) # creates a tensor of shape (batch_size,) filled with the current timestep value

        # Predict the noise in the current image
        predicted_noise = model(x, t, labels)

        alpha_bar_t = alpha_bar[time]

        # The next selected timestep; 1.0 represents a clean image
        if i + 1 < len(timesteps):
            alpha_next = alpha_bar[timesteps[i + 1]]
        else:
            alpha_next = torch.tensor(1.0, device=device)

        # Estimate the original image
        predicted_x0 = (x - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)

        predicted_x0 = predicted_x0.clamp(-1, 1)

        x = (torch.sqrt(alpha_next) * predicted_x0 + torch.sqrt(1 - alpha_next) * predicted_noise) #turns predicted_x0 and predicted_noise into the next image in the diffusion process

    return x.clamp(-1, 1)
