import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

batchSize = 64

time = 1000
beta = torch.linspace(1e-4, 0.02, time) # beta controlls amount of noise added to the image at each time step(high beta = more noise)
alpha = 1.0 - beta #alpha is percent of info remaining in the image at each time step
alpha_bar = torch.cumprod(alpha, dim=0) # also known as variance schedule

def diffuse(x0, t):
    """
    Diffuses the input image x0 at time step t by adding noise based on the variance schedule.
    Args:
        x0 (torch.Tensor): The input image tensor of shape (batch_size, channels, height, width).
        t (int): The time step at which to diffuse the image.
    """
    noise = torch.randn_like(x0) # generate noise with the same shape as the input image

    #uses view to rehape alpha_bar[t] to match the dimensions of x0 for broadcasting
    xt = torch.sqrt(alpha_bar[t]) * x0 + torch.sqrt(1 - alpha_bar[t].view(-1, 1, 1, 1)) * noise # takes original image and adds noise based on variance schedule and t.
    return xt, noise

# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(0, 1)
# ])

# dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# loader = DataLoader(dataset, batch_size=batchSize, shuffle=True)

# iterator = iter(loader)

# images, labels = next(iterator)

# xt = diffuse(images, t=500)

# plt.imshow(images[0].permute(1, 2, 0).squeeze(), cmap='gray')
# plt.title(f'Big {labels[0].item()} Energy')
# plt.axis('off')
# plt.show()

# plt.imshow(xt[0].permute(1, 2, 0).squeeze(), cmap='gray')
# plt.title(f'Big {labels[0].item()} Energy')
# plt.axis('off')
# plt.show()