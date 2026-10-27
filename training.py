import torch
import diffusion
from model import NoisePredictor
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

batchSize = 64
lossFunction = torch.nn.MSELoss()
model = NoisePredictor(maxTimeSteps=1000)
optim = torch.optim.Adam(params=model.parameters(), lr=0.001)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0, 1)
])

dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

loader = DataLoader(dataset, batch_size=batchSize, shuffle=True)

batchNum = 0
for batch in loader:
    batchNum += 1
    images, labels = batch
    t = torch.randint(0, diffusion.time, (images.size(0),))  # Randomly select time steps for each image in the batch
    xt, noise = diffusion.diffuse(images, t)  # Diffuse the images at the selected time steps

    #forward pass
    predicted_noise = model(xt, t, labels)
    loss = lossFunction(predicted_noise, noise)  # Compute the loss between the predicted noise and the actual noise

    loss.backward()
    optim.step()
    optim.zero_grad()

    if batchNum % 100 == 0:
        print(f'Batch {batchNum}, Loss: {loss.item()}')

torch.save(model.state_dict(), 'model.pth')