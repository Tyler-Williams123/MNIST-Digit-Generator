import model
from DDIMSampler import sample_ddim
import torch
from matplotlib import pyplot as plt

batchSize = 64
generationModel = model.NoisePredictor(1000)
generationModel.load_state_dict(torch.load("model.pth"))
generationModel.eval()

labels = torch.randint(0, 10, (64,))
clean = sample_ddim(model=generationModel, labels=labels)

plt.imshow(clean[0].detach().numpy().squeeze(0), cmap='gray') #had to detach grad required tensor
plt.axis('off')
plt.show()