import model
from DDIMSampler import sample_ddim
import torch
from matplotlib import pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

batchSize = 1
generationModel = model.NoisePredictor(1000)
generationModel.load_state_dict(torch.load("model.pth"))
generationModel.eval().to(device)

labels = torch.randint(0, 10, (1,))
clean = sample_ddim(model=generationModel, labels=labels, steps=20, device=device)
clean = (clean + 1) / 2 #transform from [-1, 1] to [0, 1] for visualization

plt.imshow(clean[0].cpu().detach().numpy().squeeze(0), cmap='gray') #had to detach grad required tensor
plt.title(f"Generated Image for Label: {labels.item()}")
plt.axis('off')
plt.show()