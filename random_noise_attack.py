import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import foolbox as fb
from torchvision import models

# Ensure we're using the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a built-in PyTorch model (e.g., VGG16)
model = models.vgg16(pretrained=False)  # set pretrained=True if you want a pretrained model
model = model.to(device)
model = model.eval()

# Load CIFAR-10 dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

testset = datasets.CIFAR10(root='./data', train=False,
                           download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=True, num_workers=2)

# Wrap the PyTorch model with a Foolbox model
fmodel = fb.PyTorchModel(model, bounds=(-1, 1))

# Choose an target to attack
for data in testloader:
    images, labels = data
    break

images = images.to(device)
labels = labels.to(device)

# Create the attack
attack = fb.attacks.LinearSearchBlendedUniformNoiseAttack(directions=1000, steps=1000)

# Apply the attack
advs = attack.run(fmodel, images, labels)

# Check if the attack was successful
# print("Attack success:", success.float().mean())

import matplotlib.pyplot as plt
import numpy as np

# Function to unnormalize and convert tensor target to numpy
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # Convert (C, H, W) to (H, W, C)
    plt.show()

# Get the adversarial target tensor
adv = advs[0].cpu()

# Plot original target
print('Original Image')
imshow(images[0].cpu())

# Plot adversarial target
print('Adversarial Image')
imshow(adv)

