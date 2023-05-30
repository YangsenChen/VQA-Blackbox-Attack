import torch
import torch.nn.functional as F


def random_directions(input_shape, num_directions):
    return [torch.randn(input_shape) for _ in range(num_directions)]


def rgf_attack(model, input, label, num_directions, num_steps, epsilon, device):
    input_shape = input.shape
    input = input.to(device)
    label = label.to(device)

    # Make sure the model is in evaluation mode.
    model.eval()

    # Generate random directions.
    directions = random_directions(input_shape, num_directions)
    directions = [direction.to(device) for direction in directions]

    # Initialize minimal perturbation with a large value.
    min_perturbation = torch.full_like(input, fill_value=float('inf')).to(device)
    min_distance = float('inf')

    for direction in directions:
        alpha = torch.linspace(0., 1., num_steps).to(device)

        for a in alpha:
            perturbation = a * direction

            # Clip the perturbation to make sure it's in the valid epsilon-ball.
            perturbation = torch.clamp(perturbation, -epsilon, epsilon)

            adv_input = input + perturbation

            # Ensure the adversarial example is valid.
            adv_input = torch.clamp(adv_input, 0, 1)

            # Compute the output after the perturbation.
            output = model(adv_input)

            # Compute the distance between the output and the target.
            distance = F.cross_entropy(output, label)

            if distance < min_distance:
                min_distance = distance
                min_perturbation = perturbation

    return min_perturbation


import torchvision
import torchvision.transforms as transforms
import torch
from torch.autograd import Variable

# Load a pretrained ResNet model.
model = torchvision.models.resnet50(pretrained=True)
model.eval()

# Load the CIFAR10 dataset.
transform = transforms.Compose(
    [transforms.Resize((224, 224)),  # ResNet requires the input size of (224, 224)
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False, num_workers=2)

# Get a single target from the testloader.
data_iter = iter(testloader)
images, labels = next(data_iter)

# Use a GPU if available.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
images, labels = images.to(device), labels.to(device)

# Verify the classification of the original target.
outputs = model(Variable(images))
_, predicted = torch.max(outputs.data, 1)
print('Predicted before attack: ', predicted.item())

# Apply the RGF attack.
epsilon = 0.03  # Max perturbation for each pixel
num_directions = 20  # Number of random directions
num_steps = 100  # Number of steps

perturbation = rgf_attack(model, images, labels, num_directions, num_steps, epsilon, device)
adv_images = torch.clamp(images + perturbation, 0, 1)

# Verify the classification after the attack.
outputs = model(Variable(adv_images))
_, predicted = torch.max(outputs.data, 1)
print('Predicted after attack: ', predicted.item())
