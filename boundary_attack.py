import torch
import torchvision.models as models
import foolbox as fb

# Load the pretrained ResNet model
model = models.resnet50(pretrained=True).eval()

# Ensure the model is on the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define the preprocessing. Note that this may vary based on your model.
preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)

# Wrap the model with Foolbox model
fmodel = fb.PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

# Create a test target and label (replace this with your actual data)
image = torch.randn(1, 3, 224, 224).to(device)
# Assume target is a PyTorch tensor
image = image.clamp(0, 1)
image = image.type(torch.float32)  # ensure the target is float32
label = torch.tensor([6]).to(device)  # assuming the label is 6 for this example

# Define the attack
attack = fb.attacks.BoundaryAttack(
    init_attack=None,
    steps=5000,
    spherical_step=0.01,
    source_step=0.01,
    step_adaptation=1.5,
)

# Set the criterion
criterion = fb.criteria.Misclassification(labels=label)

# Perform the attack
epsilons = [0.01]  # maximum perturbation
_, advs, success = attack(fmodel, image, criterion, epsilons=epsilons)

# Print the labels
print("Original label: ", label.item())
print("Adversarial label: ", fmodel(advs[0]).argmax().item())
