from transformers import AutoProcessor, Pix2StructForConditionalGeneration
import requests
from PIL import Image

model = Pix2StructForConditionalGeneration.from_pretrained("google/matcha-chartqa").to(0)
processor = AutoProcessor.from_pretrained("google/matcha-chartqa")
url = "https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/20294671002019.png"
image = Image.open("20294671002019.png")

inputs = processor(images=image, text="Is the sum of all 4 places greater than Laos?", return_tensors="pt").to(0)
predictions = model.generate(**inputs, max_new_tokens=512)
print(processor.decode(predictions[0], skip_special_tokens=True))


import torch
import torch.nn.functional as F


def random_directions(input_shape, num_directions):
    return [torch.randn(input_shape) for _ in range(num_directions)]


def rgf_attack(model, input, label, num_directions, num_steps, epsilon, device):
    input_shape = input.shape
    input = input.to(device)
    label = label.to(device)


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
            # todo - change this to pix2struct

            # Compute the distance between the output and the target.
            distance = F.cross_entropy(output, label)

            if distance < min_distance:
                min_distance = distance
                min_perturbation = perturbation

    return min_perturbation

