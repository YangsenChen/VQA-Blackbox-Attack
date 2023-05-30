import numpy as np
from transformers import AutoProcessor, Pix2StructForConditionalGeneration
import requests
from PIL import Image

model = Pix2StructForConditionalGeneration.from_pretrained("google/matcha-chartqa")
processor = AutoProcessor.from_pretrained("google/matcha-chartqa")
image = Image.open("20294671002019.png")
image = np.array(image) / 255.0

# def randomNoiseAttack(input_image, model, processor, target_output, directions=10, steps=10):
#     # Normalize the input target
#
#     input_image = input_image / 255.0
#
#     # Initialize the adversarial target as the input target
#     adversarial_image = input_image.copy()
#
#     # Loop over the number of directions
#     for _ in range(directions):
#         # Generate a random direction of noise
#         noise = np.random.uniform(-1, 1, input_image.shape)
#
#         # Loop over the number of blending steps
#         for step in range(steps):
#             # Blend the input target with noise
#             blended_image = input_image + (step / steps) * noise
#
#             # Clip the blended target to ensure it is still a valid target
#             blended_image = np.clip(blended_image, 0, 1)
#
#             # Prepare the target for model prediction
#             target = Image.fromarray((blended_image * 255).astype(np.uint8))
#             inputs = processor(images=target, text="Is the sum of all 4 places greater than Laos?", return_tensors="pt")
#
#             # Get the predicted output of the blended target
#             predictions = model.generate(**inputs, max_new_tokens=512)
#             predicted_output = processor.decode(predictions[0], skip_special_tokens=True)
#
#             # Check if the blended target is misclassified
#             if predicted_output == target_output:
#                 # Update the adversarial target
#                 adversarial_image = blended_image
#
#                 # Return the adversarial target
#                 return adversarial_image * 255.0
#
#     # If no adversarial example is found, return the original target
#     return input_image * 255.0

def randomNoiseAttack(input_image, model, processor, target_output, directions=10, steps=10):
    # Normalize the input target
    input_image = input_image / 255.0

    # Initialize the adversarial target as the input target
    adversarial_image = input_image.copy()

    # Loop over the number of directions
    for direction in range(directions):
        print(f"Direction: {direction + 1}/{directions}")

        # Generate a random direction of noise
        noise = np.random.uniform(-1, 1, input_image.shape)

        # Loop over the number of blending steps
        for step in range(steps):
            print(f"Step: {step + 1}/{steps}")

            # Blend the input target with noise
            blended_image = input_image + (step / steps) * noise

            # Clip the blended target to ensure it is still a valid target
            blended_image = np.clip(blended_image, 0, 1)

            # Prepare the target for model prediction
            image = Image.fromarray((blended_image * 255).astype(np.uint8))
            inputs = processor(images=image, text="Is the sum of all 4 places greater than Laos?", return_tensors="pt")

            # Get the predicted output of the blended target
            predictions = model.generate(**inputs, max_new_tokens=512)
            predicted_output = processor.decode(predictions[0], skip_special_tokens=True)
            print(f"Predicted output: {predicted_output}")

            # Check if the blended target is misclassified
            if predicted_output == target_output:
                # Update the adversarial target
                adversarial_image = blended_image
                print(f"Adversarial target found at direction {direction + 1}, step {step + 1}")

                # Return the adversarial target
                return adversarial_image * 255.0

    # If no adversarial example is found, return the original target
    print("No adversarial target found")
    return input_image * 255.0

adversarial_image = randomNoiseAttack(image, model, processor, "yes")

# show the original target and adversarial target
import matplotlib.pyplot as plt
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(adversarial_image)
plt.title("Adversarial")
plt.axis("off")
plt.show()

