
import torch
from torch import nn, optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import math
import numpy as np
from PIL import Image
import cv2
from tqdm.notebook import tqdm
from functools import partial

from transformers import AutoProcessor, Pix2StructForConditionalGeneration
from PIL import Image
processor = AutoProcessor.from_pretrained("google/matcha-chartqa")


# def eval_model(model, images):
#     images = images.astype(np.float32)
#     images = torch.from_numpy(images)
#     images = images.reshape(-1, 784)
#     outputs = model(images)
#     _, test_predicted = torch.max(outputs.data,1)
#     return test_predicted.item()


def eval_model(model, target):

    inputs = processor(images=target, text="Which country was the leading export destination for maple sugar and maple syrup from Canada in 2013?", return_tensors="pt").to(0)
    prediction = processor.decode(model.generate(**inputs, max_new_tokens=512)[0], skip_special_tokens=True)
    return prediction


def is_adversarial(oracle, target_class, image):
    if oracle(image) != target_class:
        return True
    return False


def initial_adversarial(target_, advesarial_criterion):
    while True:
        random_pertubation = np.random.uniform(0, 1, size=target_.shape).astype(np.float32)
        if advesarial_criterion(random_pertubation):
            return random_pertubation


def step_towards(target_, pertubation, epsilon):
    pertubation -= ((pertubation - target_) * epsilon)
    return pertubation


def orthogonal_pertubation(target_, pertubation, delta):
    # iid random normal
    new_pertubation = np.random.normal(0, 1, size=target_.shape)
    new_pertubation /= np.linalg.norm(new_pertubation)
    # || n_k||^2 = delta * dist(previous_adversarial, original_image), condition 2
    new_pertubation *= delta * np.linalg.norm(pertubation - target_)
    new_pertubation += pertubation
    # condition 1
    new_pertubation = new_pertubation.clip(0, 1)
    # projecting onto sphere
    new_pertubation -= target_
    new_pertubation /= np.linalg.norm(new_pertubation)
    new_pertubation *= np.linalg.norm(pertubation - target_)
    new_pertubation += target_
    return new_pertubation


def adaptive_delta_descent(is_adversarial_fn, side_step_fn, adversarial_img):
    lower_delta = 1
    upper_delta = 0
    delta = -1
    while True:
        delta = (lower_delta + upper_delta) / 2
        random_side_steps = [side_step_fn(adversarial_img, delta) for i in range(10)]
        adversarial_indices = [is_adversarial_fn(x) for x in random_side_steps]
        average = np.mean(adversarial_indices)
        if abs(average - .5) <= .05 or delta == 1:  # can no longer converge
            break
        elif delta == 0:  # cannot step anywhere that is not adversarial
            return None
        elif average > .55:
            upper_delta = delta  # increasing delta
        else:  # average < .45
            lower_delta = delta  # decreasing delta
    indices = np.where(np.array(adversarial_indices) == True)[0]
    return np.array(random_side_steps)[indices]


def adaptive_epsilon_descent(target, is_adversarial_fn, adversarial_img):
    global epsilon_upper
    epsilon_lower = 1
    epsilon_upper = 0
    epsilon = (epsilon_lower + epsilon_upper) / 2
    while epsilon_upper == 0 or abs(epsilon - epsilon_upper) > .0001:
        epsilon = (epsilon_lower + epsilon_upper) / 2
        new_pertubation = adversarial_img - ((adversarial_img - target) * epsilon)
        if is_adversarial_fn(new_pertubation):
            epsilon_upper = epsilon
        else:
            epsilon_lower = epsilon
    # print(f'{epsilon_upper = }')
    new_pertubation = adversarial_img - ((adversarial_img - target) * epsilon_upper)
    return new_pertubation


def minimum_distance_img(target, imgs):
    min_ = float('inf')
    index = 0
    for step, x in enumerate(imgs):
        dist = np.linalg.norm(x - target)
        if dist < min_:
            min_ = dist
            index = step
    return imgs[index], min_

def boundary_attack(model, oracle, target_, question_, answer_, k):
    global epsilon_upper
    # create partials so to avoid passing in same parameters every time
    # step_towards_(pertubation, epsilon)
    step_towards_ = partial(step_towards, target_)
    # oracle_(images)
    oracle_ = partial(oracle, model)
    # is_adversarial_(target)
    is_adversarial_ = partial(is_adversarial, oracle_, answer_)
    # orthogonal_pertubation_(pertubation, delta)
    orthogonal_pertubation_ = partial(orthogonal_pertubation, target_)
    # adaptive_epsilon_descent_(pertubation)
    adaptive_epsilon_descent_ = partial(adaptive_epsilon_descent, target_, is_adversarial_)
    # adaptive_delta_descent_(pertubation)
    adaptive_delta_descent_ = partial(adaptive_delta_descent, is_adversarial_, orthogonal_pertubation_)
    # min_dist_img_(images)
    min_dist_img_ = partial(minimum_distance_img, target_)
    # initial adversarial
    adversarial_img = initial_adversarial(target_, oracle_)

    # plt.imshow(adversarial_img, cmap="gray")
    # plt.show()

    # initial descent
    adversarial_img = adaptive_epsilon_descent_(adversarial_img)

    plt.imshow(adversarial_img, cmap="gray")
    plt.show()

    for i in tqdm(range(k)):
        # orthogonal step
        temp_adversarial_imgs = adaptive_delta_descent_(adversarial_img)
        if temp_adversarial_imgs is None:  # point of convergence
            continue
        # downwards step
        temp_adversarial_imgs = np.array([adaptive_epsilon_descent_(x) for x in temp_adversarial_imgs],
                                         dtype=np.float64)
        # print('\n')
        temp_adversarial_imgs1 = [x for x in temp_adversarial_imgs if is_adversarial_(x)]
        if temp_adversarial_imgs1:  # if list is not empty
            adversarial_img, min_ = min_dist_img_(temp_adversarial_imgs1)
            inputs = processor(images=adversarial_img, text=question_, return_tensors="pt").to(0)
            prediction = processor.decode(model.generate(**inputs, max_new_tokens=512)[0], skip_special_tokens=True)

            print(f'{prediction}')
            print(f'{min_} \n')

    plt.imshow(adversarial_img, cmap="gray")
    plt.show()

    return adversarial_img