"""
Script to generate Fashion Mnist-Samples. It will use a pretrained model by default. Use uv run score_matching/sample.py from the root directory to generate an image.
"""
# %%
import numpy as np
import torch
import matplotlib
import gdown
import os
from model import ScoreNet
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# %%
# Dimension of fashion mnist (greyscale, 28x28)
dim = 784
device = "cpu"
# %%
# num_sigmas: Number of discrete noise levels used for data pertubation
num_sigmas = 10
# Range in which those sigmas lie
sigma_max = 1.0
sigma_min = 0.01
# Steps per noise level
T = 100
epsilon = 2e-5
# %%
# Define sigma sequence (logarithmically decreasing)
sigmas = torch.exp(torch.linspace(np.log(sigma_max), np.log(sigma_min), num_sigmas)).to(device)

# Define model
model = ScoreNet(dim = dim, num_sigmas=num_sigmas).to(device)

# Download model
url = "https://drive.google.com/uc?id=1lFbjTqwgH2VoHrPVCKGDqstLEQJbebkh"

# get base directory
base_dir = os.path.dirname(os.path.abspath(__file__))
# stick together final output path
output_dir = os.path.join(base_dir, "models")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "score_model_mnist_pretrained.pt")

# See if model already has been downloaded
if not os.path.exists(output_path):
    print("Model hasn't been downloaded yet. Starting download...")
    gdown.download(url, output_path, quiet=False)
    print("Done.")
else:
    print("Model already exists. Skipping download.")

# Load model and put into eval mode
state_dict = torch.load(output_path, map_location=torch.device("cpu"), weights_only=False)
model.load_state_dict(state_dict)
model.eval()

# Sample noise image as starting point
x = torch.rand((1, 784), device=device)
# %%
# generate plot
plt.ion()
fig, ax = plt.subplots()

# outer loop: 10 sigma levels
for i in range(num_sigmas):
    sigma = sigmas[i]
    sigma_idx = torch.tensor([i], device=device)
    # calculate alpha (effective stepsize), decreasing over time
    alpha_i = epsilon * (sigma**2 / sigmas[-1]**2)
    # inner loop: 100 iterations per sigma level
    for t in range(T):
        with torch.no_grad():
            # predict score, include used sigma indices
            score = model(x, sigma_idx)
            # generate standard normal noise
            noise = torch.randn_like(x)
            # perform Langevin step: original x + effective stepsize (alpha_i) * prediced score (i.e direction) + noise term
            x = x + alpha_i * score + torch.sqrt(2 * alpha_i) * noise
        # remove old plot to replace with a new
        ax.clear()
        ax.imshow(x.detach().cpu().view(28, 28).clip(0, 1), cmap="gray")
        ax.set_title(f"Sigma {i+1}/{num_sigmas}, Step {t}")
        plt.pause(0.001)

plt.ioff()
plt.show()
