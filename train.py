"""
Script to train a Score Network. After a few epochs the algortihm will generate a sample via Langevin Dynamics. Specify --epochs for changing the amount of epochs (default 200)
and epochs_until_sample to influence the amount of epochs until Langevin Sampling is performed. Example: uv run score_matching/train.py --epochs 3 --epochs_until_sample 1

Note that there is a pretrained model available which I trained on 200 epochs.
"""

# %%
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from model import ScoreNet
import argparse

# %%
try:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--epochs_until_sample", type=int, default=5)
    args = parser.parse_args()
    epochs = args.epochs
    epochs_until_sample = args.epochs_until_sample
except:
    epochs = 200
    epochs_until_sample = 5

# %%
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# define initial transformation: We want a tensor and 
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

# Since we won't use test data here we will download both train and test sets and combine them.
trainset = datasets.FashionMNIST(
    root="~/.pytorch/F_MNIST_data/",
    download=True,
    train=True,
    transform=transform
)
testset = datasets.FashionMNIST(
    root="~/.pytorch/F_MNIST_data/",
    download=True,
    train=False,
    transform=transform
)
# %%

# Combine and create dataloader.
full_dataset = ConcatDataset([trainset, testset])
batch_size = 128
dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
# %%

# Dimension of our dataset (28x28 black-white images)
dim = 784
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
# Define model and optimizer
model = ScoreNet(dim = dim, num_sigmas=num_sigmas).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
# %%
# get base directory
base_dir = os.path.dirname(os.path.abspath(__file__))
# stick together final output path
output_dir_samples = os.path.join(base_dir, "samples")
os.makedirs(output_dir_samples, exist_ok=True)
output_dir_models = os.path.join(base_dir, "models")
os.makedirs(output_dir_models, exist_ok=True)

# %%
# outer loop: 10 sigma levels
for epoch in range(epochs):
    # activate train mode
    model.train()
    # progress bar
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/20")
    # inner loop: batches
    for x_batch, _ in loop:
        x_batch = x_batch.to(device)
        sigma_indices = torch.randint(0, len(sigmas), (batch_size,), device=device)
        used_sigmas = sigmas[sigma_indices].view(batch_size, *([1] * (x_batch.dim() - 1)))
        # generate noisy vector and add to batch (dependend on used sigma levels)
        noise = torch.randn_like(x_batch) * used_sigmas
        x_noisy = x_batch + noise
        x_noisy.requires_grad_(True)
        # target of gaussian
        target = -(x_noisy - x_batch) / (used_sigmas ** 2)
        # predict scores at current training step, include used sigma indices
        scores = model(x_noisy, sigma_indices)
        # calculate loss
        loss = 0.5 * ((scores - target) ** 2).sum(dim=1) * (used_sigmas.squeeze() ** 2)
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())

    # Langevin Sampler to see how good the generated pictures look after each epoch
    if epoch % epochs_until_sample == 0:
        model.eval()
        x = torch.rand((1, 784), device=device)
        for i in range(num_sigmas):
            sigma = sigmas[i]
            sigma_idx = torch.tensor([i], device=device)
            # calculate alpha (effective stepsize), decreasing over time
            alpha_i = epsilon * (sigma**2 / sigmas[-1]**2)
            # inner loop: T iterations per sigma level
            for t in range(T):
                with torch.no_grad():
                    # predict score
                    score = model(x, sigma_idx)
                    # generate standard normal noise
                    noise = torch.randn_like(x)
                    # perform Langevin step: original x + effective stepsize (alpha_i) * prediced score (i.e direction) + noise term
                    x = x + alpha_i * score + torch.sqrt(2 * alpha_i) * noise

        x_img = x.view(28, 28).detach().cpu().clip(0, 1)
        output_path_samples = os.path.join(output_dir_samples, f"sample_epoch{epoch+1}.png")
        plt.imsave(output_path_samples, x_img, cmap="gray")
        
output_path_model = os.path.join(output_dir_models, "score_model_mnist.pt")
torch.save(model.state_dict(), output_path_model)