# Fashion-MNIST Generation with Score-Based Diffusion Models

This project implements a score-based generative model for image generation on the Fashion-MNIST dataset. The model is trained using **Denoising Score Matching** and generates images through **Langevin Dynamics**, following the principles of modern diffusion and score-based generative models.

The project was originally developed as part of the graduate seminar *Dynamical Systems in Deep Learning*, which explored the connections between deep learning, stochastic processes, Bayesian inference, and dynamical systems.

## Theoretical Background

The implementation is based on the following foundational works:

- Hyvärinen (2005): *Estimation of Non-Normalized Statistical Models by Score Matching*
- Song & Ermon (2019): *Generative Modeling by Estimating Gradients of the Data Distribution*
- Song et al. (2021): *Score-Based Generative Modeling through Stochastic Differential Equations*

The core idea is to learn the **score function**, i.e. the gradient of the log-density of the data distribution. Once learned, this score field can be used to iteratively transform random noise into realistic samples.

## Dataset

All experiments use the Fashion-MNIST dataset, developed by Zalando Research as a drop-in replacement for MNIST. It consists of 28×28 grayscale images from 10 clothing categories, including shirts, coats, sneakers, dresses, and bags.

## Repository Structure

```text
score_matching/
├── train.py
└── sample.py

models/
samples/
```

## Environment Setup

This project uses `uv` for dependency and environment management.

```bash
uv venv
source .venv/bin/activate
uv sync
```

## Sampling Fashion-MNIST Images

A pretrained score network is provided for inference. During sampling, the model starts from pure Gaussian noise and iteratively denoises the image using Langevin Dynamics.

```bash
uv run score_matching/sample.py
```

The generated image is displayed interactively, allowing the denoising process to be observed step by step.

## Training a Score Network

To train a model from scratch:

```bash
uv run score_matching/train.py
```

By default, the model is trained for 200 epochs.

Intermediate samples can be generated during training:

```bash
uv run score_matching/train.py --epochs 100 --epochs_until_sample 10
```

This saves intermediate generations to the `samples/` directory and provides insight into how sample quality evolves throughout training.

## Results

The trained model successfully learns a score field over the Fashion-MNIST data distribution and is capable of generating realistic clothing images from random noise using Langevin sampling.

## Acknowledgements

This project was developed as part of the seminar *Dynamical Systems in Deep Learning*. The repository contains my individual implementation and experiments on score-based generative modeling.