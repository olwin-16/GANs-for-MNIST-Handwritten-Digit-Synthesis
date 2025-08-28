# GANs for MNIST Handwritten Digit Synthesis

<img width="860" height="375" alt="image" src="https://github.com/user-attachments/assets/5fbd439e-a893-40bc-9966-459372135ace" />

## Project Overview

This project implements a Generative Adversarial Network (GAN) using PyTorch to synthesize new images of handwritten digits based on the MNIST dataset. The GAN consists of a generator, which creates realistic digit images from random noise, and a discriminator, which distinguishes real MNIST digits from synthetically generated images. The goal is to train these networks adversarially so the generator can fool the discriminator with highly authentic handwritten digits.

## Project Structure

**Main Notebook / Scripts:** Code for training both generator and discriminator, visualizing outputs, and evaluating model performance.
**Dataset Loader:** Loads MNIST data and applies random rotation augmentation for improved generalization.
**Discriminator Network:** Convolutional architecture defined in PyTorch as described below.
**Visualization Utilities:** Functions for displaying results and batch samples directly in the notebook.

## Dataset Details

**Source:** The MNIST dataset was originally published by Yann LeCun and colleagues and is publicly available at [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
**Format:** 28x28 grayscale images of handwritten digits (0-9)
**Augmentation:** Random rotation (-20° to +20°) applied on-the-fly
**Training Split:** 60,000 images, used as training set
**Batch Size:** 128, leading to 469 batches per epoch

## Discriminator Network Architecture

The discriminator is a deep convolutional neural network designed to classify images as "real" (from MNIST) or "fake" (generated). The key details are:

<img width="413" height="345" alt="image" src="https://github.com/user-attachments/assets/206eb241-8930-4e4d-98b8-3ee842a829df" />

## Purpose

The discriminator learns to score images as genuine handwritten digits or GAN-generated fakes.
Architecture is robust to MNIST format (single channel, 28x28 images) and leverages batch normalization and leaky ReLU for stability and improved convergence.

## Generator Network Architecture

The generator creates realistic handwritten digit images from random noise vectors (z_dim = 64)  It employs transposed convolutions to progressively upsample noise into images:

<img width="418" height="345" alt="image" src="https://github.com/user-attachments/assets/ca34eb3d-5485-4bc4-b55f-88a44c2d0b20" />

## Training & Optimization

### Loss Functions:

**Discriminator:** BCEWithLogitsLoss for real (label=1) and fake (label=0) images.
**Generator:** BCEWithLogitsLoss to encourage generating images that fool the discriminator.
**Optimizers:** Adam with learning rate 0.002, betas (0.5, 0.99)

### Training Details:

For 20 epochs, alternate discriminator and generator updates per batch.
Losses tracked and shown epoch-wise along with generated sample images.

## Sample Training Output

Epoch-wise losses show gradual improvement, with generated digits becoming more realistic as training progresses.

## Setup & Installation

git clone https://github.com/olwin-16/GANs-for-MNIST-Handwritten-Digit-Synthesis.git

pip install torch torchvision matplotlib tqdm torchsummary

python gan_mnist.py

## License

This project is licensed under the **MIT License**.

## Contact

For questions or contributions, create an issue or reach out via email - olwinchristian1626@gmail.com.
