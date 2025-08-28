# GANs for MNIST Handwritten Digit Synthesis

<br>

<img width="860" height="375" alt="image" src="https://github.com/user-attachments/assets/5fbd439e-a893-40bc-9966-459372135ace" />

<br>

## Project Overview

This project implements a Generative Adversarial Network (GAN) using PyTorch to synthesize new images of handwritten digits based on the MNIST dataset. The GAN consists of a generator, which creates realistic digit images from random noise, and a discriminator, which distinguishes real MNIST digits from synthetically generated images. The goal is to train these networks adversarially so the generator can fool the discriminator with highly authentic handwritten digits.

## Project Structure

**Main Script:** Single code file containing both generator and discriminator definitions, data loading, training loop, visualization utilities, and evaluation.

**Dataset Loader:** Loads MNIST data and applies random rotation augmentation for improved generalization.

**Visualization Utilities:** Functions for displaying results and batch samples directly in the notebook.

## Dataset Details

The MNIST dataset was originally published by Yann LeCun and colleagues and is publicly available at [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

**Format:** 28x28 grayscale images of handwritten digits (0-9)

**Augmentation:** Random rotation (-20° to +20°) applied on-the-fly

**Training Split:** 60,000 images, used as training set

**Batch Size:** 128, leading to 469 batches per epoch

## Discriminator Network Architecture

The discriminator is a deep convolutional neural network designed to classify images as "real" (from MNIST) or "fake" (generated). The key details are:

| Layer          | Output Shape     | Parameters | Description                              |
|----------------|------------------|------------|------------------------------------------|
| Conv2d (1→16)  | [-1, 16, 13, 13] | 160        | 3x3 kernel, stride 2, feature extraction |
| BatchNorm2d    | [-1, 16, 13, 13] | 32         | Normalizes intermediate features         |
| LeakyReLU      | [-1, 16, 13, 13] | 0          | Non-linear activation (α=0.2)            |
| Conv2d (16→32) | [-1, 32, 5, 5]   | 12,832     | 5x5 kernel, stride 2, deeper features    |
| BatchNorm2d    | [-1, 32, 5, 5]   | 64         | Normalizes intermediate features         |
| LeakyReLU      | [-1, 32, 5, 5]   | 0          | Non-linear activation                    |
| Conv2d (32→64) | [-1, 64, 1, 1]   | 51,264     | 5x5 kernel, stride 2, global features    |
| BatchNorm2d    | [-1, 64, 1, 1]   | 128        | Normalizes global features               |
| LeakyReLU      | [-1, 64, 1, 1]   | 0          | Non-linear activation                    |
| Flatten        | [-1, 64]         | 0          | Flattens output to vector                |
| Linear (64→1)  | [-1, 1]          | 65         | Single output: real/fake score           |

**Total Trainable Parameters:** 64,545

## Purpose

The discriminator learns to score images as genuine handwritten digits or GAN-generated fakes.
Architecture is robust to MNIST format (single channel, 28x28 images) and leverages batch normalization and leaky ReLU for stability and improved convergence.

## Generator Network Architecture

The generator creates realistic handwritten digit images from random noise vectors (z_dim = 64)  It employs transposed convolutions to progressively upsample noise into images:

| Layer                     | Output Shape      | Parameters | Description                     |
|---------------------------|-------------------|------------|---------------------------------|
| ConvTranspose2d (64→256)  | [-1, 256, 3, 3]   | 147,712    | Project noise to feature map    |
| BatchNorm2d               | [-1, 256, 3, 3]   | 512        | Feature normalization           |
| ReLU                      | [-1, 256, 3, 3]   | 0          | Activation                      |
| ConvTranspose2d (256→128) | [-1, 128, 6, 6]   | 524,416    | Upsample and refine features    |
| BatchNorm2d               | [-1, 128, 6, 6]   | 256        | Feature normalization           |
| ReLU                      | [-1, 128, 6, 6]   | 0          | Activation                      |
| ConvTranspose2d (128→64)  | [-1, 64, 13, 13]  | 73,792     | Upsample                        |
| BatchNorm2d               | [-1, 64, 13, 13]  | 128        | Feature normalization           |
| ReLU                      | [-1, 64, 13, 13]  | 0          | Activation                      |
| ConvTranspose2d (64→1)    | [-1, 1, 28, 28]   | 1,025      | Final image formation           |
| Tanh                      | [-1, 1, 28, 28]   | 0          | Output normalization [-1,1]     |

**Total Trainable Parameters:** 747,841

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

### Clone the repository:

```bash
git clone https://github.com/olwin-16/GANs-for-MNIST-Handwritten-Digit-Synthesis.git
cd GANs-for-MNIST-Handwritten-Digit-Synthesis
```

### Install required dependencies:

```bash
pip install -r requirements.txt
```

### Run the main script:

```bash
python gan_mnist.py
```

## License

This project is licensed under the [MIT License](LICENSE)..

## Contact

For questions or contributions, create an issue or reach out via [Email](mailto:olwinchristian1626@gmail.com).
