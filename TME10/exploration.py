import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as dset
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn    
import torch.optim as optim

from models import VAE, ConvVAE
from vae import generate, load_mnist_dataset

device = torch.device("cuda:0")
PATH_TO_CHECKPOINT = Path("checkpoints/conv_vae/model_latent2")
PATH_TO_IMG = Path("results/exploration/")


def latent_embedding():
    model = torch.load(PATH_TO_CHECKPOINT)
    test_set = load_mnist_dataset(train=False)
    test_loader = DataLoader(test_set, shuffle=True, batch_size=500)
    samples, label = next(iter(test_loader))
    samples = samples.to(device)
    mu, logsigma = model.encoder(samples)
    mu = mu.detach().cpu()
    plt.figure(figsize=(15,15))
    plt.title(f"Encodage de l'espace latent en 2D")
    sns.scatterplot(mu[:,0], mu[:,1], hue=label, s=100, palette="Paired")
    plt.legend()
    plt.savefig(PATH_TO_IMG / "latent_embedding.png")


latent_embedding()