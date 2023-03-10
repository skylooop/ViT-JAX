### JAX
import jax
import jax.numpy as jnp
from jax import random

### Plots
import matplotlib.pyplot as plt

import numpy as np
import os


### Parser
from absl import app, flags


### Typings
import typing as tp


### DL
import torch
import torch.utils.data as data
from tensorboardX import SummaryWriter
import torchvision
from torchvision import transforms


### Utils
from utils.constants import DATA_MEANS, DATA_STD
from utils.datasets import initialize_datasets

from src.vit import img_to_patch, VisionTransformer
### Arguments
FLAGS = flags.FLAGS

flags.DEFINE_string("assets_path", default= "assets/", help="Path for logs save dir.")
flags.DEFINE_integer("seed", default=42, help="Random seed.")
flags.DEFINE_string("dataset_path", default="/home/m_bobrin/tmp", help="Path to the dataset.")
flags.DEFINE_string("logger", default='wandb', help='Logger to use. Currently only Weights n Biases')


def image_to_numpy(img) -> np.ndarray:
    img = np.array(img, dtype=np.float32)
    img = (img / 255. - DATA_MEANS) / DATA_STD
    return img


def initialize_jax(random_seed: int = 42) -> int:
    main_rng = random.PRNGKey(random_seed)
    print(f"JAX devices: {jax.devices()}")
    
    return main_rng

def numpy_to_torch(array: tp.Union[tp.Any, np.ndarray]) -> torch.Tensor:
    array = jax.device_get(array)
    tensor = torch.from_numpy(array)
    tensor = tensor.permute(0, 3, 1, 2)
    return tensor

def save_patches(CIFAR_images: np.ndarray) -> ...:
    img_patches = img_to_patch(CIFAR_images, patch_size=4)

    fig, ax = plt.subplots(CIFAR_images.shape[0], 1, figsize=(14,3))
    
    fig.suptitle("Images as input sequences of patches")
    
    for i in range(CIFAR_images.shape[0]):
        img_grid = torchvision.utils.make_grid(numpy_to_torch(img_patches[i]),
                                            nrow=64, normalize=True, pad_value=0.9)
        img_grid = img_grid.permute(1, 2, 0)
        ax[i].imshow(img_grid)
        ax[i].axis('off')
    plt.savefig(os.path.join(FLAGS.assets_path, 'patches.jpg'))
    

def main_train_vit(_: tp.Any) -> ...:
    
    main_rng = initialize_jax(FLAGS.seed)
    print("Creating dataloaders")
    train_loader, val_loader, test_loader, val_set = initialize_datasets(image_to_numpy=image_to_numpy)

    print(f"Saving samples from dataset to {FLAGS.assets_path}")

    CIFAR_images = np.stack([val_set[idx][0] for idx in range(4)], axis=0)
    img_grid = torchvision.utils.make_grid(numpy_to_torch(CIFAR_images),
                                       nrow=4, normalize=True, pad_value=0.9)
    img_grid = img_grid.permute(1, 2, 0)
    
    plt.figure(figsize=(8,8))
    plt.title("Image examples of the CIFAR10 dataset")
    plt.imshow(img_grid)
    plt.axis('off')
    
    plt.savefig(os.path.join(FLAGS.assets_path, 'example.jpg'))

    print("Initializing ViT")
    save_patches(CIFAR_images)
    
    #vit = ViT()
        
if __name__ == "__main__":
    app.run(main_train_vit)