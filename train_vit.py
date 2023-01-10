### JAX
import jax
import jax.numpy as jnp
from jax import random


import numpy as np

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

### Arguments
FLAGS = flags.FLAGS

flags.DEFINE_string("assets_path", default= "assets/", help="Path for logs save dir.")
flags.DEFINE_integer("seed", default=42, help="Random seed.")
flags.DEFINE_string("dataset_path", default="/home/m_bobrin/tmp", help="Path to the dataset.")


def image_to_numpy(img):
    img = np.array(img, dtype=np.float32)
    img = (img / 255. - DATA_MEANS) / DATA_STD
    return img


def initialize_jax(random_seed: int = 42) -> int:
    main_rng = random.PRNGKey(random_seed)
    print(f"JAX devices: {jax.devices()}")
    
    return main_rng

def numpy_to_torch(array: jnp.array) -> torch.Tensor:
    array = jax.device_get(array)
    tensor = torch.from_numpy(array)
    tensor = tensor.permute(0, 3, 1, 2)
    return tensor
    
def main_train_vit(_: tp.Any) -> ...:
    main_rng = initialize_jax(FLAGS.seed)
    print("Creating dataloaders")
    train_loader, val_loader, test_loader = initialize_datasets(image_to_numpy=image_to_numpy)
    # print(next(iter(train_loader)))

    print(f"Saving samples from dataset to {FLAGS.assets_path}")
    
    

if __name__ == "__main__":
    app.run(main_train_vit)