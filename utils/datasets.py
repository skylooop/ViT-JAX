import typing as tp
from torch.utils import data
import numpy as np
from torchvision import transforms

from torchvision.datasets import CIFAR10
import torch
from absl import flags

FLAGS = flags.FLAGS



def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)
    
    
def initialize_datasets(image_to_numpy: tp.Callable[[np.array], np.array]) -> tp.Tuple[data.DataLoader, data.DataLoader, data.DataLoader]:
    
    test_transform = image_to_numpy
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.RandomResizedCrop((32,32),scale=(0.8,1.0),ratio=(0.9,1.1)),
                                            image_to_numpy
                                        ])
    train_dataset = CIFAR10(root=FLAGS.dataset_path, train=True, transform=train_transform, download=True)
    val_dataset = CIFAR10(root=FLAGS.dataset_path, train=True, transform=test_transform, download=True)
    
    train_set, _ = torch.utils.data.random_split(train_dataset, [45000, 5000], generator=torch.Generator().manual_seed(FLAGS.seed))
    _, val_set = torch.utils.data.random_split(val_dataset, [45000, 5000], generator=torch.Generator().manual_seed(FLAGS.seed))
    
    test_set = CIFAR10(root=FLAGS.dataset_path, train=False, transform=test_transform, download=True)
    
    train_loader = data.DataLoader(train_set,
                                        batch_size=128,
                                        shuffle=True,
                                        drop_last=True,
                                        collate_fn=numpy_collate,
                                        num_workers=8,
                                        persistent_workers=True)
    
    val_loader  = data.DataLoader(val_set,
                                        batch_size=128,
                                        shuffle=False,
                                        drop_last=False,
                                        collate_fn=numpy_collate,
                                        num_workers=4,
                                        persistent_workers=True)
    
    test_loader  = data.DataLoader(test_set,
                                        batch_size=128,
                                        shuffle=False,
                                        drop_last=False,
                                        collate_fn=numpy_collate,
                                        num_workers=4,
                                        persistent_workers=True)
    
    return train_loader, val_loader, test_loader