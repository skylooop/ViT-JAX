from flax import linen as nn
import jax
import jax.numpy as jnp

import einops
import numpy as np


def img_to_patch(img: np.ndarray, patch_size: int) -> np.ndarray:
    a = einops.rearrange(img, 'b (h1 patch1) (h2 patch2) c -> b (h1 h2) patch1 patch2 c', patch1 = patch_size, patch2 = patch_size)
    return a

if __name__ == "__main__":
    img_to_patch(np.random.randn(3, 3, 28, 28))