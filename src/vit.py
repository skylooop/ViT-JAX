from flax import linen as nn
import jax
from jax import random
import jax.numpy as jnp

from absl import flags

import einops
import numpy as np

FLAGS = flags.FLAGS

# Fancy way of making patches
def img_to_patch(img: np.ndarray, patch_size: int) -> np.ndarray:
    a = einops.rearrange(img, 'b (h1 patch1) (h2 patch2) c -> b (h1 h2) (patch1 patch2 c)', patch1 = patch_size, patch2 = patch_size)
    return a

class AttentionBlock(nn.Module):
    embed_dim : int   # Dimensionality of input and attention feature vectors
    hidden_dim : int  # Dimensionality of hidden layer in feed-forward network
    num_heads : int   # Number of heads to use in the Multi-Head Attention block
    dropout_prob : float = 0.0  # Amount of dropout to apply in the feed-forward network

    def setup(self):
        self.attn = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)
        self.linear = [
            nn.Dense(self.hidden_dim),
            nn.gelu,
            nn.Dropout(self.dropout_prob),
            nn.Dense(self.embed_dim)
        ]
        self.layer_norm_1 = nn.LayerNorm()
        self.layer_norm_2 = nn.LayerNorm()
        self.dropout = nn.Dropout(self.dropout_prob)

    # Making Pre-Norm
    def __call__(self, x, train=True):
        inp_x = self.layer_norm_1(x)
        attn_out = self.attn(inputs_q=inp_x, inputs_kv=inp_x)
        x = x + self.dropout(attn_out, deterministic=not train)

        linear_out = self.layer_norm_2(x)
        for l in self.linear:
            linear_out = l(linear_out) if not isinstance(l, nn.Dropout) else l(linear_out, deterministic=not train)
        x = x + self.dropout(linear_out, deterministic=not train)
        return x

class VisionTransformer(nn.Module):
    embed_dim : int     # Dimensionality of input and attention feature vectors
    hidden_dim : int    # Dimensionality of hidden layer in feed-forward network
    num_heads : int     # Number of heads to use in the Multi-Head Attention block
    num_channels : int  # Number of channels of the input (3 for RGB)
    num_layers : int    # Number of layers to use in the Transformer
    num_classes : int   # Number of classes to predict
    patch_size : int    # Number of pixels that the patches have per dimension
    num_patches : int   # Maximum number of patches an image can have
    dropout_prob : float = 0.0  # Amount of dropout to apply in the feed-forward network

    def setup(self):
        self.input_layer = nn.Dense(self.embed_dim)
        self.transformer = [AttentionBlock(self.embed_dim,
                                           self.hidden_dim,
                                           self.num_heads,
                                           self.dropout_prob) for _ in range(self.num_layers)]
        self.mlp_head = nn.Sequential([
            nn.LayerNorm(),
            nn.Dense(self.num_classes)
        ])
        self.dropout = nn.Dropout(self.dropout_prob)

        self.cls_token = self.param('cls_token',
                                    nn.initializers.normal(stddev=1.0),
                                    (1, 1, self.embed_dim))
        self.pos_embedding = self.param('pos_embedding',
                                        nn.initializers.normal(stddev=1.0),
                                        (1, 1+self.num_patches, self.embed_dim))


    def __call__(self, x, train=True):
        # Preprocess input
        x = img_to_patch(x, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer(x)

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, axis=0)
        x = jnp.concatenate([cls_token, x], axis=1)
        x = x + self.pos_embedding[:,:T+1]

        # Apply Transforrmer
        x = self.dropout(x, deterministic=not train)
        for attn_block in self.transformer:
            x = attn_block(x, train=train)

        # Perform classification prediction
        cls = x[:,0]
        out = self.mlp_head(cls)
        return out


if __name__ == "__main__":
    
    ## Test VisionTransformer implementation
    # Example features as input
    main_rng = random.PRNGKey(42)
    main_rng, x_rng = random.split(main_rng)

    x = random.normal(x_rng, (5, 32, 32, 3))
    # Create vision transformer
    visntrans = VisionTransformer(embed_dim=128,
                                hidden_dim=512,
                                num_heads=4,
                                num_channels=3,
                                num_layers=6,
                                num_classes=10,
                                patch_size=4,
                                num_patches=64,
                                dropout_prob=0.1)
    # Initialize parameters of the Vision Transformer with random key and inputs
    main_rng, init_rng, dropout_init_rng = random.split(main_rng, 3)
    params = visntrans.init({'params': init_rng, 'dropout': dropout_init_rng}, x, True)['params']
    #print(params)
    
    main_rng, dropout_apply_rng = random.split(main_rng)
    out = visntrans.apply({'params': params}, x, train=True, rngs={'dropout': dropout_apply_rng})
    print('Out', out.shape)

    del visntrans, params