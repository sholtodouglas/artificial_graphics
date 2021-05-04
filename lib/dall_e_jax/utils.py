import jax.numpy as jnp
import PIL
from PIL import ImageOps
import io
import requests


logit_laplace_eps: float = 0.1


def map_pixels(x: jnp.ndarray) -> jnp.ndarray:
	if len(x.shape) != 4:
		raise ValueError('expected input to be 4d')
	if x.dtype != jnp.float32:
		raise ValueError('expected input to have type float')

	return (1 - 2 * logit_laplace_eps) * x + logit_laplace_eps


def unmap_pixels(x: jnp.ndarray) -> jnp.ndarray:
	if len(x.shape) != 4:
		raise ValueError('expected input to be 4d')
	if x.dtype != jnp.float32:
		raise ValueError('expected input to have type float')

	return jnp.clip((x - logit_laplace_eps) / (1 - 2 * logit_laplace_eps), 0, 1)


def download_image(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return PIL.Image.open(io.BytesIO(resp.content))
