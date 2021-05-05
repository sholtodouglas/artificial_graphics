import jax.numpy as jnp
import PIL
from PIL import ImageOps
import io
import requests
import numpy as np
from jax import jit

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

def preprocess(img, target_image_size=256):
	'''
	x = preprocess(download_image('https://github.com/sholtodouglas/artificial_graphics/raw/main/tests/670.jpg'))
	'''
	img = ImageOps.fit(img, [target_image_size,] * 2, method=0, bleed=0.0, centering=(0.5, 0.5))

	img = np.expand_dims(np.transpose(np.array(img).astype(np.float32)/255, (2, 0, 1)), 0)
	return map_pixels(img)

@jit
def preprocess_batch(b: dict, logit_laplace_eps = 0.1) -> dict:
  imgs = b['img']/255.0
  b['img'] = jnp.transpose(imgs, axes = [0, 3, 1, 2])
  b['img'] = (1 - 2 * logit_laplace_eps) * b['img'] + logit_laplace_eps
  return b

