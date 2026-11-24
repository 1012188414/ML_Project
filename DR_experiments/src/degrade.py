"""Image degradations and the image-quality signal, shared by training (robust augmentation) and testing.

All degradations act on the padded, resized uint8 photo (model input size) before normalisation.
  blur_s{sigma}       Gaussian blur, standard deviation sigma pixels (scaled with the input size)
  down_x{f}           bilinear downsampling by factor f, then bilinear upsampling back
  jpeg_q{q}           JPEG compression at quality q                        (never used in training)
  noise_s{sigma}      additive Gaussian noise, sigma in grey levels (0-255) (never used in training)
  contrast_c{f}       contrast reduced to f inside the fundus region        (never used in training)

Quality signal q_lap: log variance of the Laplacian inside the (eroded) fundus region, so the bright rim of
the fundus against the black padding does not dominate. Higher = sharper. Known blind spot: noise and JPEG
blocking add high-frequency content and raise q_lap even though quality drops.
q_oracle: true severity of a blur / downsampling degradation scaled to [0, 1] (0 = clean); undefined for
the other degradations and for real low-quality photos.
"""
import io

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

from common import AUG_PROB, BLUR_SIGMA, DOWN_FACTOR


def parse(condition):
    """'blur_s2' -> ('blur', 2.0); 'clean' -> ('clean', 0.0)."""
    if condition == 'clean':
        return 'clean', 0.0
    kind, value = condition.split('_', 1)
    return kind, float(value[1:])


def _scale(pixels):
    return pixels.shape[0] / 384.0


def fundus_region(pixels, erode=True):
    grey = pixels.astype(np.float32).mean(axis=2)
    smooth = ndimage.gaussian_filter(grey, sigma=max(1.0, 3 * _scale(pixels)))
    mask = smooth > 20
    if erode:
        mask = ndimage.binary_erosion(mask, iterations=max(1, round(12 * _scale(pixels))))
    return mask


def apply(pixels, kind, value, seed=0):
    """Return a degraded copy of an HxWx3 uint8 photo."""
    if kind == 'clean':
        return pixels
    if kind == 'blur':
        image = Image.fromarray(pixels).filter(ImageFilter.GaussianBlur(radius=float(value) * _scale(pixels)))
        return np.asarray(image, dtype=np.uint8)
    if kind == 'down':
        image = Image.fromarray(pixels)
        size = image.size[0]
        small = max(1, round(size / float(value)))
        image = image.resize((small, small), Image.Resampling.BILINEAR).resize((size, size), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.uint8)
    if kind == 'jpeg':
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format='JPEG', quality=int(value))
        buffer.seek(0)
        with Image.open(buffer) as image:
            return np.asarray(image.convert('RGB'), dtype=np.uint8)
    if kind == 'noise':
        rng = np.random.default_rng(seed)
        noisy = pixels.astype(np.float32) + rng.normal(0.0, float(value), pixels.shape)
        return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)
    if kind == 'contrast':
        mask = fundus_region(pixels, erode=False)
        if not mask.any():
            return pixels
        out = pixels.astype(np.float32)
        mean = out[mask].mean(axis=0)
        out[mask] = mean + float(value) * (out[mask] - mean)
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)
    raise ValueError(f'Unknown degradation {kind}')


def severity(kind, value):
    if kind == 'clean':
        return 0.0
    if kind == 'blur':
        return float(value) / BLUR_SIGMA[1]
    if kind == 'down':
        return (float(value) - 1.0) / (DOWN_FACTOR[1] - 1.0)
    return float('nan')


def sample_spec(rng, family, force=False):
    """Random training degradation. family: 'both' | 'blur' | 'down'. force=True always degrades."""
    if not force and rng.random() >= AUG_PROB:
        return 'clean', 0.0
    kind = family if family in ('blur', 'down') else ('blur' if rng.random() < 0.5 else 'down')
    low, high = BLUR_SIGMA if kind == 'blur' else DOWN_FACTOR
    return kind, float(rng.uniform(low, high))


def q_lap(pixels):
    grey = pixels.astype(np.float32).mean(axis=2)
    lap = ndimage.laplace(grey)
    mask = fundus_region(pixels)
    values = lap[mask] if mask.sum() >= 100 else lap.ravel()
    return float(np.log(values.var() + 1e-6))
