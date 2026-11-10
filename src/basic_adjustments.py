"""Explicit full-resolution basic edits; not Lightroom or camera emulation.

All-zero settings return the input unchanged. Colour and spatial operations
use float64 linear-light RGB, with a single final quantization to the input
bit depth. Spatial kernels operate on the original pixel grid, in bounded
tiles with their complete finite Gaussian support. Alpha is never modified.
"""
import math
from functools import lru_cache

import numpy as np


DEFAULT = dict.fromkeys(('contrast', 'saturation', 'vibrance', 'warmth', 'tint',
                         'clarity', 'texture', 'dehaze', 'sharpen', 'soften'), 0.0)
_LUMA = np.array((0.2126, 0.7152, 0.0722), dtype=np.float64)
# With the largest normal-resolution radius (48), 224 core rows fit a
# complete padded vertical convolution in a 512-point FFT. Larger tiles
# would jump to 1024 points; smaller ones repeat more overlap work.
_TILE_HEIGHT = 224
_TILE_WIDTH = 1024


def validate(value=None):
    value = {} if value is None else value
    if not isinstance(value, dict) or set(value) - set(DEFAULT):
        raise ValueError('基础调整格式无效')
    result = DEFAULT.copy()
    for key, v in value.items():
        low = 0 if key in ('sharpen', 'soften') else -100
        if type(v) not in (int, float) or not math.isfinite(v) or not low <= v <= 100:
            raise ValueError('基础调整参数超出范围')
        result[key] = float(v)
    return result


def neutral(value=None):
    return not any(validate(value).values())


def _scale(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 16:
        raise ValueError('基础调整的图像比例无效')
    return float(value)


def _radius(sigma):
    # Below this scale all off-centre weights are below float64 precision.
    return 0 if sigma < 0.05 else int(math.ceil(4 * sigma))


def halo(value=None, scale=1.0):
    """Finite support, in pixels of the supplied array, needed around a crop.

    All spatial terms read the same pointwise-adjusted source, so supports
    take their maximum, rather than accumulating across sequential filters.
    """
    v = validate(value); scale = _scale(scale)
    return max((_radius(s * scale) if v[k] else 0)
               for k, s in (('clarity', 12.0), ('texture', 2.0), ('sharpen', 1.0), ('soften', 6.0)))


def _cancel(cancelled):
    if cancelled and cancelled():
        from native_backend import RenderCancelled
        raise RenderCancelled()


def _decode(v):
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def _encode(v):
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.0031308, 12.92 * v, 1.055 * v ** (1 / 2.4) - 0.055)


def _luma(rgb):
    return rgb[..., 0] * _LUMA[0] + rgb[..., 1] * _LUMA[1] + rgb[..., 2] * _LUMA[2]


def _gamut(y, chroma):
    """Keep luminance and hue direction; reduce chroma only to fit the gamut."""
    y = np.clip(y, 0.0, 1.0)
    positive = np.maximum(np.maximum(np.maximum(chroma[..., 0], chroma[..., 1]), chroma[..., 2]), 0.0)
    negative = np.maximum(-np.minimum(np.minimum(chroma[..., 0], chroma[..., 1]), chroma[..., 2]), 0.0)
    gain = np.minimum(1.0, np.minimum((1.0 - y) / np.maximum(positive, 1e-30),
                                    y / np.maximum(negative, 1e-30)))
    return np.clip(y[..., None] + chroma * gain[..., None], 0.0, 1.0)


def _colour(rgb, values):
    linear = _decode(rgb)
    if not any(values[k] for k in ('contrast', 'saturation', 'vibrance', 'warmth', 'tint', 'dehaze')):
        return linear
    y = _luma(linear)
    chroma = linear - y[..., None]
    if values['contrast']:
        # A monotonic S-shaped change of luminance, not three unrelated
        # channel curves. Endpoints and a neutral grey axis remain stable.
        perceptual = _encode(y)
        perceptual += (values['contrast'] / 100.0) * (2 * perceptual - 1) * perceptual * (1 - perceptual)
        y = _decode(np.clip(perceptual, 0.0, 1.0))
    if values['dehaze']:
        # A manual, monotonic de-veiling tone/chroma adjustment. It is not
        # physical haze reconstruction and cannot recover clipped pixels.
        amount = values['dehaze'] / 100.0
        y = _decode(_encode(y) ** (2.0 ** (0.45 * amount)))
        chroma *= 1.0 + 0.10 * amount
    if values['saturation']:
        chroma *= 1.0 + values['saturation'] / 100.0
    if values['vibrance']:
        # Prefer less-saturated colours; achromatic pixels remain achromatic.
        maximum = np.maximum(np.maximum(linear[..., 0], linear[..., 1]), linear[..., 2])
        minimum = np.minimum(np.minimum(linear[..., 0], linear[..., 1]), linear[..., 2])
        saturation = (maximum - minimum) / np.maximum(maximum, 1e-30)
        chroma *= (1 + values['vibrance'] / 100.0 * (1 - 0.75 * saturation))[..., None]
    if values['warmth'] or values['tint']:
        # Explicit blue/amber and green/magenta colour balance. These are
        # user-editing units, not Kelvin or a reconstructed sensor white balance.
        warm = np.array((1.0, 0.0, -1.0)); warm -= float(warm @ _LUMA)
        tint = np.array((0.5, -1.0, 0.5)); tint -= float(tint @ _LUMA)
        shift = 0.30 * values['warmth'] / 100.0 * warm + 0.25 * values['tint'] / 100.0 * tint
        chroma += y[..., None] * shift
    return _gamut(y, chroma)


@lru_cache(maxsize=64)
def _kernel(sigma):
    radius = _radius(sigma)
    if not radius:
        return np.array([1.0], dtype=np.float64)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    kernel.flags.writeable = False
    return kernel


def _convolve_axis(array, kernel, axis, cancelled):
    _cancel(cancelled)
    radius = len(kernel) // 2
    if not radius:
        return array
    padding = [(0, 0)] * array.ndim; padding[axis] = (radius, radius)
    padded = np.pad(array, padding, mode='reflect')
    if len(kernel) <= 9:
        result = np.zeros_like(array)
        for n, weight in enumerate(kernel):
            _cancel(cancelled)
            selection = [slice(None)] * array.ndim
            selection[axis] = slice(n, n + array.shape[axis])
            result += padded[tuple(selection)] * weight
        return result
    # Linear, zero-padded FFT convolution of the explicitly reflected data.
    # This evaluates the same finite Gaussian as direct convolution, without
    # an O(radius) pass across each large tile for wide clarity kernels.
    full = padded.shape[axis] + len(kernel) - 1
    length = 1 << (full - 1).bit_length()
    spectrum = np.fft.rfft(padded, n=length, axis=axis)
    weights = np.fft.rfft(kernel, n=length)
    shape = [1] * array.ndim; shape[axis] = len(weights)
    spectrum *= weights.reshape(shape)
    _cancel(cancelled)
    result = np.fft.irfft(spectrum, n=length, axis=axis)
    selection = [slice(None)] * array.ndim
    selection[axis] = slice(len(kernel) - 1, len(kernel) - 1 + array.shape[axis])
    return result[tuple(selection)].copy()


def _gaussian(array, sigma, cancelled=None):
    kernel = _kernel(float(sigma))
    return _convolve_axis(_convolve_axis(array, kernel, 1, cancelled), kernel, 0, cancelled)


def _weighted_blur(array, sigma, alpha, cancelled):
    if alpha is None:
        return _gaussian(array, sigma, cancelled)
    weight = alpha if array.ndim == 2 else alpha[..., None]
    numerator = _gaussian(array * weight, sigma, cancelled)
    denominator = np.maximum(_gaussian(alpha, sigma, cancelled), 0.0)
    if array.ndim == 3:
        denominator = denominator[..., None]
    return np.divide(numerator, denominator, out=array.copy(), where=denominator > 1e-12)


def _alpha_view(alpha, shape):
    if alpha is None:
        return None, 1.0
    if not isinstance(alpha, np.ndarray):
        raise ValueError('透明度需要像素数组')
    if alpha.shape == shape + (1,):
        alpha = alpha[..., 0]
    if alpha.shape != shape:
        raise ValueError('透明度与照片尺寸不匹配')
    if alpha.dtype in (np.uint8, np.uint16):
        return alpha, float(np.iinfo(alpha.dtype).max)
    if alpha.dtype.kind != 'f' or not np.all(np.isfinite(alpha)) or np.any(alpha < 0) or np.any(alpha > 1):
        raise ValueError('透明度应为 0–1 或 8/16 位整数')
    return alpha, 1.0


def apply(rgb, value=None, cancelled=None, *, scale=1.0, alpha=None):
    """Apply explicit edits to RGB8/RGB16 without changing resolution or alpha.

    ``scale`` is the supplied array's resolution divided by the original
    photo resolution. Full-resolution crops use 1, regardless of display zoom.
    ``soften`` is Gaussian softening over the entire image, not denoising.
    """
    values = validate(value); scale = _scale(scale)
    if not isinstance(rgb, np.ndarray) or rgb.dtype not in (np.uint8, np.uint16) or rgb.ndim != 3 or rgb.shape[2] != 3 or not all(rgb.shape[:2]):
        raise ValueError('基础调整需要非空 RGB8 / RGB16 图像')
    _cancel(cancelled)
    if not any(values.values()):
        return rgb
    alpha, alpha_maximum = _alpha_view(alpha, rgb.shape[:2])
    maximum = float(np.iinfo(rgb.dtype).max)
    support = halo(values, scale)
    height, width = rgb.shape[:2]
    out = np.empty_like(rgb)
    for top in range(0, height, _TILE_HEIGHT):
        bottom = min(height, top + _TILE_HEIGHT)
        y0, y1 = max(0, top - support), min(height, bottom + support)
        for left in range(0, width, _TILE_WIDTH):
            _cancel(cancelled)
            right = min(width, left + _TILE_WIDTH)
            x0, x1 = max(0, left - support), min(width, right + support)
            base = _colour(rgb[y0:y1, x0:x1].astype(np.float64) / maximum, values)
            a = None if alpha is None else alpha[y0:y1, x0:x1].astype(np.float64) / alpha_maximum
            current = base
            if values['clarity'] or values['texture'] or values['sharpen']:
                y = _luma(base); delta = np.zeros_like(y)
                if values['clarity']:
                    detail = y - _weighted_blur(y, 12.0 * scale, a, cancelled)
                    delta += values['clarity'] / 100.0 * 0.4 * 0.1 * np.tanh(detail / 0.1)
                if values['texture']:
                    detail = (_weighted_blur(y, 0.7 * scale, a, cancelled) -
                              _weighted_blur(y, 2.0 * scale, a, cancelled))
                    delta += values['texture'] / 100.0 * 0.5 * 0.06 * np.tanh(detail / 0.06)
                if values['sharpen']:
                    detail = y - _weighted_blur(y, 1.0 * scale, a, cancelled)
                    detail *= np.abs(detail) / (np.abs(detail) + 0.002)
                    delta += values['sharpen'] / 100.0 * 0.85 * 0.08 * np.tanh(detail / 0.08)
                current = _gamut(y + delta, base - y[..., None])
            if values['soften']:
                blurred = _weighted_blur(base, 6.0 * scale, a, cancelled)
                amount = values['soften'] / 100.0
                current = current * (1.0 - amount) + blurred * amount
            selection = (slice(top - y0, bottom - y0), slice(left - x0, right - x0))
            encoded = _encode(current[selection])
            result = np.floor(encoded * maximum + 0.5).astype(rgb.dtype)
            if alpha is not None:
                transparent = alpha[top:bottom, left:right] == 0
                result[transparent] = rgb[top:bottom, left:right][transparent]
            out[top:bottom, left:right] = result
    _cancel(cancelled)
    return out
