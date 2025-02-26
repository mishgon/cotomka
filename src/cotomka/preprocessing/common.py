from typing import Tuple, Sequence
import itertools
import numpy as np
from skimage.segmentation import flood
from skimage.measure import block_reduce
from imops import pad


def get_body_box(image: np.ndarray, voxel_spacing: Tuple[float, float, float]) -> np.ndarray:
    block_size = tuple(np.int64(np.ceil(5.0 / np.array(voxel_spacing))))
    BODY_THRESHOLD_HU = -500
    mask = block_reduce(image >= BODY_THRESHOLD_HU, block_size=block_size, func=np.min)
    bbox = mask_to_bbox(mask)
    return bbox * block_size


def get_body_mask(image: np.ndarray) -> np.ndarray:
    BODY_THRESHOLD_HU = -500
    air_mask = image < BODY_THRESHOLD_HU
    air_mask = pad(air_mask, padding=1, axis=(0, 1), padding_values=True, num_threads=-1, backend='Scipy')
    body_mask = ~flood(air_mask, seed_point=(0, 0, 0))
    body_mask = body_mask[1:-1, 1:-1]
    return body_mask


def rescale_hu_piecewise(
        image: np.ndarray,
        hu_pivots: Sequence[float] = (-1000., -200., 200., 1500.),
        rescaled_pivots: Sequence[float] = (0.0, 0.4, 0.8, 1.0)
) -> np.ndarray:
    """Proposed in https://arxiv.org/abs/2102.01897.
    """
    rescaled_image = np.zeros_like(image)
    rescaled_image[image < hu_pivots[0]] = rescaled_pivots[0]
    for hu1, hu2, p1, p2 in zip(hu_pivots, hu_pivots[1:], rescaled_pivots, rescaled_pivots[1:]):
        mask = (image >= hu1) & (image < hu2)
        rescaled_image[mask] = (image[mask] - hu1) / (hu2 - hu1) * (p2 - p1) + p1
    rescaled_image[image >= hu_pivots[-1]] = rescaled_pivots[-1]
    return rescaled_image


def mask_to_bbox(mask: np.ndarray) -> np.ndarray:
    """
    Find the smallest box that contains all true values of the ``mask``.
    """
    if not mask.any():
        raise ValueError('The mask is empty.')

    start, stop = [], []
    for ax in itertools.combinations(range(mask.ndim), mask.ndim - 1):
        nonzero = np.any(mask, axis=ax)
        if np.any(nonzero):
            left, right = np.where(nonzero)[0][[0, -1]]
        else:
            left, right = 0, 0
        start.insert(0, left)
        stop.insert(0, right + 1)

    return np.array([start, stop])
