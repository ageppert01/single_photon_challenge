# photoncube_preprocess.py
import numpy as np
import torch

try:
    from visionsim.emulate.spc import spc_avg_to_rgb
    VISIONSIM_AVAILABLE = True
except ImportError:
    VISIONSIM_AVAILABLE = False


def load_photoncube(path: str):
    return np.load(path, mmap_mode="r")


def naive_sum_preprocess(
    cube: np.ndarray,
    num_frames: int = 1024,
    average: bool = False,
):
    # take last frames
    cube = cube[:, :, -num_frames:]

    # unpack bit-packed data along width axis
    cube = np.unpackbits(cube, axis=2)

    # sum across time dimension
    image = cube.sum(axis=2)

    if average:
        image = image / num_frames

    return image.astype(np.float32)


def preprocess_photoncube(
    cube: np.ndarray,
    num_frames: int = 1024,
    average: bool = False,
    invert_response: bool = True,
    invert_response_factor: float = 0.5,
):
    image = naive_sum_preprocess(cube, num_frames=num_frames, average=average)

    if invert_response:
        if not VISIONSIM_AVAILABLE:
            raise ImportError("visionsim is required for invert_response=True")

        image = spc_avg_to_rgb(image, factor=invert_response_factor)

    return image


def photoncube_file_to_tensor(
    path: str,
    num_frames: int = 1024,
    average: bool = False,
    invert_response: bool = True,
    invert_response_factor: float = 0.5,
):
    cube = load_photoncube(path)
    image = preprocess_photoncube(
        cube,
        num_frames=num_frames,
        average=average,
        invert_response=invert_response,
        invert_response_factor=invert_response_factor,
    )

    # normalize to [0,1]
    image = image - image.min()
    if image.max() > 0:
        image = image / image.max()

    # convert to torch tensor (C, H, W)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=0)
    else:
        image = image.transpose(2, 0, 1)

    return torch.from_numpy(image).float()