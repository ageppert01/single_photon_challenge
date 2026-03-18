import numpy as np
import torch

#from visionsim.emulate.spc import spc_avg_to_rgb


def load_photoncube(path: str) -> np.ndarray:
    return np.load(path, mmap_mode="r")


def naive_sum_preprocess(
    photoncube: np.ndarray,
    num_frames: int = 16,
    average: bool = False,
) -> np.ndarray:
    # Take last num_frames frames
    frames = photoncube[-num_frames:]

    # Unpack bits along channel dimension (axis=2)
    frames = np.unpackbits(frames, axis=2)

    # Sum over time
    image = frames.sum(axis=0)

    if average:
        image = image / num_frames

    return image.astype(np.float32)


def invert_response(
    image: np.ndarray,
    factor: float = 0.5,
) -> np.ndarray:
    # VisionSIM response inversion
    return False #spc_avg_to_rgb(image, factor=factor)


def preprocess_photoncube(
    photoncube: np.ndarray,
    num_frames: int = 16,
    average: bool = False,
    invert: bool = True,
    invert_factor: float = 0.5,
) -> np.ndarray:
    image = naive_sum_preprocess(
        photoncube,
        num_frames=num_frames,
        average=average,
    )

    #if invert:
    #    image = invert_response(image, factor=invert_factor)

    return image


def photoncube_file_to_tensor(
    path: str,
    num_frames: int = 16,
    average: bool = False,
    invert: bool = True,
    invert_factor: float = 0.5,
) -> torch.Tensor:
    photoncube = load_photoncube(path)

    image = preprocess_photoncube(
        photoncube,
        num_frames=num_frames,
        average=average,
        invert=invert,
        invert_factor=invert_factor,
    )

    # Normalize to [-1, 1]
    image = image / 255.0
    image = (image - 0.5) * 2.0

    # Convert to tensor (C, H, W)
    tensor = torch.from_numpy(image).permute(2, 0, 1)

    return tensor