import numpy as np
import pickle
from pathlib import Path

def save_file(data, path, overwrite = False):
    suffix = '.pickle'
    path = Path(path)
    if path.suffix != suffix:
        path = path.with_suffix(suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if overwrite:
            path.unlink()
        else:
            raise RuntimeError(f'File {path} already exists.')
    with open(path, 'wb') as file:
        pickle.dump(data, file)
    return 'File saved!'


def load_file(path):
    suffix = '.pickle'
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Not a file: {path}')
    if path.suffix != suffix:
        raise ValueError(f'Not a {suffix} file: {path}')
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data


def DataLoader(arrays, batch_size, *, seed=20):
    """
    Returns batches data.

    :param arrays: Tuple of arrays to get batches from
    :param batch_size: Batch Size
    :param key:
    :return: Tuple of batches
    """

    np.random.seed(seed)
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    assert all(array.shape[0] >= batch_size for array in arrays)
    indices = np.arange(dataset_size)

    perm = np.random.permutation(indices)
    start = 0
    end = batch_size
    while end <= dataset_size:
        batch_perm = perm[start:end]
        yield tuple(array[batch_perm] for array in arrays)
        start = end
        end = start + batch_size