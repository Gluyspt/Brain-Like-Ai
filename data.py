"""MNIST / Fashion-MNIST loading (numpy only)."""
import gzip, os, urllib.request
import numpy as np

TASKS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_cache')
_URL = {'mnist': 'https://raw.githubusercontent.com/fgnt/mnist/master/',
        'fashion': 'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/'}
_FILES = ['train-images-idx3-ubyte', 'train-labels-idx1-ubyte',
          't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte']


def _raw(name):
    npz = os.path.join(_DIR, name + '.npz')
    if os.path.exists(npz):
        d = np.load(npz)
        return d['Xtr'], d['ytr'], d['Xte'], d['yte']
    os.makedirs(_DIR, exist_ok=True)
    arrs = []
    for f in _FILES:
        p = os.path.join(_DIR, f'{name}-{f}.gz')
        if not os.path.exists(p):
            urllib.request.urlretrieve(_URL[name] + f + '.gz', p)
        with gzip.open(p, 'rb') as fh:
            b = fh.read()
        if 'images' in f:
            arrs.append(np.frombuffer(b, np.uint8, offset=16).reshape(-1, 784).copy())
        else:
            arrs.append(np.frombuffer(b, np.uint8, offset=8).astype(np.int64))
    Xtr, ytr, Xte, yte = arrs
    np.savez(npz, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    return Xtr, ytr, Xte, yte


def load(name, split):
    """Return (Xtr, ytr, Xev, yev); X float32 in [0,1] (n,784), y int64."""
    assert name in ('mnist', 'fashion') and split in ('val', 'test')
    Xtr, ytr, Xte, yte = _raw(name)
    perm = np.random.RandomState(1234).permutation(len(Xtr))
    keep, val = np.sort(perm[:-6000]), np.sort(perm[-6000:])
    f = lambda X: (X.astype(np.float32) / 255.0)
    if split == 'val':
        return f(Xtr[keep]), ytr[keep], f(Xtr[val]), ytr[val]
    return f(Xtr[keep]), ytr[keep], f(Xte), yte


def task_indices(y, classes, cap_per_class=None):
    """Ascending indices of samples whose label is in classes (optionally first cap per class)."""
    y = np.asarray(y)
    if cap_per_class is None:
        return np.flatnonzero(np.isin(y, list(classes)))
    idx = [np.flatnonzero(y == c)[:cap_per_class] for c in classes]
    return np.sort(np.concatenate(idx))
