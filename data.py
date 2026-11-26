"""MNIST / Fashion-MNIST / CIFAR-10 loading (numpy only)."""
import gzip, os, pickle, tarfile, urllib.request
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

_CIFAR_DIR = os.path.join(_DIR, 'cifar_dl')
_CIFAR_TAR = os.path.join(_CIFAR_DIR, 'cifar-10-python.tar.gz')
_OK_GLOBALS = {('numpy.core.multiarray', '_reconstruct'), ('numpy._core.multiarray', '_reconstruct'),
               ('numpy', 'ndarray'), ('numpy', 'dtype')}


class _NumpyOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _OK_GLOBALS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f'forbidden global {module}.{name}')


def _cifar_raw():
    npz = os.path.join(_DIR, 'cifar10.npz')
    if os.path.exists(npz):
        d = np.load(npz)
        return d['Xtr'], d['ytr'], d['Xte'], d['yte']
    root = os.path.realpath(_CIFAR_DIR)
    with tarfile.open(_CIFAR_TAR, 'r:gz') as tf:
        for m in tf.getmembers():
            tgt = os.path.realpath(os.path.join(root, m.name))
            if os.path.commonpath([root, tgt]) != root or not (m.isfile() or m.isdir()):
                raise RuntimeError(f'unsafe tar member {m.name}')
        tf.extractall(root)

    def batch(fn):
        with open(os.path.join(root, 'cifar-10-batches-py', fn), 'rb') as fh:
            d = _NumpyOnlyUnpickler(fh, encoding='bytes').load()
        X = np.asarray(d[b'data'], np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).reshape(-1, 3072)
        return X, np.asarray(d[b'labels'], np.int64)

    parts = [batch(f'data_batch_{i}') for i in range(1, 6)]
    Xtr = np.concatenate([p[0] for p in parts])
    ytr = np.concatenate([p[1] for p in parts])
    Xte, yte = batch('test_batch')
    np.savez(npz, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    return Xtr, ytr, Xte, yte


def load(name, split):
    """Return (Xtr, ytr, Xev, yev); X float32 in [0,1] (n,784) or (n,3072 HWC) for cifar10, y int64."""
    assert name in ('mnist', 'fashion', 'cifar10') and split in ('val', 'test')
    Xtr, ytr, Xte, yte = _cifar_raw() if name == 'cifar10' else _raw(name)
    perm = np.random.RandomState(1234).permutation(len(Xtr))
    nv = 5000 if name == 'cifar10' else 6000
    keep, val = np.sort(perm[:-nv]), np.sort(perm[-nv:])
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
