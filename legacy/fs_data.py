"""Read-only lazy features; one explicit input contract for both model families."""
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from spec import TASKS, RANGES, FEATURES


def labels(path, task):
    index = TASKS.index(task) + 1
    rows = []
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        if not line.strip(): continue
        cols = line.split()
        if len(cols) != 9 or Path(cols[0]).name != cols[0] or '\\' in cols[0]:
            raise ValueError('Invalid FS1000 label row')
        raw = np.asarray([float(v) for v in cols[1:]], dtype=np.float64)
        if not np.isfinite(raw).all() or raw[-1] <= 0: raise ValueError('Invalid label/factor')
        score = float(cols[index]) / (float(cols[8]) if task == 'PCS' else 1.) / RANGES[task]
        rows.append((cols[0], score))
    if len({r[0] for r in rows}) != len(rows): raise ValueError('Duplicate IDs')
    return rows


def validate_splits(root):
    root = Path(root)
    a = labels(root / 'train_fs1000_new.txt', 'TES')
    b = labels(root / 'val_fs1000_new.txt', 'TES')
    if len(a) != 1000 or len(b) != 247 or {v[0] for v in a} & {v[0] for v in b}:
        raise ValueError('Expected official disjoint 1000/247 split')
    return a, b


def aligned(arrays):
    v, a, f = arrays
    if v.ndim != 3 or v.shape[1:] != (15, 768) or a.ndim != 2 or a.shape[1] != 768 or f.ndim != 2 or f.shape[1] != 1024:
        raise ValueError('Unexpected feature dimensions')
    if len(v) < 8 or (len(a) - len(v), len(f) - len(v)) not in ((0, 0), (1, 1)):
        raise ValueError('Unexpected temporal misalignment')
    if any(x.dtype != np.float32 or not np.isfinite(x).all() for x in arrays):
        raise ValueError('Features must be finite float32')
    return (v.mean(1), np.array(a[:len(v)], copy=True), np.array(f[:len(v)], copy=True))


class FSData(Dataset):
    def __init__(self, root, task, train, smoke=False, external=False, crop='center95'):
        self.root = Path(root); self.train = train; self.external = external; self.crop = crop
        rows = labels(self.root / ('train_fs1000_new.txt' if train else 'val_fs1000_new.txt'), task)
        # Engineering smoke only; deterministic spread in file order, not score-selected.
        if smoke: rows = [rows[i] for i in np.linspace(0, len(rows)-1, 32, dtype=int)]
        self.rows = rows

    def __len__(self): return len(self.rows)

    @lru_cache(maxsize=64)
    def features(self, index):
        name = self.rows[index][0] + '.npy'
        return aligned([np.load(self.root / directory / name, mmap_mode='r', allow_pickle=False) for directory in FEATURES])

    def __getitem__(self, index):
        arrays = self.features(index)
        length = len(arrays[0]); valid = min(length, 95)
        if self.train or self.crop == 'center95':
            if length > 95:
                start = np.random.randint(length - 95) if self.train else (length - 95) // 2
                arrays = tuple(x[start:start+95] for x in arrays)
            elif length < 95:
                arrays = tuple(np.pad(x, ((0, 95-length), (0, 0))) for x in arrays)
        else: valid = length
        result = tuple(torch.from_numpy(np.array(x, dtype=np.float32, copy=True)) for x in arrays)
        result += (np.float32(self.rows[index][1]),)
        return result if self.external else result + ({key: valid for key in ('V', 'F', 'A')},)


def loaders(root, task, smoke=False, external=False, workers=2, crop='center95'):
    train = FSData(root, task, True, smoke, external)
    test = FSData(root, task, False, smoke, external, crop)
    return (DataLoader(train, batch_size=32, shuffle=True, num_workers=workers),
            DataLoader(test, batch_size=1, shuffle=False, num_workers=workers))
