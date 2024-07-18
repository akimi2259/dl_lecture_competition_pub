import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from glob import glob
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy import signal
from scipy.sparse import csc_matrix
from scipy.sparse import spdiags
import scipy.sparse.linalg as spla
import random

def baseline_als(y, lam = 100, p=0.3, niter=100):
  L = len(y)
  D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
  w = np.ones(L)
  for i in range(niter):
    W = sparse.spdiags(w, 0, L, L)
    Z = W + lam * D.dot(D.transpose())
    z = spsolve(Z, w*y)
    w = p * (y > z) + (1-p) * (y < z)
  return z

def SGs(y,dn=15,poly=7):
    # y as np.array, dn as int, poly as int
    n = len(y) // dn
    if n % 2 == 0:
        N = n+1
    elif n % 2 == 1:
        N = n
    else:
        print("window length can't set as odd")
    SGsmoothed = signal.savgol_filter(y, window_length=N, polyorder=poly)
    return SGsmoothed

def adjust_wave_data(y):
    y -= baseline_als(y, niter = 100, lam = 100)
    y = SGs(y, dn = 15)
    return y

def image_augmentation(
        X: np.array,
        gate: int = 10,
        width_x: int = 10,
        width_y: int = 30,
)-> np.array:
    """
    SpecAugmentをベースにしている
    Xの二次元データに対して、水平方向に拡大したり、水平方向・鉛直方向を0にすることで、データの水増しを行う関数
    Args:
        X : 拡張するデータ
        gate : 両端のマスキングを行う場合の最大値
        width_x : 水平方向にマスキングする際の最大の幅
        width_y : 鉛直方向にマスキングする際の最大の幅
    Returns:
        y: ランダムにマスキングされたデータ(np.array)
    """
    C, L = X.shape#チャネル数、データ長の取得

    if random.uniform(0,1) < 0.4:
        g_a = random.randint(0,gate)
        if random.uniform(0,1) < 0.5:
            X[:g_a, :] = 0
        if random.uniform(0,1) < 0.5:
            X[-g_a:, :] = 0

    if random.uniform(0,1) < 0.7:
        f = random.randint(0,width_x)
        c0 = random.randint(0,C-f)
        X[c0:c0+f, :] = 0
    if random.uniform(0,1) < 0.15:
        f = random.randint(0,width_x)
        c0 = random.randint(0,C-f)
        X[c0:c0+f, :] = 0
    
    if random.uniform(0,1) < 0.7:
        t = random.randint(0,width_y)
        t0 = random.randint(0,L-t)
        X[:, t0:t0+t] = 0
    if random.uniform(0,1) < 0.15:
        t = random.randint(0,width_y)
        t0 = random.randint(0,L-t)
        X[:, t0:t0+t] = 0
    return X


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", p: float = 0.3) -> None:
        super().__init__()
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.p = p
        self.split = split
        self.data_dir = data_dir
        self.num_classes = 1854
        self.num_samples = len(glob(os.path.join(data_dir, f"{split}_X", "*.npy")))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, i):
        X_path = os.path.join(self.data_dir, f"{self.split}_X", str(i).zfill(5) + ".npy")
        
        y = np.load(X_path)
        y = ((y - np.median(y)) / np.std(y)) + 0.5
        if self.split == "train":
            if random.uniform(0,1) < self.p:
                y = image_augmentation(y)
        
        X = torch.from_numpy(y)#.half()
        
        subject_idx_path = os.path.join(self.data_dir, f"{self.split}_subject_idxs", str(i).zfill(5) + ".npy")
        subject_idx = torch.from_numpy(np.load(subject_idx_path))
        
        if self.split in ["train", "val"]:
            y_path = os.path.join(self.data_dir, f"{self.split}_y", str(i).zfill(5) + ".npy")
            y = torch.from_numpy(np.load(y_path))
            
            return X, y, subject_idx
        else:
            return X, subject_idx
        
    @property
    def num_channels(self) -> int:
        return np.load(os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")).shape[0]
    
    @property
    def seq_len(self) -> int:
        return np.load(os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")).shape[1]