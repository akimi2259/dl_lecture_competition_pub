import numpy as np
#import torch
from typing import Tuple
#from termcolor import cprint
from glob import glob
import scipy
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy import signal
from scipy.sparse import csc_matrix
from scipy.sparse import spdiags
import scipy.sparse.linalg as spla
from matplotlib import pylab as plt
#os.environ["MKL_NUM_THREADS"] = "16"
from pathlib import Path
import time
from multiprocessing import Pool


def baseline_als(y, lam = 100, p=0.3, niter=100):
  """
  波形データのベースラインを求めるための関数
    y: np.array
    lam : int
    p : float
    niter : int(計算回数)
  """
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
    """
    波形データのノイズを除く関数
        y: np.array(一次元データ)
        dn : int(len(y)/dn > polyじゃないとダメ)
        poly : int(多項式の次元)
    """
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
    y -= baseline_als(y, niter = 5, lam = 100)
    y = SGs(y, dn = 15)
    return y

def wave_process_train(path):
    file_name = Path(path).name
    # データの加工
    y = np.load(path)
    y_max = np.max(y,axis=1)
    y_min = np.min(y,axis=1)
    y = (y.T-y_min)/(y_max-y_min)
    y = y.T
    y = np.apply_along_axis(adjust_wave_data, 1, y)
    processed_folder = Path("data_processed_train")
    #processed_data = process_data(file_path)
    # 加工したデータをdata_processedフォルダに保存
    save_path = processed_folder / file_name
    np.save(save_path, y)

def wave_process_test(path):
    file_name = Path(path).name
    # データの加工
    y = np.load(path)
    y_max = np.max(y,axis=1)
    y_min = np.min(y,axis=1)
    y = (y.T-y_min)/(y_max-y_min)
    y = y.T
    y = np.apply_along_axis(adjust_wave_data, 1, y)
    processed_folder = Path("data_processed_test")
    #processed_data = process_data(file_path)
    # 加工したデータをdata_processedフォルダに保存
    save_path = processed_folder / file_name
    np.save(save_path, y)

def wave_process_val(path):
    file_name = Path(path).name
    # データの加工
    y = np.load(path)
    y_max = np.max(y,axis=1)
    y_min = np.min(y,axis=1)
    y = (y.T-y_min)/(y_max-y_min)
    y = y.T
    y = np.apply_along_axis(adjust_wave_data, 1, y)
    processed_folder = Path("data_processed_val")
    #processed_data = process_data(file_path)
    # 加工したデータをdata_processedフォルダに保存
    save_path = processed_folder / file_name
    np.save(save_path, y)

'''
def main():
    start = time.time()
    n = 0
    for idx in train_X_all:
        n +=1
        file_name = Path(idx).name
        # データの加工
        y = np.load(idx)
        y_max = np.max(y,axis=1)
        y_min = np.min(y,axis=1)
        y = (y.T-y_min)/(y_max-y_min)
        y = y.T
        y = np.apply_along_axis(adjust_wave_data, 1, y)
        processed_folder = Path("data_processed")
        #processed_data = process_data(file_path)
        # 加工したデータをdata_processedフォルダに保存
        save_path = processed_folder / file_name
        np.save(save_path, y)
        if n % 1000 == 0:
            print(f"{n}個目のデータを処理しました")
    end = time.time()
    print(f"{end-start}秒かかった")
    start = time.time()
    n = 0
    for idx in test_X_all:
        n +=1
        file_name = Path(idx).name
        # データの加工
        y = np.load(idx)
        y_max = np.max(y,axis=1)
        y_min = np.min(y,axis=1)
        y = (y.T-y_min)/(y_max-y_min)
        y = y.T
        y = np.apply_along_axis(adjust_wave_data, 1, y)
        processed_folder = Path("data_processed")
        #processed_data = process_data(file_path)
        # 加工したデータをdata_processedフォルダに保存
        save_path = processed_folder / file_name
        np.save(save_path, y)
        if n % 1000 == 0:
            print(f"{n}個目のデータを処理しました")
    end = time.time()
    print(f"{end-start}秒かかった")
    start = time.time()
    n = 0
    for idx in val_X_all:
        n += 1
        file_name = Path(idx).name
        # データの加工
        y = np.load(idx)
        y_max = np.max(y,axis=1)
        y_min = np.min(y,axis=1)
        y = (y.T-y_min)/(y_max-y_min)
        y = y.T
        y = np.apply_along_axis(adjust_wave_data, 1, y)
        processed_folder = Path("data_processed")
        #processed_data = process_data(file_path)
        # 加工したデータをdata_processedフォルダに保存
        save_path = processed_folder / file_name
        np.save(save_path, y)
        if n % 1000 == 0:
            print(f"{n}個目のデータを処理しました")
    end = time.time()
    print(f"{end-start}秒かかった")
'''

if __name__ == '__main__':
    import glob
    train_X_all = glob.glob("data/train_X/*")
    test_X_all = glob.glob("data/test_X/*")
    val_X_all = glob.glob("data/val_X/*")
    p = Pool(18)
    #p.map(wave_process_train, train_X_all)
    print("end first")
    p.map(wave_process_test,test_X_all)
    print("end second")
    p.map(wave_process_val,val_X_all)
   
