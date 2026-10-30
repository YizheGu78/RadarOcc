
import numpy as np
from scipy.io import loadmat

path = "data/K-Radar-official-meta/resources/arr_doppler.mat"
meta = loadmat(path)

# 数组全部显示，不用 ... 省略
np.set_printoptions(threshold=np.inf, linewidth=160)

for name, value in meta.items():
    print(f"\n字段：{name}")
    if isinstance(value, np.ndarray):
        print(f"形状：{value.shape}，类型：{value.dtype}")
    print(value)
