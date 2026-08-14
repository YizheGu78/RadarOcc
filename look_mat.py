import scipy.io
import numpy as np

mat_path = "/home/user1/projects/RadarOcc/data/K-Radar/11/radar_tensor_8doppler/tesseract_00034.mat"

data = scipy.io.loadmat(mat_path)

# 1. 看 mat 文件里有哪些变量
print("包含的变量：")
for key, value in data.items():
    if not key.startswith("__"):
        print(
            key,
            "shape =", getattr(value, "shape", None),
            "dtype =", getattr(value, "dtype", None)
        )

# 2. 取出真正的雷达 Tensor
radar = data["arrDREA"]

print("\n原始 shape:", radar.shape)

# arrDREA = Doppler, Range, Elevation, Azimuth
d_dim, r_dim, e_dim, a_dim = radar.shape

print("Doppler:", d_dim)
print("Range:", r_dim)
print("Elevation:", e_dim)
print("Azimuth:", a_dim)