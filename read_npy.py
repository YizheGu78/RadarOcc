import numpy as np

file_path = (
    "/home/user1/projects/RadarOcc/data/RadarOcc_8doppler/58/"
    "radar_tensor_8doppler/EAsparse_00006.npz"
)

with np.load(file_path, allow_pickle=False) as data:
    print("文件类型:", type(data))
    print("包含的数组:", data.files)
    print("=" * 70)

    for key in data.files:
        arr = data[key]

        print(f"key: {key}")
        print(f"  shape: {arr.shape}")
        print(f"  dtype: {arr.dtype}")
        print(f"  ndim:  {arr.ndim}")
        print(f"  size:  {arr.size}")
        print(f"  前10个值: {arr.reshape(-1)[:10]}")

        if np.issubdtype(arr.dtype, np.number):
            print(f"  最小值: {arr.min()}")
            print(f"  最大值: {arr.max()}")
            print(f"  平均值: {arr.mean()}")

        print("-" * 70)