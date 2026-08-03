import argparse
import re
from pathlib import Path
from typing import List

import numpy as np
from mayavi import mlab


# RadarOcc配置
POINT_CLOUD_RANGE = np.array(
    [0.0, -25.6, -2.6, 51.2, 25.6, 3.0],
    dtype=np.float64,
)

OCC_SIZE = np.array(
    [128, 128, 14],
    dtype=np.int64,
)

# RadarOcc Small实际上输出3类：
# 0 = free，不保存
# 1 = Background
# 2 = Foreground
CLASS_COLORS = np.array(
    [
        [100, 150, 245, 255],  # Background：蓝色
        [255, 30, 30, 255],    # Foreground：红色
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render RadarOcc pred_c.npy using a "
            "MonoScene-style Mayavi voxel renderer."
        )
    )

    parser.add_argument(
        "input",
        help=(
            "A pred_c.npy file or a scene directory containing "
            "<lidar_token>/pred_c.npy files."
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path when rendering one pred_c.npy.",
    )

    parser.add_argument(
        "--frames-dir",
        default=None,
        help="Output directory when rendering a complete scene.",
    )

    parser.add_argument(
        "--offscreen",
        action="store_true",
        help="Render without opening a GUI window.",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Render every N-th frame.",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to render.",
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start frame index after sorting.",
    )

    parser.add_argument(
        "--azimuth",
        type=float,
        default=180.0,
    )

    parser.add_argument(
        "--elevation",
        type=float,
        default=65.0,
    )

    parser.add_argument(
        "--distance",
        type=float,
        default=82.0,
    )

    parser.add_argument(
        "--image-width",
        type=int,
        default=1600,
    )

    parser.add_argument(
        "--image-height",
        type=int,
        default=900,
    )

    return parser.parse_args()


def natural_key(text: str):
    """Sort names such as 1, 2, 10 in numerical order."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    ]


def discover_prediction_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        if input_path.name != "pred_c.npy":
            raise ValueError(
                f"Expected pred_c.npy, got {input_path.name}"
            )
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(input_path)

    # 典型目录：
    # scene_token/lidar_token/pred_c.npy
    files = list(input_path.glob("*/pred_c.npy"))

    if not files:
        files = list(input_path.rglob("pred_c.npy"))

    files.sort(
        key=lambda path: natural_key(path.parent.name)
    )

    if not files:
        raise FileNotFoundError(
            f"No pred_c.npy files found under {input_path}"
        )

    return files


def load_radarocc_sparse(path: Path):
    data = np.load(path)

    if data.ndim != 2 or data.shape[1] != 4:
        raise ValueError(
            f"{path}: expected shape (N, 4), got {data.shape}"
        )

    if len(data) == 0:
        return (
            np.empty(0),
            np.empty(0),
            np.empty(0),
            np.empty(0, dtype=np.int32),
        )

    # RadarOcc save_occ.py保存顺序：
    # [z_index, y_index, x_index, class_id]
    z_index = data[:, 0].astype(np.float64)
    y_index = data[:, 1].astype(np.float64)
    x_index = data[:, 2].astype(np.float64)
    class_id = data[:, 3].astype(np.int32)

    valid = (
        (class_id > 0)
        & (class_id < 255)
        & (x_index >= 0)
        & (x_index < OCC_SIZE[0])
        & (y_index >= 0)
        & (y_index < OCC_SIZE[1])
        & (z_index >= 0)
        & (z_index < OCC_SIZE[2])
    )

    x_index = x_index[valid]
    y_index = y_index[valid]
    z_index = z_index[valid]
    class_id = class_id[valid]

    voxel_size = (
        POINT_CLOUD_RANGE[3:] - POINT_CLOUD_RANGE[:3]
    ) / OCC_SIZE

    # 体素索引转换到体素中心的真实米制坐标
    x = (
        POINT_CLOUD_RANGE[0]
        + (x_index + 0.5) * voxel_size[0]
    )

    y = (
        POINT_CLOUD_RANGE[1]
        + (y_index + 0.5) * voxel_size[1]
    )

    z = (
        POINT_CLOUD_RANGE[2]
        + (z_index + 0.5) * voxel_size[2]
    )

    return x, y, z, class_id


def render_prediction(
    prediction_path: Path,
    output_path: Path | None,
    offscreen: bool,
    azimuth: float,
    elevation: float,
    distance: float,
    image_size: tuple[int, int],
):
    mlab.options.offscreen = offscreen

    x, y, z, class_id = load_radarocc_sparse(
        prediction_path
    )

    figure = mlab.figure(
        size=image_size,
        bgcolor=(1.0, 1.0, 1.0),
        fgcolor=(0.0, 0.0, 0.0),
    )

    # 正交投影更接近论文中的体素可视化效果
    figure.scene.parallel_projection = True

    voxel_size = (
        POINT_CLOUD_RANGE[3:] - POINT_CLOUD_RANGE[:3]
    ) / OCC_SIZE

    cube_size = float(np.min(voxel_size) * 0.92)

    if len(x) > 0:
        voxel_plot = mlab.points3d(
            x,
            y,
            z,
            class_id,
            mode="cube",
            scale_mode="none",
            scale_factor=cube_size,
            opacity=1.0,
            vmin=1,
            vmax=2,
        )

        voxel_plot.module_manager.scalar_lut_manager.lut.table = (
            CLASS_COLORS
        )

    # 雷达/车辆原点
    mlab.points3d(
        0.0,
        0.0,
        0.0,
        mode="cube",
        scale_mode="none",
        scale_factor=0.8,
        color=(0.0, 0.0, 0.0),
    )

    # 车辆朝向箭头：x正方向
    mlab.plot3d(
        [0.0, 3.0],
        [0.0, 0.0],
        [0.0, 0.0],
        color=(0.0, 0.0, 0.0),
        tube_radius=0.06,
    )

    # 占据范围边框
    mlab.outline(
        extent=[
            POINT_CLOUD_RANGE[0],
            POINT_CLOUD_RANGE[3],
            POINT_CLOUD_RANGE[1],
            POINT_CLOUD_RANGE[4],
            POINT_CLOUD_RANGE[2],
            POINT_CLOUD_RANGE[5],
        ],
        color=(0.25, 0.25, 0.25),
    )

    mlab.axes(
        extent=[
            POINT_CLOUD_RANGE[0],
            POINT_CLOUD_RANGE[3],
            POINT_CLOUD_RANGE[1],
            POINT_CLOUD_RANGE[4],
            POINT_CLOUD_RANGE[2],
            POINT_CLOUD_RANGE[5],
        ],
        xlabel="X / m",
        ylabel="Y / m",
        zlabel="Z / m",
        color=(0.1, 0.1, 0.1),
        nb_labels=5,
    )

    frame_name = prediction_path.parent.name

    mlab.title(
        f"RadarOcc Epoch 4 | Frame {frame_name}",
        size=0.22,
        height=0.94,
    )

    mlab.view(
        azimuth=azimuth,
        elevation=elevation,
        distance=distance,
        focalpoint=(25.6, 0.0, 0.0),
    )

    mlab.roll(0.0)

    if output_path is not None:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        mlab.savefig(
            str(output_path),
            figure=figure,
            size=image_size,
        )

        print(
            f"Saved: {output_path} "
            f"(occupied voxels: {len(x)})"
        )

        mlab.close(figure)
    else:
        print(
            f"Showing {prediction_path} "
            f"(occupied voxels: {len(x)})"
        )
        mlab.show()


def main():
    args = parse_args()

    input_path = Path(args.input)
    files = discover_prediction_files(input_path)

    if len(files) == 1:
        output_path = (
            Path(args.output)
            if args.output is not None
            else None
        )

        render_prediction(
            prediction_path=files[0],
            output_path=output_path,
            offscreen=args.offscreen,
            azimuth=args.azimuth,
            elevation=args.elevation,
            distance=args.distance,
            image_size=(
                args.image_width,
                args.image_height,
            ),
        )
        return

    if args.frames_dir is None:
        raise ValueError(
            "--frames-dir is required when input is a scene directory"
        )

    if args.stride <= 0:
        raise ValueError("--stride must be greater than zero")

    files = files[args.start :: args.stride]

    if args.max_frames is not None:
        files = files[: args.max_frames]

    frames_dir = Path(args.frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scene input: {input_path}")
    print(f"Frames to render: {len(files)}")
    print(f"Output directory: {frames_dir}")

    for frame_index, prediction_path in enumerate(files):
        output_path = (
            frames_dir / f"frame_{frame_index:05d}.png"
        )

        render_prediction(
            prediction_path=prediction_path,
            output_path=output_path,
            offscreen=True,
            azimuth=args.azimuth,
            elevation=args.elevation,
            distance=args.distance,
            image_size=(
                args.image_width,
                args.image_height,
            ),
        )

    print("All frames rendered.")


if __name__ == "__main__":
    main()
