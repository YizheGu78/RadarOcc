import os
import gc
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------------
INPUT_BASE_DIR = "/media/terra2024/Elements/KRadar"

# Input:
#   <INPUT_BASE_DIR>/<seq>/radar_polar_cube/*.npy, when available, or
#   <INPUT_BASE_DIR>/<seq>/radar_tesseract/*.mat (official KRadar layout)

# Output:
#   <OUTPUT_BASE_DIR>/<seq>/radar_tensor_8doppler/EAsparse_*.npz
# Keep generated data outside the KRadar source tree.
OUTPUT_BASE_DIR = "/mnt/elab-share/Datasets/RadarOcc_8doppler_hybrid125"

TOP_K = 125
SPATIAL_K = 125

# 5 x 25 = 125 spatial regions on the Elevation-Azimuth plane
N_ELE_BLOCKS = 5
N_AZI_BLOCKS = 25

# Fixed seed => reproducible spatial sampling
BASE_SEED = 20260813
MAX_WORKERS = 2


def resolve_input(clip):
    """Prefer preconverted NPY cubes, otherwise use official KRadar MAT files."""
    sequence_dir = os.path.join(INPUT_BASE_DIR, clip)
    candidates = (
        (os.path.join(sequence_dir, "radar_polar_cube"), ".npy"),
        (os.path.join(sequence_dir, "radar_tesseract"), ".mat"),
    )

    for input_dir, extension in candidates:
        if os.path.isdir(input_dir):
            return input_dir, extension

    return None, None


def load_radar_cube(file_path):
    """Load a cube without modifying its source file."""
    if file_path.endswith(".npy"):
        return np.load(file_path, mmap_mode="r", allow_pickle=False)

    if file_path.endswith(".mat"):
        from scipy.io import loadmat

        contents = loadmat(file_path, variable_names=("arrDREA",))
        if "arrDREA" not in contents:
            raise KeyError(f"MAT file has no arrDREA tensor: {file_path}")
        data = contents["arrDREA"]
        data.flags.writeable = False
        return data

    raise ValueError(f"Unsupported radar cube format: {file_path}")


def spatial_sample(top_idx, e_dim, a_dim, seed):
    """
    Split the complete E-A plane into 5 x 25 = 125 blocks.
    Select one non-Top125 point from every block.
    """
    rng = np.random.default_rng(seed)

    used = np.zeros(e_dim * a_dim, dtype=bool)
    used[top_idx] = True

    e_edges = np.linspace(0, e_dim, N_ELE_BLOCKS + 1, dtype=int)
    a_edges = np.linspace(0, a_dim, N_AZI_BLOCKS + 1, dtype=int)

    selected = []

    for ei in range(N_ELE_BLOCKS):
        for ai in range(N_AZI_BLOCKS):
            e0, e1 = e_edges[ei], e_edges[ei + 1]
            a0, a1 = a_edges[ai], a_edges[ai + 1]

            candidates = []
            for e in range(e0, e1):
                for a in range(a0, a1):
                    idx = e * a_dim + a
                    if not used[idx]:
                        candidates.append(idx)

            if len(candidates) > 0:
                idx = rng.choice(candidates)
            else:
                # Very rare fallback: use nearest still-unused point
                remaining = np.where(~used)[0]
                center_e = (e0 + e1 - 1) / 2
                center_a = (a0 + a1 - 1) / 2

                rem_e = remaining // a_dim
                rem_a = remaining % a_dim

                dist = (
                    (rem_e - center_e) ** 2
                    + (rem_a - center_a) ** 2
                )
                idx = remaining[np.argmin(dist)]

            selected.append(idx)
            used[idx] = True

    return np.asarray(selected, dtype=np.int64)


def process_file(clip, file_name, input_dir, output_dir):
    file_path = os.path.join(input_dir, file_name)
    data = load_radar_cube(file_path)

    # Expected raw layout: [Doppler, Range, Elevation, Azimuth]
    if data.ndim != 4:
        raise ValueError(
            f"Expected a 4-D [D, R, E, A] cube, got {data.shape}: {file_path}"
        )
    d_dim, r_dim, e_dim, a_dim = data.shape
    if d_dim < 3:
        raise ValueError(f"At least 3 Doppler bins are required: {file_path}")
    if e_dim * a_dim < TOP_K + SPATIAL_K:
        raise ValueError(
            f"E-A plane must contain at least {TOP_K + SPATIAL_K} cells, "
            f"got {e_dim * a_dim}: {file_path}"
        )

    # ---------------------------------------------------------------------
    # 1. Doppler mean -> [Range, Elevation, Azimuth]
    # ---------------------------------------------------------------------
    cube = np.mean(data, axis=0)
    cube_flat = cube.reshape(r_dim, -1)

    # ---------------------------------------------------------------------
    # 2. Per range: Top125 by mean power + Spatial125
    # ---------------------------------------------------------------------
    top_idx_all = np.argpartition(
        cube_flat,
        -TOP_K,
        axis=1,
    )[:, -TOP_K:]

    selected_idx_all = np.empty(
        (r_dim, TOP_K + SPATIAL_K),
        dtype=np.int64,
    )

    frame_idx = int(file_name.split("_")[-1].split(".")[0])

    for r in range(r_dim):
        top_idx = top_idx_all[r]

        spatial_idx = spatial_sample(
            top_idx,
            e_dim,
            a_dim,
            seed=BASE_SEED + int(clip) * 1000000 + frame_idx * 1000 + r,
        )

        selected_idx_all[r, :TOP_K] = top_idx
        selected_idx_all[r, TOP_K:] = spatial_idx

    # ---------------------------------------------------------------------
    # 3. Convert flattened E-A indices back to coordinates
    # ---------------------------------------------------------------------
    k = TOP_K + SPATIAL_K

    elevation_inds = selected_idx_all // a_dim
    azimuth_inds = selected_idx_all % a_dim

    range_inds = np.arange(r_dim)[:, None]

    range_inds_flat = np.repeat(range_inds, k).flatten()
    elevation_inds_flat = elevation_inds.flatten()
    azimuth_inds_flat = azimuth_inds.flatten()

    # ---------------------------------------------------------------------
    # 4. Read complete Doppler spectrum at the selected locations
    # ---------------------------------------------------------------------
    power_val = data[
        :,
        range_inds_flat,
        elevation_inds_flat,
        azimuth_inds_flat,
    ]

    # Same 8-D descriptor as official RadarOcc
    top_indices = np.argpartition(power_val, -3, axis=0)[-3:]
    n_dim = power_val.shape[1]
    top_values = power_val[top_indices, np.arange(n_dim)]

    means = np.mean(power_val, axis=0)
    variances = np.var(power_val, axis=0)

    new_data = np.vstack(
        (
            top_values,
            top_indices,
            means,
            variances,
        )
    ).reshape(8, n_dim)

    # ---------------------------------------------------------------------
    # 5. Save in exactly the same NPZ format as the original generator
    # ---------------------------------------------------------------------
    idx = file_name.split("_")[-1].split(".")[0]
    output_file = os.path.join(
        output_dir,
        f"EAsparse_{idx}.npz",
    )

    # Never replace a result from an earlier run.
    output_created = False
    try:
        with open(output_file, "xb") as output_handle:
            output_created = True
            np.savez(
                output_handle,
                range_ind=range_inds_flat,
                elevation_ind=elevation_inds_flat,
                azimuth_ind=azimuth_inds_flat,
                power_val=new_data,
            )
    except FileExistsError:
        return f"{clip} {file_name} skipped (output already exists)"
    except BaseException:
        # Remove only the incomplete file created by this call.
        if output_created:
            try:
                os.remove(output_file)
            except OSError:
                pass
        raise

    return f"{clip} {file_name} processed successfully"


def process_clip(clip):
    input_dir, extension = resolve_input(clip)

    output_dir = os.path.join(
        OUTPUT_BASE_DIR,
        clip,
        "radar_tensor_8doppler",
    )

    if input_dir is None:
        print(f"[SKIP] no supported radar cubes for sequence {clip}")
        return

    os.makedirs(output_dir, exist_ok=True)

    file_names = sorted(
        f for f in os.listdir(input_dir)
        if f.endswith(extension)
    )

    # A MAT cube is roughly 0.5 GB. Limiting concurrency avoids multi-GB spikes.
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {
            executor.submit(
                process_file,
                clip,
                file_name,
                input_dir,
                output_dir,
            ): file_name
            for file_name in file_names
        }

        count = 0

        for future in as_completed(future_to_file):
            count += 1

            try:
                print(future.result())
            except Exception as exc:
                file_name = future_to_file[future]
                raise RuntimeError(
                    f"Failed processing {clip}/{file_name}"
                ) from exc

            if count % 64 == 0:
                gc.collect()


# -------------------------------------------------------------------------
# Only generate the sequences required by the current aligned official split.
# -------------------------------------------------------------------------
TRAIN_SEQS = [
    4, 6, 7, 8, 9, 10, 11, 12, 13,
    14, 16, 17, 18, 19, 20, 27, 56,
]

VAL_SEQS = [
    1, 2,
]

TEST_SEQS = [
    3, 15, 22, 23, 55,
]

clips = TRAIN_SEQS + VAL_SEQS + TEST_SEQS
clips = [str(seq) for seq in clips]


def validate_paths():
    input_root = os.path.realpath(INPUT_BASE_DIR)
    output_root = os.path.realpath(OUTPUT_BASE_DIR)

    if not os.path.isdir(input_root):
        raise FileNotFoundError(f"KRadar input root does not exist: {input_root}")

    if os.path.commonpath((input_root, output_root)) == input_root:
        raise ValueError(
            "Refusing to run because the output directory is inside the "
            f"KRadar input tree: {output_root}"
        )

    if SPATIAL_K != N_ELE_BLOCKS * N_AZI_BLOCKS:
        raise ValueError(
            "SPATIAL_K must equal N_ELE_BLOCKS * N_AZI_BLOCKS"
        )

    available_clips = [
        clip
        for clip in clips
        if resolve_input(clip)[0] is not None
    ]
    if not available_clips:
        raise FileNotFoundError(
            "No supported radar cube directories were found. Expected "
            f"{input_root}/<sequence>/radar_polar_cube or radar_tesseract"
        )


def main():
    validate_paths()
    for clip in clips:
        process_clip(clip)


if __name__ == "__main__":
    main()
