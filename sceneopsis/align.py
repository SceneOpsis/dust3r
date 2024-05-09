from clize import run
from pathlib import Path
import sceneopsis.read_write_model as CM
from tqdm import tqdm
import numpy as np
import cv2
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def align(
    colmap_dir: Path,
    *,
    iters: int = 1000,
    closed_form: bool = False,
    conf_thres: float = 3,
):
    """
    Align depth maps created with Dust3r (i.e dust3r_preproc.py) to the sparse depth maps of colmap (created with colmap2depth).
    colmap_dir is expected to contain the following directories:
    - sparse/0: The colmap sparse reconstruction
    - depth: The depth maps created with colmap2depth
    - pointmaps: The dust3r pointmaps created with dust3r_preproc.

    Output will be saved in colmap_dir/aligned_depth
    """

    model_path = colmap_dir / "sparse/0"
    assert model_path.exists(), "Colmap dir must contain a model in sparse/0"
    depth_dir = colmap_dir / "depth"
    assert depth_dir.exists(), "Colmap dir must contain sparse depth maps in depth/"
    pointmaps_dir = colmap_dir / "pointmaps"
    assert (
        pointmaps_dir.exists()
    ), "Colmap dir must contain dust3r pointmaps in pointmaps/"

    output_dir = colmap_dir / "aligned_depth"
    output_dir.mkdir(exist_ok=True)

    cameras, _, _ = CM.read_model(model_path)

    assert len(cameras) == 1, "Only support one camera for now"
    CAMERA_ID = list(cameras.keys())[0]

    # glob all the depth maps
    depth_files = sorted(list(depth_dir.glob("*.npy")))
    # glob all pointmaps
    pointmap_files = sorted(list(pointmaps_dir.glob("pm_*.npy")))
    # glob all confidence maps
    confmap_files = sorted(list(pointmaps_dir.glob("conf_*.npy")))

    assert len(depth_files) == len(
        pointmap_files
    ), "Number of depth maps and pointmaps must match"

    calib_dims = (cameras[CAMERA_ID].height, cameras[CAMERA_ID].width)
    depth_dims = np.load(depth_files[0]).shape[:2]

    assert (
        calib_dims == depth_dims
    ), f"Calibration and sparse depth map dimensions must match, {calib_dims=} {depth_dims=}"

    # load depth maps and pointmaps and scale as needed
    sparse_depth_maps = [np.load(f) for f in depth_files]
    dense_depth_maps = [np.load(f)[:, :, 2] for f in pointmap_files]
    conf_maps = [np.load(f) for f in confmap_files]

    if dense_depth_maps[0].shape != calib_dims:
        dense_depth_maps = [
            cv2.resize(
                pm, (calib_dims[1], calib_dims[0]), interpolation=cv2.INTER_NEAREST
            )
            for pm in dense_depth_maps
        ]
        conf_maps = [
            cv2.resize(
                cm, (calib_dims[1], calib_dims[0]), interpolation=cv2.INTER_LINEAR
            )
            for cm in conf_maps
        ]

    sparse_depth_maps = np.stack(sparse_depth_maps, axis=0)
    dense_depth_maps = np.stack(dense_depth_maps, axis=0)
    conf_maps = np.stack(conf_maps, axis=0)

    sparse_depth_maps = torch.from_numpy(sparse_depth_maps)
    dense_depth_maps = torch.from_numpy(dense_depth_maps)
    conf_maps = torch.from_numpy(conf_maps)

    mask = (
        (sparse_depth_maps > 0.1)
        & (sparse_depth_maps < 1000.0)
        & (conf_maps > conf_thres)
    )

    print(
        "Sparse depth maps shape:",
        sparse_depth_maps.shape,
        torch.min(sparse_depth_maps),
        torch.max(sparse_depth_maps),
    )
    print(
        "Dense depth maps shape:",
        dense_depth_maps.shape,
        torch.min(dense_depth_maps),
        torch.max(dense_depth_maps),
    )

    print(
        "Conf maps shape:", conf_maps.shape, torch.min(conf_maps), torch.max(conf_maps)
    )

    if closed_form:
        with torch.no_grad():
            print("Aligning using closed form solution")

            scale, shift = compute_scale_and_shift(
                dense_depth_maps, sparse_depth_maps, mask=mask
            )
            print("Scale:", scale.flatten().numpy())
            print("Shift:", shift.flatten().numpy())

            scale = scale.unsqueeze(1).unsqueeze(2)
            shift = shift.unsqueeze(1).unsqueeze(2)
            depth_aligned = scale * dense_depth_maps + shift
            mse_loss = torch.nn.MSELoss()
            avg = mse_loss(depth_aligned[mask], sparse_depth_maps[mask])
            print(
                f"Average depth alignment error for depths is: {avg:3f} which is {'good' if avg<0.2 else 'bad'}"
            )
    else:  # grad_descent
        depth_aligned = grad_descent(
            dense_depth_maps, sparse_depth_maps, mask, iterations=iters
        )


# function adapted from dn-splatter
def grad_descent(
    mono_depth_tensors: torch.Tensor,
    sparse_depths: torch.Tensor,
    masks: torch.Tensor,
    iterations: int = 1000,
    lr: float = 0.1,
) -> torch.Tensor:
    """Align mono depth estimates with sparse depths.

    Returns:
        aligned_depths: tensor of scale aligned mono depths
    """
    aligned_mono_depths = []
    scales = []
    shifts = []
    for idx in tqdm(
        range(mono_depth_tensors.shape[0]),
        desc="Alignment with grad descent ...",
    ):
        scale = torch.nn.Parameter(
            torch.tensor([1.0], device=device, dtype=torch.float)
        )
        shift = torch.nn.Parameter(
            torch.tensor([0.0], device=device, dtype=torch.float)
        )

        estimated_mono_depth = mono_depth_tensors[idx, ...].float().to(device)
        sparse_depth = sparse_depths[idx].float().to(device)

        mask = masks[idx]
        estimated_mono_depth_map_masked = estimated_mono_depth[mask]
        sparse_depth_masked = sparse_depth[mask]

        mse_loss = torch.nn.MSELoss()
        optimizer = torch.optim.Adam([scale, shift], lr=lr)

        avg_err = []
        for step in range(iterations):
            optimizer.zero_grad()
            loss = mse_loss(
                scale * estimated_mono_depth_map_masked + shift, sparse_depth_masked
            )
            loss.backward()
            optimizer.step()
        avg_err.append(loss.item())
        aligned_mono_depths.append(scale * estimated_mono_depth + shift)
        scales.append(scale.item())
        shifts.append(shift.item())

    avg = sum(avg_err) / len(avg_err)
    print("Scales:", scales)
    print("Shifts:", shifts)
    print(
        f"Average depth alignment error for batch depths is: {avg:3f} which is {'good' if avg<0.2 else 'bad'}"
    )
    return torch.stack(aligned_mono_depths, dim=0)


# copy from monosdf
def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


if __name__ == "__main__":
    run(align)
