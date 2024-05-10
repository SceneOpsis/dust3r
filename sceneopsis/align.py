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
    conf_thres: float = 3,
):
    """
    Align depth maps created with Dust3r (i.e dust3r_preproc.py) to the sparse depth maps of colmap (created with colmap2depth).
    colmap_dir is expected to contain the following directories:
    - sparse/0: The colmap sparse reconstruction
    - sparse_depth: The depth maps created with colmap2depth
    - pointmaps: The dust3r pointmaps created with dust3r_preproc.

    Output will be saved in colmap_dir/aligned_depth
    """

    model_path = colmap_dir / "sparse/0"
    assert model_path.exists(), "Colmap dir must contain a model in sparse/0"
    depth_dir = colmap_dir / "sparse_depth"
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

    dust3r_dims = dense_depth_maps[0].shape

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

    masks = (
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

    depth_aligned = grad_descent(
        dense_depth_maps, sparse_depth_maps, masks, iterations=iters
    )

    depth_aligned = depth_aligned.detach().cpu().numpy()
    # depth_aligned = dense_depth_maps.detach().cpu().numpy()

    # scale back to dust3r res
    depth_aligned = [
        cv2.resize(
            da, (dust3r_dims[1], dust3r_dims[0]), interpolation=cv2.INTER_NEAREST
        )
        for da in depth_aligned
    ]

    for idx, f in tqdm(enumerate(depth_files), desc="Saving aligned depth maps..."):
        name = Path(f).stem
        np.save(output_dir / f"{name}.npy", depth_aligned[idx])


# function adapted from dn-splatter
def grad_descent(
    mono_depth_tensors: torch.Tensor,
    sparse_depths: torch.Tensor,
    masks: torch.Tensor,
    iterations: int = 300,
    lr: float = 1.0,
) -> torch.Tensor:
    """Align mono depth estimates with sparse depths.

    Returns:
        aligned_depths: tensor of scale aligned mono depths
    """
    aligned_mono_depths = []
    scales = []
    for idx in tqdm(
        range(mono_depth_tensors.shape[0]),
        desc="Alignment with grad descent ...",
    ):
        scale = torch.nn.Parameter(
            torch.tensor([1.0], device=device, dtype=torch.float)
        )

        estimated_mono_depth = mono_depth_tensors[idx, ...].float().to(device)
        sparse_depth = sparse_depths[idx].float().to(device)

        mask = masks[idx]
        estimated_mono_depth_map_masked = estimated_mono_depth[mask]
        sparse_depth_masked = sparse_depth[mask]

        mse_loss = torch.nn.MSELoss()
        optimizer = torch.optim.Adam([scale], lr=lr)

        avg_err = []
        for _ in range(iterations):
            optimizer.zero_grad()
            loss = mse_loss(
                scale * estimated_mono_depth_map_masked, sparse_depth_masked
            )
            loss.backward()
            optimizer.step()

        avg_err.append(loss.item())
        aligned_mono_depths.append(scale * estimated_mono_depth)
        scales.append(scale.detach().item())

    avg = sum(avg_err) / len(avg_err)
    print("Scales:", scales)

    print(
        f"Average depth alignment error for batch depths is: {avg:3f} which is {'good' if avg<0.2 else 'bad'}"
    )
    return torch.stack(aligned_mono_depths, dim=0)


if __name__ == "__main__":
    run(align)
