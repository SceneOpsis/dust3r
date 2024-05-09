from clize import run
from pathlib import Path
import sceneopsis.read_write_model as CM
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as pl
import cv2


# Function adapted from dn-splatter colmap_sfm_points_to_depths
def colmap2depth(
    colmap_dir: Path,
    *,
    min_depth: float = 0.001,
    max_depth: float = 1000,
    max_repoj_err: float = 2.5,
    min_n_visible: int = 5,
    debug: bool = False,
):
    """
    Create sparse depth maps from colmap SFM points

    Args:
        colmap_dir: Path to the colmap directory. Should contain "sparse/0"
        min_depth: Discard points closer than this to the camera.
        max_depth: Discard points farther than this from the camera.
        max_repoj_err: Discard points with reprojection error greater than this
          amount (in pixels).
        min_n_visible: Discard 3D points that have been triangulated with fewer
          than this many frames.
        debug: Also include debug images showing depth overlaid
          upon RGB.
    """

    recon_dir = colmap_dir / "sparse/0"
    assert recon_dir.exists(), "Colmap dir must contain a model in sparse/0"
    input_images_dir = colmap_dir / "images"

    output_dir = colmap_dir / "depth"
    if output_dir.exists():
        print("Warning: output directory already exists, overwriting...")
    else:
        output_dir.mkdir()

    if debug:
        assert (
            input_images_dir.exists()
        ), "Colmap dir must contain an images directory for debugging"
        cmap = pl.get_cmap("Reds")
        debug_path = output_dir / "debug_depth"
        debug_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {recon_dir} ...")
    cameras, images, points = CM.read_model(recon_dir)

    assert len(cameras) == 1, "Only support one camera for now"
    CAMERA_ID = list(cameras.keys())[0]
    print("Camera ID:", CAMERA_ID)

    W = cameras[CAMERA_ID].width
    H = cameras[CAMERA_ID].height

    for im_id, im_data in tqdm(images.items(), desc="Creating sparse depthmaps..."):
        # print(f"Processing {im_id} {im_data.name}")
        pids = [pid for pid in im_data.point3D_ids if pid != -1]
        xyz_world = np.array([points[pid].xyz for pid in pids])
        rotation = CM.qvec2rotmat(im_data.qvec)

        z = (rotation @ xyz_world.T)[-1] + im_data.tvec[-1]
        errors = np.array([points[pid].error for pid in pids])
        n_visible = np.array([len(points[pid].image_ids) for pid in pids])
        uv = np.array(
            [
                im_data.xys[i]
                for i in range(len(im_data.xys))
                if im_data.point3D_ids[i] != -1
            ]
        )

        idx = np.where(
            (z >= min_depth)
            & (z <= max_depth)
            & (errors <= max_repoj_err)
            & (n_visible >= min_n_visible)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < W)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < H)
        )
        z = z[idx]
        uv = uv[idx]

        uu, vv = uv[:, 0].astype(int), uv[:, 1].astype(int)
        depth = np.zeros((H, W), dtype=np.float32)
        depth[vv, uu] = z

        out_name = Path(str(im_data.name)).stem
        depth_path = output_dir / out_name
        # count non sero elements
        np.save(depth_path, depth)

        if debug:
            input_image_path = input_images_dir / im_data.name

            overlay = (cmap(depth)[:, :, :3] * 255).astype(np.uint8)
            input_image = cv2.imread(str(input_image_path))
            # print(f"{overlay.shape=}, {input_image.shape=}")

            debug_img = overlay[:, :, :3] * 255 * 0.9 + 0.1 * input_image

            out_name = out_name + ".debug.jpg"
            output_path = debug_path / out_name
            cv2.imwrite(str(output_path), debug_img.astype(np.uint8))


if __name__ == "__main__":
    run(colmap2depth)
