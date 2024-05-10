import cv2
import numpy as np
import glob
import torch
from clize import run
import os
from tqdm import tqdm

from dust3r.utils.geometry import geotrf

import sceneopsis.read_write_model as CM
from sceneopsis.utils import (
    get_intrinsics,
    clean_pointcloud,
    visualize_poses,
)

from pathlib import Path


def depth2pointmap(depth, focal, pp):
    if type(pp) == list:
        pp = np.array(pp)

    height, width = depth.shape
    fx, fy = focal
    u, v = np.meshgrid(range(width), range(height))

    z = depth
    x = (u - pp[0]) / fx * z
    y = (v - pp[1]) / fy * z

    return np.stack((x, y, z), axis=-1)


def cleanup(
    colmap_dir: str,
    *,
    conf_thr: float = 3,
    vis: bool = False,
):
    """
    Load dust3r output and colmap poses. Clean the pointmaps by projecting the depth from each camera to
    all other cameras and removing points with low confidence (follows dust3r cleanup method).

    """
    # Load colmap poses and dust3r output
    model_path = Path(colmap_dir) / "sparse" / "0"
    images_path = Path(colmap_dir) / "images"
    pointmaps_path = Path(colmap_dir) / "pointmaps"
    aligned_depth_path = Path(colmap_dir) / "aligned_depth"

    # make sure folders exist
    assert all(
        [
            model_path.exists(),
            images_path.exists(),
            pointmaps_path.exists(),
            aligned_depth_path.exists(),
        ]
    ), "Colmap folder not found or missing files"

    colmap_cameras, colmap_images, _ = CM.read_model(model_path)

    assert len(colmap_cameras) == 1, "Only one camera supported"

    print(f"{len(colmap_images)=}")
    colmap_camera = next(iter(colmap_cameras.values()))

    print("Camera ", colmap_camera)

    file_indices = sorted(colmap_images.keys())

    depth_files = sorted(list(aligned_depth_path.glob("*.npy")))
    print(f"{depth_files=}")
    # load pointmaps and confmaps
    filelist = [colmap_images[i].name for i in file_indices]
    print(f"{filelist=}")

    depths = []
    confmaps = []
    frames = []
    for f in tqdm(filelist, desc="Loading images, pointmaps and confmaps"):
        fname = Path(f).stem
        dm = np.load(aligned_depth_path / f"{fname}.npy")
        conf = np.load(pointmaps_path / f"conf_{fname}.npy")
        img = cv2.imread((images_path / f).as_posix())

        depths.append(dm)
        confmaps.append(conf)
        frames.append(img)

    # load poses
    cam2world_poses = []
    for i in file_indices:
        qvec = colmap_images[i].qvec
        tvec = colmap_images[i].tvec
        R = CM.qvec2rotmat(qvec)
        world2cam = np.r_[np.c_[R, tvec], [(0, 0, 0, 1)]]
        cam2world = np.linalg.inv(world2cam)
        cam2world_poses.append(cam2world)

    # scale camera intrinsics to dust3r resolution
    image_size = frames[0].shape[:2]
    depths_size = depths[0].shape[:2]
    calib_size = (colmap_camera.height, colmap_camera.width)

    assert (
        image_size[0] / image_size[1]
        == calib_size[0] / calib_size[1]
        == depths_size[0] / depths_size[1]
    ), "Aspect ratio mismatch"

    # scale images to dust3r resolution
    frames = [
        cv2.resize(fr, (depths_size[1], depths_size[0]), interpolation=cv2.INTER_AREA)
        for fr in frames
    ]

    scale = depths_size[0] / calib_size[0]
    print(f"Scaling intrinsics by {scale}")

    # Modify camera intrinsics to follow a different convention.
    # Coordinates of the center of the top-left pixels are by default:
    # - (0.5, 0.5) in Colmap
    # - (0,0) in OpenCV
    pp = (np.array(colmap_camera.params[2:4]) - 0.5) * scale
    focal = np.array(colmap_camera.params[0:2]) * scale

    print(f"Scaled Intrinsics: {focal=}, {pp=}")

    # NOTE undistort depths and confmaps ???

    print(f"{depths[0].shape=}")

    # convert depths to pointmaps
    pointmaps = [depth2pointmap(depth, focal, pp) for depth in depths]
    print(f"{pointmaps[0].shape=}")

    # apply clean_pointcloud
    K = get_intrinsics(focal, pp)
    new_confmaps = clean_pointcloud(cam2world_poses, K, pointmaps, confmaps)

    point_cloud = []
    colors = []
    i = 0
    for cam, points, col, conf in zip(cam2world_poses, pointmaps, frames, new_confmaps):
        mask = conf > conf_thr
        print(f"{i} ==> mask: {mask.sum()}")
        i += 1
        pc = points[mask]
        pc_world = geotrf(cam, pc)
        point_cloud.append(pc_world)
        colors.append(col[mask][:, ::-1])  # BGR to RGB

    # save new pointcloud in points3D.txt

    # Create points3D.txt
    print("Creating colmap points3D file...")
    # just store the pointcloud of the first frame for reference
    pts = np.vstack(point_cloud)
    col = np.vstack(colors)

    print("Col is ", col.shape, " type ", col.dtype)

    points3d = {}
    for idx, (pt, c) in tqdm(
        enumerate(zip(pts, col)), total=len(pts), desc="Creating points3D"
    ):
        points3d[idx] = CM.Point3D(
            id=idx,
            xyz=pt,
            rgb=c,
            error=1.0,
            image_ids=np.array([]),
            point2D_idxs=np.array([]),
        )

    colmap_bin = Path(model_path / "images.bin").exists()
    if colmap_bin:
        CM.write_points3D_binary(points3d, model_path / "points3D.bin")
    else:
        CM.write_points3D_text(points3d, model_path / "points3D.txt")

    if vis:
        focals = np.repeat(np.array(focal)[np.newaxis, :], len(cam2world_poses), axis=0)
        visualize_poses(cam2world_poses, focals, point_cloud, colors, server_port=7860)


if __name__ == "__main__":
    run(cleanup)
