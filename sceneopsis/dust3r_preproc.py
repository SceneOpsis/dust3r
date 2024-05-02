import cv2
import numpy as np
import glob
import torch
from clize import run
import os
from tqdm import tqdm

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images, rgb
from dust3r.post_process import estimate_focal_knowing_depth
from dust3r.utils.geometry import inv, geotrf
from dust3r.utils.device import to_numpy

import sceneopsis.read_write_model as CM
from sceneopsis.utils import visualize_poses
from sceneopsis.utils import get_intrinsics, clean_pointcloud

from pathlib import Path


def load_model(weights, device):
    return AsymmetricCroCo3DStereo.from_pretrained(weights).to(device)


def get_k_idx(idx, out):
    return {k: v[idx : idx + 1] for k, v in out.items()}


def get_pair_output(output, idx):
    res = []
    for k in ["view1", "view2", "pred1", "pred2"]:
        res.append(get_k_idx(idx, output[k]))

    return res


def process_pair(view1, view2, pred1, pred2, min_conf_thr=3):
    # compute camera params from raw input
    # expecting one pair of images i_j
    pair = f"{view1['idx'][0]}_{view2['idx'][0]}"
    conf = float(pred1["conf"][0].mean() * pred2["conf"][0].mean())

    print(f"Solving camera pose for pair {pair} conf {conf:.3f}")

    H, W = view1["true_shape"][0].numpy()
    pts3d_cam0 = pred1["pts3d"][0]  # 3D points predicted for first camera
    pp = np.array([W / 2, H / 2])

    # estimate_focal_knowing_depth requires a torch tensor
    # pts3d_cam0 is already a tensor
    pp_t = torch.tensor(pp, requires_grad=False)

    focal = float(
        estimate_focal_knowing_depth(pts3d_cam0[None], pp_t, focal_mode="weiszfeld")
    )
    pixels = np.mgrid[:W, :H].T.astype(np.float32)

    pts3d_cam1 = pred2["pts3d_in_other_view"][0].numpy()
    assert pts3d_cam0.shape[:2] == (H, W)
    conf_cam1 = pred2["conf"][0]
    mask_cam1 = conf_cam1 > min_conf_thr
    K = np.float32([(focal, 0, pp[0]), (0, focal, pp[1]), (0, 0, 1)])

    try:
        res = cv2.solvePnPRansac(
            pts3d_cam1[mask_cam1],
            pixels[mask_cam1],
            K,
            None,
            iterationsCount=100,
            reprojectionError=5,
            flags=cv2.SOLVEPNP_SQPNP,
        )
        success, R, T, inliers = res
        assert success

        R = cv2.Rodrigues(R)[0]  # world to cam
        pose = inv(np.r_[np.c_[R, T], [(0, 0, 0, 1)]])  # cam to world
    except:
        raise f"Failed to solve camera pose for pair {pair}"

    # TODO Compute confidence for the pair.

    return pose, focal, pp, conf


@torch.no_grad()
def compute_poses(output, min_conf_thr):

    pairs = len(output["view1"]["idx"])
    relative_poses = []
    focals = []
    pps = []
    confs = []

    absolute_poses = []
    # first camera is at origin
    absolute_poses.append(np.eye(4))

    for p_id in range(pairs):

        view1, view2, pred1, pred2 = get_pair_output(output, p_id)
        pose, focal, pp, conf = process_pair(view1, view2, pred1, pred2, min_conf_thr)
        print(f"{focal=}, {pp=}, {conf=}")
        print(pose, type(pose))

        relative_poses.append(pose)
        focals.append(focal)
        pps.append(pp)
        confs.append(conf)

        print(f"ABSOLUTE 0-{len(absolute_poses)}]")
        cam1_to_world = absolute_poses[-1] @ pose
        absolute_poses.append(cam1_to_world)
        print(absolute_poses[-1])

        cam1_world_pts3d = geotrf(cam1_to_world, pred2["pts3d_in_other_view"][0])

    # last cam focal and pp are not caltulated. Lets use the second to last.
    focals.append(focals[-1])
    pps.append(pps[-1])

    return relative_poses, focals, pps, absolute_poses


def preproc(
    in_folder: str,
    out_folder: str,
    *,
    vis: bool = False,
    colmap_txt: bool = False,
    raw_res: bool = False,
    conf_thr: float = 3,
    weights: str = "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
):
    """
    Preprocess images of a sequence with dust3r:
     - Create pairs of images (i, i+1)
     - Infer pointmaps and confidence maps
     - Compute camera poses (simple per pair PnPRansac)
     - Convert to colmap format and save

    Args:
        in_folder (str): sequence of images
        out_folder (str): colmap output folder
        vis (bool, optional): visualize the result (will block at the end). Defaults to False.
        colmap_txt (bool, optional): Create txt colmap model files. Defaults to False.
        raw_res (bool, optional): Output in dust3r resolution instead of source image res. Defaults to False.
        conf_thr (float, optional): Dust3r confidence threshold. Defaults to 3.
        weights (str, optional): Dust3r model weights. Defaults to "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth".
    """

    network_input_size = [288, 512]
    batch_size = 8
    device = "cuda"

    print("Loading model...")
    model = load_model(weights, device)
    model.eval()

    print("Loading images...")
    exp_name = "undistorted/images"
    filelist = sorted(glob.glob(f"{in_folder}/*.png"))
    print(filelist)

    # If raw_res is True, we will keep the resolution of dust3r (which is image_size).
    # Otherwise, we will rescale all the output to original resolution.
    if not raw_res:
        # Get the original res. Assuming all input frames are the same resolution
        original_res = cv2.imread(filelist[0]).shape[:2]
        print(f"Original resolution: {original_res}")
        # make sure aspect ratio is the same.
        assert (
            network_input_size[0] / network_input_size[1]
            == original_res[0] / original_res[1]
        ), "Aspect ratio mismatch"

    # TODO This is keeping all images in memory. Make it smarter.
    imgs = load_images(filelist, size=network_input_size[1])
    assert len(imgs) > 1, "Need at least 2 images"
    # we need a sliding window of 1 (0, 1), (1, 2), (2, 3), ...
    pairsid = [(i, i + 1) for i in range(len(imgs) - 1)]
    pairs = [(imgs[i], imgs[j]) for i, j in pairsid]
    print(f"{len(pairsid)=}, {pairsid=}")

    print("Inferencing...")
    output = inference(pairs, model, device, batch_size=batch_size)
    # This is to get the depth for the last camera
    # TODO: Find a more elegant way to get this last depthmap
    last_pair_output = inference(
        [(pairs[-1][1], pairs[-1][0])], model, device, batch_size=batch_size
    )

    print("Compute poses...")
    relative_poses, focals, pps, absolute_poses = compute_poses(output, conf_thr)

    # Gather pointmaps
    pointmaps = output["pred1"]["pts3d"]
    last_pointmap = last_pair_output["pred1"]["pts3d"]
    pointmaps = torch.cat([pointmaps, last_pointmap], dim=0)
    print(
        "POINTMAPS: ",
        pointmaps.shape,
        pointmaps.dtype,
        pointmaps.min(),
        pointmaps.max(),
    )

    # Gather confidence maps
    confmaps = output["pred1"]["conf"]
    last_conf = last_pair_output["pred1"]["conf"]
    confmaps = torch.cat([confmaps, last_conf], dim=0)
    print("CONFMAPS: ", confmaps.shape, confmaps.dtype, confmaps.min(), confmaps.max())

    # Gather color images (this is only for visualization and sample points3d file)
    color_imgs = torch.from_numpy(rgb(output["view1"]["img"]))
    last_color_img = torch.from_numpy(rgb(last_pair_output["view1"]["img"]))
    color_imgs = torch.cat([color_imgs, last_color_img], dim=0)

    # Prepare out_folder structure
    pointmaps_out = os.path.join(out_folder, "pointmaps")
    os.makedirs(pointmaps_out, exist_ok=True)
    colmap_out = os.path.join(out_folder, "sparse", "0")
    os.makedirs(colmap_out, exist_ok=True)
    images_out = os.path.join(out_folder, "images")
    os.makedirs(images_out, exist_ok=True)

    # Save dust3r output
    print("Saving pointmaps and confmaps...")
    # NOTE: Maybe resize to original resolution if not raw_res?
    for i, (d, c) in enumerate(zip(pointmaps, confmaps)):
        d = d.cpu().numpy()
        c = c.cpu().numpy()
        np.save(os.path.join(pointmaps_out, f"pm_{i:05d}.npy"), d)
        np.save(os.path.join(pointmaps_out, f"conf_{i:05d}.npy"), c)

    # Save color
    print("Saving color images...")
    if raw_res:
        for idx, img in enumerate(imgs):
            assert img["idx"] == idx, "Image idx mismatch"
            c = rgb(img["img"][0])
            cv2.imwrite(
                os.path.join(images_out, f"{idx:05d}.png"),
                (c[..., ::-1] * 255).astype(np.uint8),
            )
    else:  # just copy the originals
        import shutil

        for idx, f in enumerate(filelist):
            shutil.copy(f, os.path.join(images_out, f"{idx:05d}.png"))

    print("Creating colmap cameras file...")
    # This is a single camera. Following InstantSplat we get the mean focal length
    raw_focal = np.mean(focals)
    raw_principal_point = pps[0]  # all pps are the same (W/2, H/2)
    raw_height, raw_width = imgs[0]["true_shape"].flatten()

    if not raw_res:
        print(
            f"Dust3r camera params: {raw_focal=}, {raw_principal_point=}, {raw_height=}, {raw_width=}"
        )
        ratio = original_res[0] / network_input_size[0]
        focal = raw_focal * ratio
        principal_point = np.array(raw_principal_point) * ratio
        height, width = original_res
    else:
        focal = raw_focal
        principal_point = raw_principal_point
        height, width = raw_height, raw_width

    print(f"Camera params: {focal=}, {principal_point=}, {height=}, {width=}")

    cam_model = "OPENCV"
    # params computed are f, cx ,cy
    # but we need opencv so params will be
    # fx, fy, cx, cy, k1, k2, p1, p2
    # Modify camera intrinsics to follow a different convention.
    # Coordinates of the center of the top-left pixels are by default:
    # - (0.5, 0.5) in Colmap
    # - (0,0) in OpenCV
    cam_idx = 0  # single camera
    cameras = {
        cam_idx: CM.Camera(
            id=cam_idx,
            model=cam_model,
            width=width,
            height=height,
            params=[
                focal,
                focal,
                principal_point[0] + 0.5,
                principal_point[1] + 0.5,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        )
    }

    if colmap_txt:
        CM.write_cameras_text(cameras, os.path.join(colmap_out, "cameras.txt"))
    else:
        CM.write_cameras_binary(cameras, os.path.join(colmap_out, "cameras.bin"))

    # Create points3D.txt
    print("Creating colmap points3D file...")
    # just store the pointcloud of the first frame for reference

    pts = to_numpy(pointmaps[0]).reshape(-1, 3)
    col = (to_numpy(color_imgs[0]).reshape(-1, 3) * 255).astype(np.uint8)

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

    if colmap_txt:
        CM.write_points3D_text(points3d, os.path.join(colmap_out, "points3D.txt"))
    else:
        CM.write_points3D_binary(points3d, os.path.join(colmap_out, "points3D.bin"))

    # Create images.txt
    print("Creating colmap images file...")

    images = {}
    for idx, p in tqdm(
        enumerate(absolute_poses), total=len(absolute_poses), desc="Creating images"
    ):
        # print(f"{idx} {p=}")
        p = np.linalg.inv(p)  # world-to-cam
        images[idx] = CM.Image(
            id=idx,
            qvec=CM.rotmat2qvec(p[:3, :3]),
            tvec=p[:3, 3],
            camera_id=cam_idx,
            name=f"{idx:05d}.png",
            xys=np.array([]),
            point3D_ids=np.array([]),
        )

    if colmap_txt:
        CM.write_images_text(images, os.path.join(colmap_out, "images.txt"))
    else:
        CM.write_images_binary(images, os.path.join(colmap_out, "images.bin"))

    if vis:

        K = get_intrinsics([raw_focal, raw_focal], raw_principal_point)
        new_confmaps = clean_pointcloud(absolute_poses, K, pointmaps, confmaps)
        point_cloud = []
        colors = []
        i = 0
        for cam, points, col, conf in zip(
            absolute_poses, pointmaps, color_imgs, new_confmaps
        ):
            mask = conf > conf_thr
            print(f"{i} ==> mask: {mask.sum()}")
            i += 1
            pc = points[mask].cpu().numpy()
            pc_world = geotrf(cam, pc)
            point_cloud.append(pc_world)
            colors.append(col[mask].cpu().numpy())

        visualize_poses(absolute_poses, focals, point_cloud, colors, server_port=7860)


def cleanup(
    colmap_dir: str,
    *,
    conf_thr: float = 3,
    vis: bool = False,
):
    # Load colmap poses and dust3r output
    model_path = Path(colmap_dir) / "sparse" / "0"
    images_path = Path(colmap_dir) / "images"
    pointmaps_path = Path(colmap_dir) / "pointmaps"
    # make sure folders exist
    assert all(
        [model_path.exists(), images_path.exists(), pointmaps_path.exists()]
    ), "Colmap folder not found or missing files"

    colmap_cameras, colmap_images, _ = CM.read_model(model_path)

    print(f"{len(colmap_images)=}, {len(colmap_cameras)=}")

    print("Images ", colmap_images[0])
    print("Camera ", colmap_cameras[0])

    # load pointmaps and confmaps
    file_indices = list(range(len(colmap_images)))
    filelist = [colmap_images[i].name for i in file_indices]
    print(f"{filelist=}")

    pointmaps = []
    confmaps = []
    frames = []
    for f in tqdm(filelist, desc="Loading images, pointmaps and confmaps"):
        fname = Path(f).stem
        pm = np.load(pointmaps_path / f"pm_{fname}.npy")
        conf = np.load(pointmaps_path / f"conf_{fname}.npy")
        img = cv2.imread((images_path / f).as_posix())

        pointmaps.append(pm)
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
    pointmaps_size = pointmaps[0].shape[:2]
    calib_size = [colmap_cameras[0].height, colmap_cameras[0].width]

    assert (
        image_size[0] / image_size[1]
        == calib_size[0] / calib_size[1]
        == pointmaps_size[0] / pointmaps_size[1]
    ), "Aspect ratio mismatch"

    scale = pointmaps_size[0] / calib_size[0]
    print(f"Scaling intrinsics by {scale}")

    # Modify camera intrinsics to follow a different convention.
    # Coordinates of the center of the top-left pixels are by default:
    # - (0.5, 0.5) in Colmap
    # - (0,0) in OpenCV
    pp = (np.array(colmap_cameras[0].params[2:4]) - 0.5) * scale
    focal = np.array(colmap_cameras[0].params[0:2]) * scale

    print(f"Scaled Intrinsics: {focal=}, {pp=}")

    # NOTE undistort pointmaps and confmaps ???

    # Scale images to pointmap size
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    frames = [
        cv2.resize(img, tuple(pointmaps_size[::-1]), interpolation=interp)
        for img in tqdm(frames, desc="Resizing images")
    ]

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
    run(preproc, cleanup, description="Dust3r (pre)processing scripts")


# %%
import numpy as np

b = 20
a = [1, 2]

f = np.array([a])[::new, :]

# %%
