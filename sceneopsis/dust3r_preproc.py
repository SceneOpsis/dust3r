#
# Input: A folder with the frames from a video (sequence).
# Tasks:
# 1. Process the video frames using dust3r to get Depth, Conf Maps, and Camera Parameters (intrinsic and extrinsic).
# 2. convert camera Params to colmap format and
# 3. Save the output a new folder.

import sys
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


def dust3r_preproc(
    in_folder: str,
    out_folder: str,
    *,
    vis: bool = False,
    colmap_bin: bool = False,
    raw_res: bool = False,
):

    network_input_size = [288, 512]
    batch_size = 16
    device = "cuda"
    weights = "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"

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
    min_conf_thr = 3
    print(f"{min_conf_thr=}")
    relative_poses, focals, pps, absolute_poses = compute_poses(output, min_conf_thr)

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
    f = np.mean(focals)
    pp = pps[0]  # all pps are the same (W/2, H/2)
    height, width = imgs[0]["true_shape"].flatten()

    if not raw_res:
        ratio = original_res[0] / network_input_size[0]
        f = f * ratio
        pp = np.array(pp) * ratio
        height, width = original_res

    print(f"Camera params: {f=}, {pp=}, {height=}, {width=}")

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
            params=[f, f, pp[0] + 0.5, pp[1] + 0.5, 0.0, 0.0, 0.0, 0.0],
        )
    }

    if colmap_bin:
        CM.write_cameras_binary(cameras, os.path.join(colmap_out, "cameras.bin"))
    else:
        CM.write_cameras_text(cameras, os.path.join(colmap_out, "cameras.txt"))

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

    if colmap_bin:
        CM.write_points3D_binary(points3d, os.path.join(colmap_out, "points3D.bin"))
    else:
        CM.write_points3D_text(points3d, os.path.join(colmap_out, "points3D.txt"))

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

    if colmap_bin:
        CM.write_images_binary(images, os.path.join(colmap_out, "images.bin"))
    else:
        CM.write_images_text(images, os.path.join(colmap_out, "images.txt"))

    if vis:

        from sceneopsis.utils import get_intrinsics, clean_pointcloud

        K = get_intrinsics(f, pp)
        new_confmaps = clean_pointcloud(absolute_poses, K, pointmaps, confmaps)
        point_cloud = []
        colors = []
        i = 0
        for cam, points, col, conf in zip(
            absolute_poses, pointmaps, color_imgs, new_confmaps
        ):
            mask = conf > min_conf_thr
            print(f"{i} ==> mask: {mask.sum()}")
            i += 1
            pc = points[mask].cpu().numpy()
            pc_world = geotrf(cam, pc)
            point_cloud.append(pc_world)
            colors.append(col[mask].cpu().numpy())

        visualize_poses(absolute_poses, focals, point_cloud, colors, server_port=7860)


if __name__ == "__main__":
    run(dust3r_preproc)
