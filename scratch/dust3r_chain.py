# Use duster with a sequence of frames (0,N).
# Get the first frame pair from duster (0, 1)
# the pair output is depth and conf maps. Depth for each pair is relative to the first image of the pair.
# compute the pose and camera parameters from the relative depth maps.

# Once the camera params are computed continue to the next pair (1, 2) and repeat the process.
# Get the camera params relative to the first frame of the sequence.
# %%
import sys
import os
import cv2
import numpy as np
import glob

sys.path.append("../")

import copy
from matplotlib import pyplot as plt

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, rgb
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.device import to_numpy


import torch
from tqdm import tqdm

import scratch.read_write_model as CM
from scratch import visualize_scene

min_conf_thr = 3
image_size = 512
niter = 300
batch_size = 16
schedule = "linear"  # "cosine" or "linear"
device = "cuda"
weights = "../checkpoints/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"


def load_model(weights, device):
    return AsymmetricCroCo3DStereo.from_pretrained(weights).to(device)


from dust3r.cloud_opt.pair_viewer import PairViewer
from dust3r.post_process import estimate_focal_knowing_depth
from dust3r.utils.geometry import inv, geotrf


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


def compute_poses(output):
    pairs = len(output["view1"]["idx"])
    relative_poses = []
    focals = []
    pps = []
    confs = []

    absolute_poses = []
    # first camera is at origin
    absolute_poses.append(np.eye(4))

    point_cloud = []
    # first point cloud is in world origin
    point_cloud.append(output["pred1"]["pts3d"][0])

    color_imgs = []
    # first image is the first view
    color_imgs.append(rgb(output["view1"]["img"][0]))

    for p_id in range(pairs):

        view1, view2, pred1, pred2 = get_pair_output(output, p_id)
        pose, focal, pp, conf = process_pair(view1, view2, pred1, pred2)
        print(f"{focal=}, {pp=}, {conf=}")
        print(pose)

        relative_poses.append(pose)
        focals.append(focal)
        pps.append(pp)
        confs.append(conf)

        print(f"ABSOLUTE 0-{len(absolute_poses)}]")
        cam1_to_world = absolute_poses[-1] @ pose
        absolute_poses.append(cam1_to_world)
        print(absolute_poses[-1])

        cam1_world_pts3d = geotrf(cam1_to_world, pred2["pts3d_in_other_view"][0])
        point_cloud.append(cam1_world_pts3d)

        color_imgs.append(rgb(view2["img"][0]))

    # last cam focal and pp are not caltulated. Lets use the second to last.
    focals.append(focals[-1])
    pps.append(pps[-1])

    return relative_poses, focals, pps, absolute_poses, point_cloud, color_imgs


# %%
# load model
model = load_model(weights, device)

# %%

exp_name = "undistorted/images"
filelist = sorted(glob.glob(f"../res/{exp_name}/*.png"))[:4]
print(filelist)

# %%
# TODO This is keeping all images in memory. Make it smarter.
imgs = load_images(filelist, size=image_size)
assert len(imgs) > 1, "Need at least 2 images"
# we need a sliding window of 1 (0, 1), (1, 2), (2, 3), ...
pairsid = [(i, i + 1) for i in range(len(imgs) - 1)]
pairs = [(imgs[i], imgs[j]) for i, j in pairsid]
print(f"{len(pairsid)=}, {pairsid=}")

# # symetrize
# sym_pairs = []
# for a, b in pairs:
#     sym_pairs.append((a, b))
#     sym_pairs.append((b, a))

# pairs = sym_pairs

# %%

output = inference(pairs, model, device, batch_size=batch_size)

# %%

mode = (
    GlobalAlignerMode.PointCloudOptimizer
    if len(imgs) > 2
    else GlobalAlignerMode.PairViewer
)
scene = global_aligner(output, device=device, mode=mode)  # , optimize_pp=True)
print(f"{type(scene)=}")

# %%

lr = 0.01

if mode == GlobalAlignerMode.PointCloudOptimizer:
    loss = scene.compute_global_alignment(
        init="mst", niter=niter, schedule=schedule, lr=lr
    )
    print(f"Loss = {loss}")

print("Cleanining pointcloud")
scene = scene.clean_pointcloud()

print("done")


# %%

visualize_scene(scene)

# %%

view1, view2, pred1, pred2 = get_pair_output(output, 0)

pose, focal, pp, conf = process_pair(view1, view2, pred1, pred2)

print(f"{focal=}, {pp=}, {conf=}")
print(pose)

# %%

conf = float(pred1["conf"][0].mean() * pred2["conf"][0].mean())
conf

view1["img"][0].shape

# %%
conf_cam0 = pred1["conf"][0]

# %%

conf_cam0.median()

# %%
relative_poses, focals, pps, absolute_poses, point_cloud, color_imgs = compute_poses(
    output
)
# %%

from scratch import visualize_poses

ret = visualize_poses(absolute_poses, focals, point_cloud, color_imgs, server_port=7862)

# print(ret)

# np.median(focals)
# output["view1"]["true_shape"][0] * 3.75

# %%

# TODOs
#
# Convert camera poses to colmap
# Give them to colmap as initial positions.
# Get Depth maps and store them for use with 3dgs.

# %%
