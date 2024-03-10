# %%
import sys
import os
import cv2
import numpy as np
import glob

sys.path.append("../")

import copy
from matplotlib import pyplot as plt

from dust3r.inference import inference, load_model
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.device import to_numpy

import torch
from tqdm import tqdm

import scratch.read_write_model as CM

min_conf_thr = 3
image_size = 512
niter = 300
batch_size = 1
schedule = "linear"  # "cosine" or "linear"
device = "cuda"
weights = "../checkpoints/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"

# %%
# load model
model = load_model(weights, device)

# %%
# Do some processing

# filelist = ["../res/00001.png", "../res/00037.png"]
# glob file list from folder
filelist = glob.glob("../res/step64/*.png")
print(filelist)

# %%

imgs = load_images(filelist, size=image_size)
if len(imgs) == 1:
    imgs = [imgs[0], copy.deepcopy(imgs[0])]
    imgs[1]["idx"] = 1
# if scenegraph_type == "swin":
#     scenegraph_type = scenegraph_type + "-" + str(winsize)
# elif scenegraph_type == "oneref":
#     scenegraph_type = scenegraph_type + "-" + str(refid)

scenegraph_type = "complete"

pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)

output = inference(pairs, model, device, batch_size=batch_size)

mode = (
    GlobalAlignerMode.PointCloudOptimizer
    if len(imgs) > 2
    else GlobalAlignerMode.PairViewer
)
scene = global_aligner(output, device=device, mode=mode)
lr = 0.01

if mode == GlobalAlignerMode.PointCloudOptimizer:
    loss = scene.compute_global_alignment(
        init="mst", niter=niter, schedule=schedule, lr=lr
    )
    print(f"Loss = {loss}")

print("done")

# %%
# convert scene to colmap format
print("Saving frames")
output_folder = "../res/colmap_out"
os.makedirs(output_folder, exist_ok=True)

# save images
image_folder = os.path.join(output_folder, "images")
os.makedirs(image_folder, exist_ok=True)

for idx, img in enumerate(scene.imgs):
    cv2.imwrite(
        os.path.join(image_folder, f"{idx:05d}.png"),
        (img[..., ::-1] * 255).astype(np.uint8),
    )

height, width = img.shape[:2]

# %%
# Create cameras.txt
print("Creating cameras.txt")
focals = scene.get_focals().cpu().detach().numpy().flatten()
pps = scene.get_principal_points().cpu().numpy()

cam_model = "PINHOLE"
# params are f, f, cx ,cy
# Modify camera intrinsics to follow a different convention.
# Coordinates of the center of the top-left pixels are by default:
# - (0.5, 0.5) in Colmap
# - (0,0) in OpenCV

cameras = {
    idx: CM.Camera(
        id=idx,
        model=cam_model,
        width=width,
        height=height,
        params=[f, f, pp[0] + 0.5, pp[1] + 0.5],
    )
    for idx, (f, pp) in enumerate(zip(focals, pps))
}

CM.write_cameras_text(cameras, os.path.join(output_folder, "cameras.txt"))

# %%
# Create points3D.txt

imgs = scene.imgs
# 3D pointcloud from depthmap, poses and intrinsics
pts3d = to_numpy(scene.get_pts3d())
scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
mask = to_numpy(scene.get_masks())

pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
col = (col * 255).clip(0, 255).astype(np.uint8)


points3d = {}
for idx, (pt, rgb) in tqdm(
    enumerate(zip(pts, col)), total=len(pts), desc="Creating points3D.txt"
):
    points3d[idx] = CM.Point3D(
        id=idx,
        xyz=pt,
        rgb=rgb,
        error=1.0,
        image_ids=np.array([0]),
        point2D_idxs=np.array([]),
    )


CM.write_points3D_text(points3d, os.path.join(output_folder, "points3D.txt"))

# %%

# Create images.txt
print("Creating images.txt")
poses = scene.get_im_poses().cpu().detach().numpy()

images = {}
for idx, p in tqdm(enumerate(poses), total=len(poses), desc="Creating images.txt"):
    p = np.linalg.inv(p)
    images[idx] = CM.Image(
        id=idx,
        qvec=CM.rotmat2qvec(p[:3, :3]),
        tvec=p[:3, 3],
        camera_id=idx,
        name=f"{idx:05d}.png",
        xys=np.array([]),  # [[0, 0] for _ in range(1000)]),
        point3D_ids=np.array([]),  # range(1000)),
    )

CM.write_images_text(images, os.path.join(output_folder, "images.txt"))

# %%
