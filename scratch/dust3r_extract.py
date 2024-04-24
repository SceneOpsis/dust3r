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


def load_model(weights, device):
    return AsymmetricCroCo3DStereo.from_pretrained(weights).to(args.device)


# %%
# load model
model = load_model(weights, device)

# %%
# Do some processing

# filelist = ["../res/00001.png", "../res/00037.png"]
# glob file list from folder
# filelist = glob.glob("../res/2frames/*.png")
# exp_name = "2near_more"
# filelist = glob.glob(f"../res/undistorted/{exp_name}/*.png")
# print(filelist)

pairs = [
    [1, 2],
    [2, 3],
    [3, 4],
    [1, 10],
    [10, 20],
    [20, 30],
    [30, 40],
    [40, 50],
]

pairs = [
    [
        f"../res/undistorted/images/{p[0]:05d}.png",
        f"../res/undistorted/images/{p[1]:05d}.png",
    ]
    for p in pairs
]

print("Pairs: ", pairs)

# %%


def process(filelist):
    print(f"Processing {filelist}")
    imgs = load_images(filelist, size=image_size)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]["idx"] = 1

    scenegraph_type = "complete"

    pairs = make_pairs(
        imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True
    )

    output = inference(pairs, model, device, batch_size=batch_size)

    mode = (
        GlobalAlignerMode.PointCloudOptimizer
        if len(imgs) > 2
        else GlobalAlignerMode.PairViewer
    )
    scene = global_aligner(output, device=device, mode=mode)  # , optimize_pp=True)
    print(f"{type(scene)=}")

    lr = 0.01

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(
            init="mst", niter=niter, schedule=schedule, lr=lr
        )
        print(f"Loss = {loss}")

    print("Cleanining pointcloud")
    scene = scene.clean_pointcloud()

    print("done")
    return scene


# %%

from pathlib import Path

for filelist in pairs:
    scene = process(filelist)

    pts3d = to_numpy(scene.get_pts3d())
    masks = to_numpy(scene.get_masks())
    depths = to_numpy(scene.get_depthmaps())
    imgs = to_numpy(scene.imgs)

    name = ""
    for f in filelist:
        name += f"{(Path(f).stem)}_"

    name = name[:-1] + ".npz"

    print("Saving to " + name)
    pack = {"pts3d": pts3d, "masks": masks, "depths": depths, "imgs": imgs}
    np.savez("../tmp/" + name, **pack)

# %%


# visualize p3d in 3d in matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Reshape the array to 2D
p3d_reshaped = p3d.reshape(-1, 3)

# Separate the coordinates
x = p3d_reshaped[:, 0]
y = p3d_reshaped[:, 1]
z = p3d_reshaped[:, 2]

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x, y, z)

# Set labels
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Show the plot
# do not open matplot lib inline

plt.show()
# %%


pair = np.load("../tmp/00001_00002.npz")


# %%
pair.keys()
# %%
imgs = pair["imgs"]
depths = pair["depths"]
# %%
plt.imshow(depths[0])
# %%
