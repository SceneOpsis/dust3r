from dust3r.utils.device import to_numpy
from dust3r.utils.image import rgb
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
import trimesh
import numpy as np
from scipy.spatial.transform import Rotation
import os
import torch

import functools
import gradio

import matplotlib.pyplot as pl


def _convert_scene_output_to_glb(
    outdir,
    imgs,
    pts3d,
    mask,
    focals,
    cams2world,
    cam_size=0.05,
    cam_color=None,
    as_pointcloud=False,
    transparent_cams=False,
    silent=False,
):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    # full pointcloud
    if as_pointcloud:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
        pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    # add each camera
    for i, pose_c2w in enumerate(cams2world):
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(
            scene,
            pose_c2w,
            camera_edge_color,
            None if transparent_cams else imgs[i],
            focals[i],
            imsize=imgs[i].shape[1::-1],
            screen_width=cam_size,
        )

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler("y", np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    outfile = os.path.join(outdir, "scene.glb")
    if not silent:
        print("(exporting 3D scene to", outfile, ")")
    scene.export(file_obj=outfile)
    return outfile


def get_3D_model_from_scene(
    outdir,
    silent,
    scene,
    min_conf_thr=3,
    as_pointcloud=False,
    mask_sky=False,
    clean_depth=False,
    transparent_cams=False,
    cam_size=0.05,
):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    if scene is None:
        return None
    # post processes
    if clean_depth:
        scene = scene.clean_pointcloud()
    if mask_sky:
        scene = scene.mask_sky()

    # get optimized values from scene
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    # 3D pointcloud from depthmap, poses and intrinsics
    pts3d = to_numpy(scene.get_pts3d())
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
    msk = to_numpy(scene.get_masks())
    return _convert_scene_output_to_glb(
        outdir,
        rgbimg,
        pts3d,
        msk,
        focals,
        cams2world,
        as_pointcloud=as_pointcloud,
        transparent_cams=transparent_cams,
        cam_size=cam_size,
        silent=silent,
    )


def get_reconstructed_scene(
    outdir,
    scene,
    silent,
    min_conf_thr,
):

    outfile = get_3D_model_from_scene(
        outdir,
        silent,
        scene,
        min_conf_thr,
        as_pointcloud=True,
        mask_sky=False,
        clean_depth=True,
        transparent_cams=True,
        cam_size=0.01,
    )

    # also return rgb, depth and confidence imgs
    # depth is normalized with the max value for all images
    # we apply the jet colormap on the confidence maps
    rgbimg = scene.imgs
    depths = to_numpy(scene.get_depthmaps())
    confs = to_numpy([c for c in scene.im_conf])
    cmap = pl.get_cmap("jet")
    depths_max = max([d.max() for d in depths])
    depths = [d / depths_max for d in depths]
    confs_max = max([d.max() for d in confs])
    confs = [cmap(d / confs_max) for d in confs]

    imgs = []
    for i in range(len(rgbimg)):
        imgs.append(rgbimg[i])
        imgs.append(rgb(depths[i]))
        imgs.append(rgb(confs[i]))

    return scene, outfile, imgs


def visualize_scene(
    scene,
    tmpdirname="/tmp",
    server_name="0.0.0.0",
    server_port=7860,
    silent=False,
    min_conf_thr=3,
):

    recon_fun = functools.partial(
        get_reconstructed_scene, tmpdirname, scene, silent, min_conf_thr
    )

    with gradio.Blocks(
        css=""".gradio-container {margin: 0 !important; min-width: 100%};""",
        title="DUSt3R Demo",
    ) as demo:
        # scene state is save so that you can change conf_thr, cam_size... without rerunning the inference
        scene_state = gradio.State(None)
        gradio.HTML('<h2 style="text-align: center;">DUSt3R Demo</h2>')
        with gradio.Column():
            run_btn = gradio.Button("Run")
            outmodel = gradio.Model3D()
            outgallery = gradio.Gallery(
                label="rgb,depth,confidence", columns=3, height="100%"
            )
            # events
            run_btn.click(
                fn=recon_fun,
                inputs=[],
                outputs=[scene_state, outmodel, outgallery],
            )
    demo.launch(share=False, server_name=server_name, server_port=server_port)


def visualize_poses(
    poses,
    focals,
    point_cloud,
    color_imgs,
    server_name="0.0.0.0",
    server_port=7860,
):

    recon_fun = functools.partial(
        trimesh_cameras, poses, focals, point_cloud, color_imgs
    )

    with gradio.Blocks(
        css=""".gradio-container {margin: 0 !important; min-width: 100%};""",
        title="DUSt3R Demo",
    ) as demo:
        # scene state is save so that you can change conf_thr, cam_size... without rerunning the inference
        scene_state = gradio.State(None)
        gradio.HTML('<h2 style="text-align: center;">Camera Poses</h2>')
        with gradio.Column():
            run_btn = gradio.Button("Run")
            outmodel = gradio.Model3D()
            # events
            run_btn.click(
                fn=recon_fun,
                inputs=[],
                outputs=[outmodel],
            )
    demo.launch(
        share=False,
        server_name=server_name,
        server_port=server_port,
        inline=False,
    )

    return demo


def trimesh_cameras(
    poses, focals, point_cloud, color_imgs, cam_size=0.05, cam_color=None
):
    scene = trimesh.Scene()
    for i, pose_c2w in enumerate(poses):
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(
            scene,
            pose_c2w,
            camera_edge_color,
            None,
            focals[i],
            imsize=(512, 512),
            screen_width=cam_size,
        )

    # add point cloud
    pts = np.concatenate(point_cloud)
    col = np.concatenate(color_imgs)
    pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
    scene.add_geometry(pct)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler("y", np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(OPENGL @ rot))

    outfile = os.path.join("/tmp", "scene.glb")
    print("(exporting 3D scene to", outfile, ")")
    scene.export(file_obj=outfile)
    return outfile
