import torch
from dust3r.post_process import estimate_focal_knowing_depth
from dust3r.utils.geometry import inv


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

    for p_id in tqdm(range(pairs), desc="Computing poses"):

        view1, view2, pred1, pred2 = get_pair_output(output, p_id)
        pose, focal, pp, conf = process_pair(view1, view2, pred1, pred2, min_conf_thr)
        print(f"{focal=}, {pp=}, {conf=}")
        print(pose, type(pose))

        relative_poses.append(pose)
        focals.append(focal)
        pps.append(pp)
        confs.append(conf)

        cam1_to_world = absolute_poses[-1] @ pose
        absolute_poses.append(cam1_to_world)
        print(absolute_poses[-1])

        cam1_world_pts3d = geotrf(cam1_to_world, pred2["pts3d_in_other_view"][0])

    # last cam focal and pp are not caltulated. Lets use the second to last.
    focals.append(focals[-1])
    pps.append(pps[-1])

    return relative_poses, focals, pps, absolute_poses
