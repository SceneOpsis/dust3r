
# Extract dust3r depths and align to colmap

**Input:** RGB sequence. Single camera.

1. **dust3r_preproc** to extract pointmaps and confidence maps from dust3r. Optionally you can estimate pairwise poses with PnP (sliding window for a sequence).
2. Either refine the extracted poses or run colmap (or other SfM) to get a colmap solution from the RGB frames.
3. **colmap2depth** to create sparse depth maps from the colmap solution of step 2.
4. **align** to align duster data with colmap. Compute per frame _scale_ and _shift_ and store output dence depthmaps to aligned_depth folder. Aligned depthmaps are scaled to dust3r resolution.
5. **cleanup** to cleanup the depthmaps and convert them to colmap pointcloud (points3D)

