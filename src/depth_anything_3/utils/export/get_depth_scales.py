import os
import json
import numpy as np
import pycolmap
import cv2
from tqdm import tqdm

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def get_depth_scales(
    reconstruction,
    depth_maps,              # shape: (T, H, W)
    point_max_error=5.0,
    max_rows_per_remap=32000,
):
    cameras = reconstruction.cameras
    images = reconstruction.images
    points3d_dict = reconstruction.points3D

    max_id = max(points3d_dict.keys())
    points3d_ordered = np.zeros((max_id + 1, 3), dtype=np.float64)
    points3d_error_ordered = np.full((max_id + 1,), np.inf, dtype=np.float64)

    for pid, p in points3d_dict.items():
        points3d_ordered[pid] = np.asarray(p.xyz)
        points3d_error_ordered[pid] = p.error

    results = []

    for key in tqdm(images.keys()):
        image_meta = images[key]
        cam_intrinsic = cameras[image_meta.camera_id]

        depth = depth_maps[key - 1]  # assuming image_id = fidx+1
        invmonodepthmap = (1.0 / depth).astype(np.float32)

        Hm, Wm = invmonodepthmap.shape

        pts_idx = image_meta.point3D_ids.astype(np.int64)
        mask = (pts_idx >= 0) & (pts_idx < len(points3d_ordered))
        pts_idx = pts_idx[mask]
        valid_xys = image_meta.xys[mask]

        pts_errors = points3d_error_ordered[pts_idx]
        good = pts_errors < point_max_error
        pts_idx = pts_idx[good]
        valid_xys = valid_xys[good]

        if len(pts_idx) < 10:
            continue

        # world → camera
        R = qvec2rotmat(image_meta.qvec)
        pts = points3d_ordered[pts_idx]
        pts = np.dot(pts, R.T) + image_meta.tvec

        z = pts[..., 2]
        good_z = z > 0
        if good_z.sum() < 10:
            continue

        pts = pts[good_z]
        valid_xys = valid_xys[good_z]
        invcolmapdepth = 1.0 / pts[..., 2]

        # scale to depth resolution
        sx = Wm / cam_intrinsic.width
        sy = Hm / cam_intrinsic.height

        maps = valid_xys.copy().astype(np.float32)
        maps[:, 0] *= sx
        maps[:, 1] *= sy

        inside = (
            (maps[:, 0] >= 0) & (maps[:, 1] >= 0) &
            (maps[:, 0] < Wm) & (maps[:, 1] < Hm)
        )

        maps = maps[inside]
        invcolmapdepth = invcolmapdepth[inside]

        if len(maps) < 10:
            continue

        # ---- chunked cv2.remap ----
        x = np.clip(maps[:, 0], 0, Wm - 1 - 1e-6).astype(np.float32)
        y = np.clip(maps[:, 1], 0, Hm - 1 - 1e-6).astype(np.float32)

        invmono_chunks = []
        for start in range(0, x.shape[0], max_rows_per_remap):
            end = min(start + max_rows_per_remap, x.shape[0])
            mapx = x[start:end].reshape(-1, 1)
            mapy = y[start:end].reshape(-1, 1)
            vals = cv2.remap(
                invmonodepthmap,
                mapx, mapy,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            ).reshape(-1)
            invmono_chunks.append(vals)

        invmonodepth = np.concatenate(invmono_chunks)

        # robust statistics
        t_colmap = np.median(invcolmapdepth)
        s_colmap = np.mean(np.abs(invcolmapdepth - t_colmap))

        t_mono = np.median(invmonodepth)
        s_mono = np.mean(np.abs(invmonodepth - t_mono))

        if s_mono < 1e-12:
            scale, offset = 0, 0
        else:
            scale = s_colmap / s_mono
            offset = t_colmap - t_mono * scale

        results.append({
            "image_name": image_meta.name,
            "scale": float(scale),
            "offset": float(offset),
        })

    # summary
    scales = [r["scale"] for r in results]
    if len(scales) > 0:
        print("scale mean:", np.mean(scales))
        print("scale std :", np.std(scales))

    return results
