# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pycolmap
import cv2 as cv
import numpy as np
import gsplat
import shutil
import torch
import json
import time

from PIL import Image

from depth_anything_3.specs import Prediction
from depth_anything_3.utils.logger import logger

from .glb import _depths_to_world_points_with_colors

from .common import AsyncNDArraySaver, depths_to_world_points_with_colors_torch

import imageio

from .get_depth_scales import get_depth_scales


def export_to_colmap(
    prediction: Prediction,
    export_dir: str,
    image_paths: list[str],
    masks = None,
    num_max_points: int = 1_000_000,
    conf_thresh_percentile: float = 40.0,
    process_res_method: str = "upper_bound_resize",
    verbose:bool = True,
) -> None:
    # # 1. Data preparation
    conf_thresh = np.percentile(prediction.conf, conf_thresh_percentile)
    # conf_thresh = -10000
    points, colors = depths_to_world_points_with_colors_torch(
        prediction.depth,
        prediction.intrinsics,
        prediction.extrinsics,  # w2c
        prediction.processed_images,
        prediction.conf,
        conf_thresh,
    )
    num_points = len(points)
    if verbose:
        logger.info(f"conf_thresh_percentile: {conf_thresh_percentile}")
        logger.info(f"Exporting to COLMAP with {num_points} points")
    num_frames = len(prediction.processed_images)
    h, w = prediction.processed_images.shape[1:3]
    points_xyf = _create_xyf(num_frames, h, w)
    conf_mask = prediction.conf >= conf_thresh
    conf_mask_flat = conf_mask.reshape(-1)
    conf_indices = np.where(conf_mask_flat)[0]
    points_xyf = points_xyf[conf_mask]
    # points_xyf = points_xyf[prediction.conf >= -1000]
    if verbose:
        print("Threshold percentile :", conf_thresh_percentile)
        print("Conf Threshold :", conf_thresh)
        print("Conf filtered :", points_xyf.shape[0])
        print(points_xyf.shape, points.shape, colors.shape)

    points, points_xyf, colors, filter_idices = _filter_and_downsample(points, points_xyf, colors, num_max_points)

    final_indices = conf_indices[filter_idices]

    final_mask_flat = np.zeros_like(conf_mask_flat, dtype=bool)
    final_mask_flat[final_indices] = True
    final_mask = final_mask_flat.reshape(conf_mask.shape)
    if verbose:
        print("Mask Shape:", final_mask.shape)
        print("Mask sum:", final_mask.sum())

    num_points = len(points)
    if verbose:
        logger.info(f"After filtering and downsampling, exporting {num_points} points to COLMAP")

    start_time = time.time()
    # 2. Set Reconstruction
    reconstruction = pycolmap.Reconstruction()

    point3d_ids = []
    for vidx in range(num_points):
        point3d_id = reconstruction.add_point3D(points[vidx], pycolmap.Track(), colors[vidx])
        point3d_ids.append(point3d_id)

    for fidx in range(num_frames):
        # orig_w, orig_h = Image.open(image_paths[fidx]).size

        intrinsic = prediction.intrinsics[fidx]
        # if process_res_method.endswith("resize"):
        #     intrinsic[:1] *= orig_w / w
        #     intrinsic[1:2] *= orig_h / h
        # elif process_res_method == "crop":
        #     raise NotImplementedError("COLMAP export for crop method is not implemented")
        # else:
        #     raise ValueError(f"Unknown process_res_method: {process_res_method}")

        pycolmap_intri = np.array(
            [intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]]
        )

        extrinsic = prediction.extrinsics[fidx]
        cam_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(extrinsic[:3, :3]), extrinsic[:3, 3])

        # set and add camera
        camera = pycolmap.Camera()
        camera.camera_id = fidx + 1
        camera.model = pycolmap.CameraModelId.PINHOLE
        camera.width = w
        camera.height = h
        camera.params = pycolmap_intri
        reconstruction.add_camera(camera)

        # set and add rig (from camera)
        rig = pycolmap.Rig()
        rig.rig_id = camera.camera_id
        rig.add_ref_sensor(camera.sensor_id)
        reconstruction.add_rig(rig)

        # set image
        image = pycolmap.Image()
        image.image_id = fidx + 1
        image.camera_id = camera.camera_id

        # set and add frame (from image)
        frame = pycolmap.Frame()
        frame.frame_id = image.image_id
        frame.rig_id = camera.camera_id
        frame.add_data_id(image.data_id)
        frame.rig_from_world = cam_from_world
        reconstruction.add_frame(frame)

        # set point2d and update track
        point2d_list = []
        points_in_frame = points_xyf[:, 2].astype(np.int32) == fidx
        for vidx in np.where(points_in_frame)[0]:
            point2d = points_xyf[vidx][:2]
            # point2d[0] *= orig_w / w
            # point2d[1] *= orig_h / h
            point3d_id = point3d_ids[vidx]
            point2d_list.append(pycolmap.Point2D(point2d, point3d_id))
            reconstruction.point3D(point3d_id).track.add_element(
                image.image_id, len(point2d_list) - 1
            )

        # set and add image
        image.frame_id = image.image_id
        image.name = os.path.basename(image_paths[fidx])
        image.points2D = pycolmap.Point2DList(point2d_list)
        reconstruction.add_image(image)
    if verbose:
        print(f"[Time Info] Colmap processing took {time.time() - start_time}")
    # # Save depth scales
    # print("Depth scale estimation...")
    # depth_scales = get_depth_scales(reconstruction, prediction.depth)

    # depth_params = {
    #     depth_param["image_name"]: {"scale": depth_param["scale"], "offset": depth_param["offset"]}
    #     for depth_param in depth_scales if depth_param != None
    # }

    # depth_scale_path = os.path.join(export_dir, "colmap", "estimated_depth_scales.json")
    # with open(depth_scale_path, "w") as f:
    #     json.dump(depth_params, f, indent=4, ensure_ascii=False)
    # print("Saved to `{}`".format(depth_scale_path))

    start_time = time.time()
    # 3. Export
    os.makedirs(os.path.join(export_dir, "colmap", "sparse"), exist_ok=True)
    reconstruction.write(os.path.join(export_dir, "colmap", "sparse"))

    assert prediction.processed_images.shape[:3] == prediction.depth.shape,\
        f"prediction shape mismatch (img, depth) : {prediction.processed_images.shape[:3]} {prediction.depth.shape}"

    # Save Processed Images
    dst_dir = os.path.join(export_dir, "colmap", "images")
    os.makedirs(dst_dir, exist_ok=True)
    for fidx, image_path in enumerate(image_paths):
        filename = os.path.basename(image_path)
        dst_path = os.path.join(dst_dir, filename)
        imageio.imwrite(dst_path, prediction.processed_images[fidx])

    # # Save Processed Masks
    # if masks is not None:
    #     dst_dir = os.path.join(export_dir, "colmap", "masks")
    #     os.makedirs(dst_dir, exist_ok=True)
    #     masks = masks.permute(0, 2, 3, 1).contiguous()
    #     masks = (masks != 0).to(torch.uint8) * 255
    #     for i, mask in enumerate(masks):
    #         filename = f"{i:06d}.png"
    #         dst_path = os.path.join(dst_dir, filename)
    #         imageio.imwrite(dst_path, mask)

    # # copy images
    # if export_dir.split("/")[-1] == "scene_recon":
    #     src_dir = os.path.join(export_dir, "input_images")
    # else:
    #     src_dir = os.path.join(export_dir, "..", "input_images")
    # dst_dir = os.path.join(export_dir, "colmap", "images")
    
    # for i in range(prediction.processed_images.shape[0]):
    #     filename = f"{i:05d}.png"
    #     src_path = os.path.join(src_dir, filename)
    #     dst_path = os.path.join(dst_dir, filename)
    #     if os.path.exists(src_path) and os.path.exists(dst_path):
    #         shutil.copy2(src_path, dst_path)

    # # Save depth maps
    # depth_dir = os.path.join(export_dir, "colmap", "estimated_depths_da3")
    # os.makedirs(depth_dir, exist_ok=True)
    # for fidx in range(num_frames):
    #     image_name = os.path.basename(image_paths[fidx])
    #     output_path = os.path.join(depth_dir, f"{image_name}.npy")
    #     depth = prediction.depth[fidx]
    #     # normalized_depth = (depth - depth.min()) / (depth.max() - depth.min())
    #     np.save(output_path, depth)

    # # Save point masks
    # mask_dir = os.path.join(export_dir, "colmap", "point_masks")
    # os.makedirs(mask_dir, exist_ok=True)
    # for fidx in range(num_frames):
    #     image_name = os.path.basename(image_paths[fidx])
    #     output_path = os.path.join(mask_dir, f"{image_name}")
    #     mask = final_mask[fidx]
    #     mask_img = (mask * 255).astype(np.uint8)
    #     imageio.imwrite(output_path, mask_img)

    if verbose:
        print(f"[Time Info] Colmap saving took {time.time() - start_time}")

def _create_xyf(num_frames, height, width):
    """
    Creates a grid of pixel coordinates and frame indices (fidx) for all frames.
    """
    # Create coordinate grids for a single frame
    y_grid, x_grid = np.indices((height, width), dtype=np.int32)
    x_grid = x_grid[np.newaxis, :, :]
    y_grid = y_grid[np.newaxis, :, :]

    # Broadcast to all frames
    x_coords = np.broadcast_to(x_grid, (num_frames, height, width))
    y_coords = np.broadcast_to(y_grid, (num_frames, height, width))

    # Create frame indices and broadcast
    f_idx = np.arange(num_frames, dtype=np.int32)[:, np.newaxis, np.newaxis]
    f_coords = np.broadcast_to(f_idx, (num_frames, height, width))

    # Stack coordinates and frame indices
    points_xyf = np.stack((x_coords, y_coords, f_coords), axis=-1)

    return points_xyf

def _filter_and_downsample(points: np.ndarray, points_xyf:np.ndarray, colors: np.ndarray, num_max: int):
    if points.shape[0] == 0:
        return points, colors
    finite = np.isfinite(points).all(axis=1)
    points, colors = points[finite], colors[finite]
    if points_xyf is not None:
        points_xyf = points_xyf[finite]
    out_idx = np.where(finite)[0]
    if points.shape[0] > num_max:
        idx = np.random.choice(points.shape[0], num_max, replace=False)
        points, colors = points[idx], colors[idx]
        if points_xyf is not None:
            points_xyf = points_xyf[idx]
        out_idx = out_idx[idx]
    return points, points_xyf, colors, out_idx