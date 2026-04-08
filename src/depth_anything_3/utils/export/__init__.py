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
import numpy as np
import time

from depth_anything_3.specs import Prediction
from depth_anything_3.utils.export.gs import export_to_gs_ply, export_to_gs_video

from .colmap import export_to_colmap
from .depth_vis import export_to_depth_vis, export_to_depth_video
from .feat_vis import export_to_feat_vis
from .glb import export_to_glb, _compute_alignment_transform_first_cam_glTF_center_by_points
from .npz import export_to_mini_npz, export_to_npz
from .ply import export_to_ply
from .conf import export_to_conf
from .sky import export_to_sky
from .depth import export_to_depth

from .filter import _depths_to_world_points_with_colors, _create_xyf, _filter_and_downsample 

def postprocess_prediction(prediction, export_formats, **kwargs):
    check_formats = ["colmap", "ply", "glb"]

    # breakpoint()
    start_time = time.time()

    process_filtering = False
    for fmt in check_formats:
        if fmt in export_formats:
            process_filtering = True
            settings = kwargs.get(fmt, {})
            num_max_points = settings.get("num_max_points", 1_000_000)
            conf_thresh_percentile = settings.get("conf_thresh_percentile", 40.0)
            break
    
    if not process_filtering:
        return kwargs
    
    conf_thresh = np.percentile(prediction.conf, conf_thresh_percentile)
    points, colors = _depths_to_world_points_with_colors(
        prediction.depth,
        prediction.intrinsics,
        prediction.extrinsics,  # w2c
        prediction.processed_images,
        prediction.conf,
        conf_thresh,
    )

    if "glb" in export_formats:
        import trimesh
        A = _compute_alignment_transform_first_cam_glTF_center_by_points(
            prediction.extrinsics[0], points
        )

        kwargs["glb"].update({"align_matrix": A})

        if points.shape[0] > 0:
            points = trimesh.transform_points(points, A)

    imgs = prediction.processed_images
    num_frames = len(imgs)
    h, w = imgs.shape[1:3]

    conf_mask = prediction.conf >= conf_thresh
    conf_mask_flat = conf_mask.reshape(-1)
    conf_indices = np.where(conf_mask_flat)[0]

    points_xyf = _create_xyf(num_frames, h, w)
    points_xyf = points_xyf[conf_mask]

    points, points_xyf, colors, filter_idices = _filter_and_downsample(points, colors, num_max_points, points_xyf)

    final_indices = conf_indices[filter_idices]

    final_mask_flat = np.zeros_like(conf_mask_flat, dtype=bool)
    final_mask_flat[final_indices] = True
    final_mask = final_mask_flat.reshape(conf_mask.shape)
    print("Mask Shape:", final_mask.shape)
    print("Mask sum:", final_mask.sum())

    num_points = len(points)
    print(f"After filtering and downsampling, exporting {num_points} points")

    data = {
        "points": points,
        "colors": colors,
        "points_xyf": points_xyf,
        "filtering_mask": final_mask
    }

    for fmt in check_formats:
        if fmt in export_formats:
            if fmt in kwargs.keys():
                kwargs[fmt].update(data)
            else:
                kwargs[fmt] = data

    print(f"[Time Info] Postprocess took {time.time() - start_time}")

    return kwargs

def return_kwargs(kwargs):
    return kwargs

def export(
    prediction: Prediction,
    export_format: str,
    export_dir: str,
    **kwargs,
):
    if "-" in export_format:
        export_formats = export_format.split("-")
        # print(export_formats)
        # print(kwargs.keys())
        # kwargs = postprocess_prediction(prediction, export_formats, **kwargs)
        # print(kwargs.keys())
            
        for export_format in export_formats:
            export(prediction, export_format, export_dir, **kwargs)
        return  # Prevent falling through to single-format handling

    start_time = time.time()
    if export_format == "glb":
        export_to_glb(prediction, export_dir, **kwargs.get(export_format, {}))
    elif export_format == "mini_npz":
        export_to_mini_npz(prediction, export_dir)
    elif export_format == "npz":
        export_to_npz(prediction, export_dir)
    elif export_format == "feat_vis":
        export_to_feat_vis(prediction, export_dir, **kwargs.get(export_format, {}))
    elif export_format == "depth_vis":
        export_to_depth_vis(prediction, export_dir)
    elif export_format == "gs_ply":
        export_to_gs_ply(prediction, export_dir, **kwargs.get(export_format, {}))
    elif export_format == "gs_video":
        export_to_gs_video(prediction, export_dir, **kwargs.get(export_format, {}))
    elif export_format == "colmap":
        export_to_colmap(prediction, export_dir, **kwargs.get(export_format, {}))
    elif export_format == "ply":
        export_to_ply(prediction, export_dir, **kwargs.get(export_format, {}))
    elif export_format == "conf":
        export_to_conf(prediction, export_dir, **kwargs.get(export_format, {}))
    elif export_format == "sky":
        export_to_sky(prediction, export_dir, **kwargs.get(export_format, {}))
    elif export_format == "depth_video":
        export_to_depth_video(prediction, export_dir)
    elif export_format == "depth":
        export_to_depth(prediction, export_dir)
    else:
        raise ValueError(f"Unsupported export format: {export_format}")

    print(f"[Time Info] Exporting {export_format} took {time.time() - start_time}")

__all__ = [
    export,
]
