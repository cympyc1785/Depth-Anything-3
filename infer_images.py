import os, sys
from tqdm import tqdm
import subprocess
import numpy as np
import random
import time
import json

from depth_anything_3.api import DepthAnything3

from depth_anything_3.services.input_handlers import (
    ImagesHandler,
    InputHandler,
    VideoHandler,
    parse_export_feat,
)

from depth_anything_3.utils.constants import (
    DEFAULT_EXPORT_DIR,
    DEFAULT_MODEL,
)

def save_camera_params(extrinsics, intriniscs, save_path):
    data = []

    assert save_path.endswith(".json")

    extrinsics = np.array(extrinsics)
    intrinsics = np.array(intriniscs)
    assert len(extrinsics) == len(intriniscs)

    for i in range(len(extrinsics)):
        rotation = extrinsics[i][:3, :3].astype(float)
        position = extrinsics[i][:3, 3].astype(float)
        fx = float(intrinsics[i][0, 0])
        fy = float(intrinsics[i][1, 1])
        cx = float(intrinsics[i][0, 2])
        cy = float(intrinsics[i][1, 2])

        data.append({
            "idx": i,
            "rotation": rotation.tolist(),
            "position": position.tolist(),
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy
        })
    
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)

def get_camera_params_from_npz(cam_param_path):
    cam_params = np.load(cam_param_path, allow_pickle=True)
    if 'extrinsics' in cam_params.keys():
        ext = cam_params['extrinsics']
    elif 'poses' in cam_params.keys():
        ext = cam_params['poses']
    else:
        raise KeyError("No extrinsic key", cam_params)
    intrinsics = cam_params['intrinsics']

    return ext, intrinsics

def convert_camera(export_dir):
    npz_path = os.path.join(export_dir, "exports/npz/results.npz")
    if not os.path.exists(npz_path):
        npz_path = os.path.join(export_dir, "exports/mini_npz/results.npz")
    save_camera_path = os.path.join(export_dir, "cameras.json")
    if os.path.exists(npz_path):
        ext, intrinsics = get_camera_params_from_npz(npz_path)
        save_camera_params(ext, intrinsics, save_camera_path)


def run_da3_video(
    model,
    images_dir: str,
    frame_range: tuple[int, int] = (0, 1000),
    frame_interval: int = 1,
    mask_dir: str = None,
    image_extensions: str = "png,jpg,jpeg",
    export_dir: str = DEFAULT_EXPORT_DIR,
    export_format: str = "glb",
    process_res: int = 504,
    process_res_method: str = "upper_bound_resize",
    export_feat: str = "",
    auto_cleanup: bool = False,
    # Pose condition
    extrinsics: np.ndarray = None,
    intrinsics: np.ndarray = None,
    align_to_input_ext_scale: bool = True,
    # Pose estimation options
    use_ray_pose: bool = False,
    ref_view_strategy: str = "saddle_balanced",
    # GLB export options
    conf_thresh_percentile: float = 40.0,
    num_max_points: int = 1_000_000,
    show_cameras: bool = True,
    # Feat_vis export options
    feat_vis_fps: int = 15,
):
    import time
    start_time = time.time()
    """Run depth estimation on video by extracting frames and processing them."""
    # Handle export directory
    export_dir = InputHandler.handle_export_dir(export_dir, auto_cleanup)


    image_files = ImagesHandler.process(images_dir, image_extensions, frame_range, frame_interval)

    # Process mask input
    mask_files = None
    if mask_dir is not None:
        mask_files = ImagesHandler.process(mask_dir, image_extensions)

    # Parse export_feat parameter
    export_feat_layers = parse_export_feat(export_feat)

    inference_kwargs = {
        "image": image_files,
        "mask": mask_files,
        "export_dir": export_dir,
        "export_format": export_format,
        "process_res": process_res,
        "process_res_method": process_res_method,
        "export_feat_layers": export_feat_layers,
        "align_to_input_ext_scale": align_to_input_ext_scale,
        "use_ray_pose": use_ray_pose,
        "ref_view_strategy": ref_view_strategy,
        "conf_thresh_percentile": conf_thresh_percentile,
        "num_max_points": num_max_points,
        "show_cameras": show_cameras,
        "feat_vis_fps": feat_vis_fps,
        "infer_gs": "gs" in export_format,
    }

    # Add pose data (if exists)
    if extrinsics is not None:
        inference_kwargs["extrinsics"] = extrinsics
    if intrinsics is not None:
        inference_kwargs["intrinsics"] = intrinsics

    print(f"Running inference on {len(image_files)} images...")

    prediction = model.inference(**inference_kwargs)

    convert_camera(export_dir)

    print(f"Results saved to {export_dir}")
    print(f"Export format: {export_format}")

    print("Ended. It took", time.time() - start_time)

    return prediction

model = DepthAnything3.from_pretrained("./models/DA3NESTED-GIANT-LARGE").to("cuda")
model.eval()

# Infer Single
# root_dir = "/NHNHOME/WORKSPACE/0226010013_A/cympyc1785/Depth-Anything-3/tartanair"
# export_root_dir = "/NHNHOME/WORKSPACE/0226010013_A/cympyc1785/Depth-Anything-3/tartanair_output"
# chunk_i = 7
# scene_name = "scene0000_00"
# scene_dir = os.path.join(root_dir, scene_name)
# export_scenename = scene_name + f"_{chunk_i:02d}"
# export_dir = os.path.join(export_root_dir, export_scenename)
# images_dir = os.path.join(scene_dir, "images")
# run_da3_video(model, images_dir,
#                         frame_range=(chunk_i * chunk_len, (chunk_i + 1) * chunk_len),
#                         frame_interval=frame_interval,
#                         export_dir=export_dir,
#                         export_format="mini_npz-ply-images",
#                         auto_cleanup=False,
#                         )
# exit()

root_dir = "/NHNHOME/WORKSPACE/0226010013_A/cympyc1785/scenetok/DATA/DL3DV/DL3DV-960/train"
export_root_dir = "/NHNHOME/WORKSPACE/0226010013_A/cympyc1785/scenetok/DATA/DL3DV/DL3DV-960/train"
splits = ["1K", "11K"]
# for split in sorted(os.listdir(root_dir)):
for split in splits:
    split_dir = os.path.join(root_dir, split)
    for scene_name in tqdm(sorted(os.listdir(split_dir))):
        scene_dir = os.path.join(split_dir, scene_name)
        images_dir = os.path.join(scene_dir, "images")
        if not os.path.exists(images_dir):
            images_dir = os.path.join(scene_dir, "images_4")
        if not os.path.exists(images_dir):
            images_dir = os.path.join(scene_dir, "images_8")
        if not os.path.exists(images_dir):
            images_dir = os.path.join(scene_dir, "rgb")
        if not os.path.exists(images_dir):
            continue
        frame_len = len(os.listdir(images_dir))
        export_dir = os.path.join(export_root_dir, split, scene_name, "da3")
        

        if os.path.exists(export_dir) and os.path.exists(os.path.join(export_dir, "exports/mini_npz/results.npz")):
            continue

        run_da3_video(model, images_dir,
                        frame_range=(0, frame_len),
                        export_dir=export_dir,
                        export_format="mini_npz",
                        auto_cleanup=True,
                        )

# root_dir = "/NHNHOME/WORKSPACE/0226010013_A/cympyc1785/Depth-Anything-3/tartanair"
# export_root_dir = "/NHNHOME/WORKSPACE/0226010013_A/cympyc1785/Depth-Anything-3/tartanair_output"
# frame_interval = 1
# chunk_len = 300 * frame_interval
# splits = ["Easy_right"]
# for split in splits:
#     split_dir = os.path.join(root_dir, split)
#     for scene_name in tqdm(sorted(os.listdir(split_dir))):
#         scene_dir = os.path.join(split_dir, scene_name)
#         frame_len = len(os.listdir(os.path.join(scene_dir, "images")))

#         chunk_num = frame_len // chunk_len
#         if chunk_num == 0:
#             chunk_num = 1 

#         for chunk_i in range(chunk_num):
#             export_scenename = scene_name + f"_{chunk_i:02d}"

#             export_dir = os.path.join(export_root_dir, split, export_scenename)
#             images_dir = os.path.join(scene_dir, "images")

#             if os.path.exists(export_dir) and os.path.exists(os.path.join(export_dir, "cameras.json")):
#                 continue

#             run_da3_video(model, images_dir,
#                             frame_range=(chunk_i * chunk_len, min((chunk_i + 1) * chunk_len, frame_len)),
#                             frame_interval=frame_interval,
#                             export_dir=export_dir,
#                             export_format="mini_npz-ply-images",
#                             auto_cleanup=True,
#                             )