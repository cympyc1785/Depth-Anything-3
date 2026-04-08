import os
from typing import Literal, Optional
import moviepy.editor as mpy
import torch
import json
import numpy as np

from depth_anything_3.utils.gsply_helpers import load_ply
from depth_anything_3.model.utils.gs_renderer import run_renderer_in_chunk_w_trj_mode
from depth_anything_3.utils.layout_helpers import hcat, vcat
from depth_anything_3.utils.visualize import vis_depth_map_tensor
from depth_anything_3.utils.geometry import affine_inverse, as_homogeneous

from depth_anything_3.specs import Prediction, Gaussians

from tqdm import tqdm

VIDEO_QUALITY_MAP = {
    "low": {"crf": "28", "preset": "veryfast"},
    "medium": {"crf": "23", "preset": "medium"},
    "high": {"crf": "18", "preset": "slow"},
}

def load_gs(root_path):
    device = "cuda"
    gaussian_path = os.path.join(root_path, "gs_ply", "0000.ply")
    scale_factor_path = os.path.join(root_path, "gs_ply", "scale_factor.json")
    camera_path = os.path.join(root_path, "camera_params.npz")

    ply_data = load_ply(gaussian_path)

    gaussians = Gaussians(
        means = ply_data["means"].unsqueeze(0).to(device),
        scales = ply_data["scales"].unsqueeze(0).to(device),
        rotations=ply_data["rotations"].unsqueeze(0).to(device),
        harmonics=ply_data["harmonics"].unsqueeze(0).to(device),
        opacities=ply_data["opacities"].unsqueeze(0).to(device),
    )

    with open(scale_factor_path, "r") as f:
        scale_factor = json.load(f)

    camera_params = np.load(camera_path)

    extrinsics = torch.from_numpy(camera_params["extrinsics"]).unsqueeze(0).to(device)
    # extrinsics = affine_inverse(as_homogeneous(extrinsics))
    intrinsics = torch.from_numpy(camera_params["intrinsics"]).unsqueeze(0).to(device)

    return gaussians, scale_factor, extrinsics, intrinsics

def export_to_gs_video(
    gaussians: Gaussians,
    scale_factor: float,
    export_dir: str,
    extrinsics: Optional[torch.Tensor] = None,  # render views' world2cam, "b v 4 4"
    intrinsics: Optional[torch.Tensor] = None,  # render views' unnormed intrinsics, "b v 3 3"
    out_image_hw: Optional[tuple[int, int]] = None,  # render views' resolution, (h, w)
    chunk_size: Optional[int] = 4,
    trj_mode: Literal[
        "original",
        "smooth",
        "interpolate",
        "interpolate_smooth",
        "wander",
        "dolly_zoom",
        "extend",
        "wobble_inter",
    ] = "original",
    color_mode: Literal["RGB+D", "RGB+ED"] = "RGB+ED",
    vis_depth: Optional[Literal["hcat", "vcat"]] = None,
    enable_tqdm: Optional[bool] = True,
    output_name: Optional[str] = None,
    video_quality: Literal["low", "medium", "high"] = "high",
) -> None:
    gs_world = gaussians
    tgt_extrs = extrinsics
    if scale_factor is not None:
        tgt_extrs[:, :, :3, 3] /= scale_factor
                
    tgt_intrs = intrinsics
    # if render resolution is not provided, render the input ones
    if out_image_hw is not None:
        H, W = out_image_hw
    else:
        H, W = int(2 * intrinsics[0, 0, 1, 2].item()), int(2 * intrinsics[0, 0, 0, 2].item())

    # if single views, render wander trj
    if tgt_extrs.shape[1] <= 1:
        trj_mode = "wander"
        # trj_mode = "dolly_zoom"

    color, depth = run_renderer_in_chunk_w_trj_mode(
        gaussians=gs_world,
        extrinsics=tgt_extrs,
        intrinsics=tgt_intrs,
        image_shape=(H, W),
        chunk_size=chunk_size,
        trj_mode=trj_mode,
        use_sh=True,
        color_mode=color_mode,
        enable_tqdm=enable_tqdm,
    )

    # save as video
    ffmpeg_params = [
        "-crf",
        VIDEO_QUALITY_MAP[video_quality]["crf"],
        "-preset",
        VIDEO_QUALITY_MAP[video_quality]["preset"],
        "-pix_fmt",
        "yuv420p",
    ]  # best compatibility

    os.makedirs(os.path.join(export_dir, "gs_video"), exist_ok=True)
    for idx in range(color.shape[0]):
        video_i = color[idx]
        if vis_depth is not None:
            depth_i = vis_depth_map_tensor(depth[0])
            cat_fn = hcat if vis_depth == "hcat" else vcat
            video_i = torch.stack([cat_fn(c, d) for c, d in zip(video_i, depth_i)])
        frames = list(
            (video_i.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        )  # T x H x W x C, uint8, numpy()

        fps = 10
        clip = mpy.ImageSequenceClip(frames, fps=fps)
        output_name = f"{idx:04d}_{trj_mode}" if output_name is None else output_name
        save_path = os.path.join(export_dir, f"gs_video/{output_name}.mp4")
        # clip.write_videofile(save_path, codec="libx264", audio=False, bitrate="4000k")
        clip.write_videofile(
            save_path,
            codec="libx264",
            audio=False,
            fps=fps,
            ffmpeg_params=ffmpeg_params,
        )
    return

if __name__ == "__main__":
    ROOT_PATH = "/data1/cympyc1785/SceneData/DynamicVerse/scenes"

    with open("rerender.txt", "r") as f:
        errors = f.readlines()

    errors = [("DAVIS/boat")]

    for error in tqdm(errors):
        split, scene_name = error.split(" ")[0].split("/")
        scene_dir = os.path.join(ROOT_PATH, split, scene_name)
        da3_dir = os.path.join(scene_dir, "da3/inpainted")

        gaussians, scale_factor, extrinsics, intrinsics = load_gs(da3_dir)

        export_to_gs_video(
            gaussians=gaussians,
            scale_factor=scale_factor,
            export_dir=os.path.join(da3_dir, "gs_video_custom"),
            extrinsics=extrinsics,
            intrinsics=intrinsics,
        )
