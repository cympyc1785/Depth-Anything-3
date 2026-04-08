import os, sys
from tqdm import tqdm
import subprocess
import numpy as np
import random
import time

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

def run_da3_video(
    model,
    video_path: str,
    fps: float = 1.0,
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

    # Process input
    image_files = VideoHandler.process(video_path, export_dir, fps)

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

    print(f"Results saved to {export_dir}")
    print(f"Export format: {export_format}")

    print("Ended. It took", time.time() - start_time)

    return prediction

def get_process_time(model, img_len_lower_bound, img_len_upper_bound, sample_num=20):
    from collections import defaultdict
    def get_tasks(root_dir, splits, lower, upper):
        tasks = []
        for split in splits:
            split_dir = os.path.join(root_dir, split)
            for scene_name in sorted(os.listdir(split_dir)):
                scene_dir = os.path.join(split_dir, scene_name)
                images_dir = os.path.join(scene_dir, "rgb")
                if not os.path.exists(images_dir):
                    images_dir = os.path.join(scene_dir, "images")
                img_len = len(os.listdir(images_dir))
                if img_len < lower or img_len > upper:
                    continue
                tasks.append((scene_dir, split, scene_name))
        return tasks
    
    def get_image_stat(tasks):
        frame_num_dict = defaultdict(int)
        for task in tasks:
            scene_dir, split, scene_name = task
            images_dir = os.path.join(scene_dir, "rgb")
            if not os.path.exists(images_dir):
                images_dir = os.path.join(scene_dir, "images")
            if not os.path.exists(images_dir):
                print("no img", images_dir)
                continue
            img_len = len(os.listdir(images_dir))
            img_len = img_len // 10 * 10
            frame_num_dict[img_len] += 1
        sorted_frame_num_dict = dict(sorted(frame_num_dict.items()))
        return sorted_frame_num_dict

    def get_all_tasks(lower, upper):
        root_dir = "/data1/cympyc1785/SceneData/DynamicVerse/scenes"
        splits = ["DAVIS", "MOSE", "MVS-Synth", "SAV", "VOST", "dynamic_replica", "spring", "uvo", "youtube_vis"]
        tasks = get_tasks(root_dir, splits, lower, upper)

        root_dir = "/data1/cympyc1785/SceneData/DynamicVerse/data/dynpose-100k"
        splits = [f"dynpose-{i:04d}" for i in range(0, 90)]
        tasks += get_tasks(root_dir, splits, lower, upper)

        # root_dir = "/data1/cympyc1785/SceneData/DL3DV/scenes"
        # splits = [f"{i}K" for i in range(1, 8)]
        # tasks += get_tasks(root_dir, splits)

        return tasks

    tasks = get_all_tasks(0, 1000)
    frame_num_dict = get_image_stat(tasks)

    estimated_process_time = 0

    for i in range(img_len_lower_bound//10, img_len_upper_bound//10):
        lower = i * 10
        upper = (i + 1) * 10
        tasks = get_all_tasks(lower, upper)
        tasks = random.sample(tasks, sample_num)
        t1 = time.time()

        for task in tasks:
            scene_dir, split, scene_name = task
            export_dir = "/data1/cympyc1785/3d_recon/Depth-Anything-3/tmp"
            inpainted_video_path = os.path.join(scene_dir, "inpaint_result.mp4")
            mask_dir = os.path.join(scene_dir, "mask")

            run_da3_video(model, inpainted_video_path, fps=10, mask_dir=mask_dir,
                export_dir=export_dir,
                export_format="npz",
                auto_cleanup=True,
                )
        
        avg_time = (time.time() - t1) / sample_num
        print(f"avg time for {lower}:", avg_time)
        e_t = frame_num_dict[lower] * avg_time
        print(f"estimated_time", e_t)
        estimated_process_time += e_t

    print(estimated_process_time, "sec")
    print(estimated_process_time/3600, "hours")
    print(estimated_process_time/3600/24, "days")

    exit()

model = DepthAnything3.from_pretrained("./models/DA3NESTED-GIANT-LARGE").to("cuda")
model.eval()

# get_process_time(model, 300, 310)

# # Infer Single
# scene_dir = "/data1/cympyc1785/3d_recon/Depth-Anything-3/SceneData/DynamicVerse/scenes/dynpose-100k/dynpose-0005/a12197a3-d097-489a-9656-35893ad1ac02"
# export_dir = os.path.join(scene_dir, "da3")
# inpainted_video_path = os.path.join(scene_dir, "inpaint_result.mp4")
# mask_dir = os.path.join(scene_dir, "mask")
# run_da3_video(model, inpainted_video_path, fps=10, mask_dir=mask_dir,
#                 export_dir=f"{scene_dir}/da3/inpainted",
#                 export_format="mini_npz-ply-colmap",
#                 auto_cleanup=True,
#                 )
# exit()



# splits = ["MOSE", "MVS-Synth", "SAV", "VOST", "dynamic_replica", "spring", "uvo"]
# splits = ["dynamic_replica", "spring", "uvo"]



# # Error Rerun
# with open("rerun.txt", "r") as f:
#     errors = f.readlines()
# tasks = []
# for error in errors:
#     split, scene_name = error.split(" ")[0].split("/")
#     tasks.append((split, scene_name))

# TARGET_PATH = "/data1/cympyc1785/SceneData/DynamicVerse/scenes/dynpose-100k" # 여기 기준으로 찾아서 돌림
TARGET_PATH = "/data4/cympyc1785/data/DynamicVerse/dynpose-100k"
EXPORT_ROOT_PATH = "/data4/cympyc1785/data/DynamicVerse/dynpose-100k"

splits = [f"dynpose-{i:04d}" for i in range(67, 73)]

# inpainted_video_path = "/data1/cympyc1785/3d_recon/Depth-Anything-3/SceneData/DynamicVerse/scenes/DAVIS/mallard-fly/inpaint_result.mp4"
# export_dir = "/data1/cympyc1785/3d_recon/Depth-Anything-3/SceneData/DynamicVerse/scenes/DAVIS/blackswan/tmp"
# mask_dir = "/data1/cympyc1785/3d_recon/Depth-Anything-3/SceneData/DynamicVerse/scenes/DAVIS/blackswan/mask"
# run_da3_video(model, inpainted_video_path, fps=10, mask_dir=mask_dir,
#             export_dir=export_dir,
#             export_format="npz",
#             auto_cleanup=True,
#             )
# exit()

root_dir = "/data1/cympyc1785/3d_recon/Depth-Anything-3/SceneData/sintel/dataset"
export_root_dir = "/data1/cympyc1785/data/motion_dataset/DynamicVerse/data/sintel"
ref_dir = export_root_dir
# video_dir = os.path.join(root_dir, "videos")
video_dir = os.path.join(root_dir, "inpainted_videos")

for vid_name in tqdm(sorted(os.listdir(video_dir))):
    scene_name = vid_name.split(".")[0]

    if scene_name not in os.listdir(ref_dir):
        print("skip", scene_name)
        continue

    video_path = os.path.join(video_dir, vid_name)
    mask_dir = os.path.join(root_dir, "mask", scene_name)
    # export_dir = os.path.join(root_dir, "da3_preds_no_inpaint", scene_name)
    export_dir = os.path.join(export_root_dir, scene_name, f"da3_inpaint")
    
    run_da3_video(model, video_path, fps=24, mask_dir=mask_dir,
        export_dir=export_dir,
        export_format="depth",
        auto_cleanup=True,
        )
exit()


# Add All Tasks
# ckpt = ("dynpose-0012", "b97590f5-f42f-4a2c-9fc7-620b25a41488")
ckpt = None
# dynpose-0058/35315b4f-67a3-4107-923c-200d56bd3e68
is_ckpt_reached = False
tasks = []
# for split in sorted(os.listdir(TARGET_PATH)):
for split in splits:
    data_dir = os.path.join(TARGET_PATH, split)
    for scene_name in sorted(os.listdir(data_dir)):
        if ckpt is None or is_ckpt_reached:
            tasks.append((split, scene_name))
        elif split == ckpt[0] and scene_name == ckpt[1]:
            is_ckpt_reached = True
            tasks.append((split, scene_name))
    
    print("Task Registered:", len(tasks))


for task in tqdm(tasks):
    split, scene_name = task
    scene_dir = os.path.join(TARGET_PATH, split, scene_name)
    export_dir = os.path.join(EXPORT_ROOT_PATH, split, scene_name, "da3/inpainted")
    # original_video_path = os.path.join(scene_dir, "video_input.mp4")
    inpainted_video_path = os.path.join(scene_dir, "inpaint_result.mp4")
    mask_dir = os.path.join(scene_dir, "mask")
    
    try:
        run_da3_video(model, inpainted_video_path, fps=10, mask_dir=mask_dir,
            export_dir=export_dir,
            export_format="npz",
            auto_cleanup=True,
            )
        # os.symlink(export_dir, os.path.join(scene_dir, "da3/inpainted"))
        with open("da3_ckpt.txt", "w") as f:
            f.write(f"{split}/{scene_name}")
    except Exception as e:
        with open("errors.txt", "a+") as f:
            f.write(f"{split}/{scene_name} {e}\n")