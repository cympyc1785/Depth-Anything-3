import os
import numpy as np
import argparse
from pathlib import Path
from depth_anything_3.specs import Prediction
import imageio
import trimesh

from .colmap import export_to_colmap
from .ply import export_to_ply
from .common import depths_to_world_points_with_colors_torch, _filter_and_downsample

def _as_homogeneous44(ext: np.ndarray) -> np.ndarray:
    """
    Accept (4,4) or (3,4) extrinsic parameters, return (4,4) homogeneous matrix.
    """
    if ext.shape == (4, 4):
        return ext
    if ext.shape == (3, 4):
        H = np.eye(4, dtype=ext.dtype)
        H[:3, :4] = ext
        return H
    raise ValueError(f"extrinsic must be (4,4) or (3,4), got {ext.shape}")

def load_prediction(npz_path, load_image=True):
    if not os.path.exists(npz_path):
        raise ValueError("Path Not Found", npz_path)
    
    data = np.load(npz_path)

    prediction = Prediction(
        depth=data["depth"],
        conf=data["conf"],
        extrinsics=data["extrinsics"],
        intrinsics=data["intrinsics"],
        is_metric=0,
    )

    if load_image:
        if "image" not in data:
            export_dir = str(Path(npz_path).parents[2])
            images_dir = os.path.join(export_dir, "colmap/images")
            images = []
            for imgname in sorted(os.listdir(images_dir)):
                img_path = os.path.join(images_dir, imgname)
                img = imageio.imread(img_path)
                images.append(img)
            images = np.stack(images)
        else:
            images = data["image"]

        prediction.processed_images = images
    

    return prediction

def get_ply_from_npz(
    npz_path: str,
    num_max_points: int = 1_000_000,
    conf_thresh_percentile: float = 40.0,
    verbose: bool = True,
):
    prediction = load_prediction(npz_path)

    assert (
        prediction.processed_images is not None
    ), "Export to GLB: prediction.processed_images is required but not available"

    images_u8 = prediction.processed_images  # (N,H,W,3) uint8

    conf_thr = np.percentile(prediction.conf, conf_thresh_percentile)

    points, colors = depths_to_world_points_with_colors_torch(
        prediction.depth,
        prediction.intrinsics,
        prediction.extrinsics,  # w2c
        images_u8,
        prediction.conf,
        conf_thr,
        alignment_transform=True,
    )

    points, colors = _filter_and_downsample(points, colors, num_max_points)

    if points.shape[0] > 0:
        pc = trimesh.points.PointCloud(vertices=points, colors=colors)
    else:
        print("\n\npoint cloud is empty!\n\n")
        return None

    return pc

def export_from_npz_only_camera(npz_path, export_dir):
    prediction = load_prediction(npz_path, load_image=False)

    os.makedirs(export_dir, exist_ok=True)

    w2c_ext = np.zeros((len(prediction.extrinsics), 4, 4), dtype=np.float32)
    for i in range(len(prediction.extrinsics)):
        w2c_ext[i] = _as_homogeneous44(prediction.extrinsics[i])

    np.savez(os.path.join(export_dir, "camera_params.npz"),
             extrinsics=prediction.extrinsics,
             intrinsics=prediction.intrinsics)

def export_from_npz(npz_path, export_dir, verbose=True, device="cuda", conf_thresh_percentile=40.0):
    prediction = load_prediction(npz_path)

    images_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(npz_path))), "input_images")

    img_paths = []
    for img_name in sorted(os.listdir(images_dir)):
        dir = os.path.join(images_dir, img_name)
        img_paths.append(dir)

    # export_to_colmap(
    #     prediction=prediction,
    #     export_dir=export_dir,
    #     image_paths=img_paths,
    #     num_max_points=100_000,
    #     verbose=verbose,
    # )

    export_to_ply(
        prediction=prediction,
        export_dir=export_dir,
        export_depth_vis=False,
        verbose=verbose,
        conf_thresh_percentile=conf_thresh_percentile,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_path", type=str)
    args = parser.parse_args()

    export_dir = str(Path(args.npz_path).parents[2])

    # export_from_npz(args.npz_path, export_dir)

    get_ply_from_npz(args.npz_path)



