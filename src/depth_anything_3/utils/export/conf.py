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
import imageio
import numpy as np

from depth_anything_3.specs import Prediction
from depth_anything_3.utils.visualize import visualize_conf


def export_to_conf(
    prediction: Prediction,
    export_dir: str,
):
    # Use prediction.processed_images, which is already processed image data
    if prediction.processed_images is None:
        raise ValueError("prediction.processed_images is required but not available")

    images_u8 = prediction.processed_images  # (N,H,W,3) uint8

    vis_frames = []

    os.makedirs(os.path.join(export_dir, "conf_vis"), exist_ok=True)
    for idx in range(prediction.conf.shape[0]):
        conf_vis = visualize_conf(prediction.conf[idx])
        image_vis = images_u8[idx]
        conf_vis = conf_vis.astype(np.uint8)
        image_vis = image_vis.astype(np.uint8)
        vis_image = np.concatenate([image_vis, conf_vis], axis=1)
        save_path = os.path.join(export_dir, f"conf_vis/{idx:04d}.jpg")
        imageio.imwrite(save_path, vis_image, quality=95)

        vis_frames.append(vis_image)
    
    save_path = os.path.join(export_dir, f"conf_vis/conf_vis.mp4")
    imageio.mimsave(save_path, vis_frames, fps=10)