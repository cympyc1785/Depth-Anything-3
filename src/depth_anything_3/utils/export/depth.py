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
import numpy as np

from depth_anything_3.specs import Prediction
from depth_anything_3.utils.parallel_utils import async_call


# @async_call
def export_to_depth(
    prediction: Prediction,
    export_dir: str,
):
    output_file = os.path.join(export_dir, "depth.npy")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    np.save(output_file, np.round(prediction.depth, 6))

    w2c_ext = np.zeros((len(prediction.extrinsics), 4, 4), dtype=np.float32)
    for i in range(len(prediction.extrinsics)):
        w2c_ext[i] = _as_homogeneous44(prediction.extrinsics[i])
    
    cam_out_path = os.path.join(export_dir, "camera_params.npz")
    np.savez(cam_out_path,
             extrinsics=w2c_ext,
             intrinsics=prediction.intrinsics)

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