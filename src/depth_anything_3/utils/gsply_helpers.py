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
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from einops import rearrange, repeat
from plyfile import PlyData, PlyElement
from torch import Tensor

from depth_anything_3.specs import Gaussians


def construct_list_of_attributes(num_rest: int) -> list[str]:
    attributes = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(3):
        attributes.append(f"f_dc_{i}")
    for i in range(num_rest):
        attributes.append(f"f_rest_{i}")
    attributes.append("opacity")
    for i in range(3):
        attributes.append(f"scale_{i}")
    for i in range(4):
        attributes.append(f"rot_{i}")
    return attributes

def unproject_from_depth(depths, extrinsics, intrinsics):
    if depths.ndim == 4:
        depths = depths.squeeze(-1)  # [B,H,W]

    B, H, W = depths.shape
    device = depths.device
    dtype = depths.dtype

    # pixel grid
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij"
    )  # [H,W]

    xs = xs.unsqueeze(0).expand(B, -1, -1)  # [B,H,W]
    ys = ys.unsqueeze(0).expand(B, -1, -1)

    fx = intrinsics[:, 0, 0].unsqueeze(-1).unsqueeze(-1)
    fy = intrinsics[:, 1, 1].unsqueeze(-1).unsqueeze(-1)
    cx = intrinsics[:, 0, 2].unsqueeze(-1).unsqueeze(-1)
    cy = intrinsics[:, 1, 2].unsqueeze(-1).unsqueeze(-1)

    z = depths
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy

    xyz_cam = torch.stack([x, y, z], dim=-1)  # [B,H,W,3]

    # to homogeneous
    ones = torch.ones((B, H, W, 1), device=device, dtype=dtype)
    xyz_cam_h = torch.cat([xyz_cam, ones], dim=-1)  # [B,H,W,4]

    # camera -> world
    R_w2c = extrinsics[:, :3, :3]
    t_w2c = extrinsics[:, :3, 3]

    R_c2w = R_w2c.transpose(-1, -2)
    t_c2w = (-R_c2w @ t_w2c.unsqueeze(-1)).squeeze(-1)
    
    c2w = torch.eye(4, device=extrinsics.device, dtype=extrinsics.dtype).unsqueeze(0).repeat(B, 1, 1)  # [B,4,4]
    c2w[:, :3, :3] = R_c2w
    c2w[:, :3, 3]  = t_c2w

    xyz_world = torch.einsum(
        "bij,bhwj->bhwi",
        c2w,
        xyz_cam_h
    )  # [B,H,W,4]

    return xyz_world[..., :3]

def export_ply(
    means: Tensor,  # "gaussian 3"
    scales: Tensor,  # "gaussian 3"
    rotations: Tensor,  # "gaussian 4"
    harmonics: Tensor,  # "gaussian 3 d_sh"
    opacities: Tensor,  # "gaussian"
    # offsets: Tensor,
    path: Path,
    shift_and_scale: bool = False,
    save_sh_dc_only: bool = True,
    match_3dgs_mcmc_dev: Optional[bool] = False,
):
    if shift_and_scale:
        # Shift the scene so that the median Gaussian is at the origin.
        means = means - means.median(dim=0).values

        # Rescale the scene so that most Gaussians are within range [-1, 1].
        scale_factor = means.abs().quantile(0.95, dim=0).max()
        means = means / scale_factor
        scales = scales / scale_factor
        # offsets = offsets / scale_factor

    rotations = rotations.detach().cpu().numpy()

    # Since current model use SH_degree = 4,
    # which require large memory to store, we can only save the DC band to save memory.
    f_dc = harmonics[..., 0]
    f_rest = harmonics[..., 1:].flatten(start_dim=1)

    if match_3dgs_mcmc_dev:
        sh_degree = 3
        n_rest = 3 * (sh_degree + 1) ** 2 - 3
        f_rest = repeat(
            torch.zeros_like(harmonics[..., :1]), "... i -> ... (n i)", n=(n_rest // 3)
        ).flatten(start_dim=1)
        dtype_full = [
            (attribute, "f4")
            for attribute in construct_list_of_attributes(num_rest=n_rest)
            if attribute not in ("nx", "ny", "nz")
        ]
    else:
        dtype_full = [
            (attribute, "f4")
            for attribute in construct_list_of_attributes(
                0 if save_sh_dc_only else f_rest.shape[1]
            )
        ]
    elements = np.empty(means.shape[0], dtype=dtype_full)
    attributes = [
        means.detach().cpu().numpy(),
        torch.zeros_like(means).detach().cpu().numpy(),
        f_dc.detach().cpu().contiguous().numpy(),
        f_rest.detach().cpu().contiguous().numpy(),
        opacities[..., None].detach().cpu().numpy(),
        scales.log().detach().cpu().numpy(),
        rotations,
        # offsets.detach().cpu().numpy(),
    ]
    if match_3dgs_mcmc_dev:
        attributes.pop(1)  # dummy normal is not needed
    elif save_sh_dc_only:
        attributes.pop(3)  # remove f_rest from attributes

    attributes = np.concatenate(attributes, axis=1)
    elements[:] = list(map(tuple, attributes))
    path.parent.mkdir(exist_ok=True, parents=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)

def load_ply(path: Path):
    """
    Loader matched EXACTLY to your export_ply()/save_gaussian_ply().

    - scales: stored as log(scales) -> returned as linear scales (exp)
    - rotations: stored as rot_0..rot_3 in the SAME order as written -> returned unchanged order
    - opacities: stored as whatever you passed (in your case: inverse_sigmoid(opacity), i.e. logits)
                 -> returned as logits (DO NOT sigmoid here)
    - harmonics: if DC-only -> (N,3,1) with DC at [:,:,0]
    """
    ply = PlyData.read(str(path))
    if "vertex" not in ply:
        raise ValueError(f"PLY has no 'vertex' element: {path}")

    v = ply["vertex"].data
    names = v.dtype.names or ()

    def req(*cols):
        missing = [c for c in cols if c not in names]
        if missing:
            raise ValueError(f"Missing columns in PLY: {missing}. Available: {sorted(names)}")

    # means
    req("x", "y", "z")
    means = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

    # scales: stored as log(scales)
    req("scale_0", "scale_1", "scale_2")
    scales_log = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1).astype(np.float32)
    scales = np.exp(scales_log).astype(np.float32)

    # rotations: EXACT column order as written by export_ply()
    # (Your export uses construct_list_of_attributes(), which for 3DGS is typically rot_0..rot_3)
    req("rot_0", "rot_1", "rot_2", "rot_3")
    rotations = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype(np.float32)

    # opacity: EXACT as stored (in your pipeline, this is logits if inv_opacity=True at save time)
    if "opacity" in names:
        opacities = np.asarray(v["opacity"], dtype=np.float32)
    elif "alpha" in names:
        opacities = np.asarray(v["alpha"], dtype=np.float32)
    else:
        raise ValueError(f"Missing opacity column (expected 'opacity' or 'alpha'). Available: {sorted(names)}")

    # SH DC
    req("f_dc_0", "f_dc_1", "f_dc_2")
    f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32)

    # Optional SH rest
    rest_cols = [c for c in names if c.startswith("f_rest_")]
    # numeric sort f_rest_0, f_rest_1, ...
    rest_cols = sorted(rest_cols, key=lambda s: int(s.split("_")[-1]))

    if len(rest_cols) == 0:
        harmonics = f_dc[:, :, None].astype(np.float32)  # (N,3,1)
    else:
        f_rest = np.stack([v[c] for c in rest_cols], axis=1).astype(np.float32)  # (N,M)
        if f_rest.shape[1] % 3 != 0:
            raise ValueError(f"f_rest dim must be multiple of 3, got {f_rest.shape[1]}")
        K = 1 + (f_rest.shape[1] // 3)
        harmonics = np.zeros((means.shape[0], 3, K), dtype=np.float32)
        harmonics[:, :, 0] = f_dc
        harmonics[:, :, 1:] = f_rest.reshape(means.shape[0], -1, 3).transpose(0, 2, 1)

    return {
        "means": torch.from_numpy(means),
        "scales": torch.from_numpy(scales),
        "rotations": torch.from_numpy(rotations),
        "opacities": torch.from_numpy(opacities),   # logits 그대로
        "harmonics": torch.from_numpy(harmonics),
        "raw": v,
    }



def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def save_gaussian_ply(
    gaussians: Gaussians,
    save_path: str,
    ctx_depth: torch.Tensor,  # depth of input views; for getting shape and filtering, "v h w 1"
    shift_and_scale: bool = False,
    save_sh_dc_only: bool = True,
    gs_views_interval: int = 1,
    inv_opacity: Optional[bool] = True,
    prune_by_depth_percent: Optional[float] = 1.0,
    prune_border_gs: Optional[bool] = True,
    match_3dgs_mcmc_dev: Optional[bool] = False,
):
    b = gaussians.means.shape[0]
    assert b == 1, "must set batch_size=1 when exporting 3D gaussians"
    src_v, out_h, out_w, _ = ctx_depth.shape

    # extract gs params
    world_means = gaussians.means
    world_shs = gaussians.harmonics
    world_rotations = gaussians.rotations
    gs_scales = gaussians.scales
    gs_opacities = inverse_sigmoid(gaussians.opacities) if inv_opacity else gaussians.opacities

    # Create a mask to filter the Gaussians.

    # TODO: prune the sky region here

    # throw away Gaussians at the borders, since they're generally of lower quality. <- view depth 기준
    if prune_border_gs:
        mask = torch.zeros_like(ctx_depth, dtype=torch.bool)
        gstrim_h = int(8 / 256 * out_h)
        gstrim_w = int(8 / 256 * out_w)
        mask[:, gstrim_h:-gstrim_h, gstrim_w:-gstrim_w, :] = 1
    else:
        mask = torch.ones_like(ctx_depth, dtype=torch.bool)

    # trim the far away point based on depth;
    if prune_by_depth_percent is not None and prune_by_depth_percent < 1:
        in_depths = ctx_depth
        d_percentile = torch.quantile(
            in_depths.view(in_depths.shape[0], -1), q=prune_by_depth_percent, dim=1
        ).view(-1, 1, 1)
        d_mask = (in_depths[..., 0] <= d_percentile).unsqueeze(-1)
        mask = mask & d_mask

    # # --- scale based pruning ---
    # prune_by_scale_percent = 0.99
    # scales_are_log = False
    # if prune_by_scale_percent is not None and prune_by_scale_percent < 1:
    #     scales = gaussians.scales  # [1, N, 3] or [N,3]

    #     if scales.ndim == 3:
    #         scales = scales[0]     # -> [N,3]

    #     scales_lin = scales.exp() if scales_are_log else scales  # [N,3]

    #     scales_vhw = rearrange(
    #         scales_lin, "(v h w) c -> v h w c", v=src_v, h=out_h, w=out_w
    #     )  # [V,H,W,3]

    #     scale_metric = scales_vhw.max(dim=-1).values  # [V,H,W]

    #     thr = torch.quantile(
    #         scale_metric.view(src_v, -1), q=prune_by_scale_percent, dim=1
    #     ).view(-1, 1, 1)  # [V,1,1]

    #     s_mask = (scale_metric <= thr).unsqueeze(-1)  # [V,H,W,1]
    #     mask = mask & s_mask
    
    
    mask = mask.squeeze(-1)  # v h w

    # helper fn, must place after mask
    def trim_select_reshape(element):
        selected_element = rearrange(
            element[0], "(v h w) ... -> v h w ...", v=src_v, h=out_h, w=out_w
        )
        selected_element = selected_element[::gs_views_interval][mask[::gs_views_interval]]
        return selected_element

    export_ply(
        means=trim_select_reshape(world_means),
        scales=trim_select_reshape(gs_scales),
        rotations=trim_select_reshape(world_rotations),
        harmonics=trim_select_reshape(world_shs),
        opacities=trim_select_reshape(gs_opacities),
        # offsets=trim_select_reshape(gaussians.offsets),
        path=Path(save_path),
        shift_and_scale=shift_and_scale,
        save_sh_dc_only=save_sh_dc_only,
        match_3dgs_mcmc_dev=match_3dgs_mcmc_dev,
    )
