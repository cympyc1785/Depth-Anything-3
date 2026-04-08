import os
import queue
import threading
import concurrent.futures
from concurrent.futures.thread import ThreadPoolExecutor
import torch
import numpy as np
import cv2
import trimesh


class AsyncTensorSaver:
    def __init__(self, maxsize: int = 16):
        self.queue = queue.Queue(maxsize=maxsize)
        self.thread = threading.Thread(target=self._save_tensor_from_queue)
        self.thread.start()

    def _save_tensor_from_queue(self):
        while True:
            t = self.queue.get()
            if t is None:
                break
            tensor, path = t
            self._sync_save_tensor_to_file(tensor, path)

    def save(self, tensor, path):
        self.queue.put((tensor.cpu(), path))

    def stop(self):
        self.queue.put(None)
        self.thread.join()

    @staticmethod
    def _sync_save_tensor_to_file(tensor, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(tensor, path + ".tmp")
        os.rename(path + ".tmp", path)


class AsyncNDArraySaver:
    def __init__(self, maxsize: int = 16):
        self.queue = queue.Queue(maxsize=maxsize)
        self.thread = threading.Thread(target=self._save_ndarray_from_queue)
        self.thread.start()

    def _save_ndarray_from_queue(self):
        while True:
            t = self.queue.get()
            if t is None:
                break
            ndarray, path = t
            self._sync_save_ndarray_to_file(ndarray, path)

    def save(self, ndarray, path):
        self.queue.put((ndarray, path))

    def stop(self):
        self.queue.put(None)
        self.thread.join()

    @staticmethod
    def _sync_save_ndarray_to_file(ndarray, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, ndarray)


class AsyncImageSaver:
    def __init__(self, maxsize=16, is_rgb: bool = False):
        self.queue = queue.Queue(maxsize=maxsize)
        self.thread = threading.Thread(target=self._save_image_from_queue)
        self.thread.start()

        self.is_rgb = is_rgb

    def save(self, image, path, processor=None):
        self.queue.put((image, path, processor))

    def _save_image_from_queue(self):
        while True:
            i = self.queue.get()
            if i is None:
                break
            image, path, func = i
            self._sync_save_image(image, path, func, self.is_rgb)

    def stop(self):
        self.queue.put(None)
        self.thread.join()

    @staticmethod
    def _sync_save_image(image, path, func, is_rgb):
        dot_index = path.rfind(".")
        ext = path[dot_index:]

        tmp_path = f"{path[:dot_index]}.tmp{ext}"

        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)

        if func is not None:
            image = func(image)
        if image.shape[-1] == 3 and is_rgb is True:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(tmp_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        os.rename(tmp_path, path)


class AsyncImageReader:
    def __init__(self, image_list: list, maxsize=16):
        self.image_list = image_list.copy()
        self.queue = queue.Queue(maxsize=maxsize)

        self.tpe = ThreadPoolExecutor(max_workers=4)

        self.is_finished = False

        self.thread = threading.Thread(target=self._read_images)
        self.thread.start()


    def get(self):
        return self.queue.get()

    def _read_images(self):
        results = self.tpe.map(self._read_image_to_queue, self.image_list)
        try:
            for _ in results:
                pass
        except concurrent.futures.CancelledError:
            pass
        self.tpe.shutdown()
        self.is_finished = True

    def _read_image_to_queue(self, image_path):
        self.queue.put((image_path, cv2.imread(image_path)))

    def stop(self):
        while self.is_finished is False:
            try:
                self.queue.get(block=False)
            except:
                pass
            self.tpe.shutdown(wait=False, cancel_futures=True)


def find_files(dir: str, extensions: list[str], as_relative_path: bool = True) -> list[str]:
    from glob import glob

    image_list = []
    for ext in extensions:
        image_list += list(glob(os.path.join(dir, "**/*.{}".format(ext)), recursive=True))
    image_list.sort()

    dir_len = len(dir)
    if as_relative_path is True:
        image_list = [i[dir_len:].lstrip("/") for i in image_list]

    return image_list

def compute_alignment_transform_first_cam_gltf_center_by_points_torch(
    ext_w2c0,         # (4,4) or (3,4) np.ndarray or torch.Tensor
    points_world,     # (P,3) np.ndarray or torch.Tensor
    *,
    device="cuda",
    dtype=torch.float64,
    median_chunk=2_000_000,   # sampling size when P is huge
    return_torch=False,
):
    # ---- to torch ----
    if not torch.is_tensor(ext_w2c0):
        ext_w2c0 = torch.as_tensor(ext_w2c0)
    if not torch.is_tensor(points_world):
        points_world = torch.as_tensor(points_world)

    ext_w2c0 = ext_w2c0.to(device=device, dtype=dtype)
    points_world = points_world.to(device=device, dtype=dtype)

    # ---- ext_w2c0 -> (4,4) ----
    if ext_w2c0.shape == (3, 4):
        pad = torch.tensor([0, 0, 0, 1], device=device, dtype=dtype).view(1, 4)
        w2c0 = torch.cat([ext_w2c0, pad], dim=0)
    elif ext_w2c0.shape == (4, 4):
        w2c0 = ext_w2c0
    else:
        raise ValueError(f"ext_w2c0 must be (4,4) or (3,4), got {tuple(ext_w2c0.shape)}")

    # ---- CV → glTF axis transform (flip Y, Z) ----
    M = torch.eye(4, device=device, dtype=dtype)
    M[1, 1] = -1.0
    M[2, 2] = -1.0

    A_no_center = M @ w2c0  # (4,4)

    # ---- compute center (median) ----
    P = points_world.shape[0]
    if P == 0:
        center = torch.zeros(3, device=device, dtype=dtype)
    else:
        if P <= median_chunk:
            pts = points_world
        else:
            idx = torch.randint(0, P, (median_chunk,), device=device)
            pts = points_world.index_select(0, idx)

        ones = torch.ones((pts.shape[0], 1), device=device, dtype=dtype)
        pts_h = torch.cat([pts, ones], dim=1)             # (m,4)
        pts_tmp = (A_no_center @ pts_h.t()).t()[:, :3]    # (m,3)

        center = pts_tmp.median(dim=0).values             # (3,)

    # ---- centering ----
    T_center = torch.eye(4, device=device, dtype=dtype)
    T_center[:3, 3] = -center

    A = T_center @ A_no_center

    if return_torch:
        return A

    # ---- return numpy ----
    return A.detach().cpu().numpy()

def to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    else:
        return x

def _filter_and_downsample(points: np.ndarray, colors: np.ndarray, num_max: int):
    if points.shape[0] == 0:
        return points, colors
    finite = np.isfinite(points).all(axis=1)
    points, colors = points[finite], colors[finite]
    if int(points.shape[0]) > num_max:
        idx = np.random.choice(points.shape[0], num_max, replace=False)
        points, colors = points[idx], colors[idx]
    return points, colors

def depths_to_world_points_with_colors_torch(
    depth,          # (N,H,W)  torch.Tensor (float32/float16 OK)
    K,              # (N,3,3)  torch.Tensor
    ext_w2c,         # (N,4,4) or (N,3,4) torch.Tensor
    images_u8,       # (N,H,W,3) torch.uint8
    conf=None,       # (N,H,W) torch.Tensor or None
    conf_thr=1.0,
    *,
    device="cuda",
    stride=1,                 # >1이면 다운샘플링
    max_points_per_frame=None,  # int or None (랜덤 샘플링)
    chunk_size=1_000_000,     # 너무 큰 M일 때 matmul을 chunk로
    return_torch=False,
    alignment_transform=False,
):
    # ---- move / normalize ----
    depth = to_torch(depth).to(device)
    K = to_torch(K).to(device)
    ext_w2c = to_torch(ext_w2c).to(device)
    images_u8 = to_torch(images_u8).to(torch.uint8).to(device)
    if conf is not None:
        conf = to_torch(conf).to(device)

    if depth.dim() != 3:
        raise ValueError(f"depth must be (N,H,W), got {tuple(depth.shape)}")
    N, H, W = depth.shape

    # ext_w2c -> (N,4,4)
    if ext_w2c.shape[-2:] == (3, 4):
        pad = torch.tensor([0, 0, 0, 1], device=device, dtype=ext_w2c.dtype).view(1, 1, 4).repeat(N, 1, 1)
        ext_w2c = torch.cat([ext_w2c, pad], dim=1)
    elif ext_w2c.shape[-2:] != (4, 4):
        raise ValueError(f"ext_w2c must be (N,4,4) or (N,3,4), got {tuple(ext_w2c.shape)}")

    # build pixel grid (downsample by stride)
    us = torch.arange(0, W, stride, device=device)
    vs = torch.arange(0, H, stride, device=device)
    vv, uu = torch.meshgrid(vs, us, indexing="ij")  # (h', w')
    ones = torch.ones_like(uu, dtype=torch.float32)

    # pix: (P,3) float32
    pix = torch.stack([uu, vv, ones], dim=-1).reshape(-1, 3).to(torch.float32)  # (P,3)
    P = pix.shape[0]

    # flatten indices mapping from downsample grid to original image
    # idx in original H*W space for color/depth gather
    flat_idx = (vv * W + uu).reshape(-1)  # (P,)

    pts_list = []
    col_list = []

    # precompute K_inv, c2w per frame
    # (safe: torch.linalg.inv on GPU)
    K_inv = torch.linalg.inv(K.to(torch.float32))  # (N,3,3)
    c2w = torch.linalg.inv(ext_w2c.to(torch.float32))  # (N,4,4)

    for i in range(N):
        d = depth[i].reshape(-1)  # (H*W,)
        # gather only downsampled pixels
        d_ds = d.index_select(0, flat_idx)  # (P,)
        valid = torch.isfinite(d_ds) & (d_ds > 0)

        if conf is not None:
            c = conf[i].reshape(-1).index_select(0, flat_idx)  # (P,)
            valid = valid & (c >= conf_thr)

        vidx = torch.nonzero(valid, as_tuple=False).squeeze(1)  # (M,)
        if vidx.numel() == 0:
            continue

        if max_points_per_frame is not None and vidx.numel() > max_points_per_frame:
            perm = torch.randperm(vidx.numel(), device=device)[:max_points_per_frame]
            vidx = vidx.index_select(0, perm)

        # rays = K_inv @ pix^T  but only for vidx
        pix_sel = pix.index_select(0, vidx)  # (M,3)

        # chunked compute to avoid huge intermediate
        M = pix_sel.shape[0]
        start = 0
        while start < M:
            end = min(start + chunk_size, M)
            pix_chunk = pix_sel[start:end]  # (m,3)
            d_chunk = d_ds.index_select(0, vidx[start:end]).to(torch.float32)  # (m,)

            rays = (K_inv[i] @ pix_chunk.t()).t()  # (m,3)
            Xc = rays * d_chunk.unsqueeze(1)       # (m,3)

            Xc_h = torch.cat([Xc, torch.ones((Xc.shape[0], 1), device=device, dtype=torch.float32)], dim=1)  # (m,4)
            Xw = (c2w[i] @ Xc_h.t()).t()[:, :3].contiguous()  # (m,3)

            # colors (uint8)
            cols = images_u8[i].reshape(-1, 3).index_select(0, flat_idx.index_select(0, vidx[start:end]))
            pts_list.append(Xw.to(torch.float32))
            col_list.append(cols.to(torch.uint8))

            start = end

    if len(pts_list) == 0:
        pts = torch.zeros((0, 3), device=device, dtype=torch.float32)
        cols = torch.zeros((0, 3), device=device, dtype=torch.uint8)
    else:
        pts = torch.cat(pts_list, dim=0)
        cols = torch.cat(col_list, dim=0)

    # if alignment_transform:
    #     A = compute_alignment_transform_first_cam_gltf_center_by_points_torch(
    #         ext_w2c[0], pts, return_torch=True
    #     )
        
    #     if return_torch:
    #         return pts, cols, A
    #     return pts.detach().cpu().numpy(), cols.detach().cpu().numpy(), A.detach().cpu().numpy()

    if return_torch:
        return pts, cols
    return pts.detach().cpu().numpy(), cols.detach().cpu().numpy()