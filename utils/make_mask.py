import cv2
import os
import numpy as np
from PIL import Image

def _resize_longest_side(img: Image.Image, target_size: int) -> Image.Image:
        w, h = img.size
        longest = max(w, h)
        if longest == target_size:
            return img
        scale = target_size / float(longest)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        arr = cv2.resize(np.asarray(img), (new_w, new_h), interpolation=interpolation)
        return Image.fromarray(arr)

def resize_mask(mask, ref_img):
    h, w = ref_img.shape[:2]
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask_resized

def make_mask(ref_img):
    h, w = ref_img.shape[:2]
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    mask[: h // 2, : w // 2, :] = (0, 188, 188)
    return mask

def export_mask(imgs_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(imgs_dir)):
        img_path = os.path.join(imgs_dir, name)
        img = Image.open(img_path).convert("RGB")
        mask = _resize_longest_side(img, target_size=504)
        mask.save(os.path.join(out_dir, name))

root_dir = "/data1/cympyc1785/SceneData/sintel/dataset"
mask_root_dir = os.path.join(root_dir, "mask")

for scene_name in tqdm(sorted(os.listdir(mask_root_dir))):
    imgs_dir = os.path.join(mask_root_dir, scene_name)
    out_dir = os.path.join(root_dir, "resized_mask", scene_name)
    export_mask(imgs_dir, out_dir)