import os
import numpy as np

from depth_anything_3.specs import Prediction
from PIL import Image


def export_images(
    prediction: Prediction,
    export_dir: str,
):
    # Use prediction.processed_images, which is already processed image data
    if prediction.processed_images is None:
        raise ValueError("prediction.processed_images is required but not available")

    images = prediction.processed_images  # (N,H,W,3) uint8
    
    imgs_folder = os.path.join(export_dir, "images")
    os.makedirs(imgs_folder, exist_ok=True)

    for i, image in enumerate(images):
        output_path = os.path.join(imgs_folder, f"{i:05d}.png")
        Image.fromarray(image).save(output_path)
