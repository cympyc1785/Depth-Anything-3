import os
import time
from pathlib import Path
from depth_anything_3.utils.export.export_from_npz import export_from_npz, get_ply_from_npz




npz_path = "/data1/cympyc1785/3d_recon/Depth-Anything-3/SceneData/DynamicVerse/scenes/MOSE/32dc49d4/da3/inpainted/exports/mini_npz/results.npz"
export_dir = str(Path(npz_path).parents[2])

start_time = time.time()
result = get_ply_from_npz(npz_path)

print(time.time() - start_time)
print(result)