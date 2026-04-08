cd /data1/cympyc1785/STTN

ROOT_PATH=/data1/cympyc1785/SceneData/DynamicVerse/scenes/DAVIS
SCENE_NAME=blackswan

# python infer.py \
# --images_dir "${ROOT_PATH}/${SCENE_NAME}/rgb" \
# --mask "${ROOT_PATH}/${SCENE_NAME}/mask"  \
# --ckpt checkpoints/sttn.pth

python infer_original.py \
--video "${ROOT_PATH}/${SCENE_NAME}/video.mp4" \
--mask "${ROOT_PATH}/${SCENE_NAME}/mask"  \
--ckpt checkpoints/sttn.pth