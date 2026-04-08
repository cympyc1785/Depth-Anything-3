VIDEO_NAME="soccer_1"
ROOT_PATH="/data1/cympyc1785/SceneData/${VIDEO_NAME}/scene_recon"

da3 gradio \
    --model-dir ./models/DA3NESTED-GIANT-LARGE \
    --workspace-dir $ROOT_PATH \
    --gallery-dir $ROOT_PATH \
    --cache-examples