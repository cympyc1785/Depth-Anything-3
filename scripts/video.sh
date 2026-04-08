# This can be a Hugging Face repository or a local directory
# If you encounter network issues, consider using the following mirror: export HF_ENDPOINT=https://hf-mirror.com
# Alternatively, you can download the model directly from Hugging Face
export MODEL_DIR=./models/DA3NESTED-GIANT-LARGE
export FPS=10

# DAVIS/blackswan
# DAVIS/bike-packing
# youtube_vis/0ae1ff65a5
export ROOT_DIR="/data1/cympyc1785/SceneData/DynamicVerse/scenes/DAVIS/blackswan"
export SAVE_DIR="${ROOT_DIR}/da3"
mkdir -p $SAVE_DIR

# # CLI video processing with feature visualization
# da3 video "${ROOT_DIR}/video.mp4" \
#     --fps ${FPS} \
#     --model-dir ${MODEL_DIR} \
#     --export-dir ${SAVE_DIR} \
#     --export-format npz-ply-colmap-conf-sky \
#     --conf-thresh-percentile 40.0 \
#     --auto-cleanup
#     # --process-res-method lower_bound_resize \

# da3 video "$ROOT_DIR/inpaint_result.mp4" \
#     --fps 10 \
#     --model-dir ${MODEL_DIR} \
#     --export-dir ${ROOT_DIR}/da3/inpainted_test \
#     --export-format npz \
#     --conf-thresh-percentile 40.0 \
#     --mask-dir ${ROOT_DIR}/mask \
#     --auto-cleanup

ROOT_DIR=/data1/cympyc1785/3d_recon/Depth-Anything-3/SceneData/DynamicVerse/scenes/DAVIS/bike-packing
da3 video "$ROOT_DIR/noisy_render_0_49_mask.mp4" \
    --fps 10 \
    --model-dir ${MODEL_DIR} \
    --export-dir ${ROOT_DIR}/rgbd \
    --export-format depth_video \
    --conf-thresh-percentile 40.0 \
    --mask-dir ${ROOT_DIR}/mask \
    --auto-cleanup