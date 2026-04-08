export MODEL_DIR=./models/DA3NESTED-GIANT-LARGE
# This can be a Hugging Face repository or a local directory
# If you encounter network issues, consider using the following mirror: export HF_ENDPOINT=https://hf-mirror.com
# Alternatively, you can download the model directly from Hugging Face
export GALLERY_DIR=workspace/gallery
mkdir -p $GALLERY_DIR

# # CLI auto mode with backend reuse
# da3 backend --model-dir ${MODEL_DIR} --gallery-dir ${GALLERY_DIR} # Cache model to gpu
# da3 auto assets/examples/SOH \
#     --export-format glb \
#     --export-dir ${GALLERY_DIR}/TEST_BACKEND/SOH \
#     --use-backend

# CLI video processing with feature visualization
# da3 video assets/examples/robot_unitree.mp4 \
# da3 video assets/videos/parkour_1.mp4 \
#     --fps 15 \
#     --use-backend \
#     --export-dir ${GALLERY_DIR}/TEST_BACKEND/parkour_1 \
#     --export-format glb-feat_vis \
#     --feat-vis-fps 15 \
#     --process-res-method lower_bound_resize \
#     --export-feat "11,21,31"

# CLI auto mode without backend reuse
da3 auto workspace/gallery/TEST_BACKEND/parkour_1/input_images \
    --export-format npz-glb-gs_ply \
    --export-dir ${GALLERY_DIR}/TEST_CLI/parkour_1 \
    --model-dir ${MODEL_DIR}