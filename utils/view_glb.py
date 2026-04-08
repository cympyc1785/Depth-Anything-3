import gradio as gr
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--video-name", type=str, default="parkour_1", help="Name of the video.")
args = parser.parse_args()

video_name = args.video_name
root_path = f"/data1/cympyc1785/Depth-Anything-3/SceneData/{video_name}/scene_recon"

def show_glb():
    return f"{root_path}/scene.glb"

demo = gr.Interface(
    fn=show_glb,
    inputs=None,
    outputs=gr.Model3D(label="3D Viewer")
)

demo.launch(allowed_paths=[f"{root_path}"])