#
# app.py
#

import subprocess
from pathlib import Path
import modal

# Define the Modal resources
app = modal.App(name="comfyui-app")
volume = modal.Volume.from_name("comfyui-models-vol") # We are using the volume we created earlier
custom_nodes_volume = modal.Volume.from_name("comfyui-custom-nodes-vol") # Volume for custom nodes

# Define the environment image
# This is the most important part: we are building a container with everything ComfyUI needs.
comfyui_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .run_commands(
        "cd /root && git clone https://github.com/comfyanonymous/ComfyUI.git",
        "cd /root/ComfyUI && pip install -r requirements.txt",
        # Ensure the models and custom_nodes directories are empty to allow volume mounting
        "rm -rf /root/ComfyUI/models/*",
        "rm -rf /root/ComfyUI/custom_nodes/*",
    )
)

@app.function(
    image=comfyui_image,
    gpu="any",  # Request a GPU. "any" is fine, or specify one like "L40S" for more power.
    volumes={"/root/ComfyUI/models": volume, "/root/ComfyUI/custom_nodes": custom_nodes_volume}, # Mount the volumes to the correct paths
    scaledown_window=300, # Keep the container alive for 5 minutes after last request
    max_containers=10, # Updated from concurrency_limit for maximum concurrent containers
)
@modal.web_server(8188, startup_timeout=300) # This exposes port 8188 (ComfyUI's default) to the web, with a 5-minute startup timeout
def run_comfyui():
    # This command launches the ComfyUI server.
    # The `--listen` argument tells it to be accessible from the network, not just localhost.
    cmd = "cd /root/ComfyUI && python main.py --listen 0.0.0.0"
    subprocess.Popen(cmd, shell=True)
