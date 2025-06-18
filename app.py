#
# app.py
#

import subprocess
from pathlib import Path
import modal

# Define the Modal resources
app = modal.App(name="comfyui-app")
volume = modal.Volume.from_name("comfyui-models-vol") # We are using the volume we created earlier

# Define the environment image
# This is the most important part: we are building a container with everything ComfyUI needs.
comfyui_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .run_commands(
        "cd /root && git clone https://github.com/comfyanonymous/ComfyUI.git",
        "cd /root/ComfyUI && pip install -r requirements.txt",
        # Install OpenCV headless for comfyui_controlnet_aux compatibility
        "pip install opencv-python-headless",
        # --- Install Custom Nodes Here ---
        # Use `git clone` to add any custom nodes you want.
        # Make sure they go into the `custom_nodes` directory.
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/ltdrdata/ComfyUI-Manager.git",
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git",
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git",
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/Acly/comfyui-inpaint-nodes.git",
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/Acly/comfyui-tooling-nodes.git",
        # Ensure the models directory is empty to allow volume mounting
        "rm -rf /root/ComfyUI/models/*",
    )
)

@app.function(
    image=comfyui_image,
    gpu="any",  # Request a GPU. "any" is fine, or specify one like "L40S" for more power.
    volumes={"/root/ComfyUI/models": volume}, # Mount the models volume to the correct path
    scaledown_window=300, # Keep the container alive for 5 minutes after last request
    max_containers=10, # Updated from concurrency_limit for maximum concurrent containers
)
@modal.web_server(8188, startup_timeout=300) # This exposes port 8188 (ComfyUI's default) to the web, with a 5-minute startup timeout
def run_comfyui():
    # This command launches the ComfyUI server.
    # The `--listen` argument tells it to be accessible from the network, not just localhost.
    cmd = "cd /root/ComfyUI && python main.py --listen 0.0.0.0"
    subprocess.Popen(cmd, shell=True)
