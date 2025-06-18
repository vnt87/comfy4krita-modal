#
# download_models.py (Updated for complete SDXL setup)
#

import os
from pathlib import Path

import modal
# Import requests and shutil only when running on Modal to avoid local import errors
try:
    import requests
    import shutil
except ImportError:
    requests = None
    shutil = None

# --- Configuration for a Complete SDXL Setup and Additional Models ---
MODELS = {
    # 1. SDXL Base Model
    "sdxl-base-1.0": {
        "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "filenames": ["sd_xl_base_1.0.safetensors"],
        "target_dir": "checkpoints"
    },

    # 2. SDXL Refiner Model
    "sdxl-refiner-1.0": {
        "repo_id": "stabilityai/stable-diffusion-xl-refiner-1.0",
        "filenames": ["sd_xl_refiner_1.0.safetensors"],
        "target_dir": "checkpoints"
    },

    # 3. SDXL VAE (Improves color and detail)
    "sdxl-vae": {
        "repo_id": "stabilityai/sdxl-vae",
        "filenames": ["sdxl_vae.safetensors"],
        "target_dir": "vae"
    },

    # 4. OpenCLIP ViT-bigG-14 (Primary Text Encoder)
    "clip-vit-bigg-14": {
        "repo_id": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        "filenames": ["open_clip_pytorch_model.bin"],
        "target_dir": "clip"
    },

    # 5. CLIP ViT-L/14 (Secondary Text Encoder)
    "clip-vit-l-14": {
        "repo_id": "openai/clip-vit-large-patch14",
        "filenames": ["pytorch_model.bin"],
        "target_dir": "clip"
    },

    # --- Additional Models by Category ---

    # Clip Vision Models
    "ip-adapter-image-encoder": {
        "repo_id": "h94/IP-Adapter",
        "filenames": ["models/image_encoder/model.safetensors"],
        "target_dir": "clip_vision"
    },
    "sigclip-vision-384": {
        "repo_id": "Comfy-Org/sigclip_vision_384",
        "filenames": ["sigclip_vision_patch14_384.safetensors"],
        "target_dir": "clip_vision"
    },

    # Upscale Models
    "nmkd-superscale-sp": {
        "repo_id": "gemasai/4x_NMKD-Superscale-SP_178000_G",
        "filenames": ["4x_NMKD-Superscale-SP_178000_G.pth"],
        "target_dir": "upscale_models"
    },
    "omnisr-x2": {
        "repo_id": "Acly/Omni-SR",
        "filenames": ["OmniSR_X2_DIV2K.safetensors"],
        "target_dir": "upscale_models"
    },
    "omnisr-x3": {
        "repo_id": "Acly/Omni-SR",
        "filenames": ["OmniSR_X3_DIV2K.safetensors"],
        "target_dir": "upscale_models"
    },
    "omnisr-x4": {
        "repo_id": "Acly/Omni-SR",
        "filenames": ["OmniSR_X4_DIV2K.safetensors"],
        "target_dir": "upscale_models"
    },
    "hat-srx4": {
        "repo_id": "Acly/hat",
        "filenames": ["HAT_SRx4_ImageNet-pretrain.pth"],
        "target_dir": "upscale_models"
    },
    "real-hat-gan-sharper": {
        "repo_id": "Acly/hat",
        "filenames": ["Real_HAT_GAN_sharper.pth"],
        "target_dir": "upscale_models"
    },

    # IP Adapter Models
    "ip-adapter-sdxl-vit-h": {
        "repo_id": "h94/IP-Adapter",
        "filenames": ["sdxl_models/ip-adapter_sdxl_vit-h.safetensors"],
        "target_dir": "ipadapter"
    },

    # LoRA Models
    "hyper-sdxl-8steps-cfg": {
        "repo_id": "ByteDance/Hyper-SD",
        "filenames": ["Hyper-SDXL-8steps-CFG-lora.safetensors"],
        "target_dir": "loras"
    },
    "ip-adapter-faceid-plusv2-sdxl": {
        "repo_id": "h94/IP-Adapter-FaceID",
        "filenames": ["ip-adapter-faceid-plusv2_sdxl.bin"],
        "target_dir": "loras"
    },

    # Inpaint Models
    "fooocus-inpaint-head": {
        "repo_id": "lllyasviel/fooocus_inpaint",
        "filenames": ["fooocus_inpaint_head.pth"],
        "target_dir": "inpaint"
    },
    "fooocus-inpaint-patch": {
        "repo_id": "lllyasviel/fooocus_inpaint",
        "filenames": ["inpaint_v26.fooocus.patch"],
        "target_dir": "inpaint"
    },
    "mat-places512": {
        "repo_id": "Acly/MAT",
        "filenames": ["MAT_Places512_G_fp16.safetensors"],
        "target_dir": "inpaint"
    },

    # ControlNet Models
    "controlnet-union-sdxl-promax": {
        "repo_id": "xinsir/controlnet-union-sdxl-1.0",
        "filenames": ["diffusion_pytorch_model_promax.safetensors"],
        "target_dir": "controlnet"
    },
    "controlnet-qrcode-monster": {
        "repo_id": "monster-labs/control_v1p_sdxl_qrcode_monster",
        "filenames": ["diffusion_pytorch_model.safetensors"],
        "target_dir": "controlnet"
    },
    "flux1-dev-controlnet-union-pro": {
        "repo_id": "ABDALLALSWAITI/FLUX.1-dev-ControlNet-Union-Pro-2.0-fp8",
        "filenames": ["diffusion_pytorch_model.safetensors"],
        "target_dir": "controlnet"
    },
    "flux1-dev-controlnet-inpainting-beta": {
        "repo_id": "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta",
        "filenames": ["diffusion_pytorch_model.safetensors"],
        "target_dir": "controlnet"
    },
    "mistoline-flux-dev": {
        "repo_id": "TheMistoAI/MistoLine_Flux.dev",
        "filenames": ["mistoline_flux.dev_v1.safetensors"],
        "target_dir": "controlnet"
    },

    # Style Models
    "flux1-redux-dev": {
        "repo_id": "files.interstice.cloud/models",
        "filenames": ["flux1-redux-dev.safetensors"],
        "target_dir": "style_models",
        "direct_url": "https://files.interstice.cloud/models/flux1-redux-dev.safetensors"
    },

    # Checkpoints
    "realvisxl-v5.0": {
        "repo_id": "SG161222/RealVisXL_V5.0",
        "filenames": ["RealVisXL_V5.0_fp16.safetensors"],
        "target_dir": "checkpoints"
    },
    "flux1-dev-fp8": {
        "repo_id": "Comfy-Org/flux1-dev",
        "filenames": ["flux1-dev-fp8.safetensors"],
        "target_dir": "checkpoints"
    },
    "zavychromaxl-v80": {
        "repo_id": "misri/zavychromaxl_v80",
        "filenames": ["zavychromaxl_v80.safetensors"],
        "target_dir": "checkpoints"
    },
    "dreamshaper-xl-v2-turbo": {
        "repo_id": "Lykon/dreamshaper-xl-v2-turbo",
        "filenames": ["DreamShaperXL_Turbo_v2.safetensors"],
        "target_dir": "checkpoints"
    },
}
# --- End Configuration ---

# Define the Modal resources
app = modal.App(name="comfyui-model-downloader")
volume = modal.Volume.from_name("comfyui-models-vol", create_if_missing=True)

# This is the image we'll use to run the download function.
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "huggingface_hub",
    "requests"
)

# This is the function that will be executed on Modal.
@app.function(
    image=image,
    volumes={"/models": volume},
    timeout=1800  # 30 minutes, downloading can be slow
)
def download_all_models():
    from huggingface_hub import hf_hub_download
    
    # Ensure requests and shutil are available when running on Modal
    try:
        import requests
        import shutil
    except ImportError as e:
        print(f"Error importing requests or shutil: {e}. Direct URL downloads will not work.")
        requests = None
        shutil = None
    
    print("Starting model download process...")
    for model_name, model_info in MODELS.items():
        # Create the full target path inside the Volume
        target_dir_path = Path("/models") / model_info["target_dir"]
        
        for filename in model_info["filenames"]:
            # Correctly handle nested paths from Hugging Face
            source_path = Path(filename)
            # Use model_name as a prefix to ensure unique filenames
            unique_filename = f"{model_name}_{source_path.name}"
            final_target_path = target_dir_path / unique_filename
            
            final_target_path.parent.mkdir(parents=True, exist_ok=True)

            if final_target_path.exists():
                print(f"-> File {final_target_path.name} already exists. Skipping.")
                continue

            if "direct_url" in model_info and requests is not None and shutil is not None:
                print(f"   -> Downloading from direct URL for {model_name}...")
                response = requests.get(model_info["direct_url"], stream=True)
                if response.status_code == 200:
                    temp_path = Path("/tmp") / source_path.name
                    with open(temp_path, 'wb') as f:
                        shutil.copyfileobj(response.raw, f)
                    shutil.move(temp_path, final_target_path)
                    print(f"   -> Downloaded to {final_target_path}")
                else:
                    print(f"   -> Failed to download from {model_info['direct_url']}. Status code: {response.status_code}")
            else:
                print(f"   -> Downloading {model_info['repo_id']}/{filename}...")
                # Download the file to a temporary location
                temp_download_path = Path("/tmp") / source_path.name
                try:
                    hf_hub_download(
                        repo_id=model_info["repo_id"],
                        filename=filename,
                        local_dir=Path("/tmp"),
                        local_dir_use_symlinks=False, # Important for Modal Volumes
                    )
                    # Check if the file exists before moving
                    if temp_download_path.exists():
                        shutil.move(temp_download_path, final_target_path)
                        print(f"   -> Downloaded to {final_target_path}")
                    else:
                        print(f"   -> Error: Temporary file {temp_download_path} not found after download.")
                except Exception as e:
                    print(f"   -> Error downloading {model_info['repo_id']}/{filename}: {str(e)}")

    print("All models have been downloaded.")
    volume.commit() # Save the changes to the volume

# This lets you run the function from your command line.
@app.local_entrypoint()
def main():
    print("Calling the model downloader function on Modal...")
    download_all_models.remote()
    print("✅ Model download function has been executed.")
