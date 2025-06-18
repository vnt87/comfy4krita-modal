# ComfyUI on Modal Deployment for Krita AI

This project provides a streamlined setup for deploying [ComfyUI](https://github.com/comfyanonymous/ComfyUI), a powerful graphical user interface for Stable Diffusion, on the [Modal](https://modal.com/) platform. It leverages Modal's serverless infrastructure to run ComfyUI with GPU acceleration, making it accessible via a web interface. The primary purpose of this deployment is to create a high-performance ComfyUI instance to be used in conjunction with Krita AI, enhancing digital painting and image generation capabilities within Krita. The setup includes a comprehensive collection of Stable Diffusion models, custom nodes for extended functionality, and scripts to automate model downloading and application deployment.

## Project Overview

ComfyUI is a node-based interface for Stable Diffusion, allowing users to create complex image generation workflows. This project deploys ComfyUI on Modal, utilizing a custom Docker image with pre-installed dependencies and custom nodes. A persistent volume stores a wide array of models, including SDXL base and refiner models, ControlNet models, LoRAs, and more, ensuring they are readily available for use. The deployment supports GPU acceleration for efficient processing and can scale to handle multiple concurrent containers, making it an ideal backend for Krita AI's image generation needs.

## Prerequisites

Before deploying this project, ensure you have the following set up:

- **Modal CLI**: Install the Modal CLI by following the [official installation guide](https://modal.com/docs/guide). Ensure you are logged in with your Modal credentials.
- **Python Environment**: A Python environment (version 3.11 recommended) to run the Modal scripts locally if needed.
- **Git**: Installed on your system to clone repositories or manage version control.
- **Access Permissions**: Ensure your Modal account has sufficient permissions and credits for GPU usage and volume storage.

## Installation and Deployment

Follow these steps to deploy ComfyUI on Modal:

1. **Clone or Download the Repository**:
   Clone this repository to your local machine or download the project files.

   ```bash
   git clone https://github.com/vnt87/comfy4krita-modal.git
   cd comfy4krita-modal
   ```

2. **Run the Deployment Script**:
   Execute the provided bash script to download models and deploy the application. This script automates the process for you.

   ```bash
   ./deploy_comfyui.sh
   ```

   The script will first run the model downloader to populate the volume with necessary models and then deploy the ComfyUI application.

3. **Manual Deployment (Optional)**:
   If you prefer manual control or need to troubleshoot, you can run the commands individually:

   - Download models to the volume:
     ```bash
     modal run download_models.py
     ```
   - Deploy the ComfyUI application:
     ```bash
     modal deploy app.py
     ```

4. **Access the Deployment URL**:
   After successful deployment, Modal will output a URL for your ComfyUI instance (typically in the format `https://[your-username]--comfyui-app-run-comfyui.modal.run`). This URL points to the web interface running on port 8188.

## GPU Options and Pricing

When deploying ComfyUI on Modal, you can specify a GPU type in `app.py` by modifying the `gpu` parameter in the function definition (e.g., changing `gpu="any"` to `gpu="L40S"` for a specific model). Below is a table of available GPU options, their estimated power tiers, and hourly costs as per Modal's pricing:

| GPU Model          | Power Tier         | Hourly Cost   |
|--------------------|--------------------|---------------|
| Nvidia B200        | High-End           | $6.25 / h     |
| Nvidia H200        | High-End           | $4.54 / h     |
| Nvidia H100        | High-End           | $3.95 / h     |
| Nvidia A100, 80 GB | High-Mid Range     | $2.50 / h     |
| Nvidia A100, 40 GB | Mid-Range          | $2.10 / h     |
| Nvidia L40S        | Mid-Range          | $1.95 / h     |
| Nvidia A10G        | Entry-Mid Range    | $1.10 / h     |
| Nvidia L4          | Entry-Level        | $0.80 / h     |
| Nvidia T4          | Basic              | $0.59 / h     |

Select a GPU based on your performance needs and budget. Higher-end GPUs like the Nvidia B200 or H100 offer superior processing power for faster image generation, which can be particularly beneficial for complex workflows in Krita AI.

## Usage

Once deployed, access the ComfyUI web interface via the URL provided by Modal. Here's how to get started:

- **Open the Interface**: Navigate to the deployment URL in your web browser. You should see the ComfyUI graphical interface.
- **Load Models**: Models are stored in the mounted volume under `/root/ComfyUI/models`. Use the interface to load checkpoints, VAEs, LoRAs, or other models from their respective directories.
- **Create Workflows**: Use the node-based editor to design image generation workflows. Connect nodes for text encoding, diffusion, image processing, and more.
- **Generate Images**: Run your workflow to generate images using Stable Diffusion. Adjust parameters like steps, CFG scale, and resolution as needed.
- **Save and Share**: Save your workflows as JSON files for reuse, and download generated images directly from the interface.

Note: The container remains active for 5 minutes after the last request (configurable in `app.py`), so you can return to it without redeploying if within this window.

## Models and Custom Nodes

This deployment includes a comprehensive set of models for various Stable Diffusion tasks, downloaded automatically by `download_models.py`. Below is a categorized list of included models with their source repository IDs or direct download URLs from Hugging Face or other repositories:

### Checkpoints
- **SDXL Base 1.0** (`sd_xl_base_1.0.safetensors`): The primary Stable Diffusion XL model for high-quality image generation.  
  *Source*: [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- **SDXL Refiner 1.0** (`sd_xl_refiner_1.0.safetensors`): A refiner model to enhance details in generated images.  
  *Source*: [stabilityai/stable-diffusion-xl-refiner-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0)
- **RealVisXL V5.0** (`RealVisXL_V5.0_fp16.safetensors`): A realistic vision model for photorealistic outputs.  
  *Source*: [SG161222/RealVisXL_V5.0](https://huggingface.co/SG161222/RealVisXL_V5.0)
- **Flux1-dev-fp8** (`flux1-dev-fp8.safetensors`): A development version of the Flux1 model.  
  *Source*: [Comfy-Org/flux1-dev](https://huggingface.co/Comfy-Org/flux1-dev)
- **ZavyChromaXL V8.0** (`zavychromaxl_v80.safetensors`): A model for vibrant and detailed outputs.  
  *Source*: [misri/zavychromaxl_v80](https://huggingface.co/misri/zavychromaxl_v80)
- **DreamShaper XL V2 Turbo** (`DreamShaperXL_Turbo_v2.safetensors`): A turbo version for faster generation with quality.  
  *Source*: [Lykon/dreamshaper-xl-v2-turbo](https://huggingface.co/Lykon/dreamshaper-xl-v2-turbo)

### VAEs
- **SDXL VAE** (`sdxl_vae.safetensors`): Improves color and detail in SDXL generations.  
  *Source*: [stabilityai/sdxl-vae](https://huggingface.co/stabilityai/sdxl-vae)

### CLIP Text Encoders
- **OpenCLIP ViT-bigG-14** (`open_clip_pytorch_model.bin`): Primary text encoder for SDXL.  
  *Source*: [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
- **CLIP ViT-L/14** (`pytorch_model.bin`): Secondary text encoder for compatibility.  
  *Source*: [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)

### CLIP Vision Models
- **IP-Adapter Image Encoder** (`model.safetensors`): For image-to-text encoding.  
  *Source*: [h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter)
- **SigCLIP Vision 384** (`sigclip_vision_patch14_384.safetensors`): Specialized vision model for CLIP.  
  *Source*: [Comfy-Org/sigclip_vision_384](https://huggingface.co/Comfy-Org/sigclip_vision_384)

### Upscale Models
- **NMKD Superscale SP** (`4x_NMKD-Superscale-SP_178000_G.pth`): For upscaling images.  
  *Source*: [gemasai/4x_NMKD-Superscale-SP_178000_G](https://huggingface.co/gemasai/4x_NMKD-Superscale-SP_178000_G)
- **OmniSR X2, X3, X4** (`OmniSR_X2_DIV2K.safetensors`, etc.): Multiple upscale resolutions.  
  *Source*: [Acly/Omni-SR](https://huggingface.co/Acly/Omni-SR)
- **HAT SRx4** (`HAT_SRx4_ImageNet-pretrain.pth`): High-quality upscaling.  
  *Source*: [Acly/hat](https://huggingface.co/Acly/hat)
- **Real HAT GAN Sharper** (`Real_HAT_GAN_sharper.pth`): Enhanced sharpness in upscaling.  
  *Source*: [Acly/hat](https://huggingface.co/Acly/hat)

### IP Adapter Models
- **IP-Adapter SDXL ViT-H** (`ip-adapter_sdxl_vit-h.safetensors`): For image prompt adaptation.  
  *Source*: [h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter)

### LoRA Models
- **Hyper-SDXL 8steps CFG** (`Hyper-SDXL-8steps-CFG-lora.safetensors`): For faster generation.  
  *Source*: [ByteDance/Hyper-SD](https://huggingface.co/ByteDance/Hyper-SD)
- **IP-Adapter FaceID PlusV2 SDXL** (`ip-adapter-faceid-plusv2_sdxl.bin`): For face recognition and adaptation.  
  *Source*: [h94/IP-Adapter-FaceID](https://huggingface.co/h94/IP-Adapter-FaceID)

### Inpaint Models
- **Fooocus Inpaint Head and Patch** (`fooocus_inpaint_head.pth`, `inpaint_v26.fooocus.patch`): For inpainting tasks.  
  *Source*: [lllyasviel/fooocus_inpaint](https://huggingface.co/lllyasviel/fooocus_inpaint)
- **MAT Places512** (`MAT_Places512_G_fp16.safetensors`): For scene inpainting.  
  *Source*: [Acly/MAT](https://huggingface.co/Acly/MAT)

### ControlNet Models
- **ControlNet Union SDXL Promax** (`diffusion_pytorch_model_promax.safetensors`): Multi-purpose ControlNet.  
  *Source*: [xinsir/controlnet-union-sdxl-1.0](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0)
- **ControlNet QRCode Monster** (`diffusion_pytorch_model.safetensors`): For QR code generation.  
  *Source*: [monster-labs/control_v1p_sdxl_qrcode_monster](https://huggingface.co/monster-labs/control_v1p_sdxl_qrcode_monster)
- **Flux1-dev ControlNet Union Pro and Inpainting Beta** (`diffusion_pytorch_model.safetensors`): Advanced control models.  
  *Source*: [ABDALLALSWAITI/FLUX.1-dev-ControlNet-Union-Pro-2.0-fp8](https://huggingface.co/ABDALLALSWAITI/FLUX.1-dev-ControlNet-Union-Pro-2.0-fp8) and [alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta](https://huggingface.co/alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta)
- **MistoLine Flux.dev** (`mistoline_flux.dev_v1.safetensors`): Line art control.  
  *Source*: [TheMistoAI/MistoLine_Flux.dev](https://huggingface.co/TheMistoAI/MistoLine_Flux.dev)

### Style Models
- **Flux1 Redux Dev** (`flux1-redux-dev.safetensors`): For stylistic outputs.  
  *Source*: [files.interstice.cloud/models](https://files.interstice.cloud/models/flux1-redux-dev.safetensors) (Direct URL: https://files.interstice.cloud/models/flux1-redux-dev.safetensors)

### Custom Nodes
The following custom nodes are pre-installed in the ComfyUI setup for additional functionality:
- **ComfyUI-Manager**: For managing workflows and nodes.
- **comfyui_controlnet_aux**: Auxiliary tools for ControlNet integration.
- **ComfyUI_IPAdapter_plus**: Enhanced IP Adapter functionalities.
- **comfyui-inpaint-nodes**: Specialized nodes for inpainting.
- **comfyui-tooling-nodes**: General utility nodes for workflow enhancement.

## Integration with Krita AI

This ComfyUI deployment on Modal is designed to work seamlessly with Krita AI, enhancing your digital painting workflow with powerful GPU-accelerated image generation. Follow these steps to set up Krita, install the Krita AI plugin, and connect it to your ComfyUI instance:

### Setting Up Krita and Krita AI Plugin

1. **Install Krita**:
   - Download and install Krita from the official website [krita.org](https://krita.org/en/download/krita-desktop/). Krita is available for Windows, macOS, and Linux.
   - Follow the installation instructions specific to your operating system.

2. **Install Krita AI Plugin**:
   - Visit the Krita AI plugin repository or download page (e.g., [Krita AI Diffusion on GitHub](https://github.com/Acly/krita-ai-diffusion)).
   - Follow the installation guide provided in the repository. Typically, this involves:
     - Downloading the plugin files.
     - Placing them in Krita's plugin directory (e.g., `~/.local/share/krita/pykrita` on Linux or similar paths on other OS).
     - Enabling the plugin via Krita's settings under `Settings > Configure Krita > Python Plugin Manager`.
   - Restart Krita to ensure the plugin is loaded.

### Connecting Krita AI to ComfyUI on Modal

1. **Obtain Your ComfyUI Deployment URL**:
   - After deploying ComfyUI using `modal deploy app.py` or the `deploy_comfyui.sh` script, note the URL provided by Modal (e.g., `https://[your-username]--comfyui-app-run-comfyui.modal.run`).
   - This URL is the endpoint for your ComfyUI API, which Krita AI will connect to.

2. **Configure Krita AI Plugin**:
   - Open Krita and navigate to the Krita AI plugin settings (usually under `Tools > Scripts` or a dedicated AI panel).
   - Enter the ComfyUI deployment URL in the plugin's configuration field for the server or API endpoint.
   - If the plugin requires additional settings like API keys or specific workflow configurations, consult the Krita AI documentation. For this setup, ensure the plugin is set to use a remote ComfyUI server.
   - Test the connection to confirm that Krita AI can communicate with your Modal-hosted ComfyUI instance. You should see available models or a successful connection message.

3. **Using ComfyUI with Krita AI**:
   - **Select AI Tools in Krita**: Use the Krita AI plugin interface to access image generation tools. You can typically select areas on your canvas or input text prompts for generation.
   - **Generate Images**: The plugin will send requests to your ComfyUI instance on Modal, leveraging its GPU power and model library to process the request. Results are sent back to Krita and appear on your canvas or as layers.
   - **Iterate and Refine**: Adjust prompts, select different models (e.g., SDXL or inpaint models for specific tasks), or tweak parameters directly in Krita. The remote ComfyUI instance handles the heavy computation, allowing for rapid iteration.
   - **Inpainting and Editing**: Use ComfyUI's inpaint models and ControlNet for precise edits. Select a region on your canvas, send it to ComfyUI via Krita AI, and receive edited results based on your prompt or mask.

### Tips for Optimal Integration
- Ensure your Modal container remains active by setting a longer `scaledown_window` in `app.py` if you plan extended Krita sessions.
- Use high-performance GPUs (like Nvidia H100 or A100) for faster response times in Krita AI, especially for complex workflows.
- If you encounter connection issues, verify that your ComfyUI deployment is active in the Modal dashboard and that the URL is correctly entered in Krita AI settings.

For further details on Krita AI functionalities, refer to its official documentation or GitHub repository. This integration allows artists to harness Stable Diffusion's capabilities directly within Krita, powered by a robust Modal backend.

## Troubleshooting

Here are solutions to common issues you might encounter:

- **Model Download Failures**: If `download_models.py` fails for certain models, check your internet connection or Modal's status. You can rerun `modal run download_models.py` to retry downloading missing models. Ensure your Modal account has enough storage quota for the volume.
- **Deployment Errors**: If `modal deploy app.py` fails, check the error logs for issues with the image build or GPU availability. Ensure your Modal plan supports GPU usage and that the `comfyui-models-vol` volume exists.
- **Interface Not Accessible**: If the provided URL doesn't work, verify that the deployment is active in your Modal dashboard. The container may have scaled down after 5 minutes of inactivity; redeploy if necessary.
- **Missing Models in Interface**: Confirm that the model downloader completed successfully. Models should appear in subdirectories under `/root/ComfyUI/models` in the ComfyUI interface.

For further assistance, refer to the [Modal documentation](https://modal.com/docs) or [ComfyUI GitHub issues](https://github.com/comfyanonymous/ComfyUI/issues).

## Contributing and Customization

To customize this deployment:

- **Add or Remove Models**: Edit `download_models.py` to modify the `MODELS` dictionary. Add new models with their Hugging Face repository IDs and target directories, or remove unnecessary ones. Rerun `modal run download_models.py` to update the volume.
- **Install Additional Custom Nodes**: Modify `app.py` to include more `git clone` commands in the image build section for custom nodes. Redeploy with `modal deploy app.py` to apply changes.
- **Adjust Container Settings**: In `app.py`, tweak parameters like `scaledown_window` for container persistence or `max_containers` for concurrency limits based on your needs.

Contributions to improve this setup are welcome. Please fork the repository, make your changes, and submit a pull request with a detailed description of your updates.

## License and Credits

[License information to be added based on project owner's preference. Currently a placeholder.]

**Credits**:
- **ComfyUI**: Developed by [comfyanonymous](https://github.com/comfyanonymous/ComfyUI), providing the core interface for Stable Diffusion workflows.
- **Modal**: The serverless platform hosting this deployment, enabling GPU-accelerated computing.
- **Model Creators**: Various model weights are sourced from Hugging Face repositories credited to their respective authors and organizations like Stability AI, OpenAI, and others listed in `download_models.py`.

Thank you for using this ComfyUI on Modal deployment setup. For further customization or support, reach out via the repository's issue tracker or Modal's community channels.
