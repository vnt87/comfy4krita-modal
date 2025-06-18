#!/bin/bash

# Script to run the model downloader and deploy the ComfyUI application on Modal

echo "Starting the model download process on Modal..."
modal run download_models.py
echo "Model download process completed."

echo "Deploying the ComfyUI application on Modal..."
modal deploy app.py
echo "Deployment of ComfyUI application completed."

echo "All steps completed successfully."
