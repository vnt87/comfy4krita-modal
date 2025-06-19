#!/bin/bash

# Script to run the model downloader and deploy the ComfyUI application on Modal

# Check if Modal is installed
if ! command -v modal &> /dev/null; then
    echo "Modal is not installed. Installing Modal using pipx..."
    pipx install modal
    echo "Modal installed successfully."
    echo "Running Modal setup..."
    modal setup
    echo "Modal setup completed."
else
    echo "Modal is already installed."
fi

echo "Starting the model download process on Modal..."
modal run download_models.py
echo "Model download process completed."

echo "Deploying the ComfyUI application on Modal..."
modal deploy app.py
echo "Deployment of ComfyUI application completed."

echo "All steps completed successfully."
