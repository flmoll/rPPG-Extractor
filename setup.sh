#!/bin/bash

# Check if a mode argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 {conda|uv}"
    exit 1
fi

MODE=$1

# Function to set up using conda
conda_setup() {
    echo "Setting up using conda..."
    conda remove --name rppg-toolbox --all -y || exit 1
    conda create -n rppg-toolbox python=3.8 -y || exit 1
    source "$(conda info --base)/etc/profile.d/conda.sh" || exit 1
    conda activate rppg-toolbox || exit 1
    pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121
    pip install transformers==4.46.3 || exit 1
    pip install neurokit2==0.2.10 || exit 1
    pip install pycryptodome==3.22.0 || exit 1
    pip install thop==0.1.1.post2209072238 || exit 1
    pip install -r requirements.txt --no-build-isolation || exit 1
    pip install vitallens==0.4.2 || exit 1
}


# Function to set up using uv
uv_setup() {
    rm -rf .venv || exit 1
    uv venv --python 3.8 || exit 1
    source .venv/bin/activate || exit 1
    uv pip install setuptools wheel || exit 1
    uv pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121 || exit 1
    uv pip install transformers==4.46.3 || exit 1
    uv pip install neurokit2==0.2.10 || exit 1
    uv pip install pycryptodome==3.22.0 || exit 1
    uv pip install thop==0.1.1.post2209072238 || exit 1
    uv pip install -r requirements.txt --no-build-isolation || exit 1
    uv pip install vitallens==0.4.2 || exit 1
}

# Execute the appropriate setup based on the mode
case $MODE in
    conda)
        conda_setup
        ;;
    uv)
        uv_setup
        ;;
    *)
        echo "Invalid mode: $MODE"
        echo "Usage: $0 {conda|uv}"
        exit 1
        ;;
esac
