FROM mcr.microsoft.com/devcontainers/python:1-3.12-bullseye

# Set labels
LABEL user="flmoll"

USER root

RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
#RUN curl -s -L https://nvidia.github.io/libnvidia-container/debian10/libnvidia-container.list | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
RUN echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/deb/$(dpkg --print-architecture) /" | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
RUN echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/debian10/$(dpkg --print-architecture) /" | tee -a /etc/apt/sources.list.d/nvidia-container-toolkit.list

RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/3bf863cc.pub | gpg --dearmor -o /usr/share/keyrings/nvidia-build-tools.gpg
RUN echo "deb [signed-by=/usr/share/keyrings/nvidia-build-tools.gpg] https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/ /" | tee /etc/apt/sources.list.d/nvidia-build-tools.list

# Install NVIDIA Container Toolkit and other dependencies
RUN apt-get update && \
    apt-get install -y nvidia-container-toolkit cuda-nvcc-12-2 libcusparse-dev-12-2 libcublas-dev-12-2 libcusolver-dev-12-2 build-essential ffmpeg v4l-utils git-lfs jupyter-notebook && \
    apt-get clean

ARG USERNAME=vscode
ARG USER_UID=1005
ARG USER_GID=$USER_UID

RUN groupmod --gid $USER_GID $USERNAME \
    && usermod --uid $USER_UID --gid $USER_GID $USERNAME \
    && chown -R $USER_UID:$USER_GID /home/$USERNAME

USER $USERNAME

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Set up Python dependencies
#COPY requirements.txt /workspaces/requirements.txt
#RUN pip3 install -r /workspaces/requirements.txt

# Set the working directory
WORKDIR /workspaces

# Ensure NVIDIA runtime is used
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

# Set the default command
CMD ["bash"]