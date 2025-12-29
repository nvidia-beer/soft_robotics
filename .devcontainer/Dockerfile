FROM nvcr.io/nvidia/tensorflow:22.02-tf2-py3

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for PyGame rendering
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    xvfb \
    x11-utils \
    && rm -rf /var/lib/apt/lists/*

# Install core packages (pin scipy<1.12 for numpy compatibility)
RUN pip install --no-cache-dir \
    "scipy>=1.7.0,<1.12" \
    matplotlib>=3.3.0 \
    pygame \
    pyvirtualdisplay \
    tqdm \
    gymnasium

# Install Nengo ecosystem with nengo-dl (TensorFlow GPU acceleration)
# nengo-dl 3.6.0 is compatible with TensorFlow 2.3-2.11 (container has 2.7.0)
RUN pip install --no-cache-dir \
    nengo>=3.2.0 \
    nengo-gui>=0.5.0 \
    nengo-dl==3.6.0

# Install Warp for FEM physics
RUN pip install --no-cache-dir warp-lang

# Set working directory
WORKDIR /workspace
