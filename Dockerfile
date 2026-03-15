# Use a versioned base image for stability (instead of :latest)
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04


# 2. Install python and system dependencies
# We use --no-install-recommends to keep the image small
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    git \
    libcudnn8 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# 3. Install python packages
# Note: Using pip3 specifically for Ubuntu
RUN pip3 install --no-cache-dir jupyter \
    tensorflow['and-cuda'] \
    opencv-python-headless \
    docopt \
    matplotlib \
    numpy \
    dahuffman \
    datasets