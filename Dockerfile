# Use a versioned base image for stability (instead of :latest)
FROM tensorflow/tensorflow:latest-gpu-jupyter


# 2. Install python and system dependencies
# We use --no-install-recommends to keep the image small (libs are for kaleido so  on same line)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    git \
    libcudnn8 \
    graphviz \
    libnss3 libatk-bridge2.0-0 libcups2 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libxkbcommon0 libpango-1.0-0 libcairo2 libasound2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/
COPY . /app

# 3. Install python packages
# Upgrade pip first, then install from requirements.txt
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt && \
    plotly_get_chrome -y || yes | plotly_get_chrome