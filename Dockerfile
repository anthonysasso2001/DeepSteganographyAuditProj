FROM tensorflow/tensorflow:latest-gpu-jupyter

RUN apt-get update && apt-get install -y

RUN pip install --no-cache-dir tensorflow[and-cuda] ipykernel opencv-python-headless docopt matplotlib numpy dahuffman datasets 