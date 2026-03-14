FROM python:latest

RUN apt-get update && apt-get install -y

RUN pip install --no-cache-dir jupyter tensorflow[and-cuda] ipykernel opencv-python-headless docopt matplotlib numpy dahuffman datasets
RUN cd /app && jupyter notebook