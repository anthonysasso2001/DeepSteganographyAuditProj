# Deep Stenography Audit Project

A simple audit of "A deep learning-driven multilayered steganographic approach  for enhanced data security" (Sanjalawe Y. et al) using PyTorch against a model extraction attack to prove insecurities

## Main Argument/goal of study
- Given that the Deep Stenography decoder network is extracted and duplicated, then the benefits of the LSB/Huffman classical layer are limited, and they can be possibly easily circumvented by using a Huffman decoder / CNN-based decoder

## running code
make sure to use
`docker build . -t steganography-audit`
then to access (--rm optional)
`docker run -it --gpus all -p 8888:8888 steganography-audit`
check if gpu is found: `python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices())"`
then run below to run jupyter and allow access from vscode
`jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root`