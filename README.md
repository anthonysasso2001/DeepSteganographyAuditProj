# Deep Stenography Audit Project

A simple audit of "A deep learning-driven multilayered steganographic approach  for enhanced data security" (Sanjalawe Y. et al) using PyTorch against a model extraction attack to prove insecurities

## Main Argument/goal of study
- Given that the Deep Stenography decoder network is extracted and duplicated, then the benefits of the LSB/Huffman classical layer are limited, and they can be possibly easily circumvented by using a Huffman decoder / CNN-based decoder

## running code
Should work with `docker-compose up --build` and then use `http://localhost:8888/?token={token in .env file}` to connect once container is running

## steganalysis
run `python3 run_aletheia_postprocess.py --run-data-dir /app/data/baseline/0` inside docker with 0 switched for whichever ablation is being targeted