#! /bin/bash

pip3 install \
scipy \
einops \
flask-restful \
nltk \
pytest \
pytest-cov \
pytest_mock \
pytest-csv \
pytest-random-order \
sentencepiece \
wrapt \
zarr \
wandb \
tensorstore==0.1.45 \
pytest_mock \
pybind11 \
setuptools==69.5.1 \
datasets \
tiktoken \
pynvml

pip3 install "huggingface_hub[cli]"
python3 -m nltk.downloader punkt_tab

pip3 install transformers


git clone --recursive https://github.com/ROCm/TransformerEngine -b wen/gfx950_bf16
cd TransformerEngine 
NVTE_ROCM_ARCH=gfx950 pip install .
