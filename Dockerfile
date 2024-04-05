# Start your image with a ubuntu base image
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# The /app directory should act as the main application directory
WORKDIR /app

# Install node packages, install serve, build the app, and remove dependencies at the end
RUN apt update\
    && apt -y install sudo\
    && apt upgrade -y

# dependencies
RUN sudo apt -y update\
    && sudo apt install -y software-properties-common curl wget tar build-essential git htop libxrender1\
    && sudo apt install -y libgl1-mesa-glx xvfb

# python
RUN sudo apt install -y python3.10 python3-dev python3-doc python3-pip python3-venv\
    && ln -s /usr/bin/python3.10 /usr/bin/python

RUN sudo apt update\
    && sudo apt upgrade -y\
    && sudo apt clean\
    && rm -rf /var/lib/apt/lists/*\
    && sudo apt autoremove

# python packages
RUN pip3 install --upgrade pip wheel\
    && pip3 install numpy scipy pandas\
    && pip3 install matplotlib plotly seaborn wandb

RUN pip3 install scikit-learn lifelines statsmodels

RUN pip3 install torch\
    && pip3 install torchvision\
    && pip3 install torchaudio

RUN pip3 install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

RUN pip3 install equinox optax\
    && pip3 install jax-dataloader

EXPOSE 3000

# Start the app
CMD ["nvidia-smi"]