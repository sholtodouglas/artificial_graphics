```shell
gcloud alpha compute tpus tpu-vm create lfp1 --zone=europe-west4-a --accelerator-type=v3-8 --version=v2-alpha --project learning-from-play-303306

gcloud alpha compute tpus tpu-vm ssh lfp1 --zone europe-west4-a --project learning-from-play-303306 -- -L 8888:localhost:8888
```

```
# Git config
git config --global user.email "sholto.douglas1@gmail.com"
git config --global user.name "Sholto Douglas"

# create python 3.9 env
sudo apt update
sudo apt install python3.9 -y
sudo  apt-get install python3.9-dev python3.9-venv -y

# create a venv
python3.9 -m venv 39

# activate it
source 39/bin/activate


# Clone
git clone https://github.com/sholtodouglas/artificial_graphics
cd artificial_graphics
pip install --upgrade pip
pip install .
# keep these out of requirements as somewhat time specific
pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install "protobuf==3.20.1"
```

```
wget https://storage.googleapis.com/scenic-bucket/baselines/ResNet50_ImageNet1k -P models
```
