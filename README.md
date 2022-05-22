```shell
gcloud alpha compute tpus tpu-vm create lfp1 --zone=europe-west4-a --accelerator-type=v3-8 --version=v2-alpha

gcloud alpha compute tpus tpu-vm ssh lfp1 --zone europe-west4-a --project learning-from-play-303306 -- -L 8888:localhost:8888

wget https://storage.googleapis.com/scenic-bucket/baselines/ResNet50_ImageNet1k -P models
cd scenic
pip install .

cd 
```
