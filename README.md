# artificial_graphics


tmux
export BUCKET_NAME=lfp_europe_west4_a

git clone https://github.com/sholtodouglas/artificial_graphics
cd artificial_graphics

    
pip install pathy -q
pip install wandb -q
pip install -q git+https://github.com/huggingface/transformers.git timm
pip install -q pytorch-lightning
pip install natsort
pip install pycocotools


mkdir data
gsutil -m cp -r dir gs://$BUCKET_NAME/data/rgb_simple_ppt/ data


python3 train_lfp.py \
angles&rgb  \
--train_dataset rgb_ppt/train \
--test_dataset rgb_ppt/val \
-s LOCAL \
--bucket_name lfp_europe_west4_a