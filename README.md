# artificial_graphics


tmux
export BUCKET_NAME=lfp_europe_west4_a

git clone https://github.com/sholtodouglas/artificial_graphics
cd artificial_graphics

    
pip install pathy -q
pip install wandb -q
pip install -q git+https://github.com/huggingface/transformers.git timm
pip install -q pytorch-lightning


mkdir data
gsutil -m cp -r dir gs://$BUCKET_NAME/data/unity data