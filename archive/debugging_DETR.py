import argparse
from natsort import natsorted
from pathlib import Path
from pathy import Pathy
import torch

parser = argparse.ArgumentParser(description='AG training arguments')
parser.add_argument('run_name')
parser.add_argument('--train_datasets', nargs='+', help='Training dataset names')
parser.add_argument('--test_datasets', nargs='+', help='Testing dataset names')
parser.add_argument('-c', '--colab', default=False, action='store_true', help='Enable if using colab environment')
parser.add_argument('-s', '--data_source', default='DRIVE', help='Source of training data')
parser.add_argument('-d', '--device', default='TPU', help='Hardware device to train on')
parser.add_argument('-b', '--batch_size', default=16, type=int)
parser.add_argument('-lr', '--learning_rate', type=float, default=2e-4)
parser.add_argument('-t', '--train_steps', type=int, default=1000)
parser.add_argument('--bucket_name', help='GCS bucket name to stream data from')
parser.add_argument('--tpu_name', help='GCP TPU name') # Only used in the script on GCP



### Sample local config
args = parser.parse_args('''
dummy_run2 
--train_dataset rgb_ppt/train
--test_dataset rgb_ppt/val
-c
-s GCS
--bucket_name lfp_europe_west4_a
'''.split())

DATA_BASE = 'sample_data/custom/'

import torchvision
import os
from lib.DETR import CocoDetection



# Based on the class defined above, we create training and validation datasets.
from transformers import DetrFeatureExtractor

feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")

#train_dataset = CocoDetection(img_folder=f'{DATA_BASE}/train', feature_extractor=feature_extractor)
val_dataset = CocoDetection(img_folder=f'{DATA_BASE}/val', feature_extractor=feature_extractor, train=False)

# print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(val_dataset))

from torch.utils.data import DataLoader

import subprocess 

def dld_model():
    STORAGE_PATH = Pathy(f'gs://{args.bucket_name}')
    command = ["gsutil", "cp", str(STORAGE_PATH/f'saved_models/{args.run_name}'), "saved_models/"]
    
    try:
        subprocess.call(command)
    except:
        print("try this:  ", " ".join(command))

dld_model()

import pytorch_lightning as pl
from transformers import DetrConfig
from lib.DETR import DetrForObjectDetection
import torch


cats = val_dataset.coco.cats
id2label = {k: v['name'] for k,v in cats.items()}
config = DetrConfig.from_pretrained("facebook/detr-resnet-50", num_labels=len(id2label))
model = DetrForObjectDetection(config)

device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def load(model):
    state_dict = torch.load(f'saved_models/{args.run_name}', map_location=torch.device(device))
    model.load_state_dict(state_dict)
    
# load(model)

def collate_fn(batch):
  pixel_values = [item[0] for item in batch]
  encoding = feature_extractor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")
  labels = [item[1] for item in batch]
  batch = {}
  batch['pixel_values'] = encoding['pixel_values']
  batch['pixel_mask'] = encoding['pixel_mask']
  batch['labels'] = labels
  return batch

batch = collate_fn([val_dataset[2], val_dataset[3]])
pixel_values = batch["pixel_values"] # typiccally resied to 1, 750, 1333
pixel_mask = batch["pixel_mask"] # typiccally resied to 1, 750, 1333
labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

print(outputs)