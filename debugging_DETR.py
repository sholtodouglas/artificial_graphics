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

DATA_BASE = 'data/rgb_simple_ppt/'

import torchvision
import os

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, feature_extractor, train=True):
        ann_file = os.path.join(img_folder, "custom_train.json" if train else "custom_val.json")
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super(CocoDetection, self).__getitem__(idx)
        
        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        rotation = torch.as_tensor([a['rotation'] for a in target['annotations']])
        fill = torch.as_tensor([a['fill'] for a in target['annotations']])
        target = encoding["target"][0] # remove batch dimension
        target['rotation'] = rotation
        target['fill'] = fill
        return pixel_values, target


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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def load(model):
    state_dict = torch.load(f'saved_models/{args.run_name}', map_location=torch.device(device))
    model.load_state_dict(state_dict)
    
load(model)

def collate_fn(batch):
  pixel_values = [item[0] for item in batch]
  encoding = feature_extractor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")
  labels = [item[1] for item in batch]
  batch = {}
  batch['pixel_values'] = encoding['pixel_values']
  batch['pixel_mask'] = encoding['pixel_mask']
  batch['labels'] = labels
  return batch

batch = collate_fn([val_dataset[63], val_dataset[58]])
pixel_values = batch["pixel_values"] # typiccally resied to 1, 750, 1333
pixel_mask = batch["pixel_mask"] # typiccally resied to 1, 750, 1333
labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

print(outputs)