


# %%
import argparse
from natsort import natsorted

print('------------------------------------------a')
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils
print('------------------------------------------b')

parser = argparse.ArgumentParser(description='AG training arguments')
parser.add_argument('run_name')
parser.add_argument('--train_datasets', nargs='+', help='Training dataset names')
parser.add_argument('--test_datasets', nargs='+', help='Testing dataset names')
parser.add_argument('-c', '--colab', default=False, action='store_true', help='Enable if using colab environment')
parser.add_argument('-s', '--data_source', default='DRIVE', help='Source of training data')
parser.add_argument('-d', '--device', default='TPU', help='Hardware device to train on')
parser.add_argument('-b', '--batch_size', default=4, type=int)
parser.add_argument('-log', '--log_steps', default=1, type=int)
parser.add_argument('-lr', '--learning_rate', type=float, default=2e-4)
parser.add_argument('-t', '--train_steps', type=int, default=1000)
parser.add_argument('--bucket_name', help='GCS bucket name to stream data from')
parser.add_argument('--tpu_name', help='GCP TPU name') # Only used in the script on GCP
args = parser.parse_args()




### Sample local config
# args = parser.parse_args('''
# angles&rgb 
# --train_dataset rgb_ppt/train
# --test_dataset rgb_ppt/val
# -s LOCAL
# --bucket_name lfp_europe_west4_a
# '''.split())


# %%
from pathlib import Path
from pathy import Pathy
import os
import requests
import json
import pprint
import logging
import numpy as np
import tensorflow as tf
import time


import torchvision
import os
import torch 

from transformers import DetrFeatureExtractor
from torch.utils.data import DataLoader
print('------------------------------------------c')

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
        
        target = encoding["labels"][0] # remove batch dimension
        target['rotation'] = rotation
        target['fill'] = fill
        return pixel_values, target

pp = pprint.PrettyPrinter(indent=4)
# In[4]:



print('Using local setup')
WORKING_PATH = Path.cwd()
print(f'Working path: {WORKING_PATH}')

# Change working directory to artificial_graphics
os.chdir(WORKING_PATH)
import lib

print('Reading data from local filesystem')
STORAGE_PATH = WORKING_PATH

# print(f'Storage path: {STORAGE_PATH}')
# TRAIN_DATA_PATHS = [STORAGE_PATH/'data'/x for x in args.train_datasets]
# TEST_DATA_PATHS = [STORAGE_PATH/'data'/x for x in args.test_datasets]

DATA_BASE = 'data/rgb_simple_ppt'


print('Feat')
feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
print('OverFeat')
from transformers import DetrConfig
from lib.DETR import DetrForObjectDetection
import torch
def collate_fn(batch):
  pixel_values = [item[0] for item in batch]
  encoding = feature_extractor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")
  labels = [item[1] for item in batch]
  batch = {}
  batch['pixel_values'] = encoding['pixel_values']
  batch['pixel_mask'] = encoding['pixel_mask']
  batch['labels'] = labels
  return batch

def create_model():
  model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
  state_dict = model.state_dict()
  # Remove class weights
  del state_dict["class_labels_classifier.weight"]
  del state_dict["class_labels_classifier.bias"]
  # define new model with custom class classifier
  config = DetrConfig.from_pretrained("facebook/detr-resnet-50", num_labels=len(id2label))
  model = DetrForObjectDetection(config)
  model.load_state_dict(state_dict, strict=False)
  return model


def common_step(batch, device, model):
    pixel_values = batch["pixel_values"]
    pixel_mask = batch["pixel_mask"]
    labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

    outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

    loss = outputs.loss
    loss_dict = outputs.loss_dict

    return loss, loss_dict

#########################################################################################################################################
def _train_update(device, step, loss, tracker, epoch, writer):
  test_utils.print_training_update(
      device,
      step,
      loss.item(),
      tracker.rate(),
      tracker.global_rate(),
      epoch,
      summary_writer=writer)


def train_imagenet():
  print('==> Preparing data..')
  
  train_dataset = CocoDetection(img_folder=f'{DATA_BASE}/train', feature_extractor=feature_extractor)
  test_dataset = CocoDetection(img_folder=f'{DATA_BASE}/val', feature_extractor=feature_extractor, train=False)

  train_sampler, test_sampler = None, None
  if xm.xrt_world_size() > 1:
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True)
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=False)

    train_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=args.batch_size, sampler = train_sampler, shuffle=True, num_workers = 3)
    test_loader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=args.batch_size, sampler= test_sampler, shuffle=True, num_workers = 3)

  torch.manual_seed(42)

  device = xm.xla_device()
  model = create_model().to(device)
  writer = None
  if xm.is_master_ordinal():
    writer = test_utils.get_summary_writer('/tmp/')
  optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=1e-4)


  def train_loop_fn(loader, epoch):
    tracker = xm.RateTracker()
    model.train()
    for step, batch in enumerate(loader):
      optimizer.zero_grad()
      loss, _ = common_step(batch, device, model)
      loss.backward()
      xm.optimizer_step(optimizer)
      tracker.add(args.batch_size)
      if step % args.log_steps == 0:
        xm.add_step_closure(
            _train_update, args=(device, step, loss, tracker, epoch, writer))

  def test_loop_fn(loader, epoch):
    total_samples, correct = 0, 0
    model.eval()
    for step, batch in enumerate(loader):
      loss, _ = common_step(batch, device, model)

    #   if step % args.log_steps == 0:
    #     xm.add_step_closure(
    #         test_utils.print_test_update, args=(device, None, epoch, step))
    # accuracy = 100.0 * correct.item() / total_samples
    # accuracy = xm.mesh_reduce('test_accuracy', accuracy, np.mean)
    return loss

  train_device_loader = pl.MpDeviceLoader(train_loader, device)
  test_device_loader = pl.MpDeviceLoader(test_loader, device)

  for epoch in range(1, 5):
    xm.master_print('Epoch {} train begin {}'.format(epoch, test_utils.now()))
    train_loop_fn(train_device_loader, epoch)
    xm.master_print('Epoch {} train end {}'.format(epoch, test_utils.now()))
    # if not FLAGS.test_only_at_end or epoch == FLAGS.num_epochs:
    #   accuracy = test_loop_fn(test_device_loader, epoch)
    #   xm.master_print('Epoch {} test end {}, Accuracy={:.2f}'.format(epoch, test_utils.now(), accuracy))
    # if FLAGS.metrics_debug:
    #   xm.master_print(met.metrics_report())

  test_utils.close_summary_writer(writer)


def _mp_fn(index, flags):
  print('ok')
  torch.set_default_tensor_type('torch.FloatTensor')
  train_imagenet()


if __name__ == '__main__':
  print('Beginning')
  xmp.spawn(_mp_fn, args=(args,), nprocs=8)
