from flask import Flask
import numpy as np
np.set_printoptions(precision=2)
import argparse
from natsort import natsorted
from pathlib import Path
from pathy import Pathy
from PIL import Image
from lib.app_interface import get_readout, check_ag_create
from lib.viz import img_format
from apscheduler.schedulers.background import BackgroundScheduler   
import time
import os

# Load up the existing powerpoint stuff
import win32com.client
from lib.ppt_interface import PPT_shapes, add_shape, add_line, add_title, add_textBox, convert_line_xyhw_to_points, read_slide, write_slide
shape_manager = PPT_shapes()

slide_height, slide_width = 540, 960
IMAGE_RES_DIVISOR = 1



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
sketches
--bucket_name lfp_europe_west4_a
'''.split())

import pytorch_lightning as pl
from transformers import DetrConfig, DetrFeatureExtractor
from lib.DETR import DetrForObjectDetection, CocoDetection
import torch

feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
DATA_BASE = 'data/custom/'
val_dataset = CocoDetection(img_folder=f'{DATA_BASE}/val', feature_extractor=feature_extractor, train=False)
cats = val_dataset.coco.cats
id2label = {k: v['name'] for k,v in cats.items()}
config = DetrConfig.from_pretrained("facebook/detr-resnet-50", num_labels=len(id2label))
model = DetrForObjectDetection(config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_ = model.to(device)


def load(model):
    state_dict = torch.load(f'saved_models/{args.run_name}', map_location=torch.device(device))
    model.load_state_dict(state_dict)
    
load(model)
print('Model loaded')

Application = win32com.client.Dispatch('PowerPoint.Application')


def test_job():
    current_slide = Application.ActiveWindow.View.Slide
    if check_ag_create(current_slide):
        print("Creating")
        # Move the @AG tags off page
        # Take a photo, save it, load it
        image_path = os.getcwd()+'\working_img.jpg'
        current_slide.Export(image_path, 'JPG', slide_width, slide_height)
        image = Image.open(image_path)
        pixel_values = img_format(image, feature_extractor).to(device).unsqueeze(0)
        outputs = model(pixel_values=pixel_values, pixel_mask=None)
        readout = get_readout(image, outputs, id2label, threshold=0.8)
        write_slide(readout, Application.ActivePresentation, shape_manager, idx=Application.ActiveWindow.View.Slide.SlideIndex)
        # Feed this into the model
        # Create the predictions!
    print('I am working...')

t0 = time.time()
while(1):
    if time.time() > t0+5:
        test_job()
        t0 = time.time()