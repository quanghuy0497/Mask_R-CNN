import os
import cv2
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage
import glob

# Root directory of the project
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
from mrcnn import parse_args

args = parse_args.parse_args()

import dataset
import training


MODEL_DIR = os.path.join(ROOT_DIR, "logs")

weight_path = args.weight

config = training.CustomConfig()
dataset_dir = os.path.join(ROOT_DIR, args.dataset)
image_path = os.path.join(ROOT_DIR, args.image)

class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
#DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0


# Load validation dataset
dataset = dataset.CustomDataset()
dataset.load_custom(dataset_dir, "val")

dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))


model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)


print("Loading weights ", weight_path)
model.load_weights(weight_path, by_name=True)


n = 0
positive = 0
negative = 0


for image_name in sorted(os.listdir(image_path)):
  if image_name.endswith(('.jpg','.jpeg')):
    image = skimage.io.imread(os.path.join(image_path, image_name))
    if image.ndim != 3:
      image = skimage.color.gray2rgb(image)
    if image.shape[-1] == 4:
      image = image[..., :3]
    check = 0
    
    results = model.detect([image], verbose=1)
    r = results[0]
    n += 1

    if r["rois"].shape[0]:
      positive += 1
      check = 1
    else: negative += 1

    if check:
      print(image_name," - Positive")
    else: print(image_name," - Negative")

    # visualize.display_instances(image_name,image, r['rois'], r['masks'], r['class_ids'], dataset.class_names, r['scores'])
    visualize.save_image(image_name, image, r['rois'], r['masks'], r['class_ids'], dataset.class_names, r['scores'])

print("Total: ", n ," images")
print("Postivie: ", positive, " - ",positive/n * 100,"%")
print("Negative: ", negative, " - ",negative/n * 100,"%")
