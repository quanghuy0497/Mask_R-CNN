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
import dlib
import json
from skimage.measure import find_contours

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
sys.path.append(os.path.join("train"))
import dataset
import training


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

weight_path = args.weight

config = training.CustomConfig()
dataset_dir = os.path.join(ROOT_DIR, args.dataset)
image_path = os.path.join(ROOT_DIR, args.image)

# Override the training configurations with a few
# changes for inferencing.
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

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet


# Load validation dataset
dataset = dataset.CustomDataset()
dataset.load_custom(dataset_dir, "val")
dataset.prepare()


predicted_class = ['BG', 'fg', 'mg', 'b', 'a'] #vlass label

# Create model in inference mode
#with tf.device(DEVICE):
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)
# Load weights
#print("Loading weights ", custom_WEIGHTS_PATH)
model.load_weights(weight_path, by_name=True)

data = {}
for filename in sorted(os.listdir(image_path)):
	if filename.endswith(('.jpg','.jpeg')):
		image = skimage.io.imread(os.path.join(image_path, filename))
		height, width = image.shape[:2]

		if image.ndim != 3:
			image = skimage.color.gray2rgb(image)
		if image.shape[-1] == 4:
			image = image[..., :3]

		results = model.detect([image], verbose=1)
		r = results[0]

		# masks: [height, width, num_instances]

		N = len(r['class_ids'])
		
		size = os.path.getsize(os.path.join(image_path, filename))
		regions = []
		
		for i in range(N):
			if (r['class_ids'][i] <= 4):
				# Extract mask 
				masks = r['masks']
				mask = masks[:, :, i]

				# Create mask polygon
				# Create padding = 1 to ensure proper polygons for masks that touch image edges.
				padded_mask = np.zeros(
	                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
				padded_mask[1:-1, 1:-1] = mask

				contours = find_contours(padded_mask, 0.5)
				for verts in contours:
					# Subtract the padding and flip (y, x) to (x, y)
					verts = np.fliplr(verts) - 1
				# Coordinate of polygon mask
				x = verts[:,0].tolist()
				y = verts[:,1].tolist()
				if (len(x) < 200): continue
				
				label  = str(predicted_class[r['class_ids'][i]])

				regions.append({"shape_attributes":{"name": "polygon", "all_points_x":x,"all_points_y":y},"region_attributes":{"porn":label}})

		id_image = filename + str(size)
		data.update({id_image:{"filename":filename,"size":size,"regions":regions,"file_attributes":{}}})

# Create annotation file
import json
with open(image_path + 'annotate.json', 'w') as outfile:
    json.dump(data, outfile,separators=(',', ':'))
print("Annotating json file saved as: ", image_path)
