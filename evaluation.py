import logging
import warnings
import os
logging.getLogger("tensorflow").setLevel(logging.ERROR)
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
from sklearn.metrics import confusion_matrix
import skimage
import glob
import itertools

ROOT_DIR = os.getcwd()

sys.path.append(ROOT_DIR)
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

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

weight_path = args.weight

config = training.CustomConfig()
dataset_dir = os.path.join(ROOT_DIR, args.dataset)

class InferenceConfig(config.__class__):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                            config=config)

model.load_weights(weight_path, by_name=True)


dataset = dataset.CustomDataset()
dataset.load_custom(dataset_dir, "test")
dataset.prepare()
print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

mAP_pascal = []
mAP_coco = []
mprecision = []
mrecall = []
moverlap = []

for image_id in dataset.image_ids:
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
       modellib.load_image_gt(dataset, config,
                              image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, config), 0)
    results = model.detect([image], verbose=0)

    r = results[0]
    index = [x for x in range(len(r['class_ids'])) if r['class_ids'][x] <= 4]

    AP_pascal, precision, recall, overlap =utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                        r["rois"][index], r["class_ids"][index], r["scores"][index], r['masks'][...,index])
    AP_coco = utils.compute_ap_range(gt_bbox, gt_class_id, gt_mask,
		                r["rois"][index], r["class_ids"][index], r["scores"][index], r['masks'][...,index])

    # if r["rois"].shape[0]:
    #     # Precision-Recall curve
    #     visualize.plot_precision_recall(AP, precision, recall)
    #     # Grid of ground truth objects and their predictions
    #     visualize.plot_overlaps(gt_class_id, r['class_ids'], r['scores'], overlap, dataset.class_names)

    mAP_pascal.append(AP_pascal)
    mAP_coco.append(AP_coco)
    mprecision.append(precision)
    mrecall.append(recall)
    moverlap.append(overlap)


print("mAP PASCAL",np.mean(mAP_pascal) * 100,"%")
print("mAP COCO:",np.mean(mAP_coco) * 100,"%")
print("mean Precision: ", np.mean(mprecision) * 100, "%")
print("mean Recall: ", np.mean(mrecall) * 100, "%")
print("mean Overlaps: ", np.mean(moverlap) * 100, "%")
