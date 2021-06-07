"""
    # Train a new model starting from pre-trained weights
    python3 training.py --dataset=/path/to/dataset --weight=/path/to/pretrained/weight.h5
    # Resume training a model
    python3 training.py --dataset=/path/to/dataset --continue_train=/path/to/latest/weights.h5
"""

import logging
import warnings
import os
logging.getLogger("tensorflow").setLevel(logging.ERROR)
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
import matplotlib.pyplot as plt
import imgaug

# Root directory of the project
ROOT_DIR = os.getcwd()
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import parse_args
import dataset

############################################################
#  Args Configurations
############################################################

args = parse_args.parse_args()
# config parameter
pretrained_weight = os.path.join(ROOT_DIR, args.weight)
dataset_path = os.path.join(ROOT_DIR, args.dataset)
logs = os.path.join(ROOT_DIR, "logs")

if args.continue_train == "None":
    continue_train = args.continue_train
else:
    continue_train = os.path.join(ROOT_DIR, args.continue_train)

############################################################
#  Configurations
############################################################

class CustomConfig(Config):

    NAME = "custom_dataset"

    IMAGES_PER_GPU = 1

    IMAGE_MAX_DIM = 512

    NUM_CLASSES = 1 + 4

    STEPS_PER_EPOCH = 750

    VALIDATION_STEPS = 250

    DETECTION_MIN_CONFIDENCE = 0.9

    LEARNING_RATE = 0.001

    DETECTION_NMS_THRESHOLD = 0.2

    TRAIN_ROIS_PER_IMAGE = 200

    MAX_GT_INSTANCES = 50

    DETECTION_MAX_INSTANCES = 50

############################################################
#  Training 
############################################################

def train(model):
    """Train the model."""
    epoch_count = 0

    # training cross-validation with 5 fold
    for i in range(5):
        # Training dataset.
        print("Training fold", i)

        dataset_train = dataset.CustomDataset()
        dataset_train.load_custom_K_fold(dataset_path, "train", i)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = dataset.CustomDataset()
        dataset_val.load_custom_K_fold(dataset_path, "val", i)
        dataset_val.prepare()

        
        augmentation = imgaug.augmenters.Sometimes(0.5, [
                         imgaug.augmenters.Fliplr(0.5),
                         imgaug.augmenters.Flipud(0.5)])

        model_inference = modellib.MaskRCNN(mode="inference", config=config,model_dir=logs)

        mAP_callback = modellib.MeanAveragePrecisionCallback(model, model_inference, dataset_val, 
                                                        calculate_at_every_X_epoch=25, dataset_limit=500, verbose=1)
        # Training - Stage 1
        epoch_count += 20
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE *2,
                    epochs= epoch_count,
                    layers='heads',
                    custom_callbacks=[mAP_callback])
                    #augmentation=augmentation)

        epoch_count += 10
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs= epoch_count,
                    layers='4+',
                    custom_callbacks=[mAP_callback],
                    augmentation=augmentation)

############################################################
#  Training
############################################################

if __name__ == '__main__':
    
    print("Pre-trained weight: ", pretrained_weight)
    print("Dataset: ", dataset_path)
    print("Logs: ", logs)
    print("Continue Train: ", continue_train)

    # Configurations
    config = CustomConfig()
    config.display()

    # Create model
    model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=logs)
    
    if continue_train.lower() == "none":
        weights_path = pretrained_weight
    else:
        weights_path = continue_train

    # Load weights
    print("Loading weights ", weights_path)
    if continue_train == "None":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    train(model)