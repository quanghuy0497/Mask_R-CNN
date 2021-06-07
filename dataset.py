import os
import cv2
import json
import skimage.draw
import numpy as np
from mrcnn import model as modellib, utils
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split

class CustomDataset(utils.Dataset):

    def resize_image(image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)  
        return img
        
    def load_custom(self, dataset_dir, subset):
        """Load a subset of the custom_dataset dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes
        self.add_class("custom_dataset", 1, "fg")
        self.add_class("custom_dataset", 2, "mg")
        self.add_class("custom_dataset", 3, "b")
        self.add_class("custom_dataset", 4, "a")
        # Train or validation dataset?
        assert subset in ["train", "val", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        annotations1 = json.load(open(os.path.join(dataset_dir, "via_export_json.json")))
        # print(annotations1)
        annotations = list(annotations1.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # print(a)
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygons = [r['shape_attributes'] for r in a['regions']]
            names = [r['region_attributes'] for r in a['regions']]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            #print(image_path)
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "custom_dataset",  ## for a single class just add the name here
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons, names = names)    

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a custom_dataset dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "custom_dataset":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        class_names = info["names"]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            try:
                mask[rr, cc, i] = 1
            except:
                print(info['path'])

        class_ids = np.zeros([len(info["polygons"])])
        for i, p in enumerate(class_names):
            if p['custom_dataset'] == 'fg':
                class_ids[i] = 1
            elif p['custom_dataset'] == 'mg':
                class_ids[i] = 2
            elif p['custom_dataset'] == 'b':
                class_ids[i] = 3
            elif p['custom_dataset'] == 'a':
                class_ids[i] = 4
           #assert code here to extend to other labels
        class_ids = class_ids.astype(int)


        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "custom_dataset":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

############################################################
#  K-fold cross validation
############################################################
    def load_custom_K_fold(self, dataset_path, subset, fold):
        # Add classes
        self.add_class("custom_dataset", 1, "fg")
        self.add_class("custom_dataset", 2, "mg")
        self.add_class("custom_dataset", 3, "b")
        self.add_class("custom_dataset", 4, "a")

        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_path, 'train')        #split train folder to k-fold train and val

        N_Folds = 5
        
        annotations = []

        annotation = json.load(open(os.path.join(dataset_dir, "via_export_json.json")))
        
        annotation = list(annotation.values())  # don't need the dict keys

        k_fold = KFold(n_splits = N_Folds, random_state = 42, shuffle = True)

        for i, (train, val) in enumerate(k_fold.split(annotation)):
            if subset == "train" and fold == i:
                for index in train:
                    annotations.append(annotation[index])

            elif subset == "val" and fold == i:
                for index in val:
                    annotations.append(annotation[index])

        annotations = [a for a in annotations if a['regions']]

        for a in annotations:
            # print(a)
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygons = [r['shape_attributes'] for r in a['regions']]
            names = [r['region_attributes'] for r in a['regions']]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            if os.path.exists(image_path):
                image = skimage.io.imread(image_path)
                height, width = image.shape[:2]

                self.add_image(
                    "custom_dataset",  ## for a single class just add the name here
                    image_id=a['filename'],  # use file name as a unique image id
                    path=image_path,
                    width=width, height=height,
                    polygons=polygons, names = names)
