## ----------------------------
## == | Library Imports | == ##
## ----------------------------

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

import os
from PIL import Image
import pandas as pd
import torch

import local_utility as utils



## ------------------------------------------------- ##
## == | Global Variables & Scene Initialisation | == ##
## ------------------------------------------------- ##

# Image size
image_size = 416

## ------------------------------------------
## == | Dataset Setup & Configuration | == ##
## ------------------------------------------
class Task2Dataset(torch.utils.data.Dataset):
    def __init__(
        self, csv_file, image_dir, label_dir, anchors,
        image_size=416, grid_sizes=[75, 100, 125],
        n_classes=1, transform=None
    ):
        # Setup image labels, names & files
        self.label_list = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.label_dir = label_dir
        
        # Define image info | Size, given grid sizes for scales and transform assignment
        self.image_size = image_size
        self.grid_sizes = grid_sizes
        self.transform = transform

        # Anchor boxes | Number of anchor boxes | Number of anchor boxes per scale
        self.anchors = torch.tensor(anchors[0]+anchors[1]+anchors[2])
        self.n_anchors = self.anchors.shape[0]
        self.n_anchors_per_scale = self.n_anchors // 3
        
        # IoU threshold and n_classes init
        self.iou_thresh = 0.5
        self.n_classes = n_classes

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        # Getting the label path
        label_path = os.path.join(self.label_dir, self.label_list.iloc[idx, 1])
     
        # 5 columns: x, y, width, height, class_label
        bboxes = np.roll(np.loadtxt(fname=label_path,
                         delimiter=" ", ndmin=2), 4, axis=1).tolist()

        # Getting the image path
        img_path = os.path.join(self.image_dir, self.label_list.iloc[idx, 0])
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            augs = self.transform(image=image, bboxes=bboxes)
            image = augs["image"]
            bboxes = augs["bboxes"]

        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        # target : [probabilities, x, y, width, height, class_label]
        targets = [torch.zeros((self.n_anchors_per_scale, s, s, 6))
                   for s in self.grid_sizes]

        # Identify anchor box and cell for each bounding box
        for box in bboxes:
            # Calculate iou of bounding box with anchor boxes
            iou_anchors = utils.iou(torch.tensor(box[2:4]),
                              self.anchors,
                              is_pred=False)
            # Selecting the best anchor box
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box

            # At each scale, assigning the bounding box to the
            # best matching anchor box
            has_anchor = [False] * 3
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.n_anchors_per_scale
                anchor_on_scale = anchor_idx % self.n_anchors_per_scale

                # Identifying the grid size for the scale
                s = self.grid_sizes[scale_idx]

                # Identifying the cell to which the bounding box belongs
                i, j = int(s * y), int(s * x)
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

                # Check if the anchor box is already assigned
                if not anchor_taken and not has_anchor[scale_idx]:

                    # Set the probability to 1
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1

                    # Calculating the center of the bounding box relative
                    # to the cell
                    x_cell, y_cell = s * x - j, s * y - i

                    # Calculating the width and height of the bounding box
                    # relative to the cell
                    width_cell, height_cell = (width * s, height * s)

                    # Idnetify the box coordinates
                    box_coordinates = torch.tensor(
                                        [x_cell, y_cell, width_cell,
                                         height_cell]
                                    )

                    # Assigning the box coordinates to the target
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates

                    # Assigning the class label to the target
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)

                    # Set the anchor box as assigned for the scale
                    has_anchor[scale_idx] = True

                # If the anchor box is already assigned, check if the
                # IoU is greater than the threshold
                elif not anchor_taken and iou_anchors[anchor_idx] > self.iou_thresh:
                    # Set the probability to -1 to ignore the anchor box
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1

        # Return the image and the target
        return image, tuple(targets)
    

# Transform for training
train_transform = A.Compose(
    [
        # Rescale an image so that maximum side is equal to image_size
        A.LongestMaxSize(max_size=image_size),
        # Pad remaining areas with zeros
        A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT),
        
        # Random color jittering
        A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5, p=0.5),
        # Flip the image horizontally
        A.HorizontalFlip(p=0.5),
        # Normalize the image
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
        # Convert the image to PyTorch tensor
        ToTensorV2()
    ],
    # Augmentation for bounding boxes
    bbox_params=A.BboxParams(format="yolo",min_visibility=0.4,label_fields=[])
)

# Transform for testing
test_transform = A.Compose(
    [
        # Rescale an image so that maximum side is equal to image_size
        A.LongestMaxSize(max_size=image_size),
        # Pad remaining areas with zeros
        A.PadIfNeeded(
            min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT
        ),
        # Normalize the image
        A.Normalize(
            mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255
        ),
        # Convert the image to PyTorch tensor
        ToTensorV2()
    ],
    # Augmentation for bounding boxes
    bbox_params=A.BboxParams(
                    format="yolo",
                    min_visibility=0.4,
                    label_fields=[]
                )
)