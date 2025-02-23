## ----------------------------
## == | Library Imports | == ##
## ----------------------------


#from PIL import Image
import numpy as np
import torch

import matplotlib.patches as patches
import matplotlib.pyplot as plt
## ------------------------ ##
## == == ##
## ------------------------ ##

epsilon = 1e-6
class_labels = [
    "target"
]
# Defining a function to calculate Intersection over Union (IoU)
def iou(box1, box2, is_pred=True):
    if is_pred:
        # Box coordinates of prediction
        b1_x1 = box1[..., 0:1] - box1[..., 2:3] / 2
        b1_y1 = box1[..., 1:2] - box1[..., 3:4] / 2
        b1_x2 = box1[..., 0:1] + box1[..., 2:3] / 2
        b1_y2 = box1[..., 1:2] + box1[..., 3:4] / 2

        # Box coordinates of ground truth
        b2_x1 = box2[..., 0:1] - box2[..., 2:3] / 2
        b2_y1 = box2[..., 1:2] - box2[..., 3:4] / 2
        b2_x2 = box2[..., 0:1] + box2[..., 2:3] / 2
        b2_y2 = box2[..., 1:2] + box2[..., 3:4] / 2

        # Get the coordinates of the intersection rectangle
        x1 = torch.max(b1_x1, b2_x1)
        y1 = torch.max(b1_y1, b2_y1)
        x2 = torch.min(b1_x2, b2_x2)
        y2 = torch.min(b1_y2, b2_y2)
        # Make sure the intersection is at least 0
        intersection = (x2-x1).clamp(0)*(y2-y1).clamp(0)

        # Calculate the union area
        box1_area = abs((b1_x2-b1_x1)*(b1_y2-b1_y1))
        box2_area = abs((b2_x2-b2_x1)*(b2_y2-b2_y1))
        union = box1_area + box2_area - intersection

        # Calculate the IoU score
        
        iou_score = intersection/(union+epsilon)

        # Return IoU score
        return iou_score

    else:
        # IoU score based on width and height of bounding boxes
        box1 = box1.float()
        box2 = box2.float()
        # Calculate intersection area
        intersection_area = torch.min(box1[..., 0], box2[..., 0]) * torch.min(box1[..., 1], box2[..., 1])

        # Calculate union area
        box1_area = box1[..., 0] * box1[..., 1]
        box2_area = box2[..., 0] * box2[..., 1]
        union_area = box1_area + box2_area - intersection_area

        # Calculate IoU score
        iou_score = intersection_area/union_area

        # Return IoU score
        return iou_score
    
# Non-maximum suppression function to remove overlapping bounding boxes
def nms(bboxes, iou_thresh, threshold): 
    # Filter out bounding boxes with confidence below the threshold. 
    bboxes = [box for box in bboxes if box[1]>threshold] 
  
    # Sort the bounding boxes by confidence in descending order. 
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True) 
  
    # Initialize the list of bounding boxes after non-maximum suppression. 
    bboxes_nms = [] 
  
    while bboxes: 
        # Get the first bounding box. 
        first_box = bboxes.pop(0) 
        bboxes_nms.append(first_box)
  
        # Iterate over the remaining bounding boxes and remove those that overlap significantly with the first box.
        bboxes = [box for box in bboxes if box[0] == first_box[0] and iou(
            torch.tensor(first_box[2:]), 
            torch.tensor(box[2:]),
        ) < iou_thresh]
  
    # Return bounding boxes after non-maximum suppression. 
    return bboxes_nms


# Function to convert cells to bounding boxes
def convert_cells_to_bboxes(predictions, anchors, s, is_predictions=True):
    # Batch size used on predictions
    batch_size = predictions.shape[0]
    # Number of anchors
    num_anchors = len(anchors)
    # List of all the predictions
    box_predictions = predictions[..., 1:5]

    # If the input is predictions then we will pass the x and y coordinate
    # through sigmoid function and width and height to exponent function and
    # calculate the score and best class.
    if is_predictions:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = torch.exp(
            box_predictions[..., 2:]) * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)

    # Else we will just calculate scores and best class.
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]

    # Calculate cell indices
    cell_indices = (
        torch.arange(s)
        .repeat(predictions.shape[0], 3, s, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )

    # Calculate x, y, width and height with proper scaling
    x = 1 / s * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / s * (box_predictions[..., 1:2] +
                 cell_indices.permute(0, 1, 3, 2, 4))
    width_height = 1 / s * box_predictions[..., 2:4]

    # Concatinating the values and reshaping them in
    # (BATCH_SIZE, num_anchors * S * S, 6) shape
    converted_bboxes = torch.cat(
        (best_class, scores, x, y, width_height), dim=-1
    ).reshape(batch_size, num_anchors * s * s, 6)

    # Returning the reshaped and converted bounding box list
    return converted_bboxes.tolist()

# Function to plot images with bounding boxes and class labels
def plot_image(image, boxes):
    # Getting the color map from matplotlib
    colour_map = plt.get_cmap("tab20b")
    # Getting 20 different colors from the color map for 20 different classes
    colors = [colour_map(i) for i in np.linspace(0, 1, len(class_labels))]

    # Reading the image with OpenCV
    img = np.array(image)
    # Getting the height and width of the image
    h, w, _ = img.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Add image to plot
    ax.imshow(img)

    # Plotting the bounding boxes and labels over the image
    for box in boxes:
        # Get the class from the box
        class_pred = box[0]
        # Get the center x and y coordinates
        box = box[2:]
        # Get the upper left corner coordinates
        upper_left_x = box[0]-box[2]/2
        upper_left_y = box[1]-box[3]/2

        # Create a Rectangle patch with the bounding box
        rect = patches.Rectangle(
            (upper_left_x*w, upper_left_y*h),
            box[2]*w,
            box[3]*h,
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none",
        )

        # Add the patch to the Axes
        ax.add_patch(rect)

        # Add class name to the patch
        plt.text(
            upper_left_x*w,
            upper_left_y*h,
            s=class_labels[int(class_pred)],
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(class_pred)], "pad": 0},
        )

    # Display the plot
    plt.show()

