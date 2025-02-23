## ----------------------------
## == | Library Imports | == ##
## ----------------------------
## -- Progress | Completion progress bar
from tqdm import tqdm
import torch
import local_model as lm
import local_dataset as ld
import torch.optim as optim

num_classes = 1 
image_size = 416                                     
s = [image_size//32, image_size//16, image_size//8]  


ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
    

]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    # Device setup
model = lm.YOLOv3(num_classes=num_classes).to(device)                       # Model framework setup

## -- Model vars
load_model = True
num_epochs = 20
gbl_lr = 1e-4

optimizer = optim.Adam(model.parameters(), lr=gbl_lr)        # Model optimizer setup
loss_fn = lm.YOLOLoss()                                         # Model loss/criterion function setup
## -------------------------
## == | Code Section | == ##
## -------------------------

# Defining the scaler for mixed precision training
scaler = torch.cuda.amp.GradScaler()

# Define the train function to train the model
def training_loop(loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    # Creating a progress bar
    progress_bar = tqdm(loader, leave=True)

    # Initializing a list to store the losses
    losses = []

    # Iterating over the training data
    for _, (x, y) in enumerate(progress_bar):
        x = x.to(device)
        y0, y1, y2 = (
            y[0].to(device),
            y[1].to(device),
            y[2].to(device),
        )

        with torch.cuda.amp.autocast():
            # Getting the model predictions
            outputs = model(x)
            # Calculating the loss at each scale
            loss = (
                loss_fn(outputs[0], y0, scaled_anchors[0])
                +loss_fn(outputs[1], y1, scaled_anchors[1])
                +loss_fn(outputs[2], y2, scaled_anchors[2])
            )

        # Add the loss to the list
        losses.append(loss.item())

        # Reset gradients
        optimizer.zero_grad()

        # Backpropagate the loss
        scaler.scale(loss).backward()

        # Optimization step
        scaler.step(optimizer)

        # Update the scaler for next iteration
        scaler.update()

        # update progress bar with loss
        mean_loss = sum(losses) / len(losses)
        progress_bar.set_postfix(loss=mean_loss)

# Defining the train dataset
train_dataset = ld.Task2Dataset(
    csv_file= "Dataset.Code/labels.csv",
    image_dir= "TrainingImages/positives",
    label_dir= "TrainingImages/positives/annotations",
    anchors=ANCHORS,
    transform=ld.train_transform
)

# Defining the train data loader
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size = 1,
    shuffle = False
)

# Scaling the anchors
scaled_anchors = (
    torch.tensor(ANCHORS) *
    torch.tensor(s).unsqueeze(1).unsqueeze(1).repeat(1,3,2)
).to(device)

# Training the model
for e in range(1, num_epochs+1):
    print("Current Epoch:", e ,"/", num_epochs)
    training_loop(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)
