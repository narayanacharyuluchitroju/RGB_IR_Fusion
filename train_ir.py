import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import xml.etree.ElementTree as ET
import torchvision
from load_dataset_rgb import DroneVehicleDataset

print("train.py")

def calculate_iou(pred_box, gt_box):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1_max = max(pred_box[0], gt_box[0])
    y1_max = max(pred_box[1], gt_box[1])
    x2_min = min(pred_box[2], gt_box[2])
    y2_min = min(pred_box[3], gt_box[3])

    # Compute the area of intersection
    inter_area = max(0, x2_min - x1_max) * max(0, y2_min - y1_max)

    # Compute the area of both boxes
    pred_box_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    gt_box_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])

    # Compute the area of union
    union_area = pred_box_area + gt_box_area - inter_area

    # Compute IoU
    iou = inter_area / union_area if union_area > 0 else 0

    return iou

# Define the model
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # Modify the first convolutional layer to accept 1-channel (grayscale) images instead of 3-channel RGB
    model.backbone.body.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# Paths to your datasets
train_img_dir = 'train/trainimg'
train_label_dir = 'train/trainlabel'
val_img_dir = 'val/valimg'
val_label_dir = 'val/vallabel'

# Define transformations, including ToTensor
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale (1 channel)
    transforms.ToTensor(),  # Convert PIL Image to PyTorch Tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize (for grayscale images)
])

# Dataset class
train_dataset = DroneVehicleDataset(train_img_dir, train_label_dir, transform=transform)
val_dataset = DroneVehicleDataset(val_img_dir, val_label_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Initialize the model
num_classes = 6  # Background + 5 vehicle types
model = get_model(num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: ', device)
model.to(device)

# Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# IoU threshold for true positives
iou_threshold = 0.5

# Training loop
num_epochs = 10
train_losses_per_epoch = []

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_running_loss = 0.0

    for images, targets in train_loader:
        # Move images and targets to GPU
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        train_running_loss += losses.item()

        # Backward pass and optimization
        losses.backward()
        optimizer.step()

    avg_train_loss = train_running_loss / len(train_loader)
    train_losses_per_epoch.append(avg_train_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss}")

    # Validation phase (IoU-based evaluation)
    model.eval()
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass for predictions (no loss)
            outputs = model(images)

            for i, output in enumerate(outputs):
                pred_boxes = output['boxes'].cpu().numpy()
                pred_labels = output['labels'].cpu().numpy()
                true_boxes = targets[i]['boxes'].cpu().numpy()
                true_labels = targets[i]['labels'].cpu().numpy()

                for pred_box in pred_boxes:
                    best_iou = 0
                    for true_box in true_boxes:
                        iou = calculate_iou(pred_box, true_box)
                        if iou > best_iou:
                            best_iou = iou

                    if best_iou >= iou_threshold:
                        true_positives += 1
                    else:
                        false_positives += 1

                # Count false negatives (ground truth boxes without matching predictions)
                false_negatives += len(true_boxes) - true_positives

    # Calculate precision, recall, and F1 score
    precision = true_positives / (true_positives + false_positives + 1e-6)
    recall = true_positives / (true_positives + false_negatives + 1e-6)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)

    print(f"Validation - Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")

    # Save the model
    model_save_path = os.path.join('saved_models', f'faster_rcnn_epoch_{epoch + 1}.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

# Plot training loss curves and save as image
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses_per_epoch, label="Training Loss", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss per Epoch")
plt.legend()
plt.grid(True)
plt.savefig("training_loss_plot.png")
plt.close()
