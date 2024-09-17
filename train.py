import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from load_dataset_rgb import DroneVehicleDataset


# Apply transformations without resizing
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])  # Normalize (for RGB images)
])

# Paths to your datasets
train_img_dir = 'train/trainimg'
train_label_dir = 'train/trainlabel'
val_img_dir = 'val/valimg'
val_label_dir = 'val/vallabel'

# Create datasets and dataloaders
train_dataset = DroneVehicleDataset(train_img_dir, train_label_dir, transform=transform)
val_dataset = DroneVehicleDataset(val_img_dir, val_label_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))


# Define the model
def get_model(num_classes):
    # Load a pre-trained model on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # Replace the pre-trained head with a new one for our dataset
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# Number of classes (5 categories + 1 background)
num_classes = 6  # Background, car, truck, bus, van, freight car

model = get_model(num_classes)

# Move model to the device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"Using device: {device}")

# Define the optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Create directory to save models
model_dir = 'saved_models'
os.makedirs(model_dir, exist_ok=True)

# Training loop
num_epochs = 10
train_losses_per_epoch = []
val_losses_per_epoch = []

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_running_loss = 0.0

    for images, targets in train_loader:
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
    print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")

    # Validation phase
    model.eval()
    val_running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            val_loss_dict = model(images, targets)
            val_losses = sum(loss for loss in val_loss_dict.values())
            val_running_loss += val_losses.item()

            # Collect predictions and labels for metrics
            outputs = model(images)
            for i, output in enumerate(outputs):
                pred_labels = output['labels'].cpu().numpy()
                true_labels = targets[i]['labels'].cpu().numpy()
                all_preds.extend(pred_labels)
                all_labels.extend(true_labels)

    avg_val_loss = val_running_loss / len(val_loader)
    val_losses_per_epoch.append(avg_val_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

    # Calculate class-wise Precision, Recall, and F1-Score
    precision = precision_score(all_labels, all_preds, labels=list(range(1, num_classes)), average=None,
                                zero_division=0)
    recall = recall_score(all_labels, all_preds, labels=list(range(1, num_classes)), average=None, zero_division=0)
    f1 = f1_score(all_labels, all_preds, labels=list(range(1, num_classes)), average=None, zero_division=0)

    for idx, cls in enumerate(range(1, num_classes)):
        print(f"Class {cls}: Precision: {precision[idx]:.4f}, Recall: {recall[idx]:.4f}, F1 Score: {f1[idx]:.4f}")

    # Save the model
    model_save_path = os.path.join(model_dir, f'faster_rcnn_epoch_{epoch + 1}.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

# Plot training and validation loss curves and save as image
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses_per_epoch, label="Training Loss", marker="o")
plt.plot(range(1, num_epochs + 1), val_losses_per_epoch, label="Validation Loss", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss per Epoch")
plt.legend()
plt.grid(True)
plt.savefig("training_validation_loss_plot.png")  # Save the plot as an image file
plt.close()  # Close the plot to avoid displaying it

print("Training complete. All models and loss plot saved.")
