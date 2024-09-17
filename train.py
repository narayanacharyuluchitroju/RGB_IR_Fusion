import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import xml.etree.ElementTree as ET
import torchvision


class DroneVehicleDataset(Dataset):
    def __init__(self, img_folder, label_folder, transform=None):
        self.img_folder = img_folder
        self.label_folder = label_folder
        self.transform = transform
        self.img_files = sorted(os.listdir(self.img_folder))
        self.label_files = sorted(os.listdir(self.label_folder))

        # Filter out images without bounding boxes during initialization
        self.filtered_files = self.filter_empty_images()

    def __len__(self):
        return len(self.filtered_files)

    def filter_empty_images(self):
        """Filter out images that do not have any bounding boxes."""
        filtered_files = []
        for img_file, label_file in zip(self.img_files, self.label_files):
            label_path = os.path.join(self.label_folder, label_file)
            boxes, _ = self.parse_annotation(label_path)
            if len(boxes) > 0:  # Only keep images with at least one bounding box
                filtered_files.append((img_file, label_file))
        return filtered_files

    def parse_annotation(self, label_path):
        tree = ET.parse(label_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall('object'):
            name = obj.find('name').text.lower()  # Lowercase for consistency

            bndbox = obj.find('bndbox')
            if bndbox is not None:
                try:
                    x_min = int(bndbox.find('xmin').text) - 100
                    y_min = int(bndbox.find('ymin').text) - 100
                    x_max = int(bndbox.find('xmax').text) - 100
                    y_max = int(bndbox.find('ymax').text) - 100
                except AttributeError as e:
                    print(f"Warning: Missing xmin/xmax/ymin/ymax in {label_path}: {e}")
                    continue
                boxes.append([x_min, y_min, x_max, y_max])

                # Map label names to integers based on the five categories
                if name == 'car':
                    labels.append(1)
                elif name == 'truck':
                    labels.append(2)
                elif name == 'bus':
                    labels.append(3)
                elif name == 'van':
                    labels.append(4)
                elif name == 'feright car':
                    labels.append(5)
                else:
                    labels.append(0)  # Unknown class or ignore
                    print(f"Warning: Unknown class '{name}' in {label_path}")

        return boxes, labels

    def __getitem__(self, idx):
        img_file, label_file = self.filtered_files[idx]
        img_path = os.path.join(self.img_folder, img_file)
        label_path = os.path.join(self.label_folder, label_file)

        img = Image.open(img_path).convert("RGB")

        # Crop the image to remove the 100-pixel border on all sides (top, bottom, left, right)
        width, height = img.size
        img = img.crop((100, 100, width - 100, height - 100))  # Adjust crop dynamically

        boxes, labels = self.parse_annotation(label_path)

        # Convert boxes and labels to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        # Apply transformations (make sure ToTensor is included)
        if self.transform:
            img = self.transform(img)

        return img, target


# Define the model
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
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
    transforms.ToTensor(),  # Convert PIL Image to PyTorch Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize (for RGB images)
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
model.to(device)

# Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

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

    # Validation phase (no loss calculation, just prediction and metrics)
    model.eval()
    all_image_metrics = []  # Store precision, recall, f1 per image

    with torch.no_grad():
        for images, targets in val_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass for predictions (no loss)
            outputs = model(images)

            # Collect predictions and labels for metrics
            for i, output in enumerate(outputs):
                pred_labels = output['labels'].cpu().numpy()
                true_labels = targets[i]['labels'].cpu().numpy()

                if len(true_labels) > 0:  # Only compute metrics for non-empty ground truth
                    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels,
                                                                               average='macro', zero_division=0)
                    all_image_metrics.append((precision, recall, f1))

    # Calculate average precision, recall, and F1 over all images
    if all_image_metrics:
        avg_precision = sum(x[0] for x in all_image_metrics) / len(all_image_metrics)
        avg_recall = sum(x[1] for x in all_image_metrics) / len(all_image_metrics)
        avg_f1 = sum(x[2] for x in all_image_metrics) / len(all_image_metrics)

        print(f"Validation - Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1 Score: {avg_f1:.4f}")
    else:
        print(f"No valid predictions for validation metrics in epoch {epoch + 1}")

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
