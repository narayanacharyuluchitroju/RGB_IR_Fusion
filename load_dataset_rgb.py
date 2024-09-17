import os
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np


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

            # Check for polygon
            polygon = obj.find('polygon')
            if polygon is not None:
                try:
                    # Extract polygon coordinates
                    x_coords = [int(polygon.find(f'x{i}').text) for i in range(1, 5)]
                    y_coords = [int(polygon.find(f'y{i}').text) for i in range(1, 5)]

                    # Adjust the bounding box coordinates to remove the 100-pixel border
                    x_coords = [x - 100 for x in x_coords]  # Remove left/right border
                    y_coords = [y - 100 for y in y_coords]  # Remove top/bottom border

                except AttributeError as e:
                    print(f"Warning: Missing x/y coordinates in {label_path}: {e}")
                    continue

                # Compute the bounding box (x_min, y_min, x_max, y_max)
                x_min = min(x_coords)
                y_min = min(y_coords)
                x_max = max(x_coords)
                y_max = max(y_coords)

            # Check for bndbox if polygon is not present
            elif obj.find('bndbox') is not None:
                bndbox = obj.find('bndbox')
                try:
                    x_min = int(bndbox.find('xmin').text) - 100
                    y_min = int(bndbox.find('ymin').text) - 100
                    x_max = int(bndbox.find('xmax').text) - 100
                    y_max = int(bndbox.find('ymax').text) - 100
                except AttributeError as e:
                    print(f"Warning: Missing xmin/xmax/ymin/ymax in {label_path}: {e}")
                    continue

            # Check for point if neither polygon nor bndbox is present
            elif obj.find('point') is not None:
                point = obj.find('point')
                try:
                    x = int(point.find('x').text) - 100
                    y = int(point.find('y').text) - 100

                    # Define a small bounding box around the point (e.g., 20x20 box)
                    box_size = 10
                    x_min = x - box_size
                    y_min = y - box_size
                    x_max = x + box_size
                    y_max = y + box_size

                except AttributeError as e:
                    print(f"Warning: Missing x/y coordinates in <point> in {label_path}: {e}")
                    continue

            else:
                print(f"Warning: No <polygon>, <bndbox>, or <point> found in {label_path}")
                continue

            # Append the bounding box to the list of boxes
            boxes.append([x_min, y_min, x_max, y_max])

            # Map label names to integers based on the five categories
            if name == 'car':
                labels.append(1)
            elif name == 'truck' or name == 'truvk':
                labels.append(2)
            elif name == 'bus':
                labels.append(3)
            elif name == 'van':
                labels.append(4)
            elif name == 'feright car' or name == 'feright_car' or name == 'feright':
                labels.append(5)
            else:
                labels.append(0)  # Unknown class or ignore
                print(f"Warning: Unknown class '{name}' in {label_path}")

        return boxes, labels

    def __getitem__(self, idx):
        img_file, label_file = self.filtered_files[idx]
        img_path = os.path.join(self.img_folder, img_file)
        label_path = os.path.join(self.label_folder, label_file)

        img = Image.open(img_path).convert("RGB")  # Convert IR images accordingly

        # Crop the image to remove the 100-pixel border on all sides (top, bottom, left, right)
        width, height = img.size
        img = img.crop((100, 100, width - 100, height - 100))  # Adjust crop dynamically

        boxes, labels = self.parse_annotation(label_path)

        # Convert boxes and labels to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Filter out invalid boxes (width/height should be > 0)
        valid_boxes = []
        valid_labels = []

        for box, label in zip(boxes, labels):
            xmin, ymin, xmax, ymax = box
            if xmax > xmin and ymax > ymin:
                valid_boxes.append([xmin, ymin, xmax, ymax])
                valid_labels.append(label)

        # Handle case where all boxes are invalid
        if len(valid_boxes) == 0:
            valid_boxes = torch.zeros((0, 4), dtype=torch.float32)
            valid_labels = torch.tensor([], dtype=torch.int64)
        else:
            valid_boxes = torch.as_tensor(valid_boxes, dtype=torch.float32)
            valid_labels = torch.as_tensor(valid_labels, dtype=torch.int64)

        target = {}
        target["boxes"] = valid_boxes
        target["labels"] = valid_labels

        # Apply any transformations (including ToTensor)
        if self.transform:
            img = self.transform(img)

        return img, target
