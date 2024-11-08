import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import os

# Training, Validation, and Testing functions
def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    train_loss = 0
    total_batches = len(train_loader)

    for batch_idx, (images, targets) in enumerate(train_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        train_loss += losses.item()
        progress = (batch_idx + 1) / total_batches * 100
        print(f"Training Progress: {progress:.2f}%", end="\r")

    avg_train_loss = train_loss / total_batches
    return avg_train_loss

def validate_one_epoch(model, valid_loader, device):
    model.eval()
    val_loss = 0
    total_batches = len(valid_loader)

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(valid_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            if isinstance(loss_dict, dict):
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
                progress = (batch_idx + 1) / total_batches * 100
                print(f"Validation Progress: {progress:.2f}%", end="\r")

    avg_val_loss = val_loss / total_batches if total_batches > 0 else 0
    return avg_val_loss

def test_model(model, test_loader, device):
    model.eval()
    test_loss = 0
    total_batches = len(test_loader)

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            if isinstance(loss_dict, dict):
                losses = sum(loss for loss in loss_dict.values())
                test_loss += losses.item()
                progress = (batch_idx + 1) / total_batches * 100
                print(f"Testing Progress: {progress:.2f}%", end="\r")

    avg_test_loss = test_loss / total_batches if total_batches > 0 else 0
    print(f"Test Loss: {avg_test_loss:.4f}")
    return avg_test_loss

# Inference function to reconstruct the full value
def predict_and_read_value(model, image, device):
    model.eval()
    with torch.no_grad():
        image = image.squeeze(0).to(device)  # Remove batch dimension
        predictions = model([image])[0]
        boxes = predictions["boxes"].cpu().numpy()
        labels = predictions["labels"].cpu().numpy()
        
        sorted_indices = boxes[:, 0].argsort()
        sorted_labels = [labels[i] for i in sorted_indices]
        
        digit_map = {0: "-", 1: ".", 2: "0", 3: "1", 4: "2", 5: "3", 6: "4", 7: "5", 8: "6", 9: "7", 10: "8", 11: "9"}
        result = "".join([digit_map[label] for label in sorted_labels])
        return result

# Display the image and predicted value for comparison
def display_image_with_prediction(image_tensor, predicted_value, boxes, labels):
    # Convert the image tensor to a format that matplotlib can display
    image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # Remove batch dim and permute to [H, W, C]
    
    # Set up the figure and display the image
    plt.figure(figsize=(8, 8))
    plt.imshow(image_np)
    plt.title(f"Predicted Value: {predicted_value}", fontsize=16)
    
    # Define a color and font for boxes and labels
    color = "red"
    fontdict = {'fontsize': 12, 'color': 'white', 'weight': 'bold'}
    
    # Plot each bounding box
    for box, label in zip(boxes, labels):
        # Unpack the box coordinates
        xmin, ymin, xmax, ymax = box
        width, height = xmax - xmin, ymax - ymin
        
        # Create a rectangle for the bounding box
        rect = plt.Rectangle((xmin, ymin), width, height, edgecolor=color, facecolor="none", linewidth=2)
        plt.gca().add_patch(rect)
        
        # Add label text near the top-left corner of the bounding box
        plt.text(xmin, ymin - 5, f"{label}", fontdict=fontdict, bbox=dict(facecolor=color, alpha=0.5, pad=2))
    
    # Remove axes and show the plot
    plt.axis("off")
    plt.show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Custom Dataset Class for YOLO Format
class YoloDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_files = [f for f in os.listdir(os.path.join(root, 'images')) if f.endswith('.jpg') or f.endswith('.png')]
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root, 'images', img_name)
        label_path = os.path.join(self.root, 'labels', img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # Load annotations
        boxes = []
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                class_id, cx, cy, w, h = map(float, line.split())
                xmin = (cx - w / 2) * image.width
                ymin = (cy - h / 2) * image.height
                xmax = (cx + w / 2) * image.width
                ymax = (cy + h / 2) * image.height
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(int(class_id))

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}
        
        if self.transform:
            image = self.transform(image)
        
        return image, target

if __name__ == "__main__":
    # Define transformations
    transform = T.Compose([
        T.ToTensor()
    ])

    # Load dataset
    train_data = YoloDataset(root="dataset/train", transform=transform)
    valid_data = YoloDataset(root="dataset/valid", transform=transform)
    test_data = YoloDataset(root="dataset/test", transform=transform)

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    valid_loader = DataLoader(valid_data, batch_size=16, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    # Define the Faster R-CNN model with 12 classes + background
    num_classes = 12 + 1  # 12 classes + background
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model = model.to(device)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Consider using SGD with momentum=0.9, weight_decay=0.0005

    # Main training loop
    num_epochs = 1000
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        avg_train_loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"\nTraining Loss: {avg_train_loss:.4f}")
        avg_val_loss = validate_one_epoch(model, valid_loader, device)
        print(f"\nValidation Loss: {avg_val_loss:.4f}")
        print("-" * 30)

    # Final test
    test_model(model, test_loader, device)
    torch.save(model, './models/faster_rcnn_model.pth')

    # Example of using the inference function and displaying the result
    sample_image, _ = test_data[0]
    sample_image = sample_image.unsqueeze(0)  # Add batch dimension for model input
    predicted_value = predict_and_read_value(model, sample_image, device)

    # Run inference to get boxes and labels
    model.eval()
    with torch.no_grad():
        predictions = model([sample_image.squeeze(0).to(device)])[0]  # Get predictions for a single image
        boxes = predictions["boxes"].cpu().numpy()
        labels = predictions["labels"].cpu().numpy()


    # Display the image with overlaid boxes and predicted value
    display_image_with_prediction(sample_image, predicted_value, boxes, labels)
