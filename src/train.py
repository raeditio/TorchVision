import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import os

digit_map = {0: "-", 1: ".", 2: "0", 3: "1", 4: "2", 5: "3", 6: "4", 7: "5", 8: "6", 9: "7", 10: "8", 11: "9"}

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

def apply_nms(predictions, iou_threshold=0.3, score_threshold=0.5):
    # Extract boxes, scores, and labels from predictions
    boxes = predictions['boxes']
    scores = predictions['scores']
    labels = predictions['labels']
    
    # Filter out low-confidence boxes
    keep = scores >= score_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    
    # Apply NMS to remove overlapping boxes
    indices = torchvision.ops.nms(boxes, scores, iou_threshold)
    
    # Keep only boxes that passed NMS
    boxes = boxes[indices]
    scores = scores[indices]
    labels = labels[indices]
    
    # Keep the highest confidence box for each unique label
    unique_labels = torch.unique(labels)
    final_boxes = []
    final_scores = []
    final_labels = []
    
    for label in unique_labels:
        # Select all boxes with the current label
        label_mask = (labels == label)
        label_boxes = boxes[label_mask]
        label_scores = scores[label_mask]
        
        # Find the index of the highest scoring box for this label
        max_score_index = label_scores.argmax()
        final_boxes.append(label_boxes[max_score_index])
        final_scores.append(label_scores[max_score_index])
        final_labels.append(label)

    # Convert lists back to tensors
    final_boxes = torch.stack(final_boxes)
    final_scores = torch.stack(final_scores)
    final_labels = torch.tensor(final_labels)

    return final_boxes, final_scores, final_labels



# Inference function to reconstruct the full value
def predict_and_read_value(model, image, device):
    model.eval()
    with torch.no_grad():
        image = image.squeeze(0).to(device)  # Remove batch dimension
        predictions = model([image])[0]
        boxes = predictions["boxes"].cpu().numpy()
        labels = predictions["labels"].cpu().numpy()
        
        # Apply NMS to remove duplicate boxes
        boxes, scores, labels = apply_nms(predictions, 0.5, 0.5)
        
        sorted_indices = boxes[:, 0].argsort()
        sorted_labels = [labels[i] for i in sorted_indices]
        
        result = "".join([digit_map[label.item()] for label in sorted_labels])
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
    
    # Plot each bounding box and label only once
    for box, label in zip(boxes, labels):
        # Unpack the box coordinates
        xmin, ymin, xmax, ymax = box
        width, height = xmax - xmin, ymax - ymin
        
        # Create a rectangle for the bounding box
        rect = plt.Rectangle((xmin, ymin), width, height, edgecolor=color, facecolor="none", linewidth=2)
        plt.gca().add_patch(rect)
        
        # Add label text near the top-left corner of the bounding box
        plt.text(xmin, ymin - 5, f"{digit_map[label.item()]}", fontdict=fontdict, bbox=dict(facecolor=color, alpha=0.5, pad=2))
    
    # Remove axes and show the plot
    plt.axis("off")
    plt.show()


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define transformations
    transform = T.Compose([
        T.ToTensor()
    ])

    # Load dataset
    train_data = YoloDataset(root="dataset/train", transform=transform)
    valid_data = YoloDataset(root="dataset/valid", transform=transform)
    test_data = YoloDataset(root="dataset/test", transform=transform)

    train_loader = DataLoader(train_data, batch_size=8, shuffle=True, pin_memory=True, collate_fn=lambda x: tuple(zip(*x)))
    valid_loader = DataLoader(valid_data, batch_size=8, shuffle=False, pin_memory=True, collate_fn=lambda x: tuple(zip(*x)))
    test_loader = DataLoader(test_data, batch_size=8, shuffle=False, pin_memory=True, collate_fn=lambda x: tuple(zip(*x)))

    # Define the Faster R-CNN model with 12 classes + background
    num_classes = 12 + 1  # 12 classes + background
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model = model.to(device)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Consider using SGD with momentum=0.9, weight_decay=0.0005

    # Define a learning rate scheduler
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # Main training loop
    num_epochs = 200
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate_one_epoch(model, valid_loader, device)

        # Step the scheduler
        # scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        print("-" * 30)
        print("\n")

    # Final test
    test_model(model, test_loader, device)
    torch.save(model, './models/faster_rcnn_model.pth')

    # Example of using the inference function and displaying the result
    sample_image, _ = test_data[0]
    sample_image = sample_image.unsqueeze(0)  # Add batch dimension for model input
    predicted_value = predict_and_read_value(model, sample_image, device)

    # Run inference to get boxes and labels
    with torch.no_grad():
        predictions = model([sample_image.squeeze(0).to(device)])[0]  # Get predictions for a single image
        # Apply NMS and keep only one box per label
        boxes, scores, labels = apply_nms(predictions, iou_threshold=0.3, score_threshold=0.5)

        # Convert tensors to numpy arrays for visualization
        boxes = boxes.cpu().numpy()
        labels = labels.cpu().numpy()

    # Display the image with overlaid boxes and predicted value
    display_image_with_prediction(sample_image, predicted_value, boxes, labels)
