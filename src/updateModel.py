from train import YoloDataset, train_one_epoch, validate_one_epoch, test_model, apply_nms, predict_and_read_value, display_image_with_prediction
import torch
from torch import load
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import ToTensor, Compose
import os
import shutil

def create_backup(file_path):
    base_name = os.path.basename(file_path)
    dir_name = os.path.dirname(file_path)
    bak_file = os.path.join(dir_name, f"{base_name}.bak")
    
    # Check if .bak file already exists, and if so, find the next available numbered backup.
    if os.path.exists(bak_file):
        index = 1
        while os.path.exists(os.path.join(dir_name, f"{base_name}({index}).bak")):
            index += 1
        bak_file = os.path.join(dir_name, f"{base_name}({index}).bak")
    
    # Use shutil to copy the file
    shutil.copy2(file_path, bak_file)
    print(f"Backup created at {bak_file}")

# Usage example
# create_backup('path_to_your_file_here') # Replace with actual path

def update_model(model, optimizer, train_loader, valid_loader, device, num_epochs):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        print(f"Train Loss: {train_loss:.4f}")
        
        valid_loss = validate_one_epoch(model, valid_loader, device)
        print(f"Validation Loss: {valid_loss:.4f}")
    return model

if __name__ == "__main__":
    # Load the model
    model = load("models/faster_rcnn_model3.pth")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = model.to(device)
    
    # Load dataset
    train_data = YoloDataset("dataset3/train", transform=Compose([ToTensor()]))
    valid_data = YoloDataset("dataset3/valid", transform=Compose([ToTensor()]))
    test_data = YoloDataset("dataset3/test", transform=Compose([ToTensor()]))
    
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True, pin_memory=True, collate_fn=lambda x: tuple(zip(*x)))
    valid_loader = DataLoader(valid_data, batch_size=8, shuffle=False, pin_memory=True, collate_fn=lambda x: tuple(zip(*x)))
    test_loader = DataLoader(test_data, batch_size=8, shuffle=False, pin_memory=True, collate_fn=lambda x: tuple(zip(*x)))
    
    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Previously SGD with lr=0.005 momentum=0.9, weight_decay=0.0005

    # Define a learning rate scheduler
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # Main training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate_one_epoch(model, valid_loader, device)

        # Step the scheduler
        scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        print("-" * 30)
        print("\n")

    # Final test
    test_model(model, test_loader, device)
    # Create a backup of current model and save the updated model
    # create_backup("models/faster_rcnn_model2.pth")
    torch.save(model, 'models/faster_rcnn_model4.pth')

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
    