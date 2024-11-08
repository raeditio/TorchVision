from app import YoloDataset, train_one_epoch, validate_one_epoch, test_model, predict_and_read_value, display_image_with_prediction
import torch
from torch import load
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

def update_model(model, optimizer, train_loader, valid_loader, device, num_epochs):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        print(f"Train Loss: {train_loss:.4f}")
        
        valid_loss = validate_one_epoch(model, valid_loader, device)
        print(f"Validation Loss: {valid_loss:.4f}")
    return model

if __name__ == "__main__":
    model = load("faster_rcnn_model.pth")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = model.to(device)
    
    # Load dataset
    train_dataset = YoloDataset("dataset2/train", transform=ToTensor.compose([ToTensor()]))
    valid_dataset = YoloDataset("dataset2/valid", transform=ToTensor.compose([ToTensor()]))
    test_dataset = YoloDataset("dataset2/test", transform=ToTensor.compose([ToTensor()]))
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=train_dataset.collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, collate_fn=valid_dataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=test_dataset.collate_fn)
    
    