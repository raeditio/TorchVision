import torch
from torch import load
from torchvision.transforms import ToTensor, Compose
import shutil
from train import predict_and_read_value, display_image_with_prediction, apply_nms, YoloDataset

if __name__ == "__main__":    
    # load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load("models/faster_rcnn_model.pth")
    model = model.to(device)
    
    test_data = YoloDataset("dataset/test", transform=Compose([ToTensor()]))
    
    # Example of using the inference function and displaying the result
    sample_image, _ = test_data[10]
    sample_image = sample_image.unsqueeze(0)  # Add batch dimension for model input
    predicted_value = predict_and_read_value(model, sample_image, device)

    # Run inference to get boxes and labels
    with torch.no_grad():
        predictions = model([sample_image.squeeze(0).to(device)])[0]  # Get predictions for a single image
        # Apply NMS and keep only one box per label
        boxes, scores, labels = apply_nms(predictions, iou_threshold=0.5, score_threshold=0.5)

        # Convert tensors to numpy arrays for visualization
        boxes = boxes.cpu().numpy()
        labels = labels.cpu().numpy()

    # Display the image with overlaid boxes and predicted value
    display_image_with_prediction(sample_image, predicted_value, boxes, labels)
