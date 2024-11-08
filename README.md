# TorchVision
This project aims to fine-tune the R-CNN model on PyTorch to detect 7-segment readings from a digital screen.

This is a recreation of a previous project which I created for the same purpose using Tensorflow API.

All images in this training were sourced from Roboflow, and the power of their API's were elevated to do most preprocessing and augmentation. All iamges were also 
resized to 640 x 640 for a faster training.

At around 100 epochs, the results were nearly optimized, extracting somewhat useful readings from the data. Yet, the model continued to have issues, such as 
failing to recognize negative signs or failing to distinguish between zeros and eights.

- The optimizer algorithm was changed: SGD -> Adam
- The learning rate was therefore adjusted: 0.005 -> 0.001.
- The momentum (betas) were defaulted, as well as the weight decays
- Epoch increased the 1000.