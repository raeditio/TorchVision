# TorchVision
This project aims to fine-tune the R-CNN model on PyTorch to detect 7-segment readings from a digital screen.

This is a recreation of a previous project which I created for the same purpose using Tensorflow API.

All images in this training were sourced from Roboflow, and the power of their API's were elevated to do most preprocessing and augmentation. All iamges were also 
resized to 640 x 640 for a faster training.

At around 100 epochs, the results were nearly optimized, extracting somewhat useful readings from the data. Yet, the model continued to have issues, such as 
failing to recognize negative signs or failing to distinguish between zeros and eights.

- The optimizer algorithm was changed: SGD -> Adam
- The learning rate was therefore adjusted: 0.005 -> 0.001
- The momentum (betas) were defaulted, as well as the weight decays
- A learning rate step scheduler was introduced
- Epoch was increased to 200

The updateModel script additionally trains the saved model using a new dataset.
The app script was added to render the app on a new image.

As many deep learning projects, the importance of an abundance of high quality data is crucial to the successful training of the model.
In the latest update, the model was able to produce predictions of meaningful accuracy, but it struggled to detect negative signs due to the lack of its
representation in the dataset.

![Fails to detect negative sign](https://github.com/raeditio/Torchvision/blob/main/89.1.png?raw=true)

The model also struggled in cases of repeating digits.

![Fails to detect repeat](https://github.com/raeditio/Torchvision/blob/main/rep.png?raw=true)

In the future, the model is intended to be updated through a generated and auto-labeled dataset.