# TorchVision
This project aims to fine-tune the R-CNN model on PyTorch to detect 7-segment readings from a digital screen.

This is a recreation of a previous project that I created for the same purpose using Tensorflow API.

All images in this training were sourced from Roboflow, and the power of their APIs was elevated to do most preprocessing and augmentation. All images were also 
resized to 640 x 640 for faster training.

At around 100 epochs, the results were nearly optimized, extracting somewhat useful readings from the data. Yet, the model continued to have issues, such as 
failing to recognize negative signs or failing to distinguish between zeros and eights.

- The optimizer algorithm was changed: SGD -> Adam
- The learning rate was therefore adjusted: 0.005 -> 0.001
- The momentum (betas) were defaulted, as well as the weight decays
- A learning rate step scheduler was introduced
- Epoch was increased to 200

The updateModel script additionally trains the saved model using a new dataset.
The app script was added to render the app on a new image.

As with many deep learning projects, an abundance of high-quality data is crucial to the model's successful training.
In the latest update, the model produced predictions of meaningful accuracy. Still, it struggled to detect negative signs due to the lack of 
representation in the dataset.

<table>
  <tr>
    <td>
      <h3 style="text-align: center;">Fails to detect negative sign</h3>
      <img 
        src="https://github.com/raeditio/Torchvision/blob/main/appTest/89.1.png?raw=true" 
        alt="Negative sign" 
        width="400"
      />
    </td>
    <td>
      <h3 style="text-align: center;">Struggles with repeating digits</h3>
      <img
        src="https://github.com/raeditio/Torchvision/blob/main/appTest/rep.png?raw=true"
        alt="Repeat"
        width="400"
      />
    </td>
  </tr>
</table>


In the future, the model is intended to be updated through a generated and auto-labeled dataset.
