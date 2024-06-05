# CNN Project for Image Classification with PyTorch Lightning

This project covers image classification using deep learning. We will train and evaluate three different categories of networks on the Imagenette dataset and also perform transfer learning using the CIFAR10 dataset. The networks will be implemented using PyTorch Lightning.

## Project Structure

The project is divided into the following tasks:

1. **Basic CNN**:

   - Train a basic Convolutional Neural Network (CNN) with convolutional layers followed by fully connected layers.
   - Implement early stopping to prevent overfitting.
   - Report the chosen architecture, training loss, validation loss, and final test accuracy.

2. **All Convolutional Network**:

   - Create and train an all-convolutional network.
   - Compare the number of parameters with the basic CNN.
   - Implement early stopping to prevent overfitting.
   - Report the chosen architecture, training loss, validation loss, and final test accuracy.

3. **Regularization**:

   - Add regularization to one of the models from the previous sections using data augmentation or dropout.
   - Train the model until convergence.
   - Compare the model with and without regularization.
   - Report the chosen regularization technique and its impact on model performance.

4. **Transfer Learning**:
   - Use a pre-trained model on the Imagenette dataset and fine-tune it on the CIFAR10 dataset.
   - Compare the model trained from scratch on CIFAR10 with the pre-trained and fine-tuned model.
   - Report the training plots and final accuracy of the models.

## Prerequisites

- Python 3.6 or higher
- PyTorch
- PyTorch Lightning
- torchvision

You can install the required libraries using the following command:

```sh
pip install torch torchvision pytorch-lightning
```
