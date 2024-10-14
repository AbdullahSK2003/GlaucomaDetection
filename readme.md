
# ImprovedGlaucomaNet: Glaucoma Detection with Neural Networks

This repository contains the implementation of **ImprovedGlaucomaNet**, a neural network model for glaucoma detection. The model leverages Residual Blocks, Dense Blocks, and SE (Squeeze-and-Excitation) Blocks for robust image classification between normal retinal images and glaucomatous retinal images. It also includes training and evaluation scripts.

## Contributions

We welcome contributions from the community! If you're interested in improving the model, fixing bugs, or adding new features, follow the guidelines below.

### How to Contribute

1. **Fork the repository**: 
   Start by forking this repository to your own GitHub account. This allows you to work on your copy of the project independently.

2. **Clone the forked repository**:
   Clone your forked repository to your local machine:
   ```bash
   git clone https://github.com/your-username/ImprovedGlaucomaNet.git
   cd ImprovedGlaucomaNet
   ```

3. **Create a new branch**:
   Always work on a new branch specific to the feature/bugfix you are working on:
   ```bash
   git checkout -b your-feature-branch
   ```

4. **Set up your environment**:
   Install the required dependencies to set up the environment. You can use `pip` or `conda` to install the dependencies listed in the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

5. **Make your changes**:
   Write clean and modular code. Follow the structure and style used in the repository. Be sure to:
   - Add comments where necessary.
   - Write tests if you're adding new functionality.

6. **Test your changes**:
   Ensure that your changes do not break any existing functionality. Run the existing tests and add new tests if applicable:
   ```bash
   pytest
   ```

7. **Commit your changes**:
   Commit your changes with clear and concise commit messages:
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

8. **Push to your fork**:
   Push your branch to your forked repository:
   ```bash
   git push origin your-feature-branch
   ```

9. **Create a Pull Request (PR)**:
   Go to the original repository and click on **Pull Requests**. Click on **New Pull Request** and follow the instructions to submit your changes for review. Be sure to provide:
   - A clear title for your PR.
   - A detailed description of the changes made and why they are necessary.

### Contribution Guidelines

- **Code Style**: Please follow the PEP8 coding style for Python.
- **Documentation**: Ensure your code is well-documented with docstrings, especially for new features or functions.
- **Testing**: Write tests for any new functionality and ensure existing tests pass.
- **Bug Fixes**: Clearly reference the issue you're addressing and describe the fix implemented.
- **New Features**: Clearly describe the new feature, its purpose, and why it should be included.

### Getting Help

If you encounter any issues while contributing or have questions, feel free to open an issue in the repository, or reach out via the discussion board.

---

Thank you for your interest in contributing to **ImprovedGlaucomaNet**! We look forward to collaborating with you.

# CNN Model for Image Classification

This repository contains a convolutional neural network (CNN) for binary image classification. The model architecture utilizes a series of convolutional, batch normalization, ReLU activation, and max-pooling layers, followed by fully connected (dense) layers to produce the final output.

## Model Architecture

The architecture of the model consists of several convolutional layers, each followed by batch normalization, ReLU activations, and max-pooling layers. The model is designed to process images with the following specifications:

- **Input shape**: `[batch_size, 3, 224, 224]`
- **Number of classes**: 2 (binary classification)

### Key Layers

1. **Convolutional Layers**: 
   - Multiple Conv2d layers with varying filter sizes, ranging from 64 to 1024 channels, progressively downsample the input image.
   - Kernel size: `3x3`
   - Stride: `1`
   - Padding: `1`

2. **Batch Normalization**: Batch normalization is applied after each convolutional layer to accelerate training and improve stability.

3. **ReLU Activation**: Non-linear ReLU activation is applied after each convolutional and fully connected layer.

4. **MaxPooling Layers**: 
   - MaxPool2d layers with a `2x2` kernel are used to downsample the feature maps after each block of convolutional layers.

5. **Fully Connected Layers**: 
   - The model has 3 fully connected (dense) layers, with a final output layer of size 2 for binary classification.

6. **Dropout**: Dropout layers with a probability of `0.5` are added to prevent overfitting.

7. **Adaptive Average Pooling**: Used to aggregate spatial dimensions before the fully connected layers.

### Summary of the Model

The detailed structure of the model is as follows:



# Model Architecture Summary

This file contains the architecture summary of a Convolutional Neural Network (CNN) model.

## Layer Details

| Layer (Type)          | Output Shape        | Param #   |
|-----------------------|---------------------|-----------|
| **Conv2d-1**          | [-1, 64, 112, 112]  | 9,472     |
| **BatchNorm2d-2**     | [-1, 64, 112, 112]  | 128       |
| **ReLU-3**            | [-1, 64, 112, 112]  | 0         |
| **MaxPool2d-4**       | [-1, 64, 56, 56]    | 0         |
| **Conv2d-5**          | [-1, 128, 56, 56]   | 73,856    |
| **BatchNorm2d-6**     | [-1, 128, 56, 56]   | 256       |
| **ReLU-7**            | [-1, 128, 56, 56]   | 0         |
| **Conv2d-8**          | [-1, 128, 56, 56]   | 147,584   |
| **BatchNorm2d-9**     | [-1, 128, 56, 56]   | 256       |
| **ReLU-10**           | [-1, 128, 56, 56]   | 0         |
| **Conv2d-11**         | [-1, 128, 56, 56]   | 147,584   |
| **BatchNorm2d-12**    | [-1, 128, 56, 56]   | 256       |
| **ReLU-13**           | [-1, 128, 56, 56]   | 0         |
| **Conv2d-14**         | [-1, 128, 56, 56]   | 147,584   |
| **BatchNorm2d-15**    | [-1, 128, 56, 56]   | 256       |
| **ReLU-16**           | [-1, 128, 56, 56]   | 0         |
| **MaxPool2d-17**      | [-1, 128, 28, 28]   | 0         |
| **Conv2d-18**         | [-1, 256, 28, 28]   | 295,168   |
| **BatchNorm2d-19**    | [-1, 256, 28, 28]   | 512       |
| **ReLU-20**           | [-1, 256, 28, 28]   | 0         |
| **Conv2d-21**         | [-1, 256, 28, 28]   | 590,080   |
| **BatchNorm2d-22**    | [-1, 256, 28, 28]   | 512       |
| **ReLU-23**           | [-1, 256, 28, 28]   | 0         |
| **Conv2d-24**         | [-1, 256, 28, 28]   | 590,080   |
| **BatchNorm2d-25**    | [-1, 256, 28, 28]   | 512       |
| **ReLU-26**           | [-1, 256, 28, 28]   | 0         |
| **Conv2d-27**         | [-1, 256, 28, 28]   | 590,080   |
| **BatchNorm2d-28**    | [-1, 256, 28, 28]   | 512       |
| **ReLU-29**           | [-1, 256, 28, 28]   | 0         |
| **MaxPool2d-30**      | [-1, 256, 14, 14]   | 0         |
| **Conv2d-31**         | [-1, 512, 14, 14]   | 1,180,160 |
| **BatchNorm2d-32**    | [-1, 512, 14, 14]   | 1,024     |
| **ReLU-33**           | [-1, 512, 14, 14]   | 0         |
| **Conv2d-34**         | [-1, 512, 14, 14]   | 2,359,808 |
| **BatchNorm2d-35**    | [-1, 512, 14, 14]   | 1,024     |
| **ReLU-36**           | [-1, 512, 14, 14]   | 0         |
| **Conv2d-37**         | [-1, 512, 14, 14]   | 2,359,808 |
| **BatchNorm2d-38**    | [-1, 512, 14, 14]   | 1,024     |
| **ReLU-39**           | [-1, 512, 14, 14]   | 0         |
| **Conv2d-40**         | [-1, 512, 14, 14]   | 2,359,808 |
| **BatchNorm2d-41**    | [-1, 512, 14, 14]   | 1,024     |
| **ReLU-42**           | [-1, 512, 14, 14]   | 0         |
| **MaxPool2d-43**      | [-1, 512, 7, 7]     | 0         |
| **Conv2d-44**         | [-1, 1024, 7, 7]    | 4,719,616 |
| **BatchNorm2d-45**    | [-1, 1024, 7, 7]    | 2,048     |
| **ReLU-46**           | [-1, 1024, 7, 7]    | 0         |
| **Conv2d-47**         | [-1, 1024, 7, 7]    | 9,438,208 |
| **BatchNorm2d-48**    | [-1, 1024, 7, 7]    | 2,048     |
| **ReLU-49**           | [-1, 1024, 7, 7]    | 0         |
| **Conv2d-50**         | [-1, 1024, 7, 7]    | 9,438,208 |
| **BatchNorm2d-51**    | [-1, 1024, 7, 7]    | 2,048     |
| **ReLU-52**           | [-1, 1024, 7, 7]    | 0         |
| **Conv2d-53**         | [-1, 1024, 7, 7]    | 9,438,208 |
| **BatchNorm2d-54**    | [-1, 1024, 7, 7]    | 2,048     |
| **ReLU-55**           | [-1, 1024, 7, 7]    | 0         |
| **MaxPool2d-56**      | [-1, 1024, 3, 3]    | 0         |
| **AdaptiveAvgPool2d-57** | [-1, 1024, 1, 1] | 0         |
| **Linear-58**         | [-1, 1024]          | 1,049,600 |
| **ReLU-59**           | [-1, 1024]          | 0         |
| **Dropout-60**        | [-1, 1024]          | 0         |
| **Linear-61**         | [-1, 512]           | 524,800   |
| **ReLU-62**           | [-1, 512]           | 0         |
| **Dropout-63**        | [-1, 512]           | 0         |
| **Linear-64**         | [-1, 2]             | 1,026     |

## Model Statistics

- **Total Parameters**: 45,476,226  
- **Trainable Parameters**: 45,476,226  
- **Non-trainable Parameters**: 0  

### Memory Consumption

- **Input size**: 0.57 MB
- **Forward/backward pass size**: 90.27 MB
- **Parameters size**: 173.48 MB
- **Estimated Total Size**: 264.32 MB



<!-- export HSA_OVERRIDE_GFX_VERSION=10.3.0 -->