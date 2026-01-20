# PyTorch Brain Tumor Classification via Attention Mechanisms

This project implements a Deep Learning pipeline using PyTorch to classify brain MRI scans into four distinct categories. It features a custom architecture combining Convolutional Neural Networks (CNNs) with Spatial and Channel Attention modules.

## Dataset
The model is trained on the Brain Tumor MRI Dataset, which includes:
- **Glioma**
- **Meningioma**
- **No Tumor**
- **Pituitary**

## Model Architecture
The architecture is built from scratch to include:
1. **Base Convolutional Layers**: Standard feature extraction using `nn.Conv2d`.
2. **Channel Attention**: A Squeeze-and-Excitation style block that adaptively recalibrates channel-wise feature responses.
3. **Spatial Attention**: A module that focuses on "where" the informative part of the image is located.
4. **Classifier Head**: Fully connected layers (`nn.Linear`) that map the extracted features to the four output classes.



## Implementation Details

### Data Preprocessing
Images are processed using `torchvision.transforms`:
- Resizing to 224x224 pixels.
- Conversion to Tensors.
- Normalization using ImageNet statistics (Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]).

### Training Configuration
- **Loss Function**: CrossEntropyLoss for multi-class classification.
- **Optimizer**: Adam Optimizer.
- **Device**: Configured to run on CUDA (GPU) for accelerated training.

## Usage

### Training
The training loop handles:
1. Forward pass of batches through the model.
2. Loss calculation and backpropagation.
3. Parameter updates via the Adam optimizer.
4. Validation at the end of each epoch to track accuracy and prevent overfitting.

### Visualization
The project utilizes `matplotlib` for plotting:
- Training vs. Validation Loss.
- Training vs. Validation Accuracy.

## Results and Interpretation
The model uses Class Activation Mapping (CAM) to provide interpretability, highlighting the specific regions of the MRI scan that influenced the model's prediction.



## Dependencies
- torch
- torchvision
- numpy
- matplotlib
- pandas
- torchcam (optional for visualization)
