# PyTorch: From Scratch to Advanced Architectures

A comprehensive series of lectures and implementations focused on building Deep Learning models from the ground up using PyTorch.

## Curriculum Overview

### Phase 1: Foundations
* **Lecture 1: Tensor Basics**
    * Creation, manipulation, and memory management.
    * Understanding the difference between `.view()` and `.reshape()`.
* **Lecture 2: Autograd Engine**
    * The computational graph and `requires_grad`.
    * Manual backpropagation using `.backward()`.
* **Lecture 3: Linear Regression from Scratch**
    * Implementing $y = wx + b$ without `nn.Module`.
    * Manual gradient descent and weight updates.



### Phase 2: Neural Network Modules
* **Lecture 4: The PyTorch Workflow**
    * Using `nn.Module`, `nn.Linear`, and `torch.optim`.
    * Transitioning from manual math to automated optimizers.
* **Lecture 5: Non-Linearity & Classification**
    * Introduction to Activation Functions (ReLU, Sigmoid).
    * Binary Cross-Entropy (BCE) loss.
* **Lecture 6: Data Pipelines**
    * Building custom `Dataset` and `DataLoader` classes.
    * Batching, shuffling, and epoch management.



### Phase 3: Computer Vision & Transfer Learning
* **Lecture 7: Convolutional Neural Networks (CNN)**
    * Spatial features via `nn.Conv2d` and `nn.MaxPool2d`.
    * Building the TinyVGG architecture.
* **Lecture 8: Transfer Learning**
    * Freezing backbones and modifying classifier heads.
    * Using pre-trained weights from ImageNet.
* **Lecture 9: Model Serialization**
    * Saving and loading using `state_dict`.
    * Deployment and inference mode.



### Phase 4: Sequence Modeling & Advanced Topics
* **Lecture 10: Natural Language Processing (NLP)**
    * Tokenization and Word Embeddings.
    * Recurrent Neural Networks (RNN) and Hidden States.
* **Lecture 11: Transformers**
    * Self-Attention and Multi-Head Attention.
    * The shift from sequential to parallel processing.



## Implementation: Brain Tumor Classification
The series concludes with a real-world application based on MRI medical imaging, implementing:
- **Spatial Attention**: Focusing on localized tumor regions.
- **Channel Attention**: Weighting feature maps based on importance.
- **Interpretability**: Using Class Activation Maps (CAM) to visualize model decisions.

## Technical Requirements
- PyTorch
- Torchvision
- NumPy
- Matplotlib
