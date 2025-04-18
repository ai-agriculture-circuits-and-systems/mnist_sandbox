# MNIST Classification with PyTorch

This project implements a comprehensive deep learning framework for training and evaluating various neural network architectures on the MNIST dataset. The framework is designed to be modular and extensible, supporting a wide range of modern deep learning models.

## Project Structure

```
.
├── data/                  # MNIST dataset files
├── models/               # Model implementations
│   ├── architectures/   # Various model architectures
│   ├── base_model.py    # Base model class
│   ├── model_factory.py # Model factory for easy instantiation
│   └── logger.py        # Logging utilities
├── utils/               # Utility classes
│   ├── data_loader.py   # Data loading utilities
│   ├── trainer.py       # Training utilities
│   └── evaluator.py     # Evaluation utilities
├── main.py              # Main script
├── requirements.txt     # Project dependencies
└── README.md           # This file
```

## Supported Models

The framework currently supports the following architectures:

### Classification Models
- **Simple CNN**: Basic convolutional neural network
- **AlexNet**: Classic CNN architecture
- **VGG**: Very Deep Convolutional Networks
- **ResNet**: Residual Networks
- **DenseNet**: Densely Connected Convolutional Networks
- **MobileNet**: Lightweight CNN for mobile devices
- **EfficientNet**: Efficient and accurate scaling
- **SqueezeNet**: Lightweight CNN architecture
- **Xception**: Extreme Inception
- **Vision Transformer (ViT)**: Transformer-based vision model
- **MLP**: Multi-Layer Perceptron

### Generative Models
- **Autoencoder**: For unsupervised learning and dimensionality reduction
- **GAN**: Generative Adversarial Network

### Language Models
- **BERT**: Bidirectional Encoder Representations from Transformers
- **GPT**: Generative Pre-trained Transformer
- **RNN**: Recurrent Neural Network

## Setup

1. Install the required dependencies:
```bash
pip download --no-cache-dir -r requirements.txt -d wheels
pip install --no-index --find-links=wheels -r requirements.txt
```

2. Place your MNIST dataset files in the `data/` directory:
   - `MNISTtrain.mat`
   - `MNISTtest.mat`

## Usage

Run the training script:
```bash
python main.py
```

The script will:
1. Load and preprocess the MNIST dataset
2. Train the selected model for the specified number of epochs
3. Save the best model checkpoint to `outputs/best_model.pth`
4. Generate evaluation metrics and visualizations in the `outputs/` directory

## Adding New Models

To add a new model:

1. Create a new file in the `models/architectures/` directory
2. Implement your model class inheriting from `BaseModel`
3. Register your model in `models/model_factory.py`
4. Import and use your model through the model factory

## Features

- Modular architecture for easy extension
- Model factory pattern for flexible model instantiation
- Separate classes for training and evaluation
- Progress bars for training and evaluation
- Automatic model checkpointing
- Comprehensive evaluation metrics and visualizations
- GPU support when available
- Extensive logging and monitoring capabilities
