# MNIST Classification with PyTorch

This project implements a comprehensive deep learning framework for training and evaluating various neural network architectures on the MNIST dataset. The framework is designed to be modular and extensible, supporting a wide range of modern deep learning models.

## Project Structure

```
.
├── data/                # MNIST dataset files
├── models/              # Model implementations
│   ├── architectures/   # Various model architectures
│   ├── base_model.py    # Base model class
│   ├── model_factory.py # Model factory for easy instantiation
│   └── logger.py        # Logging utilities
├── utils/               # Training code and utilities
│   ├── main.py          # Main training entry point
│   ├── regression.py    # NAS-style regression across all models
│   ├── data_loader.py   # Data loading utilities
│   ├── trainer.py       # Training utilities
│   └── evaluator.py     # Evaluation utilities
├── scripts/             # Shell wrappers and git submodule helpers
│   ├── run.sh           # Single-model training wrapper
│   ├── regression.sh    # Regression suite wrapper
│   ├── test.sh          # Smoke-test all registered models
│   ├── init_submodules.sh
│   ├── add_submodule.sh
│   ├── remove_submodule.sh
│   └── update_submodules.sh
├── outputs/             # Checkpoints, logs, regression artifacts
├── requirements.txt     # Project dependencies
└── README.md            # This file
```

## Supported Models

The project ships with **58 models** registered in `models/model_factory.py` (`MODEL_REGISTRY`). Lists below are the CLI names passed to `--model` / `-m`.

Print the live list anytime:

```bash
python -c "from models.model_factory import ModelFactory; print('\n'.join(ModelFactory.get_available_models()))"
```

### Classification models (50)

| CLI name | Description | Default input |
|----------|-------------|---------------|
| `mlp` | Multi-layer perceptron | 28×28 |
| `simple_cnn` | Configurable shallow CNN | 28×28 |
| `lenet` | LeNet-5 style CNN | 28×28 |
| `alexnet` | AlexNet | 224×224 |
| `vgg` | VGG | 224×224 |
| `resnet` | ResNet-18 style | 28×28 |
| `wide_resnet` | WideResNet | 224×224 |
| `resnext` | ResNeXt (grouped convolutions) | 28×28 |
| `res2net` | Res2Net (multi-scale blocks) | 28×28 |
| `densenet` | DenseNet | 28×28 |
| `dpn` | Dual Path Network | 28×28 |
| `inception_resnet` | Inception-ResNet v2 style | 224×224 |
| `googlenet` | GoogLeNet / Inception | 224×224 |
| `nin` | Network in Network | 224×224 |
| `mobilenet` | MobileNet v1 | 28×28 |
| `mobilenetv2` | MobileNet v2 | 28×28 |
| `mobilenetv3` | MobileNet v3 | 28×28 |
| `mnasnet` | MNASNet | 28×28 |
| `efficientnet` | EfficientNet | 224×224 |
| `efficientnetv2` | EfficientNetV2-style MBConv | 28×28 |
| `ghostnet` | GhostNet | 28×28 |
| `repghost` | RepGhost | 28×28 |
| `squeezenet` | SqueezeNet | 28×28 |
| `xception` | Xception | 224×224 |
| `shufflenet` | ShuffleNet | 28×28 |
| `regnet` | RegNet | 28×28 |
| `repvgg` | RepVGG | 28×28 |
| `convnext` | ConvNeXt | 224×224 |
| `darknet` | Darknet / YOLO-style stacks | 28×28 |
| `cspnet` | CSPNet | 28×28 |
| `hardnet` | HarDNet | 28×28 |
| `hrnet` | HRNet (multi-resolution) | 28×28 |
| `lcnet` | PP-LCNet style | 28×28 |
| `se_resnet` | ResNet + squeeze-and-excitation | 28×28 |
| `cbam_resnet` | ResNet + CBAM attention | 28×28 |
| `eca_resnet` | ResNet + ECA attention | 28×28 |
| `coord_resnet` | ResNet + coordinate attention | 28×28 |
| `sknet` | SKNet (selective kernel) | 28×28 |
| `van` | Visual Attention Network | 28×28 |
| `poolformer` | PoolFormer (MetaFormer) | 28×28 |
| `capsnet` | Capsule Network | 28×28 |
| `vit` | Vision Transformer | 224×224 |
| `deit` | DeiT (distillation tokens) | 224×224 |
| `swin_tiny` | Swin Transformer (tiny) | 224×224 |
| `coatnet` | CoAtNet (conv + attention) | 224×224 |
| `vim_tiny` | ViM-style SSM-inspired blocks | 224×224 |
| `bert` | BERT on flattened pixels | 28×28 |
| `gpt` | GPT on flattened pixels | 28×28 |
| `lstm` | LSTM sequence model | 28×28 |
| `gru` | GRU sequence model | 28×28 |

Models marked **224×224** are listed in `LARGE_IMAGE_MODELS` in `model_factory.py`; the regression suite upsamples MNIST to 224 for those names. All other classifiers use native **28×28** input.

### Autoencoder models (4)

| CLI name | Description |
|----------|-------------|
| `simple_ae` | Fully connected autoencoder |
| `conv_ae` | Convolutional autoencoder |
| `vae` | Variational autoencoder |
| `denoising_ae` | Denoising autoencoder |

### GAN models (4)

| CLI name | Description |
|----------|-------------|
| `vanilla_gan` | Vanilla GAN |
| `dcgan` | Deep Convolutional GAN |
| `wgan` | Wasserstein GAN |
| `cgan` | Conditional GAN |

### Quick reference by category

| Category | Count | Examples |
|----------|-------|----------|
| Classical / CNN | 12 | `lenet`, `alexnet`, `vgg`, `resnet`, `densenet` |
| Lightweight | 10 | `mobilenet`, `mobilenetv2`, `mobilenetv3`, `ghostnet`, `mnasnet`, `lcnet` |
| Attention-enhanced ResNet | 5 | `se_resnet`, `cbam_resnet`, `eca_resnet`, `coord_resnet`, `sknet` |
| Modern CNN / hybrid | 14 | `convnext`, `repvgg`, `cspnet`, `van`, `poolformer`, `coatnet` |
| Transformers / sequence | 7 | `vit`, `deit`, `swin_tiny`, `bert`, `gpt`, `lstm`, `gru` |
| Specialized | 3 | `capsnet`, `vim_tiny`, `inception_resnet` |
| Autoencoders | 4 | `simple_ae`, `conv_ae`, `vae`, `denoising_ae` |
| GANs | 4 | `vanilla_gan`, `dcgan`, `wgan`, `cgan` |

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

Train a single model (run from the project root):

```bash
./scripts/run.sh -m resnet --epochs 10
./scripts/run.sh -m vit --quick-test
python -m utils.main --model densenet --epochs 5
```

Run the full regression suite (all 58 models, loss × optimizer × NAS trials):

```bash
./scripts/regression.sh -q              # quick-test subset, parallel workers
./scripts/regression.sh -m resnet,deit -q # subset of models
./scripts/regression.sh -f              # full MNIST (long)
```

Smoke-test every registered model:

```bash
./scripts/test.sh
```

Each training run will:
1. Load and preprocess the MNIST dataset
2. Train the selected model for the specified number of epochs
3. Save the best model checkpoint under `outputs/`
4. Generate evaluation metrics and visualizations

## Adding New Models

To add a new model:

1. Create a new file in `models/architectures/`
2. Implement your model class inheriting from `BaseModel`
3. Add the class to `MODEL_REGISTRY` in `models/model_factory.py`
4. Add default kwargs in `utils/regression.py` → `build_model_kwargs()` if needed
5. If the model expects 224×224 input, add its CLI name to `LARGE_IMAGE_MODELS` in `model_factory.py`

`CLASSIFICATION_MODELS`, `AUTOENCODER_MODELS`, and `GAN_MODELS` are derived from the registry automatically.

## Features

- Modular architecture for easy extension
- Model factory pattern for flexible model instantiation
- Separate classes for training and evaluation
- Progress bars for training and evaluation
- Automatic model checkpointing
- Comprehensive evaluation metrics and visualizations
- GPU support when available
- Extensive logging and monitoring capabilities