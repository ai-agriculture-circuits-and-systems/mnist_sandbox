from .architectures.alexnet import AlexNet
from .architectures.simple_cnn import SimpleCNN
from .architectures.vgg import VGG
from .architectures.resnet import ResNet
from .architectures.densenet import DenseNet
from .architectures.mobilenet import MobileNet
from .architectures.mlp import MLP
from .architectures.bert import BERTMNIST
from .architectures.gpt import GPTMNIST
from .architectures.rnn import LSTMMNIST, GRUMNIST
from .architectures.gan import VanillaGAN, DCGAN, WGAN, CGAN
from .architectures.autoencoder import SimpleAutoencoder, ConvolutionalAutoencoder, VariationalAutoencoder, DenoisingAutoencoder
from .architectures.squeezenet import SqueezeNet
from .architectures.efficientnet import EfficientNet
from .architectures.xception import Xception
from .architectures.vit import VisionTransformer
from .architectures.lenet import LeNet
from .architectures.nin import NiN
from .architectures.googlenet import GoogLeNet
from .architectures.shufflenet import ShuffleNet
from .architectures.se_resnet import SEResNet
from .architectures.wide_resnet import WideResNet
from .architectures.convnext import ConvNeXt
from .architectures.repvgg import RepVGG
from .architectures.regnet import RegNet
from .architectures.ghostnet import GhostNet
from .architectures.resnext import ResNeXt
from .architectures.res2net import Res2Net
from .architectures.cbam_resnet import CBAMResNet
from .architectures.mobilenetv3 import MobileNetV3
from .architectures.mnasnet import MNASNet
from .architectures.eca_resnet import ECAResNet
from .architectures.sknet import SKNet
from .architectures.dpn import DPN
from .architectures.lcnet import LCNet
from .architectures.capsnet import CapsNet
from .architectures.coord_resnet import CoordResNet
from .architectures.hardnet import HarDNet
from .architectures.cspnet import CSPNet
from .architectures.van import VAN
from .architectures.poolformer import PoolFormer
from .architectures.darknet import Darknet
from .architectures.inception_resnet import InceptionResNet
from .architectures.repghost import RepGhost
from .architectures.hrnet import HRNet
from .architectures.swin_tiny import SwinTiny
from .architectures.mobilenetv2 import MobileNetV2
from .architectures.efficientnetv2 import EfficientNetV2
from .architectures.deit import DeiT
from .architectures.coatnet import CoAtNet
from .architectures.vim_tiny import VimTiny
import os

# Single registry — add new architectures here only.
MODEL_REGISTRY = {
    'alexnet': AlexNet,
    'simple_cnn': SimpleCNN,
    'vgg': VGG,
    'resnet': ResNet,
    'densenet': DenseNet,
    'mobilenet': MobileNet,
    'mobilenetv2': MobileNetV2,
    'mlp': MLP,
    'bert': BERTMNIST,
    'gpt': GPTMNIST,
    'lstm': LSTMMNIST,
    'gru': GRUMNIST,
    'vanilla_gan': VanillaGAN,
    'dcgan': DCGAN,
    'wgan': WGAN,
    'cgan': CGAN,
    'simple_ae': SimpleAutoencoder,
    'conv_ae': ConvolutionalAutoencoder,
    'vae': VariationalAutoencoder,
    'denoising_ae': DenoisingAutoencoder,
    'squeezenet': SqueezeNet,
    'efficientnet': EfficientNet,
    'efficientnetv2': EfficientNetV2,
    'xception': Xception,
    'vit': VisionTransformer,
    'deit': DeiT,
    'swin_tiny': SwinTiny,
    'coatnet': CoAtNet,
    'vim_tiny': VimTiny,
    'lenet': LeNet,
    'nin': NiN,
    'googlenet': GoogLeNet,
    'shufflenet': ShuffleNet,
    'se_resnet': SEResNet,
    'wide_resnet': WideResNet,
    'convnext': ConvNeXt,
    'repvgg': RepVGG,
    'regnet': RegNet,
    'ghostnet': GhostNet,
    'repghost': RepGhost,
    'resnext': ResNeXt,
    'res2net': Res2Net,
    'cbam_resnet': CBAMResNet,
    'mobilenetv3': MobileNetV3,
    'mnasnet': MNASNet,
    'eca_resnet': ECAResNet,
    'coord_resnet': CoordResNet,
    'sknet': SKNet,
    'dpn': DPN,
    'lcnet': LCNet,
    'capsnet': CapsNet,
    'hardnet': HarDNet,
    'cspnet': CSPNet,
    'darknet': Darknet,
    'van': VAN,
    'poolformer': PoolFormer,
    'inception_resnet': InceptionResNet,
    'hrnet': HRNet,
}

AUTOENCODER_MODELS = ['simple_ae', 'conv_ae', 'vae', 'denoising_ae']
GAN_MODELS = ['vanilla_gan', 'dcgan', 'wgan', 'cgan']
CLASSIFICATION_MODELS = sorted(
    name for name in MODEL_REGISTRY if name not in AUTOENCODER_MODELS and name not in GAN_MODELS
)
# Models trained on 224x224 in regression (all others use 28x28)
LARGE_IMAGE_MODELS = frozenset([
    'alexnet',
    'vgg',
    'vit',
    'deit',
    'swin_tiny',
    'coatnet',
    'vim_tiny',
    'xception',
    'efficientnet',
    'wide_resnet',
    'convnext',
    'googlenet',
    'nin',
    'inception_resnet',
])
# Legacy alias used by sequence/RNN-style models
SMALL_IMAGE_MODELS = frozenset(AUTOENCODER_MODELS + GAN_MODELS + ['bert', 'gpt', 'capsnet'])


class ModelFactory:
    @staticmethod
    def create_model(model_name, num_classes=10, enable_logging=True, output_dir="outputs", **kwargs):
        """
        Create a model instance based on the model name.
        
        Args:
            model_name (str): Name of the model to create
            num_classes (int): Number of output classes
            enable_logging (bool): Whether to enable logging for this model
            output_dir (str): Directory to store log files
            **kwargs: Additional arguments for the model
            
        Returns:
            model: An instance of the requested model
        """
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(MODEL_REGISTRY.keys())}")
        
        model_class = MODEL_REGISTRY[model_name]
        try:
            if model_name == 'simple_cnn':
                channels = kwargs.get('channels', [32, 64, 64])
                input_size = kwargs.get('input_size', 28)
                model = model_class(num_classes=num_classes, channels=channels, input_size=input_size)
            else:
                model = model_class(num_classes=num_classes, **kwargs)
            
            if enable_logging:
                model.setup_logger(output_dir)
                model.log_model_summary()
                
            return model
        except Exception as e:
            print(f"Error creating {model_name} model: {str(e)}")
            raise
    
    @staticmethod
    def get_available_models():
        """
        Get a list of all available models.
        
        Returns:
            list: List of model names
        """
        return sorted(MODEL_REGISTRY.keys())
        
    @staticmethod
    def get_model_file_paths(model_name, output_dir="outputs", file_type="pth"):
        """
        Generate file paths for model files with class name included.
        
        Args:
            model_name (str): Name of the model
            output_dir (str): Directory to store files
            file_type (str): File extension (pth or png)
            
        Returns:
            str: Path to the file with class name included
        """
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(MODEL_REGISTRY.keys())}")
            
        class_name = MODEL_REGISTRY[model_name].__name__
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate file path with class name
        if file_type == "pth":
            return os.path.join(output_dir, f"{class_name}_best_model.pth")
        elif file_type == "png":
            return os.path.join(output_dir, f"{class_name}_confusion_matrix.png")
        else:
            raise ValueError(f"Unsupported file type: {file_type}") 