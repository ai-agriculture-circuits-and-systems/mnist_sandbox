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
import os

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
        model_map = {
            'alexnet': AlexNet,
            'simple_cnn': SimpleCNN,
            'vgg': VGG,
            'resnet': ResNet,
            'densenet': DenseNet,
            'mobilenet': MobileNet,
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
            'xception': Xception,
            'vit': VisionTransformer
        }
        
        if model_name not in model_map:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(model_map.keys())}")
        
        model_class = model_map[model_name]
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
        return [
            'alexnet',
            'simple_cnn',
            'vgg',
            'resnet',
            'densenet',
            'mobilenet',
            'mlp',
            'bert',
            'gpt',
            'lstm',
            'gru',
            'vanilla_gan',
            'dcgan',
            'wgan',
            'cgan',
            'simple_ae',
            'conv_ae',
            'vae',
            'denoising_ae',
            'squeezenet',
            'efficientnet',
            'xception',
            'vit'
        ]
        
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
        # Get the actual class name from the model map
        model_map = {
            'alexnet': 'AlexNet',
            'simple_cnn': 'SimpleCNN',
            'vgg': 'VGG',
            'resnet': 'ResNet',
            'densenet': 'DenseNet',
            'mobilenet': 'MobileNet',
            'mlp': 'MLP',
            'bert': 'BERTMNIST',
            'gpt': 'GPTMNIST',
            'lstm': 'LSTMMNIST',
            'gru': 'GRUMNIST',
            'vanilla_gan': 'VanillaGAN',
            'dcgan': 'DCGAN',
            'wgan': 'WGAN',
            'cgan': 'CGAN',
            'simple_ae': 'SimpleAutoencoder',
            'conv_ae': 'ConvolutionalAutoencoder',
            'vae': 'VariationalAutoencoder',
            'denoising_ae': 'DenoisingAutoencoder',
            'squeezenet': 'SqueezeNet',
            'efficientnet': 'EfficientNet',
            'xception': 'Xception',
            'vit': 'VisionTransformer'
        }
        
        if model_name not in model_map:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(model_map.keys())}")
            
        class_name = model_map[model_name]
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate file path with class name
        if file_type == "pth":
            return os.path.join(output_dir, f"{class_name}_best_model.pth")
        elif file_type == "png":
            return os.path.join(output_dir, f"{class_name}_confusion_matrix.png")
        else:
            raise ValueError(f"Unsupported file type: {file_type}") 