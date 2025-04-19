import torch
import os
import argparse
from models.model_factory import ModelFactory
from utils.data_loader import DataLoaderFactory
from utils.trainer import Trainer
from utils.evaluator import Evaluator
from utils.gantrainer import GANTrainer
from utils.wgantrainer import WGANtrainer
from utils.cgantrainer import CGANTrainer

def parse_args():
    parser = argparse.ArgumentParser(description='MNIST Classification with PyTorch')
    
    # Model selection
    parser.add_argument('--model', type=str, default='alexnet', 
                        choices=ModelFactory.get_available_models(), 
                        help='Model architecture to use (default: alexnet)')
    
    # Dataset parameters
    parser.add_argument('--train-path', type=str, default='data/MNISTtrain.mat',
                        help='Path to training data (default: data/MNISTtrain.mat)')
    parser.add_argument('--test-path', type=str, default='data/MNISTtest.mat',
                        help='Path to test data (default: data/MNISTtest.mat)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of workers for data loading (default: 4)')
    parser.add_argument('--quick-test', action='store_true',
                        help='Use small test dataset (100 images) for quick testing')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Size to resize images to (default: 224)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use for training (default: auto)')
    
    # Model-specific parameters
    # AlexNet parameters
    parser.add_argument('--alexnet-dropout', type=float, default=0.5,
                        help='AlexNet dropout rate (default: 0.5)')
    
    # SimpleCNN parameters
    parser.add_argument('--simple-cnn-channels', type=str, default='32,64,64',
                        help='SimpleCNN channel configuration (default: 32,64,64)')
    
    # VGG parameters
    parser.add_argument('--vgg-config', type=str, default='A',
                        choices=['A', 'B', 'C', 'D', 'E'],
                        help='VGG configuration (default: A)')
    
    # ResNet parameters
    parser.add_argument('--resnet-blocks', type=str, default='2,2,2,2',
                        help='ResNet block configuration (default: 2,2,2,2)')
    
    # DenseNet parameters
    parser.add_argument('--densenet-growth', type=int, default=12,
                        help='DenseNet growth rate (default: 12)')
    parser.add_argument('--densenet-blocks', type=str, default='3,6,12,8',
                        help='DenseNet block configuration (default: 3,6,12,8)')
    
    # MobileNet parameters
    parser.add_argument('--mobilenet-width-multiplier', type=float, default=1.0,
                        help='MobileNet width multiplier (default: 1.0)')
    
    # MLP parameters
    parser.add_argument('--mlp-hidden', type=str, default='512,256,128',
                        help='MLP hidden layer sizes (default: 512,256,128)')
    
    # Vision Transformer parameters
    parser.add_argument('--vit-patch-size', type=int, default=7,
                        help='ViT patch size (default: 7)')
    parser.add_argument('--vit-embed-dim', type=int, default=128,
                        help='ViT embedding dimension (default: 128)')
    parser.add_argument('--vit-depth', type=int, default=4,
                        help='ViT transformer depth (default: 4)')
    parser.add_argument('--vit-num-heads', type=int, default=8,
                        help='ViT number of attention heads (default: 8)')
    parser.add_argument('--vit-mlp-ratio', type=float, default=4.0,
                        help='ViT MLP ratio (default: 4.0)')
    parser.add_argument('--vit-drop-rate', type=float, default=0.0,
                        help='ViT dropout rate (default: 0.0)')
    parser.add_argument('--vit-attn-drop-rate', type=float, default=0.0,
                        help='ViT attention dropout rate (default: 0.0)')
    
    # Xception parameters
    parser.add_argument('--xception-middle-blocks', type=int, default=8,
                        help='Xception number of middle blocks (default: 8)')
    
    # EfficientNet parameters
    parser.add_argument('--efficientnet-width-mult', type=float, default=1.0,
                        help='EfficientNet width multiplier (default: 1.0)')
    parser.add_argument('--efficientnet-depth-mult', type=float, default=1.0,
                        help='EfficientNet depth multiplier (default: 1.0)')
    parser.add_argument('--efficientnet-dropout', type=float, default=0.2,
                        help='EfficientNet dropout rate (default: 0.2)')
    parser.add_argument('--efficientnet-reduction', type=int, default=4,
                        help='EfficientNet reduction ratio (default: 4)')
    
    # SqueezeNet parameters
    parser.add_argument('--squeezenet-version', type=float, default=1.1,
                        choices=[1.0, 1.1],
                        help='SqueezeNet version (default: 1.1)')
    
    # BERT model parameters
    parser.add_argument('--bert-hidden-size', type=int, default=256,
                      help='Hidden size for BERT model')
    parser.add_argument('--bert-num-layers', type=int, default=6,
                      help='Number of transformer layers for BERT model')
    parser.add_argument('--bert-num-heads', type=int, default=8,
                      help='Number of attention heads for BERT model')
    parser.add_argument('--bert-mlp-ratio', type=float, default=4.0,
                      help='MLP ratio for BERT model')
    parser.add_argument('--bert-dropout', type=float, default=0.1,
                      help='Dropout rate for BERT model')
    parser.add_argument('--bert-max-seq-length', type=int, default=50176,
                      help='Maximum sequence length for BERT model')

    # GPT model parameters
    parser.add_argument('--gpt-hidden-size', type=int, default=256,
                      help='Hidden size for GPT model')
    parser.add_argument('--gpt-num-layers', type=int, default=6,
                      help='Number of transformer layers for GPT model')
    parser.add_argument('--gpt-num-heads', type=int, default=8,
                      help='Number of attention heads for GPT model')
    parser.add_argument('--gpt-mlp-ratio', type=float, default=4.0,
                      help='MLP ratio for GPT model')
    parser.add_argument('--gpt-dropout', type=float, default=0.1,
                      help='Dropout rate for GPT model')
    parser.add_argument('--gpt-max-seq-length', type=int, default=784,
                      help='Maximum sequence length for GPT model')
    
    # RNN model parameters (LSTM and GRU)
    parser.add_argument('--rnn-hidden-size', type=int, default=128,
                      help='Hidden size for RNN models (LSTM/GRU)')
    parser.add_argument('--rnn-num-layers', type=int, default=2,
                      help='Number of layers for RNN models (LSTM/GRU)')
    parser.add_argument('--rnn-dropout', type=float, default=0.2,
                      help='Dropout rate for RNN models (LSTM/GRU)')
    parser.add_argument('--rnn-bidirectional', action='store_true',
                      help='Use bidirectional RNN models (LSTM/GRU)')
    
    # GAN model parameters
    parser.add_argument('--latent-dim', type=int, default=100,
                      help='Latent dimension for GAN models')
    parser.add_argument('--generator-hidden', type=int, default=256,
                      help='Hidden size for Vanilla GAN generator')
    parser.add_argument('--discriminator-hidden', type=int, default=256,
                      help='Hidden size for Vanilla GAN discriminator')
    parser.add_argument('--generator-channels', type=int, default=64,
                      help='Number of channels for DCGAN/WGAN/CGAN generator')
    parser.add_argument('--discriminator-channels', type=int, default=64,
                      help='Number of channels for DCGAN/WGAN/CGAN discriminator')
    
    # Autoencoder parameters
    parser.add_argument('--simple-ae-latent-dim', type=int, default=32,
                      help='Latent dimension for SimpleAutoencoder')
    parser.add_argument('--simple-ae-hidden-dims', type=str, default='128,64',
                      help='Comma-separated list of hidden dimensions for SimpleAutoencoder')
    
    parser.add_argument('--conv-ae-latent-dim', type=int, default=32,
                      help='Latent dimension for ConvolutionalAutoencoder')
    parser.add_argument('--conv-ae-channels', type=str, default='32,64,128',
                      help='Comma-separated list of channel dimensions for ConvolutionalAutoencoder')
    
    parser.add_argument('--vae-latent-dim', type=int, default=32,
                      help='Latent dimension for VariationalAutoencoder')
    parser.add_argument('--vae-hidden-dims', type=str, default='128,64',
                      help='Comma-separated list of hidden dimensions for VariationalAutoencoder')
    
    parser.add_argument('--denoising-ae-noise-factor', type=float, default=0.3,
                      help='Noise factor for DenoisingAutoencoder')
    parser.add_argument('--denoising-ae-hidden-dims', type=str, default='128,64',
                      help='Comma-separated list of hidden dimensions for DenoisingAutoencoder')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Directory to save outputs (default: outputs)')
    parser.add_argument('--save-model', action='store_true',
                        help='Save model checkpoint after training')
    parser.add_argument('--plot-confusion', action='store_true',
                        help='Plot confusion matrix after training')
    
    return parser.parse_args()

def get_device(device_arg):
    if device_arg == 'auto':
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(device_arg)

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set appropriate image size based on model type
    if args.model in ['simple_ae', 'conv_ae', 'vae', 'denoising_ae', 'vanilla_gan', 'dcgan', 'wgan', 'cgan', 'bert', 'gpt']:
        args.image_size = 28  # Use original MNIST size for these models
    else:
        args.image_size = 224  # Default size for CNN models
    
    # Initialize data loaders
    if args.quick_test:
        print("Using quick test dataset (100 images)")
        train_loader, test_loader = DataLoaderFactory.get_data_loaders(
            train_path='data/test_data/test_images.npy',
            test_path='data/test_data/test_images.npy',  # Use same file for both train and test in quick test mode
            batch_size=min(args.batch_size, 32),  # Limit batch size for small dataset
            num_workers=min(args.num_workers, 2),  # Limit workers for small dataset
            image_size=args.image_size
        )
    else:
        train_loader, test_loader = DataLoaderFactory.get_data_loaders(
            train_path=args.train_path,
            test_path=args.test_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size
        )
    
    # Define MNIST class names
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    # Prepare model-specific parameters
    model_kwargs = {}
    
    if args.model == 'alexnet':
        model_kwargs['dropout'] = args.alexnet_dropout
    elif args.model == 'simple_cnn':
        model_kwargs['channels'] = [int(x) for x in args.simple_cnn_channels.split(',')]
        model_kwargs['input_size'] = args.image_size
    elif args.model == 'vgg':
        model_kwargs['cfg'] = args.vgg_config
        model_kwargs['input_size'] = args.image_size
    elif args.model == 'resnet':
        model_kwargs['num_blocks'] = [int(x) for x in args.resnet_blocks.split(',')]
    elif args.model == 'densenet':
        model_kwargs['growth_rate'] = args.densenet_growth
        model_kwargs['block_config'] = tuple([int(x) for x in args.densenet_blocks.split(',')])
    elif args.model == 'mobilenet':
        model_kwargs['width_multiplier'] = args.mobilenet_width_multiplier
    elif args.model == 'mlp':
        model_kwargs['hidden_sizes'] = [int(x) for x in args.mlp_hidden.split(',')]
    elif args.model == 'vit':
        model_kwargs['patch_size'] = args.vit_patch_size
        model_kwargs['embed_dim'] = args.vit_embed_dim
        model_kwargs['depth'] = args.vit_depth
        model_kwargs['num_heads'] = args.vit_num_heads
        model_kwargs['mlp_ratio'] = args.vit_mlp_ratio
        model_kwargs['drop_rate'] = args.vit_drop_rate
        model_kwargs['attn_drop_rate'] = args.vit_attn_drop_rate
    elif args.model == 'xception':
        model_kwargs['middle_blocks'] = args.xception_middle_blocks
    elif args.model == 'efficientnet':
        model_kwargs['width_mult'] = args.efficientnet_width_mult
        model_kwargs['depth_mult'] = args.efficientnet_depth_mult
        model_kwargs['dropout_rate'] = args.efficientnet_dropout
        model_kwargs['reduction'] = args.efficientnet_reduction
    elif args.model == 'squeezenet':
        model_kwargs['version'] = args.squeezenet_version
    elif args.model == 'bert':
        model_kwargs = {
            'hidden_size': args.bert_hidden_size,
            'num_layers': args.bert_num_layers,
            'num_heads': args.bert_num_heads,
            'mlp_ratio': args.bert_mlp_ratio,
            'dropout': args.bert_dropout,
            'max_seq_length': args.bert_max_seq_length
        }
    elif args.model == 'gpt':
        model_kwargs = {
            'hidden_size': args.gpt_hidden_size,
            'num_layers': args.gpt_num_layers,
            'num_heads': args.gpt_num_heads,
            'mlp_ratio': args.gpt_mlp_ratio,
            'dropout': args.gpt_dropout,
            'max_seq_length': args.gpt_max_seq_length
        }
    elif args.model in ['lstm', 'gru']:
        model_kwargs = {
            'hidden_size': args.rnn_hidden_size,
            'num_layers': args.rnn_num_layers,
            'dropout': args.rnn_dropout,
            'bidirectional': args.rnn_bidirectional
        }
    elif args.model == 'vanilla_gan':
        model_kwargs = {
            'latent_dim': args.latent_dim,
            'generator_hidden': args.generator_hidden,
            'discriminator_hidden': args.discriminator_hidden
        }
    elif args.model in ['dcgan', 'wgan', 'cgan']:
        model_kwargs = {
            'latent_dim': args.latent_dim,
            'generator_channels': args.generator_channels,
            'discriminator_channels': args.discriminator_channels
        }
    elif args.model == 'simple_ae':
        model_kwargs = {
            'latent_dim': args.simple_ae_latent_dim,
            'hidden_dims': [int(x) for x in args.simple_ae_hidden_dims.split(',')]
        }
    elif args.model == 'conv_ae':
        model_kwargs = {
            'latent_dim': args.conv_ae_latent_dim,
            'channels': [int(x) for x in args.conv_ae_channels.split(',')]
        }
    elif args.model == 'vae':
        model_kwargs = {
            'latent_dim': args.vae_latent_dim,
            'hidden_dims': [int(x) for x in args.vae_hidden_dims.split(',')]
        }
    elif args.model == 'denoising_ae':
        model_kwargs = {
            'noise_factor': args.denoising_ae_noise_factor,
            'hidden_dims': [int(x) for x in args.denoising_ae_hidden_dims.split(',')]
        }
    
    # Initialize model
    try:
        model = ModelFactory.create_model(args.model, num_classes=10, **model_kwargs)
        # Move model to device before any operations
        model = model.to(device)
        print(f"Created model: {args.model}")
    except Exception as e:
        print(f"Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Initialize trainer based on model type
    if args.model in ['vanilla_gan', 'dcgan']:
        trainer = GANTrainer(model, device, learning_rate=args.lr)
        evaluator = None  # GANs don't use standard evaluation
    elif args.model == 'wgan':
        trainer = WGANtrainer(model, device, learning_rate=args.lr)
        evaluator = None  # GANs don't use standard evaluation
    elif args.model == 'cgan':
        trainer = CGANTrainer(model, device, learning_rate=args.lr)
        evaluator = None  # GANs don't use standard evaluation
    else:
        trainer = Trainer(model, device, learning_rate=args.lr)
        evaluator = Evaluator(model, device)
    
    # Get file paths with class names
    model_checkpoint_path = ModelFactory.get_model_file_paths(args.model, args.output_dir, "pth")
    confusion_matrix_path = ModelFactory.get_model_file_paths(args.model, args.output_dir, "png")
    
    # Training loop
    num_epochs = args.epochs
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        if args.model in ['vanilla_gan', 'dcgan', 'wgan', 'cgan']:
            g_loss, d_loss = trainer.train_epoch(train_loader)
            print(f"Generator Loss: {g_loss:.4f}, Discriminator Loss: {d_loss:.4f}")
            
            # Save checkpoint for GANs
            if args.save_model:
                trainer.save_checkpoint(
                    model_checkpoint_path,
                    epoch,
                    g_loss,
                    d_loss
                )
                print(f"Saved model to {model_checkpoint_path}")
        else:
            train_loss, train_acc = trainer.train_epoch(train_loader)
            print(f"Training Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
            
            # Evaluate
            if evaluator is not None:
                test_loss, test_acc, all_preds, all_targets = evaluator.evaluate(test_loader)
                print(f"Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")
                
                # Save best model
                if test_acc > best_acc and args.save_model:
                    best_acc = test_acc
                    trainer.save_checkpoint(
                        model_checkpoint_path,
                        epoch,
                        test_loss,
                        test_acc
                    )
                    print(f"Saved best model to {model_checkpoint_path}")
                
                # Plot confusion matrix for the last epoch
                if epoch == num_epochs - 1 and args.plot_confusion:
                    evaluator.plot_confusion_matrix(
                        all_preds,
                        all_targets,
                        confusion_matrix_path
                    )
                    print(f"Saved confusion matrix to {confusion_matrix_path}")

if __name__ == "__main__":
    main() 