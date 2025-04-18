#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "====================================================="
echo -e "${GREEN}MNIST Classification with PyTorch - Setup and Run Script${NC}"
echo "====================================================="

# Create necessary directories
echo -e "${YELLOW}Creating necessary directories...${NC}"
mkdir -p outputs

# Check if data files exist
if [ ! -f "data/MNISTtrain.mat" ] || [ ! -f "data/MNISTtest.mat" ]; then
    echo -e "${RED}MNIST dataset files not found in data/ directory.${NC}"
    echo "Please ensure MNISTtrain.mat and MNISTtest.mat are in the data/ directory."
    exit 1
fi

# Default values
MODEL="alexnet"
BATCH_SIZE=16
EPOCHS=1
LEARNING_RATE=0.001
DEVICE="auto"
OUTPUT_DIR="outputs"
SAVE_MODEL=true
PLOT_CONFUSION=true
QUICK_TEST=true

# Model-specific default parameters
ALEXNET_DROPOUT=0.5
SIMPLE_CNN_CHANNELS="32,64,64"
VGG_CONFIG="A"
RESNET_BLOCKS="2,2,2,2"
DENSENET_GROWTH=12
DENSENET_BLOCKS="3,6,12,8"
MOBILENET_WIDTH_MULT=1.0
MLP_HIDDEN="512,256,128"
VIT_PATCH_SIZE=7
VIT_EMBED_DIM=128
VIT_DEPTH=4
VIT_NUM_HEADS=8
VIT_MLP_RATIO=4.0
VIT_DROP_RATE=0.0
VIT_ATTN_DROP_RATE=0.0
XCEPTION_MIDDLE_BLOCKS=8
EFFICIENTNET_WIDTH_MULT=1.0
EFFICIENTNET_DEPTH_MULT=1.0
EFFICIENTNET_DROPOUT=0.2
EFFICIENTNET_REDUCTION=4
SQUEEZENET_VERSION=1.1

# BERT model options
BERT_HIDDEN_SIZE=256
BERT_NUM_LAYERS=6
BERT_NUM_HEADS=8
BERT_MLP_RATIO=4.0
BERT_DROPOUT=0.1
BERT_MAX_SEQ_LENGTH=784

# GPT model options
GPT_HIDDEN_SIZE=256
GPT_NUM_LAYERS=6
GPT_NUM_HEADS=8
GPT_MLP_RATIO=4.0
GPT_DROPOUT=0.1
GPT_MAX_SEQ_LENGTH=784

# RNN model options (LSTM and GRU)
RNN_HIDDEN_SIZE=128
RNN_NUM_LAYERS=2
RNN_DROPOUT=0.2
RNN_BIDIRECTIONAL=false

# GAN model options
LATENT_DIM=100
GENERATOR_HIDDEN=256
DISCRIMINATOR_HIDDEN=256
GENERATOR_CHANNELS=64
DISCRIMINATOR_CHANNELS=64

# Autoencoder options
simple_ae_latent_dim=32
simple_ae_hidden_dims=128,64
conv_ae_latent_dim=32
conv_ae_channels=32,64,128
vae_latent_dim=32
vae_hidden_dims=128,64
denoising_ae_noise_factor=0.3
denoising_ae_hidden_dims=128,64

# Help message
show_help() {
    echo -e "${BLUE}Usage:${NC} $0 [options]"
    echo ""
    echo -e "${BLUE}Options:${NC}"
    echo "  -h, --help                 Show this help message"
    echo "  -m, --model MODEL          Model architecture to use (default: alexnet)"
    echo "                             Available models: alexnet, simple_cnn, vgg, resnet, densenet, mobilenet, mlp, vit, xception, efficientnet, squeezenet, simple_ae, conv_ae, vae, denoising_ae, bert, gpt, lstm, gru, vanilla_gan, dcgan, wgan, cgan"
    echo "  -b, --batch-size SIZE      Batch size for training (default: 32)"
    echo "  -e, --epochs EPOCHS         Number of epochs to train (default: 10)"
    echo "  -l, --lr RATE             Learning rate (default: 0.001)"
    echo "  -d, --device DEVICE       Device to use (auto/cuda/cpu) (default: auto)"
    echo "  -o, --output-dir DIR       Directory to save outputs (default: outputs)"
    echo "  -s, --save-model           Save model checkpoint after training"
    echo "  -c, --plot-confusion       Plot confusion matrix after training"
    echo "  -q, --quick-test           Use small test dataset (100 images) for quick testing"
    echo ""
    echo -e "${BLUE}Model-specific options:${NC}"
    echo "  AlexNet:"
    echo "    --alexnet-dropout RATE  Dropout rate (default: 0.5)"
    echo ""
    echo "  SimpleCNN:"
    echo "    --simple-cnn-channels CHANNELS  Channel configuration (default: 32,64,64)"
    echo ""
    echo "  VGG:"
    echo "    --vgg-config CONFIG     Configuration (A/B/C/D/E) (default: A)"
    echo ""
    echo "  ResNet:"
    echo "    --resnet-blocks BLOCKS  Block configuration (default: 2,2,2,2)"
    echo ""
    echo "  DenseNet:"
    echo "    --densenet-growth RATE  Growth rate (default: 12)"
    echo "    --densenet-blocks BLOCKS  Block configuration (default: 3,6,12,8)"
    echo ""
    echo "  MobileNet:"
    echo "    --mobilenet-width-mult MULT  Width multiplier (default: 1.0)"
    echo ""
    echo "  MLP:"
    echo "    --mlp-hidden SIZES      Hidden layer sizes (default: 512,256,128)"
    echo ""
    echo "  Vision Transformer:"
    echo "    --vit-patch-size SIZE   Patch size (default: 7)"
    echo "    --vit-embed-dim DIM     Embedding dimension (default: 128)"
    echo "    --vit-depth DEPTH       Transformer depth (default: 4)"
    echo "    --vit-num-heads HEADS   Number of attention heads (default: 8)"
    echo "    --vit-mlp-ratio RATIO   MLP ratio (default: 4.0)"
    echo "    --vit-drop-rate RATE    Dropout rate (default: 0.0)"
    echo "    --vit-attn-drop-rate RATE  Attention dropout rate (default: 0.0)"
    echo ""
    echo "  Xception:"
    echo "    --xception-middle-blocks BLOCKS  Number of middle blocks (default: 8)"
    echo ""
    echo "  EfficientNet:"
    echo "    --efficientnet-width-mult MULT  Width multiplier (default: 1.0)"
    echo "    --efficientnet-depth-mult MULT  Depth multiplier (default: 1.0)"
    echo "    --efficientnet-dropout RATE  Dropout rate (default: 0.2)"
    echo "    --efficientnet-reduction RED  Reduction ratio (default: 4)"
    echo ""
    echo "  SqueezeNet:"
    echo "    --squeezenet-version VER  Version (1.0/1.1) (default: 1.1)"
    echo ""
    echo -e "${BLUE}BERT model options:${NC}"
    echo "  --bert-hidden-size SIZE  Hidden size for BERT model (default: 256)"
    echo "  --bert-num-layers LAYERS  Number of transformer layers for BERT model (default: 6)"
    echo "  --bert-num-heads HEADS   Number of attention heads for BERT model (default: 8)"
    echo "  --bert-mlp-ratio RATIO   MLP ratio for BERT model (default: 4.0)"
    echo "  --bert-dropout RATE      Dropout rate for BERT model (default: 0.1)"
    echo "  --bert-max-seq-length LEN  Maximum sequence length for BERT model (default: 784)"
    echo ""
    echo -e "${BLUE}GPT model options:${NC}"
    echo "  --gpt-hidden-size SIZE  Hidden size for GPT model (default: 256)"
    echo "  --gpt-num-layers LAYERS  Number of transformer layers for GPT model (default: 6)"
    echo "  --gpt-num-heads HEADS   Number of attention heads for GPT model (default: 8)"
    echo "  --gpt-mlp-ratio RATIO   MLP ratio for GPT model (default: 4.0)"
    echo "  --gpt-dropout RATE      Dropout rate for GPT model (default: 0.1)"
    echo "  --gpt-max-seq-length LEN  Maximum sequence length for GPT model (default: 784)"
    echo ""
    echo -e "${BLUE}RNN model options:${NC}"
    echo "  --rnn-hidden-size SIZE  Hidden size for RNN model (default: 128)"
    echo "  --rnn-num-layers LAYERS  Number of layers for RNN model (default: 2)"
    echo "  --rnn-dropout RATE      Dropout rate for RNN model (default: 0.2)"
    echo "  --rnn-bidirectional     Use bidirectional RNN (default: false)"
    echo ""
    echo -e "${BLUE}GAN model options:${NC}"
    echo "  --latent-dim DIM         Latent dimension for GAN model (default: 100)"
    echo "  --generator-hidden SIZE   Hidden size for generator in GAN model (default: 256)"
    echo "  --discriminator-hidden SIZE  Hidden size for discriminator in GAN model (default: 256)"
    echo "  --generator-channels CHANNELS  Channels for generator in GAN model (default: 64)"
    echo "  --discriminator-channels CHANNELS  Channels for discriminator in GAN model (default: 64)"
    echo ""
    echo -e "${BLUE}Autoencoder options:${NC}"
    echo "  SimpleAE:"
    echo "    --simple-ae-latent-dim INT     Latent dimension for SimpleAutoencoder (default: 32)"
    echo "    --simple-ae-hidden-dims STR    Comma-separated list of hidden dimensions (default: 128,64)"
    echo "  ConvAE:"
    echo "    --conv-ae-latent-dim INT       Latent dimension for ConvolutionalAutoencoder (default: 32)"
    echo "    --conv-ae-channels STR         Comma-separated list of channel dimensions (default: 32,64,128)"
    echo "  VAE:"
    echo "    --vae-latent-dim INT           Latent dimension for VariationalAutoencoder (default: 32)"
    echo "    --vae-hidden-dims STR          Comma-separated list of hidden dimensions (default: 128,64)"
    echo "  DenoisingAE:"
    echo "    --denoising-ae-noise-factor FLOAT  Noise factor for DenoisingAutoencoder (default: 0.3)"
    echo "    --denoising-ae-hidden-dims STR     Comma-separated list of hidden dimensions (default: 128,64)"
    echo ""
    echo -e "${BLUE}Examples:${NC}"
    echo "  $0 --model alexnet --batch-size 64 --epochs 20"
    echo "  $0 --model vgg --vgg-config B --lr 0.0001 --save-model"
    echo "  $0 --model resnet --resnet-blocks 3,4,6,3 --plot-confusion"
    echo "  $0 --model densenet --densenet-growth 24 --densenet-blocks 6,12,24,16"
    echo "  $0 --model mobilenet --epochs 5"
    echo "  $0 --model mlp --mlp-hidden 1024,512,256 --output-dir results"
    echo "  $0 --model vit --vit-patch-size 7 --vit-depth 6 --vit-num-heads 8"
    echo "  $0 --model xception --xception-middle-blocks 12"
    echo "  $0 --model efficientnet --efficientnet-width-mult 1.2 --efficientnet-depth-mult 1.4"
    echo "  $0 --model squeezenet --squeezenet-version 1.0"
    echo "  $0 --model bert --bert-hidden-size 256 --bert-num-layers 6 --bert-num-heads 8"
    echo "  $0 --model gpt --batch-size 16 --epochs 2 --lr 0.001 --device cpu --output-dir outputs --quick-test --gpt-hidden-size 256 --gpt-num-layers 6 --gpt-num-heads 8 --gpt-mlp-ratio 4.0 --gpt-dropout 0.1 --gpt-max-seq-length 784 --save-model --plot-confusion"
    echo "  $0 --model lstm --rnn-hidden-size 128 --rnn-num-layers 2 --rnn-bidirectional"
    echo "  $0 --model gru --rnn-hidden-size 128 --rnn-num-layers 2 --rnn-dropout 0.3"
    echo "  $0 --model vanilla_gan --latent-dim 100 --generator-hidden 256"
    echo "  $0 --model dcgan --latent-dim 100 --generator-channels 64"
    echo "  $0 --model wgan --latent-dim 100 --discriminator-channels 128"
    echo "  $0 --model cgan --latent-dim 100 --generator-channels 64"
    echo "  $0 --model simple_ae --simple-ae-latent-dim 32 --simple-ae-hidden-dims 128,64"
    echo "  $0 --model conv_ae --conv-ae-latent-dim 32 --conv-ae-channels 32,64,128"
    echo "  $0 --model vae --vae-latent-dim 32 --vae-hidden-dims 128,64"
    echo "  $0 --model denoising_ae --denoising-ae-noise-factor 0.3 --denoising-ae-hidden-dims 128,64"
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -l|--lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        -d|--device)
            DEVICE="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -s|--save-model)
            SAVE_MODEL=true
            shift
            ;;
        -c|--plot-confusion)
            PLOT_CONFUSION=true
            shift
            ;;
        -q|--quick-test)
            QUICK_TEST=true
            shift
            ;;
        --alexnet-dropout)
            ALEXNET_DROPOUT="$2"
            shift 2
            ;;
        --simple-cnn-channels)
            SIMPLE_CNN_CHANNELS="$2"
            shift 2
            ;;
        --vgg-config)
            VGG_CONFIG="$2"
            shift 2
            ;;
        --resnet-blocks)
            RESNET_BLOCKS="$2"
            shift 2
            ;;
        --densenet-growth)
            DENSENET_GROWTH="$2"
            shift 2
            ;;
        --densenet-blocks)
            DENSENET_BLOCKS="$2"
            shift 2
            ;;
        --mobilenet-width-mult)
            MOBILENET_WIDTH_MULT="$2"
            shift 2
            ;;
        --mlp-hidden)
            MLP_HIDDEN="$2"
            shift 2
            ;;
        --vit-patch-size)
            VIT_PATCH_SIZE="$2"
            shift 2
            ;;
        --vit-embed-dim)
            VIT_EMBED_DIM="$2"
            shift 2
            ;;
        --vit-depth)
            VIT_DEPTH="$2"
            shift 2
            ;;
        --vit-num-heads)
            VIT_NUM_HEADS="$2"
            shift 2
            ;;
        --vit-mlp-ratio)
            VIT_MLP_RATIO="$2"
            shift 2
            ;;
        --vit-drop-rate)
            VIT_DROP_RATE="$2"
            shift 2
            ;;
        --vit-attn-drop-rate)
            VIT_ATTN_DROP_RATE="$2"
            shift 2
            ;;
        --xception-middle-blocks)
            XCEPTION_MIDDLE_BLOCKS="$2"
            shift 2
            ;;
        --efficientnet-width-mult)
            EFFICIENTNET_WIDTH_MULT="$2"
            shift 2
            ;;
        --efficientnet-depth-mult)
            EFFICIENTNET_DEPTH_MULT="$2"
            shift 2
            ;;
        --efficientnet-dropout)
            EFFICIENTNET_DROPOUT="$2"
            shift 2
            ;;
        --efficientnet-reduction)
            EFFICIENTNET_REDUCTION="$2"
            shift 2
            ;;
        --squeezenet-version)
            SQUEEZENET_VERSION="$2"
            shift 2
            ;;
        --bert-hidden-size)
            BERT_HIDDEN_SIZE="$2"
            shift 2
            ;;
        --bert-num-layers)
            BERT_NUM_LAYERS="$2"
            shift 2
            ;;
        --bert-num-heads)
            BERT_NUM_HEADS="$2"
            shift 2
            ;;
        --bert-mlp-ratio)
            BERT_MLP_RATIO="$2"
            shift 2
            ;;
        --bert-dropout)
            BERT_DROPOUT="$2"
            shift 2
            ;;
        --bert-max-seq-length)
            BERT_MAX_SEQ_LENGTH="$2"
            shift 2
            ;;
        --gpt-hidden-size)
            GPT_HIDDEN_SIZE="$2"
            shift 2
            ;;
        --gpt-num-layers)
            GPT_NUM_LAYERS="$2"
            shift 2
            ;;
        --gpt-num-heads)
            GPT_NUM_HEADS="$2"
            shift 2
            ;;
        --gpt-mlp-ratio)
            GPT_MLP_RATIO="$2"
            shift 2
            ;;
        --gpt-dropout)
            GPT_DROPOUT="$2"
            shift 2
            ;;
        --gpt-max-seq-length)
            GPT_MAX_SEQ_LENGTH="$2"
            shift 2
            ;;
        --rnn-hidden-size)
            RNN_HIDDEN_SIZE="$2"
            shift 2
            ;;
        --rnn-num-layers)
            RNN_NUM_LAYERS="$2"
            shift 2
            ;;
        --rnn-dropout)
            RNN_DROPOUT="$2"
            shift 2
            ;;
        --rnn-bidirectional)
            RNN_BIDIRECTIONAL=true
            shift
            ;;
        --latent-dim)
            LATENT_DIM="$2"
            shift 2
            ;;
        --generator-hidden)
            GENERATOR_HIDDEN="$2"
            shift 2
            ;;
        --discriminator-hidden)
            DISCRIMINATOR_HIDDEN="$2"
            shift 2
            ;;
        --generator-channels)
            GENERATOR_CHANNELS="$2"
            shift 2
            ;;
        --discriminator-channels)
            DISCRIMINATOR_CHANNELS="$2"
            shift 2
            ;;
        --simple-ae-latent-dim)
            simple_ae_latent_dim="$2"
            shift 2
            ;;
        --simple-ae-hidden-dims)
            simple_ae_hidden_dims="$2"
            shift 2
            ;;
        --conv-ae-latent-dim)
            conv_ae_latent_dim="$2"
            shift 2
            ;;
        --conv-ae-channels)
            conv_ae_channels="$2"
            shift 2
            ;;
        --vae-latent-dim)
            vae_latent_dim="$2"
            shift 2
            ;;
        --vae-hidden-dims)
            vae_hidden_dims="$2"
            shift 2
            ;;
        --denoising-ae-noise-factor)
            denoising_ae_noise_factor="$2"
            shift 2
            ;;
        --denoising-ae-hidden-dims)
            denoising_ae_hidden_dims="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            ;;
    esac
done

# Build command
CMD="python3 main.py --model $MODEL --batch-size $BATCH_SIZE --epochs $EPOCHS --lr $LEARNING_RATE --device $DEVICE --output-dir $OUTPUT_DIR"

# Add quick test flag if enabled
if [ "$QUICK_TEST" = true ]; then
    CMD="$CMD --quick-test"
fi

# Add model-specific parameters
case $MODEL in
    "alexnet")
        CMD="$CMD --alexnet-dropout $ALEXNET_DROPOUT"
        ;;
    "simple_cnn")
        CMD="$CMD --simple-cnn-channels $SIMPLE_CNN_CHANNELS"
        ;;
    "vgg")
        CMD="$CMD --vgg-config $VGG_CONFIG"
        ;;
    "resnet")
        CMD="$CMD --resnet-blocks $RESNET_BLOCKS"
        ;;
    "densenet")
        CMD="$CMD --densenet-growth $DENSENET_GROWTH --densenet-blocks $DENSENET_BLOCKS"
        ;;
    "mobilenet")
        CMD="$CMD --mobilenet-width-multiplier $MOBILENET_WIDTH_MULT"
        ;;
    "mlp")
        CMD="$CMD --mlp-hidden $MLP_HIDDEN"
        ;;
    "vit")
        CMD="$CMD --vit-patch-size $VIT_PATCH_SIZE --vit-embed-dim $VIT_EMBED_DIM --vit-depth $VIT_DEPTH --vit-num-heads $VIT_NUM_HEADS --vit-mlp-ratio $VIT_MLP_RATIO --vit-drop-rate $VIT_DROP_RATE --vit-attn-drop-rate $VIT_ATTN_DROP_RATE"
        ;;
    "xception")
        CMD="$CMD --xception-middle-blocks $XCEPTION_MIDDLE_BLOCKS"
        ;;
    "efficientnet")
        CMD="$CMD --efficientnet-width-mult $EFFICIENTNET_WIDTH_MULT --efficientnet-depth-mult $EFFICIENTNET_DEPTH_MULT --efficientnet-dropout $EFFICIENTNET_DROPOUT --efficientnet-reduction $EFFICIENTNET_REDUCTION"
        ;;
    "squeezenet")
        CMD="$CMD --squeezenet-version $SQUEEZENET_VERSION"
        ;;
    "bert")
        CMD="$CMD --bert-hidden-size $BERT_HIDDEN_SIZE \
              --bert-num-layers $BERT_NUM_LAYERS \
              --bert-num-heads $BERT_NUM_HEADS \
              --bert-mlp-ratio $BERT_MLP_RATIO \
              --bert-dropout $BERT_DROPOUT \
              --bert-max-seq-length $BERT_MAX_SEQ_LENGTH"
        ;;
    "gpt")
        CMD="$CMD --gpt-hidden-size $GPT_HIDDEN_SIZE \
              --gpt-num-layers $GPT_NUM_LAYERS \
              --gpt-num-heads $GPT_NUM_HEADS \
              --gpt-mlp-ratio $GPT_MLP_RATIO \
              --gpt-dropout $GPT_DROPOUT \
              --gpt-max-seq-length $GPT_MAX_SEQ_LENGTH"
        ;;
    "lstm"|"gru")
        CMD="$CMD --rnn-hidden-size $RNN_HIDDEN_SIZE \
              --rnn-num-layers $RNN_NUM_LAYERS \
              --rnn-dropout $RNN_DROPOUT"
        if [ "$RNN_BIDIRECTIONAL" = true ]; then
            CMD="$CMD --rnn-bidirectional"
        fi
        ;;
    "vanilla_gan")
        CMD="$CMD --latent-dim $LATENT_DIM \
              --generator-hidden $GENERATOR_HIDDEN \
              --discriminator-hidden $DISCRIMINATOR_HIDDEN"
        ;;
    "dcgan"|"wgan"|"cgan")
        CMD="$CMD --latent-dim $LATENT_DIM \
              --generator-channels $GENERATOR_CHANNELS \
              --discriminator-channels $DISCRIMINATOR_CHANNELS"
        ;;
    "simple_ae")
        CMD="$CMD --simple-ae-latent-dim $simple_ae_latent_dim --simple-ae-hidden-dims $simple_ae_hidden_dims"
        ;;
    "conv_ae")
        CMD="$CMD --conv-ae-latent-dim $conv_ae_latent_dim --conv-ae-channels $conv_ae_channels"
        ;;
    "vae")
        CMD="$CMD --vae-latent-dim $vae_latent_dim --vae-hidden-dims $vae_hidden_dims"
        ;;
    "denoising_ae")
        CMD="$CMD --denoising-ae-noise-factor $denoising_ae_noise_factor --denoising-ae-hidden-dims $denoising_ae_hidden_dims"
        ;;
esac

# Add optional flags
if [ "$SAVE_MODEL" = true ]; then
    CMD="$CMD --save-model"
fi

if [ "$PLOT_CONFUSION" = true ]; then
    CMD="$CMD --plot-confusion"
fi

# Run the training script
echo -e "${YELLOW}Starting training with the following parameters:${NC}"
echo "  Model: $MODEL"
echo "  Batch Size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Device: $DEVICE"
echo "  Save Model: $SAVE_MODEL"
echo "  Plot Confusion Matrix: $PLOT_CONFUSION"
echo "  Quick Test Mode: $QUICK_TEST"

# Display model-specific parameters
case $MODEL in
    "alexnet")
        echo "  AlexNet Dropout: $ALEXNET_DROPOUT"
        ;;
    "simple_cnn")
        echo "  SimpleCNN Channels: $SIMPLE_CNN_CHANNELS"
        ;;
    "vgg")
        echo "  VGG Config: $VGG_CONFIG"
        ;;
    "resnet")
        echo "  ResNet Blocks: $RESNET_BLOCKS"
        ;;
    "densenet")
        echo "  DenseNet Growth Rate: $DENSENET_GROWTH"
        echo "  DenseNet Blocks: $DENSENET_BLOCKS"
        ;;
    "mobilenet")
        echo "  MobileNet Width Multiplier: $MOBILENET_WIDTH_MULT"
        ;;
    "mlp")
        echo "  MLP Hidden Layers: $MLP_HIDDEN"
        ;;
    "vit")
        echo "  ViT Patch Size: $VIT_PATCH_SIZE"
        echo "  ViT Embedding Dimension: $VIT_EMBED_DIM"
        echo "  ViT Depth: $VIT_DEPTH"
        echo "  ViT Number of Heads: $VIT_NUM_HEADS"
        echo "  ViT MLP Ratio: $VIT_MLP_RATIO"
        echo "  ViT Dropout Rate: $VIT_DROP_RATE"
        echo "  ViT Attention Dropout Rate: $VIT_ATTN_DROP_RATE"
        ;;
    "xception")
        echo "  Xception Middle Blocks: $XCEPTION_MIDDLE_BLOCKS"
        ;;
    "efficientnet")
        echo "  EfficientNet Width Multiplier: $EFFICIENTNET_WIDTH_MULT"
        echo "  EfficientNet Depth Multiplier: $EFFICIENTNET_DEPTH_MULT"
        echo "  EfficientNet Dropout: $EFFICIENTNET_DROPOUT"
        echo "  EfficientNet Reduction: $EFFICIENTNET_REDUCTION"
        ;;
    "squeezenet")
        echo "  SqueezeNet Version: $SQUEEZENET_VERSION"
        ;;
    "bert")
        echo "  BERT Hidden Size: $BERT_HIDDEN_SIZE"
        echo "  BERT Num Layers: $BERT_NUM_LAYERS"
        echo "  BERT Num Heads: $BERT_NUM_HEADS"
        echo "  BERT MLP Ratio: $BERT_MLP_RATIO"
        echo "  BERT Dropout: $BERT_DROPOUT"
        echo "  BERT Max Seq Length: $BERT_MAX_SEQ_LENGTH"
        ;;
    "gpt")
        echo "  GPT Hidden Size: $GPT_HIDDEN_SIZE"
        echo "  GPT Num Layers: $GPT_NUM_LAYERS"
        echo "  GPT Num Heads: $GPT_NUM_HEADS"
        echo "  GPT MLP Ratio: $GPT_MLP_RATIO"
        echo "  GPT Dropout: $GPT_DROPOUT"
        echo "  GPT Max Seq Length: $GPT_MAX_SEQ_LENGTH"
        ;;
    "lstm"|"gru")
        echo "  RNN Hidden Size: $RNN_HIDDEN_SIZE"
        echo "  RNN Num Layers: $RNN_NUM_LAYERS"
        echo "  RNN Dropout: $RNN_DROPOUT"
        echo "  RNN Bidirectional: $RNN_BIDIRECTIONAL"
        ;;
    "vanilla_gan")
        echo "  Latent Dimension: $LATENT_DIM"
        echo "  Generator Hidden Size: $GENERATOR_HIDDEN"
        echo "  Discriminator Hidden Size: $DISCRIMINATOR_HIDDEN"
        ;;
    "dcgan"|"wgan"|"cgan")
        echo "  Latent Dimension: $LATENT_DIM"
        echo "  Generator Channels: $GENERATOR_CHANNELS"
        echo "  Discriminator Channels: $DISCRIMINATOR_CHANNELS"
        ;;
    "simple_ae")
        echo "  SimpleAE Latent Dimension: $simple_ae_latent_dim"
        echo "  SimpleAE Hidden Dimensions: $simple_ae_hidden_dims"
        ;;
    "conv_ae")
        echo "  ConvAE Latent Dimension: $conv_ae_latent_dim"
        echo "  ConvAE Channels: $conv_ae_channels"
        ;;
    "vae")
        echo "  VAE Latent Dimension: $vae_latent_dim"
        echo "  VAE Hidden Dimensions: $vae_hidden_dims"
        ;;
    "denoising_ae")
        echo "  DenoisingAE Noise Factor: $denoising_ae_noise_factor"
        echo "  DenoisingAE Hidden Dimensions: $denoising_ae_hidden_dims"
        ;;
esac

echo ""
echo -e "${YELLOW}Running command:${NC} $CMD"
echo ""

$CMD

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Training completed successfully!${NC}"
    echo "Results are saved in the $OUTPUT_DIR directory."
else
    echo -e "${RED}Training failed. Please check the error messages above.${NC}"
    exit 1
fi 