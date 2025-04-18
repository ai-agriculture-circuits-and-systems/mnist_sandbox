#!/bin/bash

# Create or clear the log file
echo "Starting test run at $(date)" > test.log

# Array of all available models from model_factory.py
models=(
    "mlp"
    "alexnet"
    "simple_cnn"
    "vgg"
    "resnet"
    "densenet"
    "mobilenet"
    "bert"
    "gpt"
    "lstm"
    "gru"
    "vanilla_gan"
    "dcgan"
    "wgan"
    "cgan"
    "simple_ae"
    "conv_ae"
    "vae"
    "denoising_ae"
    "squeezenet"
    "efficientnet"
    "xception"
    "vit"
)

# Loop through each model and run it
for model in "${models[@]}"; do
    echo "Running model: $model" | tee -a test.log
    ./run.sh -m "$model" --quick-test 2>&1 | tee -a test.log
    echo "Finished running $model" | tee -a test.log
    echo "----------------------------------------" | tee -a test.log
done

echo "Test run completed at $(date)" | tee -a test.log

