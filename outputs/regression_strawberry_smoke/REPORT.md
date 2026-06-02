# Strawberry Regression Report

Generated: 2026-05-17T15:37:19
Dataset: strawberry (6 classes: early-turning, green, late-turning, red, turning, white)
Mode: quick-test
Max epochs: 2 | Early-stop patience: 3 | Min delta: 0.1 | NAS trials per config: 1 | Workers: 1 | Max batch: 32
Total wall time: 142.1s

Training stops when validation metric shows no significant improvement for 3 consecutive epochs.

## Classification Models (ranked by test accuracy)

| Rank | Model | Loss | Optimizer | Hyperparameters | Test Acc (%) | Test Loss | Epochs Run | Convergence Epoch |
|------|-------|------|-----------|-----------------|--------------|-----------|------------|-------------------|
| 1 | lenet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 1.6049 | 2 | 1 |
| 2 | lenet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 1.5724 | 2 | 1 |
| 3 | lenet | label_smoothing | rmsprop | batch_size=32, lr=0.01, weight_decay=0.0 | 34.55 | 1.7707 | 2 | 2 |
| 4 | lenet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 0.9881 | 2 | 2 |
| 5 | lenet | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 34.55 | 1.0234 | 2 | 2 |
| 6 | lenet | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 1.0137 | 2 | 2 |
| 7 | lenet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 29.09 | 1.7932 | 2 | 1 |
| 8 | lenet | cross_entropy | rmsprop | batch_size=32, lr=0.0001, weight_decay=0.001 | 23.64 | 1.7748 | 2 | 1 |
| 9 | lenet | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 23.64 | 1.7861 | 2 | 1 |
| 10 | lenet | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 5.45 | 1.7989 | 2 | 1 |
| 11 | lenet | focal_loss | sgd | batch_size=32, lr=0.01, weight_decay=0.001 | 5.45 | 1.2811 | 2 | 1 |
| 12 | lenet | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 3.64 | 1.7772 | 2 | 1 |

## Autoencoder Models (ranked by reconstruction loss, lower is better)

| Rank | Model | Loss | Optimizer | Hyperparameters | Recon Loss | Epochs Run | Convergence Epoch |
|------|-------|------|-----------|-----------------|------------|------------|-------------------|

## GAN Models (ranked by generator loss, lower is better)

| Rank | Model | Loss | Optimizer | Hyperparameters | G Loss | D Loss | Epochs Run | Convergence Epoch |
|------|-------|------|-----------|-----------------|--------|--------|------------|-------------------|

## Search Space

- Loss (classification): cross_entropy, label_smoothing, focal_loss
- Loss (autoencoder): mse, l1, bce
- Loss (GAN): bce, wasserstein (informational; GANs use fixed objectives)
- Optimizers: adam, sgd, adamw, rmsprop
- Hyperparameters: {"lr": [0.0001, 0.001, 0.01], "batch_size": [8, 16, 32], "weight_decay": [0.0, 0.0001, 0.001]}
