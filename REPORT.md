# Strawberry Regression Report

## Run configuration

| Setting | Value |
| --- | --- |
| Generated | 2026-05-17T20:40:32 |
| Dataset | strawberry |
| Classes | 6 |
| Class names | early-turning, green, late-turning, red, turning, white |
| Mode | quick-test |
| Max epochs | 1 |
| Early-stop patience | 3 |
| Min delta | 0.1 |
| NAS trials per config | 2 |
| Workers | 1 |
| Max batch size | 32 |
| Total wall time | 312.7s |

Training stops when validation metric shows no significant improvement for 3 consecutive epochs.

## Classification — best per model

| Rank | Model | Loss | Optimizer | Hyperparameters | Test Acc (%) | Test Loss | Epochs Run | Convergence Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | capsnet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 27.27 | 1.7918 | 1 | 1 |

## Classification — all trials (12 rows)

| Rank | Model | Loss | Optimizer | Hyperparameters | Test Acc (%) | Test Loss | Epochs Run | Convergence Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | capsnet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 27.27 | 1.7918 | 1 | 1 |
| 2 | capsnet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 3.64 | 1.7918 | 1 | 1 |
| 3 | capsnet | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 3.64 | 1.7918 | 1 | 1 |
| 4 | capsnet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 3.64 | 1.7918 | 1 | 1 |
| 5 | capsnet | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 3.64 | 1.7918 | 1 | 1 |
| 6 | capsnet | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 3.64 | 1.7918 | 1 | 1 |
| 7 | capsnet | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 3.64 | 1.7918 | 1 | 1 |
| 8 | capsnet | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 3.64 | 1.7918 | 1 | 1 |
| 9 | capsnet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 3.64 | 1.2443 | 1 | 1 |
| 10 | capsnet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 3.64 | 1.2443 | 1 | 1 |
| 11 | capsnet | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 3.64 | 1.2443 | 1 | 1 |
| 12 | capsnet | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 3.64 | 1.2443 | 1 | 1 |

## Autoencoder — best per model

| Rank | Model | Loss | Optimizer | Hyperparameters | Recon Loss | Epochs Run | Convergence Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | vae | mse | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 0.0000 | 1 | 1 |

## Autoencoder — all trials (12 rows)

| Rank | Model | Loss | Optimizer | Hyperparameters | Recon Loss | Epochs Run | Convergence Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | vae | mse | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 2 | vae | mse | sgd | batch_size=32, lr=0.01, weight_decay=0.0 | 0.0000 | 1 | 1 |
| 3 | vae | mse | adamw | batch_size=16, lr=0.0001, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 4 | vae | mse | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 5 | vae | l1 | adam | batch_size=16, lr=0.0001, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 6 | vae | l1 | sgd | batch_size=32, lr=0.01, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 7 | vae | l1 | adamw | batch_size=32, lr=0.001, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 8 | vae | l1 | rmsprop | batch_size=32, lr=0.0001, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 9 | vae | bce | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 10 | vae | bce | sgd | batch_size=32, lr=0.0001, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 11 | vae | bce | adamw | batch_size=32, lr=0.0001, weight_decay=0.0 | 0.0000 | 1 | 1 |
| 12 | vae | bce | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 0.0000 | 1 | 1 |

## GAN — best per model

| Rank | Model | Loss | Optimizer | Hyperparameters | G Loss | D Loss | Epochs Run | Convergence Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## GAN — all trials (0 rows)

| Rank | Model | Loss | Optimizer | Hyperparameters | G Loss | D Loss | Epochs Run | Convergence Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Search Space

- Loss (classification): cross_entropy, label_smoothing, focal_loss
- Loss (autoencoder): mse, l1, bce
- Loss (GAN): bce, wasserstein (informational; GANs use fixed objectives)
- Optimizers: adam, sgd, adamw, rmsprop
- Hyperparameters: {"lr": [0.0001, 0.001, 0.01], "batch_size": [8, 16, 32], "weight_decay": [0.0, 0.0001, 0.001]}
