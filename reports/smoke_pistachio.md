# Pistachio Regression Report

## Run configuration

| Setting | Value |
| --- | --- |
| Generated | 2026-05-25T16:24:14 |
| Dataset | pistachio |
| Classes | 2 |
| Class names | kirmizi, siirt |
| Mode | quick-test |
| Max epochs | 10 |
| Early-stop patience | 3 |
| Min delta | 0.1 |
| NAS trials per config | 2 |
| Workers | 3 |
| Max batch size | 16 |
| Total wall time | 411.7s |

Training stops when validation metric shows no significant improvement for 3 consecutive epochs.

## Classification — best per model

| Rank | Model | Loss | Optimizer | Hyperparameters | Test Acc (%) | Macro-F1 (%) | Test Loss | Epochs Run | Convergence Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | mobilenetv2 | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 53.33 | 40.34 | 11.1042 | 7 | 4 |
| 2 | lenet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 51.67 | 36.93 | 0.1734 | 4 | 1 |
| 3 | resnet | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.001 | 50.00 | 33.33 | 1.4494 | 4 | 1 |

## Classification — all trials (36 rows)

| Rank | Model | Loss | Optimizer | Hyperparameters | Test Acc (%) | Macro-F1 (%) | Test Loss | Epochs Run | Convergence Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | mobilenetv2 | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 53.33 | 40.34 | 11.1042 | 7 | 4 |
| 2 | lenet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 51.67 | 36.93 | 0.1734 | 4 | 1 |
| 3 | lenet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 0.7217 | 4 | 1 |
| 4 | lenet | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 50.00 | 33.33 | 0.6936 | 4 | 1 |
| 5 | lenet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.7314 | 4 | 1 |
| 6 | lenet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 0.7064 | 4 | 1 |
| 7 | lenet | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.7232 | 4 | 1 |
| 8 | lenet | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.6929 | 4 | 1 |
| 9 | lenet | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 50.00 | 33.33 | 0.7098 | 4 | 1 |
| 10 | lenet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 0.7054 | 4 | 1 |
| 11 | lenet | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 0.1771 | 4 | 1 |
| 12 | lenet | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 50.00 | 33.33 | 0.1754 | 4 | 1 |
| 13 | lenet | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 0.1744 | 4 | 1 |
| 14 | resnet | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.001 | 50.00 | 33.33 | 1.4494 | 4 | 1 |
| 15 | resnet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 0.9846 | 4 | 1 |
| 16 | resnet | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 2.9811 | 4 | 1 |
| 17 | resnet | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 50.00 | 33.33 | 8.6516 | 4 | 1 |
| 18 | resnet | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 50.00 | 33.33 | 211.5216 | 4 | 1 |
| 19 | resnet | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 50.00 | 33.33 | 0.7096 | 4 | 1 |
| 20 | resnet | label_smoothing | adamw | batch_size=16, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 1.0761 | 4 | 1 |
| 21 | resnet | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 2.6124 | 4 | 1 |
| 22 | resnet | focal_loss | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 50.00 | 33.33 | 0.2392 | 4 | 1 |
| 23 | resnet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.1893 | 4 | 1 |
| 24 | resnet | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 50.00 | 33.33 | 0.3676 | 4 | 1 |
| 25 | resnet | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 0.1898 | 4 | 1 |
| 26 | mobilenetv2 | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 1.5960 | 4 | 1 |
| 27 | mobilenetv2 | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.8264 | 4 | 1 |
| 28 | mobilenetv2 | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 12.9330 | 4 | 1 |
| 29 | mobilenetv2 | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 11.0284 | 4 | 1 |
| 30 | mobilenetv2 | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 3.0679 | 4 | 1 |
| 31 | mobilenetv2 | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 9.2044 | 4 | 1 |
| 32 | mobilenetv2 | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 0.6997 | 4 | 1 |
| 33 | mobilenetv2 | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 6.4571 | 4 | 1 |
| 34 | mobilenetv2 | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.2017 | 4 | 1 |
| 35 | mobilenetv2 | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 1.6160 | 4 | 1 |
| 36 | mobilenetv2 | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 50.00 | 33.33 | 2.9893 | 4 | 1 |

## Autoencoder — best per model

| Rank | Model | Loss | Optimizer | Hyperparameters | Recon Loss | Epochs Run | Convergence Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- |

## Autoencoder — all trials (0 rows)

| Rank | Model | Loss | Optimizer | Hyperparameters | Recon Loss | Epochs Run | Convergence Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- |

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
