# Mnist Regression Report

## Run configuration

| Setting | Value |
| --- | --- |
| Generated | 2026-05-25T16:15:06 |
| Dataset | mnist |
| Classes | 10 |
| Class names | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 |
| Mode | quick-test |
| Max epochs | 10 |
| Early-stop patience | 3 |
| Min delta | 0.1 |
| NAS trials per config | 2 |
| Workers | 3 |
| Max batch size | 16 |
| Total wall time | 154.2s |

Training stops when validation metric shows no significant improvement for 3 consecutive epochs.

## Classification — best per model

| Rank | Model | Loss | Optimizer | Hyperparameters | Test Acc (%) | Macro-F1 (%) | Test Loss | Epochs Run | Convergence Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | resnet | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 100.00 | 100.00 | 0.0137 | 9 | 6 |
| 2 | mobilenetv2 | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 100.00 | 100.00 | 0.0702 | 9 | 6 |
| 3 | lenet | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 89.00 | 88.62 | 0.1320 | 10 | 10 |

## Classification — all trials (36 rows)

| Rank | Model | Loss | Optimizer | Hyperparameters | Test Acc (%) | Macro-F1 (%) | Test Loss | Epochs Run | Convergence Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | resnet | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 100.00 | 100.00 | 0.0137 | 9 | 6 |
| 2 | mobilenetv2 | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 100.00 | 100.00 | 0.0702 | 9 | 6 |
| 3 | mobilenetv2 | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 100.00 | 100.00 | 0.0628 | 9 | 6 |
| 4 | mobilenetv2 | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 100.00 | 100.00 | 0.0637 | 10 | 8 |
| 5 | mobilenetv2 | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 100.00 | 100.00 | 0.6039 | 10 | 8 |
| 6 | mobilenetv2 | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 100.00 | 100.00 | 0.6125 | 10 | 10 |
| 7 | mobilenetv2 | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 100.00 | 100.00 | 0.6490 | 10 | 7 |
| 8 | mobilenetv2 | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 100.00 | 100.00 | 0.0014 | 10 | 8 |
| 9 | mobilenetv2 | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 100.00 | 100.00 | 0.0007 | 9 | 6 |
| 10 | mobilenetv2 | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 95.00 | 94.60 | 0.3126 | 9 | 6 |
| 11 | resnet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 92.00 | 90.67 | 0.9549 | 9 | 6 |
| 12 | lenet | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 89.00 | 88.62 | 0.1320 | 10 | 10 |
| 13 | mobilenetv2 | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 88.00 | 88.11 | 0.3050 | 7 | 4 |
| 14 | lenet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 85.00 | 83.09 | 0.7382 | 10 | 10 |
| 15 | resnet | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 82.00 | 78.85 | 1.0220 | 10 | 10 |
| 16 | lenet | focal_loss | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0 | 81.00 | 81.02 | 0.4039 | 10 | 10 |
| 17 | lenet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 77.00 | 73.57 | 0.6936 | 10 | 10 |
| 18 | resnet | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 71.00 | 66.47 | 0.9341 | 10 | 7 |
| 19 | lenet | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 69.00 | 60.36 | 1.3241 | 10 | 10 |
| 20 | lenet | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 58.00 | 44.13 | 1.5845 | 10 | 10 |
| 21 | mobilenetv2 | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 49.00 | 37.79 | 1.4484 | 10 | 10 |
| 22 | lenet | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 47.00 | 29.72 | 1.3063 | 10 | 9 |
| 23 | resnet | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 47.00 | 31.67 | 2.1478 | 10 | 8 |
| 24 | resnet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 43.00 | 25.59 | 1.5900 | 10 | 10 |
| 25 | resnet | cross_entropy | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 42.00 | 33.49 | 1.5278 | 10 | 9 |
| 26 | resnet | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 32.00 | 17.19 | 2.5453 | 6 | 3 |
| 27 | resnet | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 22.00 | 7.66 | 2.1968 | 4 | 1 |
| 28 | lenet | label_smoothing | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 16.00 | 5.49 | 2.3007 | 5 | 2 |
| 29 | lenet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 2.61 | 2.2764 | 6 | 3 |
| 30 | lenet | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.61 | 2.2964 | 4 | 1 |
| 31 | lenet | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 15.00 | 2.61 | 1.8292 | 5 | 2 |
| 32 | resnet | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.001 | 15.00 | 2.61 | 3.1553 | 4 | 1 |
| 33 | resnet | focal_loss | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 15.00 | 2.61 | 2.7911 | 4 | 1 |
| 34 | resnet | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 2.61 | 2.1810 | 4 | 1 |
| 35 | mobilenetv2 | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 2.61 | 2.4696 | 4 | 1 |
| 36 | lenet | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 11.00 | 1.98 | 2.3040 | 4 | 1 |

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
