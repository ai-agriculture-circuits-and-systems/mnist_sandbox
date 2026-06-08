# Fruits Regression Report

## Run configuration

| Setting | Value |
| --- | --- |
| Generated | 2026-06-08T01:02:43 |
| Dataset | fruits |
| Classes | 5 |
| Class names | apple, banana, grape, orange, pear |
| Mode | quick-test |
| Max epochs | 10 |
| Early-stop patience | 3 |
| Min delta | 0.1 |
| NAS trials per config | 2 |
| Workers | 3 |
| Max batch size | 16 |
| Total wall time | 312.7s |

Training stops when validation metric shows no significant improvement for 3 consecutive epochs.

## Classification — best per model

| Rank | Model | Loss | Optimizer | Hyperparameters | Test Acc (%) | Macro-F1 (%) | Test Loss | Epochs Run | Convergence Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | mobilenetv2 | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 46.67 | 42.25 | 1.0833 | 10 | 10 |
| 2 | resnet | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 45.00 | 24.44 | 1.6144 | 9 | 6 |
| 3 | lenet | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 30.00 | 16.30 | 1.0499 | 9 | 6 |

## Classification — all trials (36 rows)

| Rank | Model | Loss | Optimizer | Hyperparameters | Test Acc (%) | Macro-F1 (%) | Test Loss | Epochs Run | Convergence Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | mobilenetv2 | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 46.67 | 42.25 | 1.0833 | 10 | 10 |
| 2 | resnet | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 45.00 | 24.44 | 1.6144 | 9 | 6 |
| 3 | resnet | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 45.00 | 44.30 | 0.9915 | 10 | 8 |
| 4 | mobilenetv2 | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 45.00 | 38.06 | 1.1715 | 10 | 8 |
| 5 | resnet | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.001 | 35.00 | 30.01 | 3.2581 | 5 | 2 |
| 6 | mobilenetv2 | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 33.33 | 18.78 | 2.1573 | 7 | 4 |
| 7 | resnet | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 31.67 | 27.16 | 1.8044 | 10 | 9 |
| 8 | resnet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 31.67 | 21.20 | 1.0011 | 10 | 7 |
| 9 | lenet | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 30.00 | 16.30 | 1.0499 | 9 | 6 |
| 10 | resnet | cross_entropy | adam | batch_size=16, lr=0.001, weight_decay=0.001 | 28.33 | 24.26 | 4.3095 | 5 | 2 |
| 11 | lenet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 26.67 | 8.42 | 1.6634 | 4 | 1 |
| 12 | lenet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 26.67 | 8.42 | 1.6580 | 6 | 3 |
| 13 | lenet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 26.67 | 8.42 | 1.6191 | 4 | 1 |
| 14 | lenet | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 26.67 | 16.82 | 1.0658 | 4 | 1 |
| 15 | lenet | focal_loss | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0 | 26.67 | 8.42 | 1.0628 | 4 | 1 |
| 16 | resnet | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 26.67 | 13.97 | 4.5284 | 9 | 6 |
| 17 | resnet | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 26.67 | 8.42 | 7.7289 | 6 | 3 |
| 18 | mobilenetv2 | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 26.67 | 8.42 | 5.5403 | 6 | 3 |
| 19 | mobilenetv2 | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 26.67 | 8.42 | 1.6492 | 4 | 1 |
| 20 | mobilenetv2 | cross_entropy | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 26.67 | 8.42 | 5.8751 | 4 | 1 |
| 21 | mobilenetv2 | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 26.67 | 8.42 | 5.6803 | 4 | 1 |
| 22 | mobilenetv2 | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 26.67 | 8.42 | 2.0006 | 7 | 4 |
| 23 | resnet | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 25.00 | 12.46 | 2.5161 | 4 | 1 |
| 24 | resnet | focal_loss | adam | batch_size=16, lr=0.0001, weight_decay=0.001 | 25.00 | 16.06 | 3.7282 | 10 | 8 |
| 25 | lenet | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 23.33 | 7.57 | 1.6033 | 4 | 1 |
| 26 | lenet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 23.33 | 7.57 | 1.6458 | 4 | 1 |
| 27 | lenet | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 23.33 | 7.57 | 1.6135 | 4 | 1 |
| 28 | lenet | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 23.33 | 7.57 | 1.6073 | 4 | 1 |
| 29 | resnet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 23.33 | 8.48 | 3.3969 | 7 | 4 |
| 30 | resnet | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 21.67 | 11.94 | 2.1460 | 7 | 4 |
| 31 | lenet | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 18.33 | 6.20 | 1.6118 | 4 | 1 |
| 32 | lenet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 18.33 | 6.20 | 1.0346 | 4 | 1 |
| 33 | mobilenetv2 | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 18.33 | 6.20 | 5.6554 | 6 | 3 |
| 34 | mobilenetv2 | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 18.33 | 6.20 | 2.2802 | 4 | 1 |
| 35 | mobilenetv2 | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 18.33 | 6.20 | 2.0813 | 4 | 1 |
| 36 | mobilenetv2 | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 18.33 | 6.20 | 4.0212 | 5 | 2 |

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
