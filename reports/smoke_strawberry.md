# Strawberry Regression Report

## Run configuration

| Setting | Value |
| --- | --- |
| Generated | 2026-05-25T16:56:24 |
| Dataset | strawberry |
| Classes | 6 |
| Class names | early-turning, green, late-turning, red, turning, white |
| Mode | quick-test |
| Max epochs | 10 |
| Early-stop patience | 3 |
| Min delta | 0.1 |
| NAS trials per config | 2 |
| Workers | 3 |
| Max batch size | 16 |
| Total wall time | 2467.0s |

Training stops when validation metric shows no significant improvement for 3 consecutive epochs.

## Classification — best per model

| Rank | Model | Loss | Optimizer | Hyperparameters | Test Acc (%) | Macro-F1 (%) | Test Loss | Epochs Run | Convergence Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | lenet | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 38.18 | 13.96 | 0.9467 | 6 | 3 |
| 2 | mobilenetv2 | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 8.56 | 4.4998 | 4 | 1 |
| 3 | resnet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 8.56 | 2.1932 | 4 | 1 |

## Classification — all trials (36 rows)

| Rank | Model | Loss | Optimizer | Hyperparameters | Test Acc (%) | Macro-F1 (%) | Test Loss | Epochs Run | Convergence Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | lenet | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 38.18 | 13.96 | 0.9467 | 6 | 3 |
| 2 | lenet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 8.56 | 1.5326 | 4 | 1 |
| 3 | lenet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 8.56 | 1.5020 | 6 | 3 |
| 4 | lenet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 8.56 | 1.5183 | 7 | 4 |
| 5 | lenet | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 34.55 | 8.56 | 1.6283 | 5 | 2 |
| 6 | lenet | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 8.56 | 1.7540 | 4 | 1 |
| 7 | lenet | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 34.55 | 8.56 | 1.6341 | 5 | 2 |
| 8 | lenet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 8.56 | 1.6237 | 6 | 3 |
| 9 | lenet | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 8.56 | 0.9616 | 4 | 1 |
| 10 | lenet | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 34.55 | 8.56 | 0.9577 | 5 | 2 |
| 11 | lenet | focal_loss | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0 | 34.55 | 8.56 | 0.9850 | 4 | 1 |
| 12 | mobilenetv2 | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 8.56 | 4.4998 | 4 | 1 |
| 13 | mobilenetv2 | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 8.56 | 1.8801 | 4 | 1 |
| 14 | mobilenetv2 | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 8.56 | 22.1414 | 4 | 1 |
| 15 | mobilenetv2 | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 8.56 | 1.6131 | 5 | 2 |
| 16 | mobilenetv2 | focal_loss | adamw | batch_size=16, lr=0.0001, weight_decay=0.0001 | 34.55 | 8.56 | 1.4688 | 4 | 1 |
| 17 | mobilenetv2 | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 8.56 | 2.9774 | 6 | 3 |
| 18 | resnet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 8.56 | 2.1932 | 4 | 1 |
| 19 | resnet | cross_entropy | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 8.56 | 522.5793 | 4 | 1 |
| 20 | resnet | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 8.56 | 1.6775 | 7 | 4 |
| 21 | resnet | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 34.55 | 8.56 | 1.6232 | 6 | 3 |
| 22 | resnet | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 8.56 | 178.9508 | 4 | 1 |
| 23 | resnet | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 34.55 | 8.56 | 1.6049 | 5 | 2 |
| 24 | resnet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 34.55 | 8.56 | 1.0461 | 4 | 1 |
| 25 | resnet | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 8.56 | 1.0776 | 6 | 3 |
| 26 | mobilenetv2 | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 29.09 | 7.51 | 8.1680 | 5 | 2 |
| 27 | mobilenetv2 | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 29.09 | 7.51 | 2.0019 | 4 | 1 |
| 28 | mobilenetv2 | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 29.09 | 7.51 | 6.5850 | 4 | 1 |
| 29 | mobilenetv2 | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 29.09 | 7.51 | 1.4612 | 4 | 1 |
| 30 | resnet | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.001 | 29.09 | 7.51 | 2.2486 | 4 | 1 |
| 31 | resnet | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 29.09 | 7.51 | 6.4377 | 4 | 1 |
| 32 | resnet | focal_loss | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 29.09 | 7.51 | 3.2573 | 7 | 4 |
| 33 | resnet | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 29.09 | 7.51 | 4.9718 | 5 | 2 |
| 34 | mobilenetv2 | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 23.64 | 6.37 | 2.3546 | 5 | 2 |
| 35 | mobilenetv2 | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 23.64 | 6.37 | 5.3492 | 4 | 1 |
| 36 | lenet | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 3.64 | 1.17 | 1.8084 | 4 | 1 |

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
