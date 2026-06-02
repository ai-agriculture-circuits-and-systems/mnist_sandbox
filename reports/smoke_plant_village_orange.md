# Plant_Village_Orange Regression Report

## Run configuration

| Setting | Value |
| --- | --- |
| Generated | 2026-05-19T23:55:11 |
| Dataset | plant_village_orange |
| Classes | 2 |
| Class names | huanglongbing_citrus_greening, background_without_leaves |
| Mode | quick-test |
| Max epochs | 10 |
| Early-stop patience | 3 |
| Min delta | 0.1 |
| NAS trials per config | 2 |
| Workers | 3 |
| Max batch size | 16 |
| Total wall time | 298.9s |

Training stops when validation metric shows no significant improvement for 3 consecutive epochs.

## Classification — best per model

| Rank | Model | Loss | Optimizer | Hyperparameters | Test Acc (%) | Test Loss | Epochs Run | Convergence Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | mobilenetv2 | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 98.33 | 0.1152 | 10 | 9 |
| 2 | lenet | focal_loss | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0 | 96.67 | 0.1281 | 8 | 5 |
| 3 | resnet | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 91.67 | 0.4110 | 5 | 2 |

## Classification — all trials (36 rows)

| Rank | Model | Loss | Optimizer | Hyperparameters | Test Acc (%) | Test Loss | Epochs Run | Convergence Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | mobilenetv2 | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 98.33 | 0.1152 | 10 | 9 |
| 2 | lenet | focal_loss | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0 | 96.67 | 0.1281 | 8 | 5 |
| 3 | mobilenetv2 | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 96.67 | 2.3592 | 7 | 4 |
| 4 | mobilenetv2 | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 96.67 | 0.1139 | 10 | 8 |
| 5 | resnet | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 91.67 | 0.4110 | 5 | 2 |
| 6 | mobilenetv2 | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 91.67 | 0.2434 | 6 | 3 |
| 7 | lenet | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 90.00 | 0.8518 | 4 | 1 |
| 8 | mobilenetv2 | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 90.00 | 0.8678 | 5 | 2 |
| 9 | mobilenetv2 | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 88.33 | 0.4734 | 8 | 5 |
| 10 | resnet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 86.67 | 34.5739 | 8 | 5 |
| 11 | resnet | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 83.33 | 0.6172 | 6 | 3 |
| 12 | lenet | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 81.67 | 0.1784 | 6 | 3 |
| 13 | mobilenetv2 | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 73.33 | 1.0412 | 8 | 5 |
| 14 | lenet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 71.67 | 0.8363 | 6 | 3 |
| 15 | lenet | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 71.67 | 0.6878 | 4 | 1 |
| 16 | lenet | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 71.67 | 0.1932 | 4 | 1 |
| 17 | resnet | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.001 | 71.67 | 1.1384 | 4 | 1 |
| 18 | resnet | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 71.67 | 0.7226 | 6 | 3 |
| 19 | resnet | label_smoothing | adamw | batch_size=16, lr=0.0001, weight_decay=0.0001 | 71.67 | 0.7818 | 4 | 1 |
| 20 | resnet | focal_loss | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 71.67 | 0.2255 | 4 | 1 |
| 21 | resnet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 71.67 | 0.1406 | 5 | 2 |
| 22 | resnet | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 71.67 | 0.3013 | 4 | 1 |
| 23 | resnet | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 71.67 | 0.2583 | 5 | 2 |
| 24 | mobilenetv2 | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 71.67 | 0.6237 | 5 | 2 |
| 25 | mobilenetv2 | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 71.67 | 0.6238 | 4 | 1 |
| 26 | mobilenetv2 | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 71.67 | 1.4166 | 4 | 1 |
| 27 | mobilenetv2 | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 71.67 | 1.5772 | 4 | 1 |
| 28 | mobilenetv2 | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 71.67 | 0.2453 | 5 | 2 |
| 29 | resnet | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.001 | 30.00 | 104.3723 | 7 | 4 |
| 30 | lenet | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 28.33 | 0.7006 | 4 | 1 |
| 31 | lenet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 28.33 | 0.8599 | 4 | 1 |
| 32 | lenet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 28.33 | 0.9170 | 4 | 1 |
| 33 | lenet | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 28.33 | 0.8734 | 4 | 1 |
| 34 | lenet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 28.33 | 0.7318 | 4 | 1 |
| 35 | lenet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 28.33 | 0.1778 | 4 | 1 |
| 36 | resnet | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 28.33 | 0.7109 | 4 | 1 |

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
