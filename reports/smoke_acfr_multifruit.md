# Acfr_Multifruit Regression Report

## Run configuration

| Setting | Value |
| --- | --- |
| Generated | 2026-06-02T19:40:47 |
| Dataset | acfr_multifruit |
| Classes | 3 |
| Class names | almond, apple, mangoe |
| Mode | quick-test |
| Max epochs | 10 |
| Early-stop patience | 3 |
| Min delta | 0.1 |
| NAS trials per config | 2 |
| Workers | 3 |
| Max batch size | 16 |
| Total wall time | 308.8s |

Training stops when validation metric shows no significant improvement for 3 consecutive epochs.

## Classification — best per model

| Rank | Model | Loss | Optimizer | Hyperparameters | Test Acc (%) | Macro-F1 (%) | Test Loss | Epochs Run | Convergence Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | lenet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 85.00 | 67.72 | 0.6023 | 5 | 2 |
| 2 | mobilenetv2 | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 75.00 | 66.56 | 0.1909 | 10 | 7 |
| 3 | resnet | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 71.67 | 63.99 | 0.9230 | 9 | 6 |

## Classification — all trials (36 rows)

| Rank | Model | Loss | Optimizer | Hyperparameters | Test Acc (%) | Macro-F1 (%) | Test Loss | Epochs Run | Convergence Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | lenet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 85.00 | 67.72 | 0.6023 | 5 | 2 |
| 2 | lenet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 85.00 | 60.46 | 0.5466 | 6 | 3 |
| 3 | lenet | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 85.00 | 60.74 | 0.5705 | 4 | 1 |
| 4 | lenet | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 81.67 | 57.88 | 0.8882 | 10 | 7 |
| 5 | lenet | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 81.67 | 65.57 | 0.2217 | 5 | 2 |
| 6 | lenet | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 78.33 | 66.39 | 0.2271 | 8 | 5 |
| 7 | mobilenetv2 | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 75.00 | 66.56 | 0.1909 | 10 | 7 |
| 8 | resnet | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 71.67 | 63.99 | 0.9230 | 9 | 6 |
| 9 | resnet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 66.67 | 60.00 | 2.2338 | 10 | 7 |
| 10 | resnet | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 66.67 | 63.15 | 0.2182 | 10 | 10 |
| 11 | mobilenetv2 | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 61.67 | 43.01 | 0.8326 | 5 | 2 |
| 12 | resnet | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 56.67 | 54.93 | 0.6033 | 9 | 6 |
| 13 | lenet | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 48.33 | 21.72 | 1.0976 | 4 | 1 |
| 14 | lenet | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 48.33 | 21.72 | 1.0471 | 4 | 1 |
| 15 | lenet | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 48.33 | 21.72 | 1.0869 | 4 | 1 |
| 16 | lenet | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 48.33 | 21.72 | 0.9918 | 4 | 1 |
| 17 | lenet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 48.33 | 21.72 | 1.1417 | 4 | 1 |
| 18 | resnet | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.001 | 48.33 | 21.72 | 2.5277 | 4 | 1 |
| 19 | resnet | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 48.33 | 21.72 | 1.2378 | 4 | 1 |
| 20 | resnet | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 48.33 | 21.72 | 8.6150 | 4 | 1 |
| 21 | resnet | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 48.33 | 21.72 | 1.1574 | 4 | 1 |
| 22 | resnet | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 48.33 | 21.72 | 2.9091 | 4 | 1 |
| 23 | resnet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 48.33 | 21.72 | 0.4872 | 4 | 1 |
| 24 | mobilenetv2 | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 48.33 | 21.72 | 2.2635 | 5 | 2 |
| 25 | mobilenetv2 | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 48.33 | 21.72 | 1.1618 | 4 | 1 |
| 26 | mobilenetv2 | label_smoothing | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 48.33 | 21.72 | 1.1701 | 4 | 1 |
| 27 | mobilenetv2 | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 48.33 | 21.72 | 2.9275 | 4 | 1 |
| 28 | mobilenetv2 | cross_entropy | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 40.00 | 19.05 | 5.4476 | 4 | 1 |
| 29 | mobilenetv2 | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 40.00 | 19.05 | 3.1852 | 4 | 1 |
| 30 | mobilenetv2 | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 40.00 | 19.05 | 2.4909 | 4 | 1 |
| 31 | mobilenetv2 | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 40.00 | 19.05 | 4.8995 | 4 | 1 |
| 32 | mobilenetv2 | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 40.00 | 19.05 | 1.7224 | 4 | 1 |
| 33 | mobilenetv2 | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 40.00 | 19.05 | 2.3198 | 4 | 1 |
| 34 | resnet | cross_entropy | adam | batch_size=16, lr=0.001, weight_decay=0.001 | 20.00 | 17.33 | 2.6858 | 5 | 2 |
| 35 | lenet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 11.67 | 6.97 | 0.5086 | 4 | 1 |
| 36 | resnet | focal_loss | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 11.67 | 6.97 | 1.3954 | 4 | 1 |

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
