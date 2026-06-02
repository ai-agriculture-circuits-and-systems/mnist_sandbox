# Plant_Village_Raspberry Regression Report

## Run configuration

| Setting | Value |
| --- | --- |
| Generated | 2026-05-19T23:50:07 |
| Dataset | plant_village_raspberry |
| Classes | 2 |
| Class names | healthy, background_without_leaves |
| Mode | quick-test |
| Max epochs | 10 |
| Early-stop patience | 3 |
| Min delta | 0.1 |
| NAS trials per config | 2 |
| Workers | 3 |
| Max batch size | 16 |
| Total wall time | 264.2s |

Training stops when validation metric shows no significant improvement for 3 consecutive epochs.

## Classification — best per model

| Rank | Model | Loss | Optimizer | Hyperparameters | Test Acc (%) | Test Loss | Epochs Run | Convergence Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | mobilenetv2 | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 90.00 | 0.5969 | 10 | 7 |
| 2 | lenet | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 75.00 | 0.6899 | 6 | 3 |
| 3 | resnet | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.001 | 56.67 | 1.4333 | 4 | 1 |

## Classification — all trials (36 rows)

| Rank | Model | Loss | Optimizer | Hyperparameters | Test Acc (%) | Test Loss | Epochs Run | Convergence Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | mobilenetv2 | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 90.00 | 0.5969 | 10 | 7 |
| 2 | lenet | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 75.00 | 0.6899 | 6 | 3 |
| 3 | mobilenetv2 | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 58.33 | 0.3777 | 6 | 3 |
| 4 | lenet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 56.67 | 1.2148 | 4 | 1 |
| 5 | lenet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 56.67 | 4.2876 | 4 | 1 |
| 6 | lenet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 56.67 | 15.7670 | 4 | 1 |
| 7 | lenet | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 56.67 | 1.7744 | 4 | 1 |
| 8 | lenet | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 56.67 | 0.6874 | 5 | 2 |
| 9 | lenet | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 56.67 | 0.9138 | 4 | 1 |
| 10 | lenet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 56.67 | 1.4212 | 4 | 1 |
| 11 | lenet | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 56.67 | 0.2890 | 4 | 1 |
| 12 | lenet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 56.67 | 0.1716 | 4 | 1 |
| 13 | lenet | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 56.67 | 0.1783 | 4 | 1 |
| 14 | lenet | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 56.67 | 2.3364 | 4 | 1 |
| 15 | resnet | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.001 | 56.67 | 1.4333 | 4 | 1 |
| 16 | resnet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 56.67 | 4.0931 | 4 | 1 |
| 17 | resnet | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.001 | 56.67 | 6.2358 | 4 | 1 |
| 18 | resnet | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 56.67 | 3.4032 | 4 | 1 |
| 19 | resnet | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 56.67 | 1.8628 | 4 | 1 |
| 20 | resnet | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 56.67 | 0.9485 | 4 | 1 |
| 21 | resnet | label_smoothing | adamw | batch_size=16, lr=0.0001, weight_decay=0.0001 | 56.67 | 1.1162 | 4 | 1 |
| 22 | resnet | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 56.67 | 1.7623 | 4 | 1 |
| 23 | resnet | focal_loss | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 56.67 | 1.2082 | 4 | 1 |
| 24 | resnet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 56.67 | 0.3482 | 4 | 1 |
| 25 | resnet | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 56.67 | 1.8282 | 4 | 1 |
| 26 | resnet | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 56.67 | 0.5263 | 4 | 1 |
| 27 | mobilenetv2 | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 56.67 | 2.9301 | 5 | 2 |
| 28 | mobilenetv2 | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 56.67 | 0.6930 | 4 | 1 |
| 29 | mobilenetv2 | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 56.67 | 0.7037 | 4 | 1 |
| 30 | mobilenetv2 | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 56.67 | 2.8590 | 4 | 1 |
| 31 | mobilenetv2 | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 56.67 | 0.6886 | 4 | 1 |
| 32 | mobilenetv2 | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 56.67 | 1.3435 | 4 | 1 |
| 33 | mobilenetv2 | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 56.67 | 0.8345 | 4 | 1 |
| 34 | mobilenetv2 | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 56.67 | 0.1649 | 6 | 3 |
| 35 | mobilenetv2 | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 56.67 | 0.1772 | 4 | 1 |
| 36 | mobilenetv2 | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 56.67 | 0.2821 | 4 | 1 |

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
