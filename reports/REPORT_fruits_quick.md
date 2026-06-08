# Fruits Regression Report

## Run configuration

| Setting | Value |
| --- | --- |
| Generated | 2026-06-08T07:09:32 |
| Dataset | fruits |
| Classes | 5 |
| Class names | apple, banana, grape, orange, pear |
| Mode | quick-test |
| Max epochs | 10 |
| Early-stop patience | 3 |
| Min delta | 0.1 |
| NAS trials per config | 2 |
| Workers | 10 |
| Max batch size | 16 |
| Total wall time | 20860.5s |

Training stops when validation metric shows no significant improvement for 3 consecutive epochs.

## Classification — best per model

| Rank | Model | Loss | Optimizer | Hyperparameters | Test Acc (%) | Macro-F1 (%) | Test Loss | Epochs Run | Convergence Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | repghost | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 71.67 | 70.46 | 0.6266 | 10 | 8 |
| 2 | se_resnet | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 66.67 | 63.03 | 1.2133 | 10 | 9 |
| 3 | mnasnet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 65.00 | 55.42 | 7.3073 | 9 | 6 |
| 4 | simple_cnn | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 63.33 | 63.32 | 1.0465 | 10 | 10 |
| 5 | coord_resnet | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 61.67 | 60.50 | 0.4649 | 10 | 10 |
| 6 | cspnet | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 60.00 | 61.60 | 1.1642 | 10 | 10 |
| 7 | res2net | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 60.00 | 62.45 | 1.1967 | 10 | 9 |
| 8 | cbam_resnet | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 60.00 | 56.22 | 1.3328 | 10 | 10 |
| 9 | vim_tiny | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 60.00 | 60.30 | 1.5258 | 10 | 8 |
| 10 | coatnet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 60.00 | 59.71 | 0.9390 | 10 | 10 |
| 11 | repvgg | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 58.33 | 50.10 | 1.2193 | 10 | 10 |
| 12 | ghostnet | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 56.67 | 48.09 | 0.6358 | 10 | 7 |
| 13 | eca_resnet | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 56.67 | 49.36 | 0.5329 | 10 | 9 |
| 14 | sknet | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 56.67 | 55.35 | 1.2544 | 10 | 9 |
| 15 | convnext | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 55.00 | 54.25 | 0.8071 | 10 | 8 |
| 16 | mlp | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 55.00 | 50.86 | 1.4366 | 9 | 6 |
| 17 | darknet | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 55.00 | 48.63 | 2.2968 | 10 | 9 |
| 18 | hrnet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 53.33 | 42.91 | 1.3572 | 10 | 10 |
| 19 | resnext | cross_entropy | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 53.33 | 45.39 | 1.4089 | 10 | 9 |
| 20 | regnet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 53.33 | 41.77 | 1.8750 | 10 | 8 |
| 21 | hardnet | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 51.67 | 43.98 | 2.3597 | 10 | 7 |
| 22 | resnet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 50.00 | 43.61 | 1.0127 | 10 | 10 |
| 23 | dpn | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 50.00 | 33.95 | 1.1371 | 10 | 10 |
| 24 | lcnet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 48.33 | 39.06 | 1.3219 | 10 | 7 |
| 25 | wide_resnet | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 48.33 | 41.86 | 1.4098 | 9 | 6 |
| 26 | shufflenet | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 46.67 | 24.93 | 3.0807 | 10 | 8 |
| 27 | densenet | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 45.00 | 32.14 | 0.9433 | 10 | 8 |
| 28 | xception | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 43.33 | 30.43 | 1.5163 | 10 | 7 |
| 29 | efficientnetv2 | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 43.33 | 29.99 | 1.7343 | 10 | 7 |
| 30 | mobilenetv3 | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 41.67 | 33.46 | 1.5877 | 10 | 7 |
| 31 | mobilenetv2 | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 41.67 | 33.95 | 1.9387 | 10 | 7 |
| 32 | mobilenet | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 38.33 | 32.85 | 2.0096 | 10 | 10 |
| 33 | swin_tiny | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 38.33 | 26.51 | 0.8731 | 10 | 10 |
| 34 | capsnet | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 38.33 | 31.87 | 1.6094 | 9 | 6 |
| 35 | van | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 38.33 | 23.11 | — | 10 | 10 |
| 36 | squeezenet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 36.67 | 20.77 | 1.6663 | 10 | 9 |
| 37 | vgg | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 35.00 | 19.90 | 1.6082 | 6 | 3 |
| 38 | lenet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 35.00 | 19.41 | 0.9512 | 9 | 6 |
| 39 | inception_resnet | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 35.00 | 22.77 | 27.3197 | 10 | 9 |
| 40 | deit | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 33.33 | 17.71 | 1.6096 | 6 | 3 |
| 41 | vit | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 33.33 | 21.50 | 1.0034 | 6 | 3 |
| 42 | efficientnet | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 33.33 | 20.49 | 2.7498 | 9 | 6 |
| 43 | gpt | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 30.00 | 15.89 | 1.0298 | 6 | 3 |
| 44 | poolformer | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 30.00 | 18.96 | 1.6802 | 7 | 4 |
| 45 | alexnet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 28.33 | 15.42 | 1.6327e+07 | 5 | 2 |
| 46 | lstm | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 28.33 | 14.40 | 1.1028 | 9 | 6 |
| 47 | gru | label_smoothing | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 28.33 | 13.28 | 1.6065 | 6 | 3 |
| 48 | bert | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 28.33 | 15.15 | 1.6088 | 9 | 6 |
| 49 | nin | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 26.67 | 8.42 | 1.6527 | 6 | 3 |
| 50 | googlenet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 26.67 | 8.42 | 1.6346 | 4 | 1 |

## Classification — all trials (600 rows)

| Rank | Model | Loss | Optimizer | Hyperparameters | Test Acc (%) | Macro-F1 (%) | Test Loss | Epochs Run | Convergence Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | repghost | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 71.67 | 70.46 | 0.6266 | 10 | 8 |
| 2 | se_resnet | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 66.67 | 63.03 | 1.2133 | 10 | 9 |
| 3 | mnasnet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 65.00 | 55.42 | 7.3073 | 9 | 6 |
| 4 | se_resnet | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 65.00 | 62.69 | 0.5357 | 10 | 9 |
| 5 | simple_cnn | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 63.33 | 63.32 | 1.0465 | 10 | 10 |
| 6 | simple_cnn | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 61.67 | 62.07 | 1.4794 | 10 | 9 |
| 7 | coord_resnet | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 61.67 | 60.50 | 0.4649 | 10 | 10 |
| 8 | cspnet | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 60.00 | 61.60 | 1.1642 | 10 | 10 |
| 9 | res2net | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 60.00 | 62.45 | 1.1967 | 10 | 9 |
| 10 | cbam_resnet | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 60.00 | 56.22 | 1.3328 | 10 | 10 |
| 11 | vim_tiny | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 60.00 | 60.30 | 1.5258 | 10 | 8 |
| 12 | coatnet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 60.00 | 59.71 | 0.9390 | 10 | 10 |
| 13 | simple_cnn | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 58.33 | 51.37 | 0.5966 | 10 | 8 |
| 14 | cbam_resnet | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 58.33 | 57.61 | 1.2946 | 10 | 10 |
| 15 | repvgg | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 58.33 | 50.10 | 1.2193 | 10 | 10 |
| 16 | ghostnet | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 56.67 | 48.09 | 0.6358 | 10 | 7 |
| 17 | eca_resnet | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 56.67 | 49.36 | 0.5329 | 10 | 9 |
| 18 | se_resnet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 56.67 | 48.84 | 1.3204 | 10 | 9 |
| 19 | sknet | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 56.67 | 55.35 | 1.2544 | 10 | 9 |
| 20 | convnext | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 55.00 | 54.25 | 0.8071 | 10 | 8 |
| 21 | ghostnet | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 55.00 | 51.29 | 0.7670 | 10 | 9 |
| 22 | mlp | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 55.00 | 50.86 | 1.4366 | 9 | 6 |
| 23 | vim_tiny | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 55.00 | 49.27 | 3.3712 | 9 | 6 |
| 24 | vim_tiny | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 55.00 | 50.51 | 1.3059 | 10 | 9 |
| 25 | eca_resnet | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 55.00 | 50.47 | 1.5504 | 10 | 7 |
| 26 | repvgg | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 55.00 | 54.30 | 0.7031 | 10 | 10 |
| 27 | darknet | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 55.00 | 48.63 | 2.2968 | 10 | 9 |
| 28 | se_resnet | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 55.00 | 50.62 | 0.7075 | 10 | 8 |
| 29 | sknet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 55.00 | 43.85 | 1.3286 | 10 | 7 |
| 30 | hrnet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 53.33 | 42.91 | 1.3572 | 10 | 10 |
| 31 | resnext | cross_entropy | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 53.33 | 45.39 | 1.4089 | 10 | 9 |
| 32 | resnext | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 53.33 | 53.86 | 0.9630 | 10 | 8 |
| 33 | convnext | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 53.33 | 51.20 | 0.8348 | 10 | 7 |
| 34 | cbam_resnet | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 53.33 | 48.68 | 1.1439 | 10 | 9 |
| 35 | coord_resnet | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 53.33 | 53.34 | 0.8583 | 10 | 8 |
| 36 | vim_tiny | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 53.33 | 53.67 | 2.4027 | 10 | 8 |
| 37 | vim_tiny | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 53.33 | 46.21 | 1.3238 | 10 | 8 |
| 38 | vim_tiny | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 53.33 | 54.11 | 0.6149 | 10 | 9 |
| 39 | eca_resnet | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 53.33 | 44.72 | 1.1981 | 10 | 7 |
| 40 | eca_resnet | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 53.33 | 53.24 | 0.5111 | 10 | 10 |
| 41 | regnet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 53.33 | 41.77 | 1.8750 | 10 | 8 |
| 42 | darknet | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 53.33 | 44.63 | 1.6380 | 10 | 10 |
| 43 | hardnet | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 51.67 | 43.98 | 2.3597 | 10 | 7 |
| 44 | simple_cnn | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 51.67 | 48.95 | 1.8291 | 10 | 8 |
| 45 | cspnet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 51.67 | 39.70 | 1.0076 | 10 | 9 |
| 46 | convnext | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 51.67 | 50.39 | 1.2675 | 10 | 8 |
| 47 | convnext | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 51.67 | 47.83 | 0.8424 | 10 | 8 |
| 48 | mlp | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 51.67 | 43.13 | 1.3668 | 10 | 7 |
| 49 | cbam_resnet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 51.67 | 46.18 | 1.9014 | 10 | 9 |
| 50 | cbam_resnet | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 51.67 | 44.93 | 0.7580 | 10 | 7 |
| 51 | coord_resnet | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 51.67 | 45.42 | 1.4555 | 10 | 8 |
| 52 | vim_tiny | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 51.67 | 45.85 | 1.1582 | 10 | 10 |
| 53 | vim_tiny | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 51.67 | 46.61 | 0.7169 | 10 | 9 |
| 54 | repvgg | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 51.67 | 51.51 | 1.1579 | 10 | 10 |
| 55 | se_resnet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 51.67 | 38.41 | 3.2881 | 10 | 9 |
| 56 | sknet | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 51.67 | 50.40 | 1.2802 | 10 | 7 |
| 57 | hrnet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 50.00 | 45.48 | 1.3106 | 10 | 8 |
| 58 | convnext | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 50.00 | 41.84 | 1.7140 | 9 | 6 |
| 59 | mlp | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 50.00 | 44.02 | 1.2917 | 10 | 9 |
| 60 | mlp | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 50.00 | 43.87 | 0.6930 | 10 | 9 |
| 61 | mlp | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 50.00 | 44.37 | 0.7494 | 9 | 6 |
| 62 | resnet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 50.00 | 43.61 | 1.0127 | 10 | 10 |
| 63 | eca_resnet | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 50.00 | 40.58 | 1.9124 | 10 | 9 |
| 64 | eca_resnet | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 50.00 | 47.23 | 0.8058 | 9 | 6 |
| 65 | dpn | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 50.00 | 33.95 | 1.1371 | 10 | 10 |
| 66 | hardnet | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 48.33 | 42.71 | 0.7152 | 10 | 8 |
| 67 | simple_cnn | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 48.33 | 36.58 | 6.8443 | 7 | 4 |
| 68 | resnext | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 48.33 | 39.60 | 1.4944 | 10 | 8 |
| 69 | convnext | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 48.33 | 41.46 | 1.0825 | 10 | 9 |
| 70 | mlp | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 48.33 | 42.63 | 1.4629 | 10 | 7 |
| 71 | mlp | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 48.33 | 45.79 | 0.8532 | 10 | 10 |
| 72 | mlp | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 48.33 | 46.88 | 0.7137 | 10 | 8 |
| 73 | res2net | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 48.33 | 31.71 | 1.5804 | 10 | 9 |
| 74 | repghost | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 48.33 | 41.57 | 1.3794 | 10 | 7 |
| 75 | repghost | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 48.33 | 40.94 | 1.4450 | 10 | 8 |
| 76 | coord_resnet | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 48.33 | 40.67 | 1.3045 | 10 | 10 |
| 77 | vim_tiny | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 48.33 | 38.11 | 2.5767 | 10 | 7 |
| 78 | lcnet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 48.33 | 39.06 | 1.3219 | 10 | 7 |
| 79 | repvgg | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 48.33 | 33.68 | 1.9418 | 8 | 5 |
| 80 | se_resnet | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 48.33 | 48.15 | 0.8651 | 10 | 7 |
| 81 | wide_resnet | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 48.33 | 41.86 | 1.4098 | 9 | 6 |
| 82 | hardnet | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 46.67 | 34.83 | 1.3774 | 10 | 10 |
| 83 | shufflenet | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 46.67 | 24.93 | 3.0807 | 10 | 8 |
| 84 | shufflenet | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 46.67 | 32.78 | 0.7629 | 10 | 8 |
| 85 | resnext | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 46.67 | 33.40 | 1.3944 | 10 | 9 |
| 86 | resnext | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 46.67 | 36.45 | 1.6915 | 10 | 7 |
| 87 | mlp | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.0001 | 46.67 | 46.09 | 1.3622 | 10 | 8 |
| 88 | mlp | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 46.67 | 43.84 | 1.3149 | 10 | 10 |
| 89 | mlp | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 46.67 | 38.52 | 1.2137 | 9 | 6 |
| 90 | se_resnet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 46.67 | 44.46 | 1.3795 | 10 | 7 |
| 91 | densenet | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 45.00 | 32.14 | 0.9433 | 10 | 8 |
| 92 | hrnet | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 45.00 | 38.63 | 0.8084 | 9 | 6 |
| 93 | simple_cnn | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 45.00 | 32.57 | 10.6646 | 8 | 5 |
| 94 | simple_cnn | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 45.00 | 44.15 | 2.9192 | 10 | 7 |
| 95 | mlp | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 45.00 | 37.66 | 1.2983 | 10 | 10 |
| 96 | repghost | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 45.00 | 32.88 | 3.2460 | 9 | 6 |
| 97 | coord_resnet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 45.00 | 24.96 | 2.4948 | 9 | 6 |
| 98 | coord_resnet | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 45.00 | 39.68 | 1.2840 | 10 | 10 |
| 99 | eca_resnet | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 45.00 | 38.79 | 1.5514 | 10 | 10 |
| 100 | regnet | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 45.00 | 37.56 | 3.5917 | 10 | 8 |
| 101 | wide_resnet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 45.00 | 39.22 | 1.4190 | 10 | 9 |
| 102 | wide_resnet | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 45.00 | 33.48 | 0.8006 | 10 | 10 |
| 103 | wide_resnet | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 45.00 | 42.75 | 0.8165 | 10 | 10 |
| 104 | hardnet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 43.33 | 31.42 | 2.1335 | 8 | 5 |
| 105 | hardnet | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 43.33 | 36.83 | 2.0076 | 10 | 8 |
| 106 | xception | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 43.33 | 30.43 | 1.5163 | 10 | 7 |
| 107 | hrnet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 43.33 | 30.74 | 1.1686 | 6 | 3 |
| 108 | convnext | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 43.33 | 37.73 | 1.4309 | 10 | 9 |
| 109 | convnext | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 43.33 | 30.71 | 1.6139 | 8 | 5 |
| 110 | convnext | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 43.33 | 30.89 | 1.6624 | 10 | 8 |
| 111 | ghostnet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 43.33 | 38.03 | 1.4610 | 10 | 9 |
| 112 | res2net | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 43.33 | 37.11 | 2.1319 | 9 | 6 |
| 113 | res2net | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 43.33 | 30.31 | 4.0604 | 10 | 8 |
| 114 | cbam_resnet | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 43.33 | 30.72 | 1.4836 | 10 | 7 |
| 115 | coord_resnet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 43.33 | 30.25 | 1.3695 | 10 | 7 |
| 116 | mnasnet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 43.33 | 32.29 | 1.9678 | 9 | 6 |
| 117 | regnet | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 43.33 | 37.21 | 0.8778 | 10 | 10 |
| 118 | efficientnetv2 | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 43.33 | 29.99 | 1.7343 | 10 | 7 |
| 119 | darknet | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 43.33 | 24.12 | 1.1195 | 10 | 10 |
| 120 | se_resnet | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 43.33 | 37.87 | 1.7763 | 10 | 9 |
| 121 | wide_resnet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 43.33 | 37.05 | 2.1383 | 10 | 7 |
| 122 | sknet | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 43.33 | 35.86 | 1.6682 | 9 | 6 |
| 123 | hardnet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 41.67 | 30.39 | 1.6152 | 7 | 4 |
| 124 | mobilenetv3 | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 41.67 | 33.46 | 1.5877 | 10 | 7 |
| 125 | cspnet | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 41.67 | 39.55 | 0.9491 | 10 | 10 |
| 126 | convnext | label_smoothing | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 41.67 | 28.67 | 1.8012 | 10 | 9 |
| 127 | cbam_resnet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 41.67 | 28.66 | 1.7409 | 8 | 5 |
| 128 | vim_tiny | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 41.67 | 35.42 | 1.4811 | 10 | 10 |
| 129 | coatnet | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 41.67 | 36.42 | 2.9090 | 7 | 4 |
| 130 | repvgg | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 41.67 | 30.13 | 11.0505 | 10 | 7 |
| 131 | repvgg | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 41.67 | 28.38 | 2.3763 | 8 | 5 |
| 132 | mobilenetv2 | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 41.67 | 33.95 | 1.9387 | 10 | 7 |
| 133 | shufflenet | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 40.00 | 35.47 | 1.0598 | 10 | 9 |
| 134 | densenet | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 40.00 | 34.41 | 0.8685 | 10 | 9 |
| 135 | convnext | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 40.00 | 32.31 | 2.7205 | 8 | 5 |
| 136 | ghostnet | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 40.00 | 29.61 | 2.9128 | 8 | 5 |
| 137 | cbam_resnet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 40.00 | 24.43 | 0.9527 | 10 | 9 |
| 138 | repghost | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 40.00 | 24.85 | 1.4842 | 10 | 7 |
| 139 | repvgg | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 40.00 | 28.17 | 4.1062 | 10 | 9 |
| 140 | darknet | label_smoothing | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 40.00 | 30.14 | 2.7145 | 10 | 10 |
| 141 | mobilenet | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 38.33 | 32.85 | 2.0096 | 10 | 10 |
| 142 | resnext | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 38.33 | 20.59 | 1.0594 | 8 | 5 |
| 143 | swin_tiny | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 38.33 | 26.51 | 0.8731 | 10 | 10 |
| 144 | coord_resnet | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 38.33 | 31.30 | 1.7024 | 9 | 6 |
| 145 | capsnet | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 38.33 | 31.87 | 1.6094 | 9 | 6 |
| 146 | eca_resnet | cross_entropy | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 38.33 | 30.54 | 1.8197 | 8 | 5 |
| 147 | eca_resnet | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 38.33 | 27.36 | 1.4447 | 7 | 4 |
| 148 | van | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 38.33 | 23.11 | — | 10 | 10 |
| 149 | sknet | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 38.33 | 28.12 | 1.4805 | 10 | 8 |
| 150 | xception | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 36.67 | 31.79 | 1.3755 | 10 | 9 |
| 151 | xception | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 36.67 | 23.22 | 4.0644 | 10 | 9 |
| 152 | densenet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 36.67 | 27.39 | 1.7087 | 10 | 9 |
| 153 | hrnet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 36.67 | 20.76 | 1.4544 | 9 | 6 |
| 154 | simple_cnn | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 36.67 | 24.46 | 2.4162 | 8 | 5 |
| 155 | simple_cnn | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 36.67 | 27.62 | 203.2748 | 6 | 3 |
| 156 | cbam_resnet | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 36.67 | 20.91 | 1.0833 | 5 | 2 |
| 157 | mnasnet | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 36.67 | 24.80 | 1.0128 | 10 | 7 |
| 158 | vim_tiny | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 36.67 | 28.70 | 79.5922 | 6 | 3 |
| 159 | regnet | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 36.67 | 22.67 | 2.2832 | 10 | 7 |
| 160 | squeezenet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 36.67 | 20.77 | 1.6663 | 10 | 9 |
| 161 | mobilenetv2 | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 36.67 | 30.76 | 1.5791 | 10 | 10 |
| 162 | se_resnet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 36.67 | 27.66 | 2.9412 | 9 | 6 |
| 163 | se_resnet | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 36.67 | 23.40 | 2.1340 | 8 | 5 |
| 164 | wide_resnet | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 36.67 | 28.55 | 1.8970 | 10 | 10 |
| 165 | hardnet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 35.00 | 24.62 | 1.6337 | 10 | 7 |
| 166 | densenet | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 35.00 | 20.14 | 4.4944 | 8 | 5 |
| 167 | hrnet | cross_entropy | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 35.00 | 25.36 | 1.6040 | 10 | 8 |
| 168 | hrnet | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 35.00 | 18.67 | 3.0872 | 7 | 4 |
| 169 | hrnet | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 35.00 | 30.51 | 0.9214 | 10 | 10 |
| 170 | simple_cnn | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 35.00 | 21.89 | 1.5867 | 10 | 7 |
| 171 | res2net | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 35.00 | 19.23 | 0.9587 | 7 | 4 |
| 172 | vgg | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 35.00 | 19.90 | 1.6082 | 6 | 3 |
| 173 | lenet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 35.00 | 19.41 | 0.9512 | 9 | 6 |
| 174 | squeezenet | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 35.00 | 19.77 | 1.0665 | 7 | 4 |
| 175 | coatnet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 35.00 | 29.89 | 24.2233 | 8 | 5 |
| 176 | repvgg | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 35.00 | 30.62 | 0.8131 | 10 | 8 |
| 177 | van | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 35.00 | 22.11 | — | 8 | 5 |
| 178 | darknet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 35.00 | 30.21 | 5.2445 | 10 | 7 |
| 179 | mobilenetv2 | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 35.00 | 19.53 | 1.1954 | 10 | 8 |
| 180 | dpn | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 35.00 | 21.62 | 1.6675 | 9 | 6 |
| 181 | inception_resnet | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 35.00 | 22.77 | 27.3197 | 10 | 9 |
| 182 | deit | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 33.33 | 17.71 | 1.6096 | 6 | 3 |
| 183 | densenet | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 33.33 | 21.63 | 13.9571 | 7 | 4 |
| 184 | densenet | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 33.33 | 18.33 | 0.9946 | 10 | 8 |
| 185 | hrnet | label_smoothing | adam | batch_size=16, lr=0.0001, weight_decay=0.0001 | 33.33 | 25.84 | 1.5252 | 10 | 10 |
| 186 | simple_cnn | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 33.33 | 18.51 | 1.8591 | 5 | 2 |
| 187 | vit | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 33.33 | 21.50 | 1.0034 | 6 | 3 |
| 188 | convnext | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 33.33 | 26.64 | 2.1403 | 9 | 6 |
| 189 | ghostnet | cross_entropy | sgd | batch_size=16, lr=0.0001, weight_decay=0.0 | 33.33 | 18.98 | 1.5907 | 8 | 5 |
| 190 | efficientnet | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 33.33 | 20.49 | 2.7498 | 9 | 6 |
| 191 | vim_tiny | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 33.33 | 22.51 | 2.8213 | 7 | 4 |
| 192 | capsnet | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 33.33 | 19.62 | 1.6094 | 4 | 1 |
| 193 | regnet | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 33.33 | 22.73 | 9.7112 | 6 | 3 |
| 194 | coatnet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 33.33 | 24.44 | 1.3940 | 10 | 7 |
| 195 | repvgg | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 33.33 | 24.65 | 1.4185 | 6 | 3 |
| 196 | wide_resnet | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 33.33 | 24.87 | 15.6902 | 7 | 4 |
| 197 | dpn | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 33.33 | 22.00 | 1.8643 | 7 | 4 |
| 198 | sknet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 33.33 | 22.41 | 1.8785 | 6 | 3 |
| 199 | mobilenetv3 | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 31.67 | 15.34 | 2.6382 | 8 | 5 |
| 200 | simple_cnn | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 31.67 | 16.47 | 5.1538 | 5 | 2 |
| 201 | cspnet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 31.67 | 18.18 | 1.6844 | 6 | 3 |
| 202 | vit | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 31.67 | 19.50 | 1.4628 | 9 | 6 |
| 203 | lenet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 31.67 | 16.36 | 1.6380 | 9 | 6 |
| 204 | repghost | label_smoothing | adamw | batch_size=16, lr=0.0001, weight_decay=0.0 | 31.67 | 20.77 | 1.6756 | 5 | 2 |
| 205 | repghost | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 31.67 | 18.70 | 1.2234 | 4 | 1 |
| 206 | coord_resnet | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 31.67 | 24.48 | 2.9115 | 10 | 10 |
| 207 | resnet | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 31.67 | 19.55 | 1.8399 | 8 | 5 |
| 208 | efficientnetv2 | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 31.67 | 20.04 | 1.6534 | 4 | 1 |
| 209 | inception_resnet | label_smoothing | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 31.67 | 16.67 | 1.6843 | 9 | 6 |
| 210 | sknet | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 31.67 | 17.71 | 1.0805 | 7 | 4 |
| 211 | deit | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 30.00 | 16.26 | 1.6393 | 8 | 5 |
| 212 | deit | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 30.00 | 14.61 | 1.6604 | 5 | 2 |
| 213 | mobilenetv3 | cross_entropy | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 30.00 | 15.56 | 4.7087 | 8 | 5 |
| 214 | gpt | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 30.00 | 15.89 | 1.0298 | 6 | 3 |
| 215 | res2net | label_smoothing | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 30.00 | 17.49 | 1.9853 | 4 | 1 |
| 216 | lenet | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 30.00 | 16.04 | 1.6514 | 5 | 2 |
| 217 | lenet | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 30.00 | 18.78 | 1.0654 | 5 | 2 |
| 218 | capsnet | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 30.00 | 20.89 | 1.6094 | 5 | 2 |
| 219 | capsnet | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 30.00 | 24.89 | 1.6094 | 4 | 1 |
| 220 | squeezenet | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 30.00 | 14.41 | 1.6586 | 7 | 4 |
| 221 | coatnet | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 30.00 | 15.15 | 2.9241 | 8 | 5 |
| 222 | van | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 30.00 | 24.96 | — | 10 | 7 |
| 223 | van | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 30.00 | 23.95 | 1.1981 | 9 | 6 |
| 224 | se_resnet | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 30.00 | 17.13 | 1.7455 | 5 | 2 |
| 225 | inception_resnet | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 30.00 | 15.83 | 1178.9512 | 7 | 4 |
| 226 | poolformer | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 30.00 | 18.96 | 1.6802 | 7 | 4 |
| 227 | poolformer | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 30.00 | 15.31 | 1.1584 | 9 | 6 |
| 228 | deit | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 28.33 | 12.40 | 1.6558 | 7 | 4 |
| 229 | deit | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 28.33 | 15.48 | 1.6170 | 7 | 4 |
| 230 | alexnet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 28.33 | 15.42 | 1.6327e+07 | 5 | 2 |
| 231 | densenet | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 28.33 | 11.73 | 1.0565 | 5 | 2 |
| 232 | mobilenet | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 28.33 | 11.91 | 1.0248 | 10 | 7 |
| 233 | vit | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 28.33 | 12.22 | 1.6687 | 4 | 1 |
| 234 | ghostnet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 28.33 | 20.18 | 3.9193 | 10 | 9 |
| 235 | ghostnet | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 28.33 | 18.87 | 1.5800 | 8 | 5 |
| 236 | res2net | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 28.33 | 17.97 | 24.1624 | 7 | 4 |
| 237 | cbam_resnet | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 28.33 | 23.54 | 1.9147 | 10 | 7 |
| 238 | lenet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 28.33 | 12.22 | 1.6201 | 5 | 2 |
| 239 | lenet | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 28.33 | 14.97 | 1.6084 | 4 | 1 |
| 240 | lenet | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 28.33 | 11.20 | 1.0236 | 10 | 9 |
| 241 | mnasnet | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 28.33 | 15.68 | 2.9727 | 7 | 4 |
| 242 | lstm | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 28.33 | 14.40 | 1.1028 | 9 | 6 |
| 243 | darknet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 28.33 | 15.21 | 2.5554 | 5 | 2 |
| 244 | gru | label_smoothing | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 28.33 | 13.28 | 1.6065 | 6 | 3 |
| 245 | gru | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 28.33 | 14.49 | 1.0693 | 8 | 5 |
| 246 | bert | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 28.33 | 15.15 | 1.6088 | 9 | 6 |
| 247 | bert | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 28.33 | 15.48 | 1.0288 | 7 | 4 |
| 248 | deit | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 26.67 | 8.42 | 1.6783 | 6 | 3 |
| 249 | deit | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 26.67 | 8.42 | 1.6897 | 5 | 2 |
| 250 | deit | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 26.67 | 8.42 | 1.7088 | 7 | 4 |
| 251 | deit | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 26.67 | 8.42 | 1.0615 | 6 | 3 |
| 252 | deit | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 26.67 | 8.42 | 1.0500 | 5 | 2 |
| 253 | deit | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 26.67 | 8.42 | 1.2705 | 6 | 3 |
| 254 | deit | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 26.67 | 8.42 | 1.2016 | 6 | 3 |
| 255 | hardnet | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 26.67 | 12.25 | 9.9533 | 4 | 1 |
| 256 | hardnet | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 26.67 | 8.42 | 1.0843 | 4 | 1 |
| 257 | hardnet | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 26.67 | 19.61 | 1.3605 | 10 | 10 |
| 258 | mobilenetv3 | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 26.67 | 8.42 | 1.6400 | 4 | 1 |
| 259 | mobilenetv3 | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 26.67 | 8.42 | 1.6136 | 5 | 2 |
| 260 | mobilenetv3 | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 26.67 | 8.42 | 21.8740 | 4 | 1 |
| 261 | mobilenetv3 | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 26.67 | 8.42 | 885.2807 | 4 | 1 |
| 262 | mobilenetv3 | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 26.67 | 8.42 | 1.2485 | 4 | 1 |
| 263 | mobilenetv3 | focal_loss | sgd | batch_size=16, lr=0.0001, weight_decay=0.0001 | 26.67 | 8.42 | 1.0300 | 4 | 1 |
| 264 | shufflenet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 26.67 | 8.42 | 1.6333 | 4 | 1 |
| 265 | xception | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 26.67 | 8.42 | 2893.9940 | 4 | 1 |
| 266 | xception | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 26.67 | 8.42 | 65.7656 | 4 | 1 |
| 267 | xception | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 26.67 | 8.42 | 1.0314 | 4 | 1 |
| 268 | xception | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 26.67 | 8.42 | 1.1160 | 4 | 1 |
| 269 | xception | focal_loss | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 26.67 | 8.42 | 1.1645 | 5 | 2 |
| 270 | alexnet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 26.67 | 8.42 | 1.6959 | 6 | 3 |
| 271 | alexnet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 26.67 | 8.42 | 1.6755 | 4 | 1 |
| 272 | alexnet | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 26.67 | 8.42 | 1.6519 | 4 | 1 |
| 273 | alexnet | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 26.67 | 8.42 | 1.6080 | 4 | 1 |
| 274 | alexnet | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 26.67 | 8.42 | 1.6284 | 4 | 1 |
| 275 | alexnet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 26.67 | 8.42 | 65863.0273 | 4 | 1 |
| 276 | alexnet | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 26.67 | 8.42 | 1.1026 | 5 | 2 |
| 277 | alexnet | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 26.67 | 8.42 | 1.0448 | 4 | 1 |
| 278 | alexnet | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 26.67 | 8.42 | 1.0720 | 4 | 1 |
| 279 | alexnet | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 26.67 | 8.42 | 718606.2031 | 4 | 1 |
| 280 | densenet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 26.67 | 8.42 | 1.8624 | 4 | 1 |
| 281 | densenet | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 26.67 | 8.42 | 2.3792 | 4 | 1 |
| 282 | densenet | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 26.67 | 8.42 | 3.5184 | 4 | 1 |
| 283 | densenet | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 26.67 | 8.42 | 1.6454 | 4 | 1 |
| 284 | densenet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 26.67 | 8.42 | 1.6736 | 4 | 1 |
| 285 | hrnet | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0 | 26.67 | 22.00 | 1.7927 | 7 | 4 |
| 286 | hrnet | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 26.67 | 12.67 | 1.6621 | 6 | 3 |
| 287 | nin | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 26.67 | 8.42 | 1.6527 | 6 | 3 |
| 288 | nin | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 26.67 | 8.42 | 1.6077 | 4 | 1 |
| 289 | nin | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 26.67 | 8.42 | 1.6354 | 4 | 1 |
| 290 | nin | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0 | 26.67 | 8.42 | 1.6493 | 5 | 2 |
| 291 | nin | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 26.67 | 8.42 | 1.6512 | 7 | 4 |
| 292 | nin | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 26.67 | 8.42 | 1.6036 | 4 | 1 |
| 293 | nin | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 26.67 | 8.42 | 1.6090 | 5 | 2 |
| 294 | nin | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 26.67 | 8.42 | 1.6433 | 4 | 1 |
| 295 | nin | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 26.67 | 8.42 | 1.1205 | 4 | 1 |
| 296 | nin | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 26.67 | 8.42 | 1.0277 | 4 | 1 |
| 297 | nin | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 26.67 | 8.42 | 1.0978 | 4 | 1 |
| 298 | nin | focal_loss | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 26.67 | 8.42 | 1.0428 | 4 | 1 |
| 299 | cspnet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 26.67 | 8.42 | 1672.3115 | 4 | 1 |
| 300 | cspnet | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 26.67 | 8.42 | 1.6866 | 4 | 1 |
| 301 | gpt | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 26.67 | 8.42 | 1.6102 | 4 | 1 |
| 302 | gpt | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 26.67 | 8.42 | 1.6085 | 4 | 1 |
| 303 | gpt | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 26.67 | 8.42 | 1.6093 | 6 | 3 |
| 304 | gpt | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 26.67 | 8.42 | 1.6072 | 4 | 1 |
| 305 | gpt | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 26.67 | 8.42 | 1.6093 | 7 | 4 |
| 306 | gpt | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 26.67 | 8.42 | 1.6083 | 9 | 6 |
| 307 | gpt | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 26.67 | 8.42 | 1.6445 | 5 | 2 |
| 308 | gpt | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 26.67 | 8.42 | 1.6460 | 6 | 3 |
| 309 | gpt | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 26.67 | 8.42 | 1.0800 | 5 | 2 |
| 310 | gpt | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 26.67 | 8.42 | 1.0661 | 5 | 2 |
| 311 | gpt | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 26.67 | 8.42 | 1.0289 | 7 | 4 |
| 312 | mobilenet | cross_entropy | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 26.67 | 8.42 | 1.6009 | 4 | 1 |
| 313 | mobilenet | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 26.67 | 8.42 | 2.2424 | 7 | 4 |
| 314 | mobilenet | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 26.67 | 8.42 | 1.1946 | 4 | 1 |
| 315 | resnext | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 26.67 | 8.65 | 1.5882 | 4 | 1 |
| 316 | resnext | label_smoothing | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 26.67 | 8.42 | 1.6493 | 4 | 1 |
| 317 | resnext | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 26.67 | 8.42 | 0.9870 | 4 | 1 |
| 318 | resnext | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 26.67 | 8.42 | 1.0846 | 5 | 2 |
| 319 | vit | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 26.67 | 8.42 | 1.7156 | 4 | 1 |
| 320 | vit | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 26.67 | 8.42 | 1.7727 | 6 | 3 |
| 321 | vit | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 26.67 | 8.42 | 1.6261 | 5 | 2 |
| 322 | vit | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 26.67 | 8.42 | 2.2729 | 5 | 2 |
| 323 | vit | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 26.67 | 8.42 | 1.6402 | 5 | 2 |
| 324 | vit | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 26.67 | 8.42 | 1.7009 | 4 | 1 |
| 325 | vit | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 26.67 | 8.42 | 1.1136 | 6 | 3 |
| 326 | vit | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 26.67 | 8.42 | 1.1004 | 5 | 2 |
| 327 | vit | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 26.67 | 8.42 | 1.0778 | 4 | 1 |
| 328 | ghostnet | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 26.67 | 8.42 | 2.6025 | 4 | 1 |
| 329 | ghostnet | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 26.67 | 8.42 | 1.7232 | 4 | 1 |
| 330 | ghostnet | focal_loss | sgd | batch_size=16, lr=0.0001, weight_decay=0.0 | 26.67 | 8.42 | 1.0283 | 4 | 1 |
| 331 | ghostnet | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 26.67 | 8.65 | 1.5377 | 4 | 1 |
| 332 | res2net | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 26.67 | 8.42 | 245.1802 | 7 | 4 |
| 333 | res2net | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 26.67 | 8.42 | 2.8886 | 5 | 2 |
| 334 | vgg | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.0001 | 26.67 | 8.42 | 1.6189 | 4 | 1 |
| 335 | vgg | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 26.67 | 8.42 | 1.6439 | 4 | 1 |
| 336 | vgg | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 26.67 | 8.42 | 13218.8641 | 4 | 1 |
| 337 | vgg | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0 | 26.67 | 8.42 | 1.6048 | 7 | 4 |
| 338 | vgg | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 26.67 | 8.42 | 3.1950 | 5 | 2 |
| 339 | vgg | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 26.67 | 8.42 | 206.5458 | 4 | 1 |
| 340 | vgg | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 26.67 | 8.42 | 1.6199 | 4 | 1 |
| 341 | vgg | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 26.67 | 8.42 | 1.0530 | 4 | 1 |
| 342 | vgg | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 26.67 | 8.42 | 1.1109 | 4 | 1 |
| 343 | vgg | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 26.67 | 8.42 | 40.7573 | 7 | 4 |
| 344 | vgg | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 26.67 | 8.42 | 1.0619 | 5 | 2 |
| 345 | efficientnet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 26.67 | 8.42 | 87.7879 | 4 | 1 |
| 346 | efficientnet | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 26.67 | 8.42 | 1.6169 | 4 | 1 |
| 347 | efficientnet | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 26.67 | 8.42 | 1.8006 | 4 | 1 |
| 348 | efficientnet | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 26.67 | 8.42 | 1.6094 | 4 | 1 |
| 349 | efficientnet | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 26.67 | 8.42 | 1.2021 | 4 | 1 |
| 350 | lenet | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 26.67 | 8.42 | 1.6023 | 4 | 1 |
| 351 | lenet | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 26.67 | 8.42 | 1.6750 | 9 | 6 |
| 352 | lenet | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 26.67 | 8.42 | 1.6313 | 4 | 1 |
| 353 | lenet | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 26.67 | 8.42 | 1.6178 | 4 | 1 |
| 354 | repghost | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.001 | 26.67 | 8.42 | 1.6151 | 4 | 1 |
| 355 | repghost | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 26.67 | 8.42 | 1.6534 | 4 | 1 |
| 356 | repghost | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 26.67 | 15.08 | 3.4702 | 6 | 3 |
| 357 | swin_tiny | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 26.67 | 8.42 | 1.6621 | 5 | 2 |
| 358 | swin_tiny | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 26.67 | 8.42 | 4.9100 | 4 | 1 |
| 359 | swin_tiny | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 26.67 | 8.42 | 1.6470 | 7 | 4 |
| 360 | swin_tiny | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 26.67 | 8.42 | 1.8097 | 5 | 2 |
| 361 | swin_tiny | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 26.67 | 8.42 | 2.3233 | 5 | 2 |
| 362 | swin_tiny | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 26.67 | 8.42 | 2.0263 | 5 | 2 |
| 363 | swin_tiny | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 26.67 | 8.42 | 2.2754 | 8 | 5 |
| 364 | swin_tiny | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 26.67 | 8.42 | 5.6047 | 4 | 1 |
| 365 | swin_tiny | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 26.67 | 11.87 | 1.1477 | 7 | 4 |
| 366 | swin_tiny | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 26.67 | 8.42 | 1.2530 | 4 | 1 |
| 367 | swin_tiny | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 26.67 | 8.42 | 1.1843 | 5 | 2 |
| 368 | googlenet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 26.67 | 8.42 | 1.6346 | 4 | 1 |
| 369 | googlenet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 26.67 | 8.42 | 1.6202 | 4 | 1 |
| 370 | googlenet | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 26.67 | 8.42 | 1.6045 | 4 | 1 |
| 371 | googlenet | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 26.67 | 8.42 | 1.6151 | 4 | 1 |
| 372 | googlenet | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 26.67 | 8.42 | 1.6587 | 5 | 2 |
| 373 | googlenet | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 26.67 | 8.42 | 1.6473 | 5 | 2 |
| 374 | googlenet | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 26.67 | 8.42 | 1.0977 | 4 | 1 |
| 375 | googlenet | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 26.67 | 8.42 | 1.0684 | 4 | 1 |
| 376 | googlenet | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 26.67 | 8.42 | 1.0279 | 6 | 3 |
| 377 | mnasnet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 26.67 | 8.42 | 1.6416 | 5 | 2 |
| 378 | mnasnet | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 26.67 | 8.42 | 2.4060 | 5 | 2 |
| 379 | mnasnet | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 26.67 | 8.42 | 5.9022 | 6 | 3 |
| 380 | mnasnet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 26.67 | 8.42 | 21.7124 | 4 | 1 |
| 381 | resnet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 26.67 | 8.42 | 295.8458 | 4 | 1 |
| 382 | resnet | cross_entropy | sgd | batch_size=16, lr=0.0001, weight_decay=0.0001 | 26.67 | 8.42 | 1.6077 | 5 | 2 |
| 383 | resnet | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 26.67 | 8.42 | 3.4126 | 4 | 1 |
| 384 | resnet | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 26.67 | 8.53 | 5.4948 | 4 | 1 |
| 385 | resnet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 26.67 | 8.42 | 1.6972 | 5 | 2 |
| 386 | resnet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 26.67 | 8.42 | 1.0370 | 5 | 2 |
| 387 | resnet | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 26.67 | 8.42 | 1443.2001 | 4 | 1 |
| 388 | resnet | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 26.67 | 8.42 | 1.0994 | 4 | 1 |
| 389 | capsnet | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 26.67 | 8.42 | 1.6094 | 4 | 1 |
| 390 | capsnet | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 26.67 | 8.42 | 1.6094 | 4 | 1 |
| 391 | capsnet | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 26.67 | 8.42 | 1.0300 | 4 | 1 |
| 392 | eca_resnet | label_smoothing | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 26.67 | 8.42 | 1.6602 | 4 | 1 |
| 393 | lcnet | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 26.67 | 8.42 | 1.6091 | 4 | 1 |
| 394 | lcnet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 26.67 | 8.42 | 1.6115 | 4 | 1 |
| 395 | lcnet | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.0001 | 26.67 | 8.42 | 1.6096 | 4 | 1 |
| 396 | lcnet | focal_loss | adamw | batch_size=16, lr=0.0001, weight_decay=0.0001 | 26.67 | 8.42 | 1.0348 | 4 | 1 |
| 397 | squeezenet | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 26.67 | 8.42 | 1.6558 | 4 | 1 |
| 398 | squeezenet | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 26.67 | 8.42 | 1.6694 | 5 | 2 |
| 399 | squeezenet | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 26.67 | 8.42 | 1.6279 | 4 | 1 |
| 400 | squeezenet | label_smoothing | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 26.67 | 15.01 | 1.6472 | 4 | 1 |
| 401 | squeezenet | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 26.67 | 8.42 | 1.6289 | 4 | 1 |
| 402 | squeezenet | label_smoothing | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 26.67 | 8.42 | 1.6410 | 6 | 3 |
| 403 | squeezenet | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 26.67 | 8.42 | 1.0556 | 6 | 3 |
| 404 | squeezenet | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 26.67 | 8.42 | 1.0818 | 5 | 2 |
| 405 | squeezenet | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 26.67 | 8.42 | 1.0491 | 5 | 2 |
| 406 | coatnet | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 26.67 | 8.42 | 34.6348 | 4 | 1 |
| 407 | coatnet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 26.67 | 12.11 | 1.0969 | 4 | 1 |
| 408 | lstm | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 26.67 | 8.42 | 1.6449 | 5 | 2 |
| 409 | lstm | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 26.67 | 8.42 | 1.6046 | 5 | 2 |
| 410 | lstm | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 26.67 | 8.42 | 1.6350 | 4 | 1 |
| 411 | lstm | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 26.67 | 8.42 | 1.6131 | 4 | 1 |
| 412 | lstm | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 26.67 | 8.42 | 1.6427 | 5 | 2 |
| 413 | lstm | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 26.67 | 8.42 | 1.6005 | 4 | 1 |
| 414 | lstm | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 26.67 | 8.42 | 1.6192 | 4 | 1 |
| 415 | lstm | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 26.67 | 8.42 | 1.6158 | 4 | 1 |
| 416 | lstm | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 26.67 | 8.42 | 1.0506 | 4 | 1 |
| 417 | lstm | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 26.67 | 8.42 | 1.0617 | 5 | 2 |
| 418 | lstm | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 26.67 | 8.42 | — | 6 | 3 |
| 419 | van | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 26.67 | 8.42 | 1.6859 | 4 | 1 |
| 420 | van | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 26.67 | 8.42 | 1.6118 | 4 | 1 |
| 421 | van | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 26.67 | 8.42 | — | 4 | 1 |
| 422 | van | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 26.67 | 8.42 | 1.1091 | 5 | 2 |
| 423 | van | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 26.67 | 8.42 | 1.0286 | 4 | 1 |
| 424 | darknet | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 26.67 | 8.42 | 1.7967 | 4 | 1 |
| 425 | darknet | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 26.67 | 8.42 | 1487.5951 | 6 | 3 |
| 426 | darknet | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 26.67 | 8.42 | 70.5181 | 5 | 2 |
| 427 | gru | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 26.67 | 8.42 | 1.6538 | 4 | 1 |
| 428 | gru | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 26.67 | 8.42 | 1.6392 | 6 | 3 |
| 429 | gru | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 26.67 | 8.42 | — | 4 | 1 |
| 430 | gru | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 26.67 | 8.42 | 1.6335 | 5 | 2 |
| 431 | gru | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 26.67 | 8.42 | 1.6496 | 4 | 1 |
| 432 | gru | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 26.67 | 8.42 | — | 4 | 1 |
| 433 | gru | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 26.67 | 8.42 | 1.0479 | 4 | 1 |
| 434 | gru | focal_loss | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0 | 26.67 | 11.43 | 1.0529 | 7 | 4 |
| 435 | mobilenetv2 | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.0001 | 26.67 | 8.42 | 1.6171 | 5 | 2 |
| 436 | se_resnet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 26.67 | 8.42 | 1.6618 | 4 | 1 |
| 437 | wide_resnet | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 26.67 | 8.42 | 1.6288 | 4 | 1 |
| 438 | wide_resnet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 26.67 | 8.42 | 1.6037 | 4 | 1 |
| 439 | wide_resnet | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 26.67 | 8.65 | 1.8962 | 4 | 1 |
| 440 | wide_resnet | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 26.67 | 8.42 | 1.3956 | 4 | 1 |
| 441 | bert | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 26.67 | 8.42 | 1.6081 | 4 | 1 |
| 442 | bert | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 26.67 | 8.42 | 1.6156 | 4 | 1 |
| 443 | bert | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 26.67 | 8.42 | 1.6106 | 4 | 1 |
| 444 | bert | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 26.67 | 8.42 | 1.6073 | 4 | 1 |
| 445 | bert | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 26.67 | 8.42 | 1.6109 | 4 | 1 |
| 446 | bert | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 26.67 | 8.42 | 1.6087 | 4 | 1 |
| 447 | bert | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 26.67 | 8.42 | 1.6078 | 6 | 3 |
| 448 | bert | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 26.67 | 8.42 | 1.0288 | 6 | 3 |
| 449 | bert | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 26.67 | 8.42 | 1.0564 | 4 | 1 |
| 450 | dpn | cross_entropy | sgd | batch_size=16, lr=0.0001, weight_decay=0.0 | 26.67 | 8.42 | 1.5932 | 4 | 1 |
| 451 | dpn | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 26.67 | 8.42 | 1.6352 | 8 | 5 |
| 452 | dpn | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 26.67 | 8.42 | 4.6331 | 4 | 1 |
| 453 | dpn | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 26.67 | 11.82 | 1.8327 | 6 | 3 |
| 454 | dpn | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 26.67 | 8.42 | 184.4218 | 5 | 2 |
| 455 | dpn | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 26.67 | 8.42 | 1.2210 | 4 | 1 |
| 456 | dpn | focal_loss | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 26.67 | 8.42 | 1.1609 | 4 | 1 |
| 457 | inception_resnet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 26.67 | 8.42 | 1.2209e+07 | 4 | 1 |
| 458 | inception_resnet | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 26.67 | 8.42 | 1.6591 | 4 | 1 |
| 459 | inception_resnet | cross_entropy | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 26.67 | 8.42 | 45783.5684 | 4 | 1 |
| 460 | inception_resnet | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 26.67 | 8.42 | 1.6128 | 4 | 1 |
| 461 | poolformer | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 26.67 | 8.42 | 1.7002 | 5 | 2 |
| 462 | poolformer | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 26.67 | 8.42 | 2.5229 | 5 | 2 |
| 463 | poolformer | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 26.67 | 8.42 | 1.7212 | 9 | 6 |
| 464 | poolformer | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.001 | 26.67 | 8.42 | 1.6098 | 5 | 2 |
| 465 | poolformer | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 26.67 | 8.42 | 1.7772 | 4 | 1 |
| 466 | poolformer | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 26.67 | 8.42 | 13.3095 | 7 | 4 |
| 467 | poolformer | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 26.67 | 8.42 | 409.6294 | 6 | 3 |
| 468 | poolformer | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 26.67 | 8.42 | 13.6562 | 8 | 5 |
| 469 | poolformer | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 26.67 | 8.42 | 1.2729 | 5 | 2 |
| 470 | poolformer | focal_loss | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 26.67 | 8.42 | 1.3469 | 7 | 4 |
| 471 | sknet | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 26.67 | 8.42 | 36.2256 | 4 | 1 |
| 472 | sknet | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 26.67 | 15.19 | 1.6230 | 6 | 3 |
| 473 | mobilenetv3 | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 25.00 | 14.76 | 1.9083 | 4 | 1 |
| 474 | cbam_resnet | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 25.00 | 14.45 | 1.7063 | 4 | 1 |
| 475 | efficientnet | cross_entropy | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 25.00 | 16.13 | 1.7495 | 6 | 3 |
| 476 | resnet | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 25.00 | 13.98 | 1.6102 | 4 | 1 |
| 477 | capsnet | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 25.00 | 10.26 | 1.6094 | 4 | 1 |
| 478 | capsnet | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 25.00 | 15.07 | 1.6094 | 4 | 1 |
| 479 | capsnet | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 25.00 | 15.62 | 1.0300 | 4 | 1 |
| 480 | lcnet | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 25.00 | 8.45 | 1.6886 | 7 | 4 |
| 481 | regnet | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 25.00 | 11.20 | 1.0413 | 5 | 2 |
| 482 | dpn | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 25.00 | 12.00 | 1.6606 | 4 | 1 |
| 483 | inception_resnet | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 25.00 | 11.19 | 1.0529 | 4 | 1 |
| 484 | sknet | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 25.00 | 8.00 | 1.6830 | 5 | 2 |
| 485 | sknet | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 25.00 | 11.52 | 2.4707 | 6 | 3 |
| 486 | hardnet | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 23.33 | 7.57 | 1.1874 | 5 | 2 |
| 487 | xception | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 23.33 | 7.57 | 1.6121 | 4 | 1 |
| 488 | xception | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 23.33 | 7.57 | 2.8242 | 5 | 2 |
| 489 | xception | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 23.33 | 7.57 | 1.6096 | 4 | 1 |
| 490 | xception | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 23.33 | 7.57 | 1.7685 | 5 | 2 |
| 491 | hrnet | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 23.33 | 10.10 | 0.9730 | 4 | 1 |
| 492 | cspnet | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 23.33 | 7.57 | 1.6278 | 4 | 1 |
| 493 | cspnet | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.001 | 23.33 | 7.57 | 1.8705 | 4 | 1 |
| 494 | mobilenet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 23.33 | 7.57 | 1.7184 | 4 | 1 |
| 495 | mobilenet | cross_entropy | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 23.33 | 7.57 | 1.7027 | 5 | 2 |
| 496 | mobilenet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 23.33 | 7.57 | 1.8007 | 5 | 2 |
| 497 | mobilenet | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 23.33 | 7.57 | 2.1408 | 4 | 1 |
| 498 | mobilenet | focal_loss | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 23.33 | 7.57 | 1.0718 | 4 | 1 |
| 499 | mobilenet | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 23.33 | 7.57 | 1.4080 | 5 | 2 |
| 500 | resnext | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 23.33 | 7.57 | 1.9027 | 6 | 3 |
| 501 | res2net | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 23.33 | 7.57 | 2.7238 | 4 | 1 |
| 502 | res2net | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 23.33 | 7.57 | 2.2984 | 6 | 3 |
| 503 | res2net | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 23.33 | 7.57 | 2.9256 | 4 | 1 |
| 504 | cbam_resnet | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 23.33 | 7.57 | 2.1782 | 4 | 1 |
| 505 | efficientnet | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 23.33 | 7.57 | 1.7403 | 4 | 1 |
| 506 | efficientnet | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 23.33 | 7.57 | 1.4161 | 4 | 1 |
| 507 | efficientnet | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 23.33 | 7.57 | 1.0494 | 4 | 1 |
| 508 | coord_resnet | cross_entropy | sgd | batch_size=16, lr=0.0001, weight_decay=0.0001 | 23.33 | 7.57 | 1.6033 | 4 | 1 |
| 509 | coord_resnet | label_smoothing | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 23.33 | 7.57 | 1.7991 | 4 | 1 |
| 510 | coord_resnet | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 23.33 | 7.57 | 1.2212 | 4 | 1 |
| 511 | googlenet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 23.33 | 7.57 | 1.6025 | 4 | 1 |
| 512 | googlenet | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 23.33 | 7.57 | 1.6409 | 4 | 1 |
| 513 | googlenet | focal_loss | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 23.33 | 7.57 | 1.0295 | 4 | 1 |
| 514 | mnasnet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 23.33 | 7.57 | 1.0590 | 5 | 2 |
| 515 | resnet | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 23.33 | 7.57 | 5.2663 | 5 | 2 |
| 516 | regnet | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 23.33 | 7.57 | 1.1577 | 4 | 1 |
| 517 | regnet | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 23.33 | 7.57 | 1.9151 | 5 | 2 |
| 518 | coatnet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 23.33 | 16.84 | 5.7872 | 4 | 1 |
| 519 | coatnet | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 23.33 | 7.57 | 1.7869 | 4 | 1 |
| 520 | efficientnetv2 | label_smoothing | adam | batch_size=16, lr=0.0001, weight_decay=0.001 | 23.33 | 7.57 | 1.6484 | 4 | 1 |
| 521 | efficientnetv2 | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 23.33 | 7.57 | 1.6577 | 4 | 1 |
| 522 | efficientnetv2 | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 23.33 | 7.57 | 2.0983 | 4 | 1 |
| 523 | efficientnetv2 | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 23.33 | 7.57 | 2.2951 | 4 | 1 |
| 524 | efficientnetv2 | focal_loss | adamw | batch_size=16, lr=0.0001, weight_decay=0.0001 | 23.33 | 7.57 | 1.0873 | 4 | 1 |
| 525 | van | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 23.33 | 7.57 | 1.6560 | 4 | 1 |
| 526 | darknet | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 23.33 | 7.57 | 6.0645 | 5 | 2 |
| 527 | gru | label_smoothing | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 23.33 | 7.57 | 1.6111 | 4 | 1 |
| 528 | gru | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 23.33 | 7.57 | 1.0272 | 4 | 1 |
| 529 | mobilenetv2 | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 23.33 | 7.57 | 2.1288 | 8 | 5 |
| 530 | inception_resnet | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 23.33 | 7.57 | 1.2854 | 4 | 1 |
| 531 | sknet | cross_entropy | adam | batch_size=16, lr=0.001, weight_decay=0.001 | 23.33 | 7.57 | 3.3631 | 5 | 2 |
| 532 | capsnet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 21.67 | 14.80 | 1.0300 | 4 | 1 |
| 533 | capsnet | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 21.67 | 12.99 | 1.0300 | 4 | 1 |
| 534 | coatnet | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 21.67 | 13.70 | 3.9576 | 4 | 1 |
| 535 | darknet | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 20.00 | 11.71 | 3.9796 | 6 | 3 |
| 536 | mobilenetv3 | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 18.33 | 6.20 | 1.6680 | 4 | 1 |
| 537 | mobilenetv3 | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 18.33 | 6.20 | 2.0686 | 4 | 1 |
| 538 | shufflenet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 18.33 | 6.20 | 4.1795 | 5 | 2 |
| 539 | shufflenet | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 18.33 | 6.20 | 1.6196 | 4 | 1 |
| 540 | shufflenet | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 18.33 | 6.20 | 1.6128 | 4 | 1 |
| 541 | shufflenet | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 18.33 | 6.20 | 2.1196 | 5 | 2 |
| 542 | shufflenet | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 18.33 | 6.20 | 2.8066 | 7 | 4 |
| 543 | shufflenet | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 18.33 | 6.20 | 1.0335 | 4 | 1 |
| 544 | alexnet | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 18.33 | 6.20 | 1.6098 | 4 | 1 |
| 545 | cspnet | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 18.33 | 6.20 | 6.2485 | 5 | 2 |
| 546 | cspnet | label_smoothing | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 18.33 | 6.20 | 1.6473 | 4 | 1 |
| 547 | cspnet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 18.33 | 6.20 | 1.0451 | 4 | 1 |
| 548 | mobilenet | label_smoothing | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 18.33 | 6.20 | 1.8279 | 4 | 1 |
| 549 | resnext | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 18.33 | 6.20 | 2.7982 | 4 | 1 |
| 550 | ghostnet | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 18.33 | 6.20 | 2.0433 | 4 | 1 |
| 551 | efficientnet | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 18.33 | 8.22 | 1.6415 | 4 | 1 |
| 552 | efficientnet | focal_loss | sgd | batch_size=16, lr=0.0001, weight_decay=0.0 | 18.33 | 6.20 | 1.0328 | 4 | 1 |
| 553 | repghost | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 18.33 | 6.20 | 3.1782 | 4 | 1 |
| 554 | mnasnet | cross_entropy | adamw | batch_size=16, lr=0.0001, weight_decay=0.0001 | 18.33 | 6.20 | 1.9656 | 4 | 1 |
| 555 | mnasnet | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 18.33 | 6.20 | 3.1667 | 6 | 3 |
| 556 | mnasnet | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.001 | 18.33 | 6.20 | 5.4432 | 4 | 1 |
| 557 | eca_resnet | cross_entropy | sgd | batch_size=16, lr=0.0001, weight_decay=0.0 | 18.33 | 6.20 | 1.6445 | 4 | 1 |
| 558 | eca_resnet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 18.33 | 6.20 | 1.0762 | 4 | 1 |
| 559 | lcnet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 18.33 | 6.20 | 4.1875 | 4 | 1 |
| 560 | lcnet | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 18.33 | 6.20 | 2.5368 | 4 | 1 |
| 561 | lcnet | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 18.33 | 6.20 | 1.1010 | 4 | 1 |
| 562 | lcnet | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 18.33 | 6.20 | 1.2059 | 4 | 1 |
| 563 | lcnet | focal_loss | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0 | 18.33 | 6.20 | 1.5795 | 5 | 2 |
| 564 | regnet | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 18.33 | 6.20 | 2.0582 | 4 | 1 |
| 565 | regnet | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 18.33 | 6.20 | 1.7607 | 4 | 1 |
| 566 | coatnet | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 18.33 | 6.20 | 1.8912 | 4 | 1 |
| 567 | efficientnetv2 | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 18.33 | 6.20 | 2.3018 | 5 | 2 |
| 568 | efficientnetv2 | cross_entropy | sgd | batch_size=16, lr=0.0001, weight_decay=0.0001 | 18.33 | 6.20 | 1.6183 | 4 | 1 |
| 569 | efficientnetv2 | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 18.33 | 6.20 | 2.0041 | 4 | 1 |
| 570 | efficientnetv2 | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 18.33 | 6.20 | 1.0499 | 4 | 1 |
| 571 | efficientnetv2 | focal_loss | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 18.33 | 6.20 | 1.2108 | 5 | 2 |
| 572 | repvgg | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 18.33 | 6.20 | 2.8038 | 4 | 1 |
| 573 | repvgg | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.0 | 18.33 | 6.20 | 1.6162 | 4 | 1 |
| 574 | repvgg | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 18.33 | 6.20 | 2.1463 | 4 | 1 |
| 575 | darknet | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 18.33 | 6.20 | 2.9810 | 4 | 1 |
| 576 | mobilenetv2 | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 18.33 | 6.20 | 2.3324 | 4 | 1 |
| 577 | mobilenetv2 | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 18.33 | 6.20 | 2.6992 | 4 | 1 |
| 578 | mobilenetv2 | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 18.33 | 6.20 | 5.6359 | 5 | 2 |
| 579 | mobilenetv2 | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 18.33 | 6.20 | 2.0328 | 4 | 1 |
| 580 | mobilenetv2 | focal_loss | sgd | batch_size=16, lr=0.0001, weight_decay=0.0001 | 18.33 | 6.20 | 1.0514 | 4 | 1 |
| 581 | mobilenetv2 | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 18.33 | 6.20 | 9.2963 | 4 | 1 |
| 582 | mobilenetv2 | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 18.33 | 6.20 | 3.7984 | 4 | 1 |
| 583 | wide_resnet | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 18.33 | 6.20 | 1.9217 | 4 | 1 |
| 584 | bert | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 18.33 | 6.20 | 1.0305 | 4 | 1 |
| 585 | inception_resnet | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 18.33 | 6.20 | 1.6878 | 4 | 1 |
| 586 | inception_resnet | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 18.33 | 6.20 | 2.4200 | 4 | 1 |
| 587 | inception_resnet | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 18.33 | 6.20 | 1.9164 | 4 | 1 |
| 588 | hardnet | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 13.33 | 4.71 | 1.9021 | 4 | 1 |
| 589 | shufflenet | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 13.33 | 4.71 | 1.6383 | 4 | 1 |
| 590 | shufflenet | label_smoothing | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 13.33 | 4.71 | 1.6571 | 4 | 1 |
| 591 | cspnet | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 13.33 | 4.71 | 1.8961 | 4 | 1 |
| 592 | lenet | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 13.33 | 4.71 | 1.0488 | 4 | 1 |
| 593 | repghost | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 13.33 | 4.71 | 1.6304 | 4 | 1 |
| 594 | lcnet | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 13.33 | 4.71 | 2.1724 | 4 | 1 |
| 595 | regnet | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 13.33 | 7.32 | 1.7637 | 4 | 1 |
| 596 | regnet | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 13.33 | 4.71 | 1.8376 | 4 | 1 |
| 597 | coatnet | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 13.33 | 4.71 | 5.0219 | 4 | 1 |
| 598 | van | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 13.33 | 4.71 | 1.6334 | 4 | 1 |
| 599 | van | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 13.33 | 4.71 | 1.6405 | 4 | 1 |
| 600 | dpn | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 13.33 | 4.71 | 1.7768 | 4 | 1 |

## Autoencoder — best per model

| Rank | Model | Loss | Optimizer | Hyperparameters | Recon Loss | Epochs Run | Convergence Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | simple_ae | mse | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 0.0000 | 4 | 1 |
| 2 | denoising_ae | mse | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 0.0000 | 4 | 1 |
| 3 | vae | mse | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 0.0000 | 4 | 1 |
| 4 | conv_ae | mse | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 0.0000 | 4 | 1 |

## Autoencoder — all trials (48 rows)

| Rank | Model | Loss | Optimizer | Hyperparameters | Recon Loss | Epochs Run | Convergence Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | simple_ae | mse | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 0.0000 | 4 | 1 |
| 2 | simple_ae | mse | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 0.0000 | 4 | 1 |
| 3 | simple_ae | mse | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 0.0000 | 4 | 1 |
| 4 | simple_ae | mse | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0 | 0.0000 | 4 | 1 |
| 5 | simple_ae | l1 | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 0.0000 | 4 | 1 |
| 6 | simple_ae | l1 | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 0.0000 | 4 | 1 |
| 7 | simple_ae | l1 | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 0.0000 | 4 | 1 |
| 8 | simple_ae | l1 | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.001 | 0.0000 | 4 | 1 |
| 9 | simple_ae | bce | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 0.0000 | 4 | 1 |
| 10 | simple_ae | bce | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 0.0000 | 4 | 1 |
| 11 | simple_ae | bce | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 0.0000 | 4 | 1 |
| 12 | simple_ae | bce | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 0.0000 | 4 | 1 |
| 13 | denoising_ae | mse | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 0.0000 | 4 | 1 |
| 14 | denoising_ae | mse | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 0.0000 | 4 | 1 |
| 15 | denoising_ae | mse | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 0.0000 | 4 | 1 |
| 16 | denoising_ae | mse | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 0.0000 | 4 | 1 |
| 17 | denoising_ae | l1 | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 0.0000 | 4 | 1 |
| 18 | denoising_ae | l1 | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 0.0000 | 4 | 1 |
| 19 | denoising_ae | l1 | adamw | batch_size=16, lr=0.0001, weight_decay=0.001 | 0.0000 | 4 | 1 |
| 20 | denoising_ae | l1 | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 0.0000 | 4 | 1 |
| 21 | denoising_ae | bce | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 0.0000 | 4 | 1 |
| 22 | denoising_ae | bce | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 0.0000 | 4 | 1 |
| 23 | denoising_ae | bce | adamw | batch_size=16, lr=0.001, weight_decay=0.001 | 0.0000 | 4 | 1 |
| 24 | denoising_ae | bce | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 0.0000 | 4 | 1 |
| 25 | vae | mse | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 0.0000 | 4 | 1 |
| 26 | vae | mse | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 0.0000 | 4 | 1 |
| 27 | vae | mse | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 0.0000 | 4 | 1 |
| 28 | vae | mse | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 0.0000 | 4 | 1 |
| 29 | vae | l1 | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 0.0000 | 4 | 1 |
| 30 | vae | l1 | sgd | batch_size=16, lr=0.0001, weight_decay=0.0001 | 0.0000 | 4 | 1 |
| 31 | vae | l1 | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 0.0000 | 4 | 1 |
| 32 | vae | l1 | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 0.0000 | 4 | 1 |
| 33 | vae | bce | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 0.0000 | 4 | 1 |
| 34 | vae | bce | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 0.0000 | 4 | 1 |
| 35 | vae | bce | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 0.0000 | 4 | 1 |
| 36 | vae | bce | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 0.0000 | 4 | 1 |
| 37 | conv_ae | mse | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 0.0000 | 4 | 1 |
| 38 | conv_ae | mse | sgd | batch_size=16, lr=0.0001, weight_decay=0.0 | 0.0000 | 4 | 1 |
| 39 | conv_ae | mse | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 0.0000 | 4 | 1 |
| 40 | conv_ae | mse | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 0.0000 | 4 | 1 |
| 41 | conv_ae | l1 | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 0.0000 | 4 | 1 |
| 42 | conv_ae | l1 | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 0.0000 | 4 | 1 |
| 43 | conv_ae | l1 | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 0.0000 | 4 | 1 |
| 44 | conv_ae | l1 | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 0.0000 | 4 | 1 |
| 45 | conv_ae | bce | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 0.0000 | 4 | 1 |
| 46 | conv_ae | bce | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 0.0000 | 4 | 1 |
| 47 | conv_ae | bce | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 0.0000 | 4 | 1 |
| 48 | conv_ae | bce | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 0.0000 | 4 | 1 |

## GAN — best per model

| Rank | Model | Loss | Optimizer | Hyperparameters | G Loss | D Loss | Epochs Run | Convergence Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | vanilla_gan | bce | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 31.2500 | 171.2911 | 5 | 2 |
| 2 | wgan | bce | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 0.0551 | -0.0627 | 4 | 1 |
| 3 | cgan | bce | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 0.7168 | 1.3250 | 7 | 4 |
| 4 | dcgan | wasserstein | sgd | batch_size=16, lr=0.0001, weight_decay=0.0 | 0.9737 | 1.1202 | 5 | 2 |

## GAN — all trials (28 rows)

| Rank | Model | Loss | Optimizer | Hyperparameters | G Loss | D Loss | Epochs Run | Convergence Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | vanilla_gan | bce | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 31.2500 | 171.2911 | 5 | 2 |
| 2 | vanilla_gan | wasserstein | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 0.0000 | 100.0000 | 9 | 6 |
| 3 | vanilla_gan | wasserstein | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 0.0000 | 100.0000 | 8 | 5 |
| 4 | wgan | bce | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 0.0551 | -0.0627 | 4 | 1 |
| 5 | wgan | bce | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 0.1100 | -0.1211 | 4 | 1 |
| 6 | wgan | wasserstein | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 0.1695 | -0.2005 | 4 | 1 |
| 7 | wgan | wasserstein | adam | batch_size=16, lr=0.001, weight_decay=0.001 | 0.2866 | -0.2961 | 4 | 1 |
| 8 | vanilla_gan | bce | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 0.7616 | 39.8167 | 5 | 2 |
| 9 | cgan | bce | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 0.7168 | 1.3250 | 7 | 4 |
| 10 | cgan | wasserstein | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 0.7084 | 1.3782 | 4 | 1 |
| 11 | cgan | wasserstein | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 0.9845 | 1.5047 | 8 | 5 |
| 12 | vanilla_gan | bce | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 0.7150 | 0.7348 | 4 | 1 |
| 13 | cgan | bce | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 0.7606 | 1.2967 | 4 | 1 |
| 14 | vanilla_gan | wasserstein | sgd | batch_size=16, lr=0.0001, weight_decay=0.0001 | 0.7144 | 0.8803 | 4 | 1 |
| 15 | cgan | wasserstein | adamw | batch_size=16, lr=0.0001, weight_decay=0.001 | 0.7611 | 1.2787 | 4 | 1 |
| 16 | cgan | bce | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 0.9692 | 1.1762 | 10 | 7 |
| 17 | cgan | bce | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 0.8495 | 1.1907 | 7 | 4 |
| 18 | cgan | wasserstein | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 1.5124 | 0.6104 | 5 | 2 |
| 19 | dcgan | wasserstein | sgd | batch_size=16, lr=0.0001, weight_decay=0.0 | 0.9737 | 1.1202 | 5 | 2 |
| 20 | dcgan | bce | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 2.0341 | 0.6514 | 4 | 1 |
| 21 | vanilla_gan | wasserstein | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 1.0759 | 0.6196 | 4 | 1 |
| 22 | dcgan | wasserstein | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 3.0844 | 0.3445 | 4 | 1 |
| 23 | dcgan | wasserstein | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 7.6546 | 0.0009 | 5 | 2 |
| 24 | dcgan | bce | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 8.0017 | 0.0008 | 5 | 2 |
| 25 | dcgan | bce | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 13.4150 | 0.0297 | 4 | 1 |
| 26 | dcgan | bce | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 8.5419 | 0.5705 | 4 | 1 |
| 27 | dcgan | wasserstein | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 12.2486 | 0.0651 | 7 | 4 |
| 28 | vanilla_gan | bce | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 100.0000 | 0.0000 | 4 | 1 |

## Search Space

- Loss (classification): cross_entropy, label_smoothing, focal_loss
- Loss (autoencoder): mse, l1, bce
- Loss (GAN): bce, wasserstein (informational; GANs use fixed objectives)
- Optimizers: adam, sgd, adamw, rmsprop
- Hyperparameters: {"lr": [0.0001, 0.001, 0.01], "batch_size": [8, 16, 32], "weight_decay": [0.0, 0.0001, 0.001]}
