# Pistachio Regression Report

## Run configuration

| Setting | Value |
| --- | --- |
| Generated | 2026-06-03T14:45:42 |
| Dataset | pistachio |
| Classes | 2 |
| Class names | kirmizi, siirt |
| Mode | quick-test |
| Max epochs | 10 |
| Early-stop patience | 3 |
| Min delta | 0.1 |
| NAS trials per config | 2 |
| Workers | 10 |
| Max batch size | 16 |
| Total wall time | 39089.3s |

Training stops when validation metric shows no significant improvement for 3 consecutive epochs.

## Classification — best per model

| Rank | Model | Loss | Optimizer | Hyperparameters | Test Acc (%) | Macro-F1 (%) | Test Loss | Epochs Run | Convergence Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | coord_resnet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 86.67 | 86.61 | 5.2483 | 10 | 7 |
| 2 | convnext | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 86.67 | 86.61 | 0.1224 | 8 | 5 |
| 3 | mlp | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 86.67 | 86.53 | 0.4292 | 10 | 9 |
| 4 | deit | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 85.00 | 85.00 | 0.7381 | 4 | 1 |
| 5 | vim_tiny | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 85.00 | 85.00 | 0.1067 | 9 | 6 |
| 6 | coatnet | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 85.00 | 84.79 | 0.5410 | 10 | 9 |
| 7 | cbam_resnet | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 83.33 | 83.16 | 1.9291 | 9 | 6 |
| 8 | simple_cnn | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 81.67 | 81.54 | 0.1680 | 8 | 5 |
| 9 | hardnet | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 81.67 | 81.54 | 0.9414 | 7 | 4 |
| 10 | poolformer | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 81.67 | 81.54 | 0.1564 | 10 | 10 |
| 11 | repghost | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 80.00 | 79.64 | 0.4944 | 7 | 4 |
| 12 | efficientnetv2 | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 80.00 | 79.91 | 1.0361 | 6 | 3 |
| 13 | repvgg | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 80.00 | 79.43 | 1.0242 | 10 | 7 |
| 14 | se_resnet | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 80.00 | 79.98 | 1.1266 | 7 | 4 |
| 15 | mobilenetv3 | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 78.33 | 78.33 | 9.8496 | 5 | 2 |
| 16 | wide_resnet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 78.33 | 78.33 | 0.9925 | 10 | 7 |
| 17 | ghostnet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 76.67 | 76.56 | 1.6621 | 7 | 4 |
| 18 | res2net | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 76.67 | 76.00 | 1.1247 | 10 | 9 |
| 19 | hrnet | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0 | 75.00 | 74.94 | 0.8034 | 5 | 2 |
| 20 | efficientnet | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 75.00 | 74.83 | 0.1849 | 10 | 7 |
| 21 | sknet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 75.00 | 74.66 | 0.1929 | 7 | 4 |
| 22 | bert | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 71.67 | 69.19 | 0.1728 | 5 | 2 |
| 23 | eca_resnet | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 70.00 | 69.70 | 1.0295 | 7 | 4 |
| 24 | alexnet | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 68.33 | 66.77 | 0.1797 | 6 | 3 |
| 25 | vit | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 66.67 | 64.75 | 0.1677 | 8 | 5 |
| 26 | xception | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 63.33 | 57.64 | 2.4563 | 7 | 4 |
| 27 | resnet | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 63.33 | 62.67 | 0.1818 | 7 | 4 |
| 28 | cspnet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 63.33 | 62.67 | 0.1744 | 7 | 4 |
| 29 | van | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 63.33 | 58.75 | 0.3020 | 7 | 4 |
| 30 | dpn | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 63.33 | 60.53 | 1.4019 | 4 | 1 |
| 31 | resnext | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 61.67 | 55.06 | 0.7509 | 6 | 3 |
| 32 | inception_resnet | cross_entropy | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 61.67 | 57.39 | 410.8154 | 7 | 4 |
| 33 | vgg | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 60.00 | 52.38 | 235.2094 | 6 | 3 |
| 34 | capsnet | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 60.00 | 52.38 | 0.1733 | 7 | 4 |
| 35 | regnet | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 53.33 | 52.86 | 0.7420 | 6 | 3 |
| 36 | mobilenetv2 | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 53.33 | 48.72 | 7.9511 | 6 | 3 |
| 37 | darknet | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 51.67 | 41.51 | 5661.6242 | 7 | 4 |
| 38 | densenet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 0.6952 | 4 | 1 |
| 39 | nin | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 0.6993 | 4 | 1 |
| 40 | shufflenet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 2.9373 | 4 | 1 |
| 41 | googlenet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 50.00 | 33.33 | 0.7236 | 4 | 1 |
| 42 | mnasnet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 16.2665 | 4 | 1 |
| 43 | gpt | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.7041 | 4 | 1 |
| 44 | mobilenet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 50.00 | 33.33 | 0.8167 | 4 | 1 |
| 45 | lenet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.7181 | 4 | 1 |
| 46 | swin_tiny | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.7115 | 4 | 1 |
| 47 | lcnet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 50.00 | 33.33 | 2.9245 | 4 | 1 |
| 48 | squeezenet | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 50.00 | 33.33 | 0.7234 | 4 | 1 |
| 49 | lstm | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 0.7058 | 4 | 1 |
| 50 | gru | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 0.7102 | 4 | 1 |

## Classification — all trials (600 rows)

| Rank | Model | Loss | Optimizer | Hyperparameters | Test Acc (%) | Macro-F1 (%) | Test Loss | Epochs Run | Convergence Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | coord_resnet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 86.67 | 86.61 | 5.2483 | 10 | 7 |
| 2 | convnext | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 86.67 | 86.61 | 0.1224 | 8 | 5 |
| 3 | mlp | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 86.67 | 86.53 | 0.4292 | 10 | 9 |
| 4 | deit | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 85.00 | 85.00 | 0.7381 | 4 | 1 |
| 5 | vim_tiny | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 85.00 | 85.00 | 0.1067 | 9 | 6 |
| 6 | vim_tiny | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 85.00 | 84.90 | 0.2515 | 10 | 8 |
| 7 | convnext | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 85.00 | 84.96 | 1.4971 | 8 | 5 |
| 8 | convnext | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 85.00 | 84.90 | 0.6285 | 8 | 5 |
| 9 | mlp | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 85.00 | 85.00 | 0.5113 | 7 | 4 |
| 10 | mlp | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 85.00 | 84.65 | 0.4776 | 8 | 5 |
| 11 | mlp | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 85.00 | 84.79 | 0.5085 | 8 | 5 |
| 12 | mlp | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 85.00 | 85.00 | 0.1375 | 8 | 5 |
| 13 | mlp | focal_loss | sgd | batch_size=16, lr=0.0001, weight_decay=0.0001 | 85.00 | 84.96 | 0.1170 | 9 | 6 |
| 14 | mlp | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 85.00 | 84.90 | 0.0801 | 9 | 6 |
| 15 | coatnet | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 85.00 | 84.79 | 0.5410 | 10 | 9 |
| 16 | convnext | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 83.33 | 83.33 | 0.1227 | 8 | 5 |
| 17 | mlp | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 83.33 | 83.16 | 0.4664 | 8 | 5 |
| 18 | mlp | cross_entropy | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 83.33 | 82.86 | 0.5724 | 10 | 9 |
| 19 | mlp | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 83.33 | 83.31 | 0.1234 | 8 | 5 |
| 20 | cbam_resnet | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 83.33 | 83.16 | 1.9291 | 9 | 6 |
| 21 | simple_cnn | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 81.67 | 81.54 | 0.1680 | 8 | 5 |
| 22 | hardnet | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 81.67 | 81.54 | 0.9414 | 7 | 4 |
| 23 | vim_tiny | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 81.67 | 81.54 | 1.8259 | 9 | 6 |
| 24 | vim_tiny | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 81.67 | 81.66 | 0.5881 | 8 | 5 |
| 25 | vim_tiny | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 81.67 | 81.66 | 0.2608 | 10 | 7 |
| 26 | vim_tiny | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 81.67 | 81.54 | 0.1893 | 10 | 7 |
| 27 | mlp | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 81.67 | 81.41 | 0.4264 | 10 | 10 |
| 28 | mlp | label_smoothing | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 81.67 | 81.41 | 0.5933 | 8 | 5 |
| 29 | poolformer | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 81.67 | 81.54 | 0.1564 | 10 | 10 |
| 30 | vim_tiny | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 80.00 | 79.43 | 0.5357 | 8 | 5 |
| 31 | convnext | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 80.00 | 79.64 | 0.8994 | 7 | 4 |
| 32 | convnext | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 80.00 | 79.43 | 0.6413 | 6 | 3 |
| 33 | convnext | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 80.00 | 79.64 | 0.6203 | 8 | 5 |
| 34 | repghost | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 80.00 | 79.64 | 0.4944 | 7 | 4 |
| 35 | coatnet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 80.00 | 79.17 | 0.5990 | 10 | 7 |
| 36 | efficientnetv2 | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 80.00 | 79.91 | 1.0361 | 6 | 3 |
| 37 | repvgg | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 80.00 | 79.43 | 1.0242 | 10 | 7 |
| 38 | se_resnet | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 80.00 | 79.98 | 1.1266 | 7 | 4 |
| 39 | hardnet | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 78.33 | 77.83 | 0.6408 | 10 | 9 |
| 40 | mobilenetv3 | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 78.33 | 78.33 | 9.8496 | 5 | 2 |
| 41 | vim_tiny | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 78.33 | 78.28 | 0.7034 | 4 | 1 |
| 42 | convnext | label_smoothing | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 78.33 | 78.28 | 0.6417 | 8 | 5 |
| 43 | cbam_resnet | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 78.33 | 78.03 | 0.6676 | 10 | 8 |
| 44 | coatnet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 78.33 | 78.28 | 5.4245 | 8 | 5 |
| 45 | wide_resnet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 78.33 | 78.33 | 0.9925 | 10 | 7 |
| 46 | wide_resnet | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 78.33 | 77.83 | 0.1337 | 7 | 4 |
| 47 | simple_cnn | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 76.67 | 75.69 | 0.6319 | 9 | 6 |
| 48 | simple_cnn | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 76.67 | 76.56 | 1.0490 | 7 | 4 |
| 49 | ghostnet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 76.67 | 76.56 | 1.6621 | 7 | 4 |
| 50 | res2net | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 76.67 | 76.00 | 1.1247 | 10 | 9 |
| 51 | efficientnetv2 | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 76.67 | 75.69 | 0.7090 | 8 | 5 |
| 52 | hrnet | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0 | 75.00 | 74.94 | 0.8034 | 5 | 2 |
| 53 | hardnet | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 75.00 | 74.94 | 0.6653 | 9 | 6 |
| 54 | vim_tiny | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 75.00 | 74.66 | 2.5549 | 9 | 6 |
| 55 | convnext | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 75.00 | 73.33 | 0.5083 | 6 | 3 |
| 56 | ghostnet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 75.00 | 74.66 | 3.0262 | 8 | 5 |
| 57 | ghostnet | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 75.00 | 73.77 | 3.0326 | 6 | 3 |
| 58 | efficientnet | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 75.00 | 74.83 | 0.1849 | 10 | 7 |
| 59 | coatnet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 75.00 | 74.99 | 0.8191 | 6 | 3 |
| 60 | repvgg | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 75.00 | 74.13 | 6.4924 | 7 | 4 |
| 61 | sknet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 75.00 | 74.66 | 0.1929 | 7 | 4 |
| 62 | simple_cnn | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 73.33 | 72.85 | 0.5317 | 6 | 3 |
| 63 | coord_resnet | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 73.33 | 73.06 | 1.0328 | 10 | 7 |
| 64 | convnext | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 73.33 | 71.80 | 2.4311 | 7 | 4 |
| 65 | ghostnet | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 73.33 | 72.22 | 0.6303 | 8 | 5 |
| 66 | cbam_resnet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 73.33 | 72.57 | 0.6572 | 6 | 3 |
| 67 | cbam_resnet | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 73.33 | 73.06 | 0.2464 | 10 | 7 |
| 68 | hrnet | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 71.67 | 71.28 | 0.8026 | 9 | 6 |
| 69 | hrnet | label_smoothing | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.001 | 71.67 | 70.68 | 0.7193 | 7 | 4 |
| 70 | hrnet | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 71.67 | 69.78 | 0.1900 | 4 | 1 |
| 71 | vim_tiny | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 71.67 | 69.78 | 0.5303 | 7 | 4 |
| 72 | ghostnet | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 71.67 | 69.19 | 0.6301 | 5 | 2 |
| 73 | cbam_resnet | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 71.67 | 71.66 | 0.7258 | 7 | 4 |
| 74 | repghost | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 71.67 | 69.19 | 0.6619 | 8 | 5 |
| 75 | se_resnet | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 71.67 | 71.60 | 2.2661 | 8 | 5 |
| 76 | bert | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 71.67 | 69.19 | 0.1728 | 5 | 2 |
| 77 | hrnet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 70.00 | 69.14 | 0.6800 | 9 | 6 |
| 78 | repghost | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 70.00 | 68.27 | 3.4620 | 5 | 2 |
| 79 | repghost | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 70.00 | 67.70 | 0.7494 | 10 | 7 |
| 80 | eca_resnet | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 70.00 | 69.70 | 1.0295 | 7 | 4 |
| 81 | repvgg | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 70.00 | 68.27 | 0.2131 | 6 | 3 |
| 82 | sknet | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 70.00 | 67.70 | 0.7091 | 7 | 4 |
| 83 | alexnet | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 68.33 | 66.77 | 0.1797 | 6 | 3 |
| 84 | vim_tiny | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 68.33 | 65.57 | 0.6504 | 10 | 7 |
| 85 | vim_tiny | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 68.33 | 65.57 | 0.7610 | 4 | 1 |
| 86 | ghostnet | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 68.33 | 68.32 | 0.6569 | 7 | 4 |
| 87 | eca_resnet | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 68.33 | 68.11 | 3.3505 | 7 | 4 |
| 88 | se_resnet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 68.33 | 66.22 | 0.6860 | 6 | 3 |
| 89 | se_resnet | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 68.33 | 66.22 | 0.1656 | 7 | 4 |
| 90 | simple_cnn | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 66.67 | 62.50 | 0.6399 | 5 | 2 |
| 91 | simple_cnn | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 66.67 | 64.75 | 3.0054 | 5 | 2 |
| 92 | deit | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 66.67 | 62.50 | 0.3690 | 4 | 1 |
| 93 | coord_resnet | cross_entropy | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 66.67 | 62.50 | 0.9167 | 8 | 5 |
| 94 | vit | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 66.67 | 64.75 | 0.1677 | 8 | 5 |
| 95 | eca_resnet | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 66.67 | 64.75 | 1.0492 | 7 | 4 |
| 96 | coatnet | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 66.67 | 64.75 | 0.3255 | 10 | 7 |
| 97 | efficientnet | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 65.00 | 61.10 | 0.2119 | 7 | 4 |
| 98 | repvgg | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 65.00 | 62.67 | 4.2181 | 4 | 1 |
| 99 | wide_resnet | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 65.00 | 62.67 | 1.2740 | 5 | 2 |
| 100 | simple_cnn | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 63.33 | 62.67 | 1.2140 | 5 | 2 |
| 101 | simple_cnn | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 63.33 | 57.64 | 0.7345 | 5 | 2 |
| 102 | deit | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 63.33 | 57.64 | 0.7133 | 4 | 1 |
| 103 | xception | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 63.33 | 57.64 | 2.4563 | 7 | 4 |
| 104 | resnet | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 63.33 | 62.67 | 0.1818 | 7 | 4 |
| 105 | cspnet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 63.33 | 62.67 | 0.1744 | 7 | 4 |
| 106 | repvgg | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 63.33 | 62.29 | 1.7549 | 6 | 3 |
| 107 | van | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 63.33 | 58.75 | 0.3020 | 7 | 4 |
| 108 | wide_resnet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 63.33 | 62.67 | 0.1730 | 5 | 2 |
| 109 | dpn | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 63.33 | 60.53 | 1.4019 | 4 | 1 |
| 110 | deit | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 61.67 | 55.06 | 0.1624 | 6 | 3 |
| 111 | resnext | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 61.67 | 55.06 | 0.7509 | 6 | 3 |
| 112 | resnext | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 61.67 | 55.06 | 0.8487 | 5 | 2 |
| 113 | res2net | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 61.67 | 61.40 | 0.8901 | 7 | 4 |
| 114 | efficientnet | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 61.67 | 57.39 | 0.8681 | 7 | 4 |
| 115 | coatnet | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 61.67 | 61.66 | 4.8066 | 10 | 7 |
| 116 | dpn | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 61.67 | 56.32 | 1.6165 | 8 | 5 |
| 117 | inception_resnet | cross_entropy | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 61.67 | 57.39 | 410.8154 | 7 | 4 |
| 118 | resnext | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 60.00 | 58.86 | 5.5062 | 9 | 6 |
| 119 | vit | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 60.00 | 52.38 | 0.7140 | 5 | 2 |
| 120 | vgg | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 60.00 | 52.38 | 235.2094 | 6 | 3 |
| 121 | capsnet | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 60.00 | 52.38 | 0.1733 | 7 | 4 |
| 122 | wide_resnet | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 60.00 | 57.70 | 0.2678 | 4 | 1 |
| 123 | poolformer | focal_loss | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 60.00 | 52.38 | 0.3523 | 7 | 4 |
| 124 | hrnet | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 58.33 | 58.23 | 0.1923 | 7 | 4 |
| 125 | resnet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 58.33 | 49.58 | 0.1966 | 7 | 4 |
| 126 | capsnet | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 58.33 | 49.58 | 0.1733 | 4 | 1 |
| 127 | inception_resnet | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 58.33 | 56.88 | 68.8235 | 5 | 2 |
| 128 | deit | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 56.67 | 49.94 | 0.7202 | 8 | 5 |
| 129 | resnet | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 56.67 | 46.65 | 0.7305 | 5 | 2 |
| 130 | cspnet | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 56.67 | 52.38 | 0.9557 | 6 | 3 |
| 131 | repvgg | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 56.67 | 52.38 | 0.4076 | 7 | 4 |
| 132 | van | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 56.67 | 49.94 | 0.7024 | 6 | 3 |
| 133 | dpn | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 56.67 | 46.65 | 20.4472 | 5 | 2 |
| 134 | deit | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 55.00 | 47.25 | 0.6883 | 5 | 2 |
| 135 | resnext | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 55.00 | 45.55 | 1.0363 | 9 | 6 |
| 136 | eca_resnet | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 55.00 | 45.55 | 1.5912 | 4 | 1 |
| 137 | alexnet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 53.33 | 52.00 | 189.5234 | 6 | 3 |
| 138 | cbam_resnet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 53.33 | 42.54 | 0.6887 | 7 | 4 |
| 139 | cbam_resnet | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 53.33 | 40.34 | 3.2014 | 5 | 2 |
| 140 | regnet | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 53.33 | 52.86 | 0.7420 | 6 | 3 |
| 141 | van | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 53.33 | 52.00 | 0.6960 | 4 | 1 |
| 142 | mobilenetv2 | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 53.33 | 48.72 | 7.9511 | 6 | 3 |
| 143 | wide_resnet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 53.33 | 40.34 | 1.4622 | 4 | 1 |
| 144 | deit | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 51.67 | 36.93 | 0.6877 | 5 | 2 |
| 145 | coord_resnet | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 51.67 | 36.93 | 0.6094 | 6 | 3 |
| 146 | resnext | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 51.67 | 39.39 | 0.6966 | 7 | 4 |
| 147 | vit | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 51.67 | 36.93 | 0.7023 | 7 | 4 |
| 148 | vit | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 51.67 | 36.93 | 0.1865 | 6 | 3 |
| 149 | vgg | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 51.67 | 36.93 | 0.1733 | 4 | 1 |
| 150 | coatnet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 51.67 | 41.51 | 16.2825 | 4 | 1 |
| 151 | coatnet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 51.67 | 44.92 | 1.3508 | 8 | 5 |
| 152 | darknet | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 51.67 | 41.51 | 5661.6242 | 7 | 4 |
| 153 | wide_resnet | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 51.67 | 36.93 | 0.7336 | 6 | 3 |
| 154 | wide_resnet | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 51.67 | 36.93 | 1.0940 | 5 | 2 |
| 155 | sknet | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 51.67 | 36.93 | 1.3010 | 6 | 3 |
| 156 | alexnet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 0.7141 | 4 | 1 |
| 157 | alexnet | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 50.00 | 33.33 | 0.6932 | 4 | 1 |
| 158 | alexnet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.7101 | 4 | 1 |
| 159 | alexnet | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.7215 | 4 | 1 |
| 160 | alexnet | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.6941 | 5 | 2 |
| 161 | alexnet | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 50.00 | 33.33 | 0.7018 | 4 | 1 |
| 162 | alexnet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 8.6605 | 4 | 1 |
| 163 | alexnet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 50.00 | 33.33 | 0.1734 | 4 | 1 |
| 164 | alexnet | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 50.00 | 33.33 | 0.1749 | 4 | 1 |
| 165 | alexnet | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 544.6207 | 4 | 1 |
| 166 | densenet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 0.6952 | 4 | 1 |
| 167 | densenet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 0.7243 | 4 | 1 |
| 168 | densenet | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 50.00 | 33.33 | 0.8686 | 4 | 1 |
| 169 | densenet | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 3.5210 | 4 | 1 |
| 170 | densenet | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.7158 | 4 | 1 |
| 171 | densenet | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 50.00 | 33.33 | 0.7121 | 4 | 1 |
| 172 | densenet | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 15.4217 | 4 | 1 |
| 173 | densenet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 0.6945 | 4 | 1 |
| 174 | densenet | focal_loss | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 50.00 | 33.33 | 0.1794 | 4 | 1 |
| 175 | densenet | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 0.2074 | 4 | 1 |
| 176 | densenet | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 0.1883 | 4 | 1 |
| 177 | densenet | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 0.1733 | 4 | 1 |
| 178 | hrnet | cross_entropy | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 50.00 | 33.33 | 0.6934 | 4 | 1 |
| 179 | hrnet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.8356 | 4 | 1 |
| 180 | hrnet | label_smoothing | adam | batch_size=16, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.6933 | 4 | 1 |
| 181 | hrnet | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 0.7083 | 4 | 1 |
| 182 | hrnet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 0.3322 | 4 | 1 |
| 183 | hrnet | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 50.00 | 33.33 | 0.1971 | 4 | 1 |
| 184 | nin | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 0.6993 | 4 | 1 |
| 185 | nin | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.6932 | 4 | 1 |
| 186 | nin | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 0.7318 | 4 | 1 |
| 187 | nin | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.7123 | 4 | 1 |
| 188 | nin | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 50.00 | 33.33 | 0.7232 | 4 | 1 |
| 189 | nin | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 50.00 | 33.33 | 0.6932 | 4 | 1 |
| 190 | nin | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 50.00 | 33.33 | 0.6952 | 4 | 1 |
| 191 | nin | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 50.00 | 33.33 | 0.7083 | 4 | 1 |
| 192 | nin | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.1802 | 4 | 1 |
| 193 | nin | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.1733 | 4 | 1 |
| 194 | nin | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.1777 | 4 | 1 |
| 195 | nin | focal_loss | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 50.00 | 33.33 | 0.1818 | 4 | 1 |
| 196 | simple_cnn | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 68.1225 | 4 | 1 |
| 197 | simple_cnn | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 50.00 | 33.33 | 0.6932 | 4 | 1 |
| 198 | simple_cnn | label_smoothing | adamw | batch_size=16, lr=0.0001, weight_decay=0.0 | 50.00 | 33.33 | 0.6819 | 4 | 1 |
| 199 | simple_cnn | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.6004 | 4 | 1 |
| 200 | deit | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 50.00 | 33.33 | 0.7289 | 4 | 1 |
| 201 | deit | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 50.00 | 33.33 | 0.7141 | 4 | 1 |
| 202 | deit | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 50.00 | 33.33 | 0.6880 | 4 | 1 |
| 203 | deit | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 0.1706 | 4 | 1 |
| 204 | deit | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 50.00 | 33.33 | 0.2713 | 4 | 1 |
| 205 | hardnet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 0.6634 | 4 | 1 |
| 206 | hardnet | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.7143 | 4 | 1 |
| 207 | hardnet | cross_entropy | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 2142.2988 | 4 | 1 |
| 208 | hardnet | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.7168 | 4 | 1 |
| 209 | hardnet | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 0.8386 | 4 | 1 |
| 210 | hardnet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 0.6987 | 4 | 1 |
| 211 | hardnet | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 50.00 | 33.33 | 0.1744 | 4 | 1 |
| 212 | hardnet | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.3660 | 4 | 1 |
| 213 | hardnet | focal_loss | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.1960 | 4 | 1 |
| 214 | mobilenetv3 | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 50.00 | 33.33 | 0.6968 | 4 | 1 |
| 215 | mobilenetv3 | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 0.6933 | 4 | 1 |
| 216 | mobilenetv3 | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.001 | 50.00 | 33.33 | 0.7807 | 4 | 1 |
| 217 | mobilenetv3 | label_smoothing | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 50.00 | 33.33 | 0.6933 | 4 | 1 |
| 218 | mobilenetv3 | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.6932 | 4 | 1 |
| 219 | mobilenetv3 | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 0.9089 | 4 | 1 |
| 220 | mobilenetv3 | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 50.00 | 33.33 | 510.0258 | 4 | 1 |
| 221 | mobilenetv3 | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 0.1819 | 4 | 1 |
| 222 | mobilenetv3 | focal_loss | sgd | batch_size=16, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.1733 | 4 | 1 |
| 223 | mobilenetv3 | focal_loss | adamw | batch_size=16, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.1757 | 4 | 1 |
| 224 | mobilenetv3 | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 50.00 | 33.33 | 0.6185 | 4 | 1 |
| 225 | shufflenet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 2.9373 | 4 | 1 |
| 226 | shufflenet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 50.00 | 33.33 | 0.6936 | 4 | 1 |
| 227 | shufflenet | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 3.6007 | 4 | 1 |
| 228 | shufflenet | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 50.00 | 33.33 | 0.6970 | 4 | 1 |
| 229 | shufflenet | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.7023 | 4 | 1 |
| 230 | shufflenet | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 50.00 | 33.33 | 0.6941 | 4 | 1 |
| 231 | shufflenet | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.6932 | 4 | 1 |
| 232 | shufflenet | label_smoothing | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 0.7127 | 4 | 1 |
| 233 | shufflenet | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 50.00 | 33.33 | 0.1769 | 4 | 1 |
| 234 | shufflenet | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.1733 | 4 | 1 |
| 235 | shufflenet | focal_loss | adamw | batch_size=16, lr=0.0001, weight_decay=0.001 | 50.00 | 33.33 | 0.1733 | 4 | 1 |
| 236 | shufflenet | focal_loss | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.1807 | 4 | 1 |
| 237 | xception | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 411.9195 | 4 | 1 |
| 238 | xception | cross_entropy | sgd | batch_size=16, lr=0.0001, weight_decay=0.0 | 50.00 | 33.33 | 0.6931 | 4 | 1 |
| 239 | xception | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 50.00 | 33.33 | 226.2980 | 4 | 1 |
| 240 | xception | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 31.2834 | 4 | 1 |
| 241 | xception | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.6931 | 4 | 1 |
| 242 | xception | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.7572 | 4 | 1 |
| 243 | xception | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 50.00 | 33.33 | 4.3608 | 4 | 1 |
| 244 | xception | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 2307.5179 | 4 | 1 |
| 245 | xception | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.1734 | 4 | 1 |
| 246 | xception | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 0.2295 | 4 | 1 |
| 247 | xception | focal_loss | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 50.00 | 33.33 | 0.1763 | 4 | 1 |
| 248 | coord_resnet | cross_entropy | sgd | batch_size=16, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.6947 | 4 | 1 |
| 249 | coord_resnet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 1.8285 | 4 | 1 |
| 250 | coord_resnet | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 50.00 | 33.33 | 0.6965 | 4 | 1 |
| 251 | coord_resnet | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 35.3189 | 4 | 1 |
| 252 | coord_resnet | label_smoothing | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 1.2354 | 4 | 1 |
| 253 | coord_resnet | focal_loss | adam | batch_size=16, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.4515 | 4 | 1 |
| 254 | coord_resnet | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 0.2035 | 4 | 1 |
| 255 | coord_resnet | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 0.1788 | 4 | 1 |
| 256 | googlenet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 50.00 | 33.33 | 0.7236 | 4 | 1 |
| 257 | googlenet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 50.00 | 33.33 | 0.7191 | 4 | 1 |
| 258 | googlenet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 50.00 | 33.33 | 0.6951 | 4 | 1 |
| 259 | googlenet | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.7032 | 4 | 1 |
| 260 | googlenet | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.7344 | 4 | 1 |
| 261 | googlenet | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.6932 | 4 | 1 |
| 262 | googlenet | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 0.7163 | 4 | 1 |
| 263 | googlenet | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.7102 | 4 | 1 |
| 264 | googlenet | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.1813 | 4 | 1 |
| 265 | googlenet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 50.00 | 33.33 | 0.1744 | 4 | 1 |
| 266 | googlenet | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 0.1763 | 4 | 1 |
| 267 | googlenet | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 0.1789 | 4 | 1 |
| 268 | mnasnet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 16.2665 | 4 | 1 |
| 269 | mnasnet | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 50.00 | 33.33 | 0.7031 | 4 | 1 |
| 270 | mnasnet | cross_entropy | adamw | batch_size=16, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.8922 | 4 | 1 |
| 271 | mnasnet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 0.6932 | 4 | 1 |
| 272 | mnasnet | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 50.00 | 33.33 | 3.2479 | 4 | 1 |
| 273 | mnasnet | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.7106 | 4 | 1 |
| 274 | mnasnet | label_smoothing | adamw | batch_size=16, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.8734 | 4 | 1 |
| 275 | mnasnet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 0.7113 | 4 | 1 |
| 276 | mnasnet | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 16.8008 | 4 | 1 |
| 277 | mnasnet | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 50.00 | 33.33 | 0.2220 | 4 | 1 |
| 278 | mnasnet | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 3.0016 | 4 | 1 |
| 279 | mnasnet | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 3.5906 | 4 | 1 |
| 280 | resnet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 67.3696 | 4 | 1 |
| 281 | resnet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 50.00 | 33.33 | 0.8963 | 4 | 1 |
| 282 | resnet | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 50.00 | 33.33 | 3.1415 | 4 | 1 |
| 283 | resnet | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 2.9796 | 4 | 1 |
| 284 | resnet | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 50.00 | 33.33 | 0.6953 | 4 | 1 |
| 285 | resnet | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 50.00 | 33.33 | 1.9769 | 4 | 1 |
| 286 | resnet | label_smoothing | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 50.00 | 33.33 | 0.8619 | 4 | 1 |
| 287 | resnet | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 50.00 | 33.33 | 0.2553 | 4 | 1 |
| 288 | resnet | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.3956 | 4 | 1 |
| 289 | cspnet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 6.9572 | 4 | 1 |
| 290 | cspnet | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.6938 | 4 | 1 |
| 291 | cspnet | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 9.7094 | 4 | 1 |
| 292 | cspnet | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 0.7221 | 4 | 1 |
| 293 | cspnet | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 1.2400 | 4 | 1 |
| 294 | cspnet | label_smoothing | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.7024 | 4 | 1 |
| 295 | cspnet | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 50.00 | 33.33 | 1.2325 | 4 | 1 |
| 296 | cspnet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 50.00 | 33.33 | 0.1736 | 4 | 1 |
| 297 | cspnet | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 4.6890 | 4 | 1 |
| 298 | cspnet | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 50.00 | 33.33 | 566.8199 | 4 | 1 |
| 299 | gpt | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.7041 | 4 | 1 |
| 300 | gpt | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.6949 | 4 | 1 |
| 301 | gpt | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.6937 | 4 | 1 |
| 302 | gpt | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 50.00 | 33.33 | 0.6956 | 4 | 1 |
| 303 | gpt | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 50.00 | 33.33 | 0.6940 | 4 | 1 |
| 304 | gpt | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 50.00 | 33.33 | 0.6933 | 4 | 1 |
| 305 | gpt | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 0.7348 | 4 | 1 |
| 306 | gpt | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 0.7044 | 4 | 1 |
| 307 | gpt | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 0.1764 | 4 | 1 |
| 308 | gpt | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 50.00 | 33.33 | 0.1792 | 4 | 1 |
| 309 | gpt | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.1740 | 4 | 1 |
| 310 | gpt | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 50.00 | 33.33 | 0.1757 | 4 | 1 |
| 311 | mobilenet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 50.00 | 33.33 | 0.8167 | 4 | 1 |
| 312 | mobilenet | cross_entropy | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 50.00 | 33.33 | 0.6937 | 4 | 1 |
| 313 | mobilenet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 50.00 | 33.33 | 0.6935 | 4 | 1 |
| 314 | mobilenet | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 50.00 | 33.33 | 1.0762 | 4 | 1 |
| 315 | mobilenet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 1.0480 | 4 | 1 |
| 316 | mobilenet | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 50.00 | 33.33 | 0.6939 | 4 | 1 |
| 317 | mobilenet | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 0.6934 | 4 | 1 |
| 318 | mobilenet | label_smoothing | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.8288 | 4 | 1 |
| 319 | mobilenet | focal_loss | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 50.00 | 33.33 | 0.1742 | 4 | 1 |
| 320 | mobilenet | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.1981 | 4 | 1 |
| 321 | mobilenet | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 0.1729 | 4 | 1 |
| 322 | mobilenet | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 0.1784 | 4 | 1 |
| 323 | resnext | cross_entropy | sgd | batch_size=16, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.7085 | 4 | 1 |
| 324 | resnext | label_smoothing | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 0.6993 | 4 | 1 |
| 325 | resnext | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 50.00 | 33.33 | 0.7462 | 4 | 1 |
| 326 | resnext | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 0.5532 | 4 | 1 |
| 327 | resnext | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 0.1893 | 4 | 1 |
| 328 | resnext | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 1.1891 | 4 | 1 |
| 329 | resnext | focal_loss | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 0.2549 | 4 | 1 |
| 330 | vit | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 50.00 | 33.33 | 0.7650 | 4 | 1 |
| 331 | vit | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 0.6946 | 4 | 1 |
| 332 | vit | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 0.7167 | 4 | 1 |
| 333 | vit | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 1.4485 | 4 | 1 |
| 334 | vit | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 0.7107 | 4 | 1 |
| 335 | vit | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.7300 | 4 | 1 |
| 336 | vit | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 0.1769 | 4 | 1 |
| 337 | vit | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.1813 | 4 | 1 |
| 338 | convnext | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 50.00 | 33.33 | 4.0818 | 4 | 1 |
| 339 | convnext | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 50.00 | 33.33 | 0.6029 | 4 | 1 |
| 340 | ghostnet | cross_entropy | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 50.00 | 33.33 | 0.6931 | 4 | 1 |
| 341 | ghostnet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 50.00 | 33.33 | 0.6870 | 4 | 1 |
| 342 | ghostnet | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.6995 | 4 | 1 |
| 343 | ghostnet | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.7841 | 4 | 1 |
| 344 | ghostnet | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 0.2584 | 4 | 1 |
| 345 | ghostnet | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 0.1732 | 4 | 1 |
| 346 | res2net | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 0.7997 | 4 | 1 |
| 347 | res2net | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 2.3958 | 4 | 1 |
| 348 | res2net | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 0.7659 | 4 | 1 |
| 349 | res2net | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 3719.3168 | 4 | 1 |
| 350 | res2net | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.0 | 50.00 | 33.33 | 0.7015 | 4 | 1 |
| 351 | res2net | label_smoothing | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.7331 | 4 | 1 |
| 352 | res2net | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 50.00 | 33.33 | 1.3873 | 4 | 1 |
| 353 | res2net | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 0.1774 | 4 | 1 |
| 354 | res2net | focal_loss | adamw | batch_size=16, lr=0.0001, weight_decay=0.001 | 50.00 | 33.33 | 0.6500 | 4 | 1 |
| 355 | res2net | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 50.00 | 33.33 | 64.2608 | 4 | 1 |
| 356 | vgg | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.6866 | 4 | 1 |
| 357 | vgg | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 50.00 | 33.33 | 0.6929 | 4 | 1 |
| 358 | vgg | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 1.3281 | 4 | 1 |
| 359 | vgg | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.7079 | 4 | 1 |
| 360 | vgg | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 5.1407 | 4 | 1 |
| 361 | vgg | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 0.6938 | 4 | 1 |
| 362 | vgg | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 1.7700 | 4 | 1 |
| 363 | vgg | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.7296 | 4 | 1 |
| 364 | vgg | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.1794 | 4 | 1 |
| 365 | vgg | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 3.7623 | 4 | 1 |
| 366 | cbam_resnet | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.7099 | 4 | 1 |
| 367 | cbam_resnet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 1.5231 | 4 | 1 |
| 368 | cbam_resnet | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 50.00 | 33.33 | 0.6762 | 4 | 1 |
| 369 | cbam_resnet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.1720 | 4 | 1 |
| 370 | cbam_resnet | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 0.1774 | 4 | 1 |
| 371 | efficientnet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 50.00 | 33.33 | 1.0192 | 4 | 1 |
| 372 | efficientnet | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 0.6933 | 4 | 1 |
| 373 | efficientnet | cross_entropy | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 0.6928 | 4 | 1 |
| 374 | efficientnet | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.6944 | 4 | 1 |
| 375 | efficientnet | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 50.00 | 33.33 | 0.6934 | 4 | 1 |
| 376 | efficientnet | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.6933 | 4 | 1 |
| 377 | efficientnet | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.7658 | 4 | 1 |
| 378 | efficientnet | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 50.00 | 33.33 | 0.2141 | 4 | 1 |
| 379 | efficientnet | focal_loss | sgd | batch_size=16, lr=0.0001, weight_decay=0.0 | 50.00 | 33.33 | 0.1746 | 4 | 1 |
| 380 | lenet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.7181 | 4 | 1 |
| 381 | lenet | cross_entropy | sgd | batch_size=16, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.6937 | 4 | 1 |
| 382 | lenet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 50.00 | 33.33 | 0.7381 | 4 | 1 |
| 383 | lenet | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 50.00 | 33.33 | 0.7117 | 4 | 1 |
| 384 | lenet | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.6964 | 4 | 1 |
| 385 | lenet | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 0.7023 | 4 | 1 |
| 386 | lenet | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 50.00 | 33.33 | 0.7191 | 4 | 1 |
| 387 | lenet | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 50.00 | 33.33 | 0.6997 | 4 | 1 |
| 388 | lenet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 0.1771 | 4 | 1 |
| 389 | lenet | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 50.00 | 33.33 | 0.1760 | 4 | 1 |
| 390 | lenet | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.1788 | 4 | 1 |
| 391 | lenet | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 0.1755 | 4 | 1 |
| 392 | repghost | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.001 | 50.00 | 33.33 | 0.7004 | 4 | 1 |
| 393 | repghost | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 50.00 | 33.33 | 0.6908 | 4 | 1 |
| 394 | repghost | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.7724 | 4 | 1 |
| 395 | repghost | label_smoothing | adamw | batch_size=16, lr=0.0001, weight_decay=0.0 | 50.00 | 33.33 | 0.6974 | 4 | 1 |
| 396 | repghost | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 0.9042 | 4 | 1 |
| 397 | repghost | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 50.00 | 33.33 | 0.2450 | 4 | 1 |
| 398 | repghost | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.3287 | 4 | 1 |
| 399 | repghost | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 1.2412 | 4 | 1 |
| 400 | swin_tiny | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.7115 | 4 | 1 |
| 401 | swin_tiny | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 50.00 | 33.33 | 2.7720 | 4 | 1 |
| 402 | swin_tiny | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 50.00 | 33.33 | 0.8659 | 4 | 1 |
| 403 | swin_tiny | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 50.00 | 33.33 | 0.8370 | 4 | 1 |
| 404 | swin_tiny | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 50.00 | 33.33 | 2.5791 | 4 | 1 |
| 405 | swin_tiny | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.6964 | 4 | 1 |
| 406 | swin_tiny | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 50.00 | 33.33 | 0.6933 | 4 | 1 |
| 407 | swin_tiny | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 50.00 | 33.33 | 1.6790 | 4 | 1 |
| 408 | swin_tiny | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 1.4596 | 4 | 1 |
| 409 | swin_tiny | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 0.5769 | 4 | 1 |
| 410 | swin_tiny | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 50.00 | 33.33 | 0.3132 | 4 | 1 |
| 411 | swin_tiny | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 50.00 | 33.33 | 0.1811 | 4 | 1 |
| 412 | capsnet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 0.6931 | 4 | 1 |
| 413 | capsnet | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.6931 | 4 | 1 |
| 414 | capsnet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 0.6931 | 4 | 1 |
| 415 | capsnet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 0.6931 | 4 | 1 |
| 416 | capsnet | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 0.6931 | 4 | 1 |
| 417 | capsnet | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 0.6931 | 4 | 1 |
| 418 | capsnet | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 0.6931 | 4 | 1 |
| 419 | capsnet | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 0.6931 | 4 | 1 |
| 420 | capsnet | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.1733 | 4 | 1 |
| 421 | capsnet | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 50.00 | 33.33 | 0.1733 | 4 | 1 |
| 422 | eca_resnet | cross_entropy | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 1.6901 | 4 | 1 |
| 423 | eca_resnet | cross_entropy | sgd | batch_size=16, lr=0.0001, weight_decay=0.0 | 50.00 | 33.33 | 0.6954 | 4 | 1 |
| 424 | eca_resnet | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 50.00 | 33.33 | 3.4974 | 4 | 1 |
| 425 | eca_resnet | label_smoothing | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 0.7023 | 4 | 1 |
| 426 | eca_resnet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 50.00 | 33.33 | 0.5365 | 4 | 1 |
| 427 | eca_resnet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 50.00 | 33.33 | 0.1784 | 4 | 1 |
| 428 | eca_resnet | focal_loss | adamw | batch_size=16, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.3455 | 4 | 1 |
| 429 | eca_resnet | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 50.00 | 33.33 | 0.6770 | 4 | 1 |
| 430 | lcnet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 50.00 | 33.33 | 2.9245 | 4 | 1 |
| 431 | lcnet | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 50.00 | 33.33 | 0.6947 | 4 | 1 |
| 432 | lcnet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 50.00 | 33.33 | 0.6961 | 4 | 1 |
| 433 | lcnet | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 0.7198 | 4 | 1 |
| 434 | lcnet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 1.0599 | 4 | 1 |
| 435 | lcnet | label_smoothing | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 0.6932 | 4 | 1 |
| 436 | lcnet | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 0.6942 | 4 | 1 |
| 437 | lcnet | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 0.9549 | 4 | 1 |
| 438 | lcnet | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 50.00 | 33.33 | 0.1735 | 4 | 1 |
| 439 | lcnet | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 50.00 | 33.33 | 0.1736 | 4 | 1 |
| 440 | lcnet | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 0.1739 | 4 | 1 |
| 441 | lcnet | focal_loss | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.1875 | 4 | 1 |
| 442 | regnet | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.7041 | 4 | 1 |
| 443 | regnet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 0.7516 | 4 | 1 |
| 444 | regnet | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 1.3328 | 4 | 1 |
| 445 | regnet | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 50.00 | 33.33 | 1.5694 | 4 | 1 |
| 446 | regnet | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 50.00 | 33.33 | 0.8210 | 4 | 1 |
| 447 | regnet | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 0.6987 | 4 | 1 |
| 448 | regnet | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 50.00 | 33.33 | 0.6895 | 4 | 1 |
| 449 | regnet | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.3097 | 4 | 1 |
| 450 | regnet | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 0.2243 | 4 | 1 |
| 451 | regnet | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 50.00 | 33.33 | 0.2806 | 4 | 1 |
| 452 | regnet | focal_loss | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.001 | 50.00 | 33.33 | 0.1873 | 4 | 1 |
| 453 | squeezenet | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 50.00 | 33.33 | 0.7234 | 4 | 1 |
| 454 | squeezenet | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 0.7347 | 4 | 1 |
| 455 | squeezenet | cross_entropy | adamw | batch_size=16, lr=0.0001, weight_decay=0.0 | 50.00 | 33.33 | 0.7228 | 4 | 1 |
| 456 | squeezenet | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.7203 | 4 | 1 |
| 457 | squeezenet | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.6982 | 4 | 1 |
| 458 | squeezenet | label_smoothing | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 0.7552 | 4 | 1 |
| 459 | squeezenet | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 50.00 | 33.33 | 0.7117 | 4 | 1 |
| 460 | squeezenet | label_smoothing | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.7215 | 4 | 1 |
| 461 | squeezenet | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 50.00 | 33.33 | 0.1757 | 4 | 1 |
| 462 | squeezenet | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 50.00 | 33.33 | 0.1880 | 4 | 1 |
| 463 | squeezenet | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 50.00 | 33.33 | 0.1814 | 4 | 1 |
| 464 | squeezenet | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.1756 | 4 | 1 |
| 465 | coatnet | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 1.4888 | 4 | 1 |
| 466 | coatnet | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 50.00 | 33.33 | 10.9309 | 4 | 1 |
| 467 | coatnet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 50.00 | 33.33 | 0.1815 | 4 | 1 |
| 468 | coatnet | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 50.00 | 33.33 | 0.8896 | 4 | 1 |
| 469 | efficientnetv2 | cross_entropy | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 50.00 | 33.33 | 1.2665 | 4 | 1 |
| 470 | efficientnetv2 | cross_entropy | sgd | batch_size=16, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.6932 | 4 | 1 |
| 471 | efficientnetv2 | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.8215 | 4 | 1 |
| 472 | efficientnetv2 | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 50.00 | 33.33 | 0.8495 | 4 | 1 |
| 473 | efficientnetv2 | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 50.00 | 33.33 | 0.6934 | 4 | 1 |
| 474 | efficientnetv2 | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 1.1734 | 4 | 1 |
| 475 | efficientnetv2 | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 50.00 | 33.33 | 0.8790 | 4 | 1 |
| 476 | efficientnetv2 | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 0.1789 | 4 | 1 |
| 477 | efficientnetv2 | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 1.3032 | 4 | 1 |
| 478 | efficientnetv2 | focal_loss | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 50.00 | 33.33 | 0.1799 | 4 | 1 |
| 479 | lstm | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 0.7058 | 4 | 1 |
| 480 | lstm | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 50.00 | 33.33 | 0.6934 | 4 | 1 |
| 481 | lstm | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 50.00 | 33.33 | 0.7292 | 4 | 1 |
| 482 | lstm | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.001 | 50.00 | 33.33 | 0.6946 | 4 | 1 |
| 483 | lstm | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 0.7080 | 4 | 1 |
| 484 | lstm | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 0.6987 | 4 | 1 |
| 485 | lstm | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.7274 | 4 | 1 |
| 486 | lstm | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 0.7087 | 4 | 1 |
| 487 | lstm | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 0.3873 | 4 | 1 |
| 488 | lstm | focal_loss | sgd | batch_size=16, lr=0.0001, weight_decay=0.0 | 50.00 | 33.33 | 0.1740 | 4 | 1 |
| 489 | lstm | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 0.1792 | 4 | 1 |
| 490 | lstm | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 0.2103 | 4 | 1 |
| 491 | repvgg | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 50.00 | 33.33 | 0.6927 | 4 | 1 |
| 492 | repvgg | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 50.00 | 33.33 | 1.4436 | 4 | 1 |
| 493 | repvgg | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.8086 | 4 | 1 |
| 494 | repvgg | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.0 | 50.00 | 33.33 | 0.6925 | 4 | 1 |
| 495 | repvgg | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 50.00 | 33.33 | 3.2282 | 4 | 1 |
| 496 | repvgg | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 50.00 | 33.33 | 0.4684 | 4 | 1 |
| 497 | van | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 50.00 | 33.33 | 0.7835 | 4 | 1 |
| 498 | van | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 50.00 | 33.33 | 0.6937 | 4 | 1 |
| 499 | van | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.7008 | 4 | 1 |
| 500 | van | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.7273 | 4 | 1 |
| 501 | van | label_smoothing | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 0.6935 | 4 | 1 |
| 502 | van | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 0.7224 | 4 | 1 |
| 503 | van | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 0.1825 | 4 | 1 |
| 504 | van | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 50.00 | 33.33 | 0.1751 | 4 | 1 |
| 505 | van | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 0.1751 | 4 | 1 |
| 506 | darknet | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 50.00 | 33.33 | 4.4823 | 4 | 1 |
| 507 | darknet | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 50.00 | 33.33 | 5630.3272 | 4 | 1 |
| 508 | darknet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 1.7741 | 4 | 1 |
| 509 | darknet | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 2.4573 | 4 | 1 |
| 510 | darknet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 0.9594 | 4 | 1 |
| 511 | darknet | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 50.00 | 33.33 | 0.7072 | 4 | 1 |
| 512 | darknet | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 1.7969 | 4 | 1 |
| 513 | darknet | label_smoothing | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 50.00 | 33.33 | 0.7879 | 4 | 1 |
| 514 | darknet | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 48.7353 | 4 | 1 |
| 515 | darknet | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.8948 | 4 | 1 |
| 516 | darknet | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.2866 | 4 | 1 |
| 517 | gru | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 0.7102 | 4 | 1 |
| 518 | gru | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 0.7198 | 4 | 1 |
| 519 | gru | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 0.7005 | 4 | 1 |
| 520 | gru | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 0.7040 | 4 | 1 |
| 521 | gru | label_smoothing | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 50.00 | 33.33 | 0.6934 | 4 | 1 |
| 522 | gru | label_smoothing | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 0.6932 | 4 | 1 |
| 523 | gru | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 0.7075 | 4 | 1 |
| 524 | gru | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 0.7057 | 4 | 1 |
| 525 | gru | focal_loss | adam | batch_size=16, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.1756 | 4 | 1 |
| 526 | gru | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.1733 | 4 | 1 |
| 527 | gru | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 50.00 | 33.33 | 0.1737 | 4 | 1 |
| 528 | gru | focal_loss | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.1793 | 4 | 1 |
| 529 | mobilenetv2 | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 50.00 | 33.33 | 3.4265 | 4 | 1 |
| 530 | mobilenetv2 | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 1.1215 | 4 | 1 |
| 531 | mobilenetv2 | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 4.7062 | 4 | 1 |
| 532 | mobilenetv2 | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 1.9013 | 4 | 1 |
| 533 | mobilenetv2 | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 4.7524 | 4 | 1 |
| 534 | mobilenetv2 | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.6941 | 4 | 1 |
| 535 | mobilenetv2 | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 3.0697 | 4 | 1 |
| 536 | mobilenetv2 | focal_loss | adam | batch_size=16, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.5048 | 4 | 1 |
| 537 | mobilenetv2 | focal_loss | sgd | batch_size=16, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.1738 | 4 | 1 |
| 538 | mobilenetv2 | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 50.00 | 33.33 | 9.2281 | 4 | 1 |
| 539 | mobilenetv2 | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 50.00 | 33.33 | 3.9590 | 4 | 1 |
| 540 | se_resnet | cross_entropy | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 50.00 | 33.33 | 1.7076 | 4 | 1 |
| 541 | se_resnet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 0.9102 | 4 | 1 |
| 542 | se_resnet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 50.00 | 33.33 | 1.7194 | 4 | 1 |
| 543 | se_resnet | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 0.7099 | 4 | 1 |
| 544 | se_resnet | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 2.2013 | 4 | 1 |
| 545 | se_resnet | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 50.00 | 33.33 | 0.2210 | 4 | 1 |
| 546 | se_resnet | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 0.5339 | 4 | 1 |
| 547 | se_resnet | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 50.00 | 33.33 | 1.7153 | 4 | 1 |
| 548 | wide_resnet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 50.00 | 33.33 | 13.1059 | 4 | 1 |
| 549 | wide_resnet | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.7129 | 4 | 1 |
| 550 | wide_resnet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 50.00 | 33.33 | 0.6942 | 4 | 1 |
| 551 | wide_resnet | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.7410 | 4 | 1 |
| 552 | bert | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 50.00 | 33.33 | 0.6934 | 4 | 1 |
| 553 | bert | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 0.7265 | 4 | 1 |
| 554 | bert | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 0.7079 | 4 | 1 |
| 555 | bert | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 50.00 | 33.33 | 0.6948 | 4 | 1 |
| 556 | bert | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 50.00 | 33.33 | 0.7025 | 4 | 1 |
| 557 | bert | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 50.00 | 33.33 | 0.6927 | 4 | 1 |
| 558 | bert | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.6941 | 4 | 1 |
| 559 | bert | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.6952 | 4 | 1 |
| 560 | bert | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 50.00 | 33.33 | 0.1744 | 4 | 1 |
| 561 | bert | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 50.00 | 33.33 | 0.1741 | 4 | 1 |
| 562 | bert | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 0.1801 | 4 | 1 |
| 563 | dpn | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 5.8504 | 4 | 1 |
| 564 | dpn | cross_entropy | sgd | batch_size=16, lr=0.0001, weight_decay=0.0 | 50.00 | 33.33 | 0.6934 | 4 | 1 |
| 565 | dpn | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.8276 | 4 | 1 |
| 566 | dpn | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 50.00 | 33.33 | 0.6931 | 4 | 1 |
| 567 | dpn | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 50.00 | 33.33 | 0.7091 | 4 | 1 |
| 568 | dpn | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 50.00 | 33.33 | 0.6936 | 4 | 1 |
| 569 | dpn | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 0.1783 | 4 | 1 |
| 570 | dpn | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 8.1881 | 4 | 1 |
| 571 | dpn | focal_loss | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.3591 | 4 | 1 |
| 572 | inception_resnet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 3.6346e+07 | 4 | 1 |
| 573 | inception_resnet | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.6986 | 4 | 1 |
| 574 | inception_resnet | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 0.8204 | 4 | 1 |
| 575 | inception_resnet | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 50.00 | 33.33 | 0.7207 | 4 | 1 |
| 576 | inception_resnet | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 50.00 | 33.33 | 0.6974 | 4 | 1 |
| 577 | inception_resnet | label_smoothing | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 0.7155 | 4 | 1 |
| 578 | inception_resnet | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 50.00 | 33.33 | 0.2821 | 4 | 1 |
| 579 | inception_resnet | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 0.1746 | 4 | 1 |
| 580 | inception_resnet | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 83143.0449 | 4 | 1 |
| 581 | inception_resnet | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 50.00 | 33.33 | 0.2470 | 4 | 1 |
| 582 | poolformer | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 50.00 | 33.33 | 0.7753 | 4 | 1 |
| 583 | poolformer | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 50.00 | 33.33 | 1.7770 | 4 | 1 |
| 584 | poolformer | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.6988 | 4 | 1 |
| 585 | poolformer | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 50.00 | 33.33 | 0.6866 | 4 | 1 |
| 586 | poolformer | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 1.1456 | 4 | 1 |
| 587 | poolformer | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.8152 | 4 | 1 |
| 588 | poolformer | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 0.8186 | 4 | 1 |
| 589 | poolformer | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 14.6352 | 4 | 1 |
| 590 | poolformer | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.1836 | 4 | 1 |
| 591 | poolformer | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 1.9193 | 4 | 1 |
| 592 | sknet | cross_entropy | adam | batch_size=16, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 0.9344 | 4 | 1 |
| 593 | sknet | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 50.00 | 33.33 | 0.7680 | 4 | 1 |
| 594 | sknet | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.8103 | 4 | 1 |
| 595 | sknet | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 50.00 | 33.33 | 0.6939 | 4 | 1 |
| 596 | sknet | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 50.00 | 33.33 | 0.7197 | 4 | 1 |
| 597 | sknet | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 50.00 | 33.33 | 0.7631 | 4 | 1 |
| 598 | sknet | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 50.00 | 33.33 | 0.2142 | 4 | 1 |
| 599 | sknet | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 50.00 | 33.33 | 0.2037 | 4 | 1 |
| 600 | sknet | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 50.00 | 33.33 | 81.6506 | 4 | 1 |

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
| 1 | dcgan | bce | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 0.0000 | 100.0000 | 10 | 10 |
| 2 | wgan | bce | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 0.0446 | -0.0557 | 4 | 1 |
| 3 | cgan | wasserstein | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 0.7370 | 1.3271 | 4 | 1 |
| 4 | vanilla_gan | bce | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 0.6913 | 1.0036 | 4 | 1 |

## GAN — all trials (28 rows)

| Rank | Model | Loss | Optimizer | Hyperparameters | G Loss | D Loss | Epochs Run | Convergence Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | dcgan | bce | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 0.0000 | 100.0000 | 10 | 10 |
| 2 | wgan | bce | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 0.0446 | -0.0557 | 4 | 1 |
| 3 | wgan | bce | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 0.1064 | -0.1141 | 4 | 1 |
| 4 | wgan | wasserstein | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 0.1446 | -0.1722 | 4 | 1 |
| 5 | wgan | wasserstein | adam | batch_size=16, lr=0.001, weight_decay=0.001 | 0.2101 | -0.2210 | 4 | 1 |
| 6 | cgan | wasserstein | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 0.7370 | 1.3271 | 4 | 1 |
| 7 | cgan | bce | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 1.4086 | 0.6773 | 9 | 6 |
| 8 | vanilla_gan | bce | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 0.6913 | 1.0036 | 4 | 1 |
| 9 | cgan | bce | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 0.7794 | 1.2656 | 6 | 3 |
| 10 | cgan | bce | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 0.9109 | 1.0544 | 4 | 1 |
| 11 | vanilla_gan | wasserstein | sgd | batch_size=16, lr=0.0001, weight_decay=0.0001 | 0.7072 | 1.2223 | 4 | 1 |
| 12 | dcgan | wasserstein | sgd | batch_size=16, lr=0.0001, weight_decay=0.0 | 0.9688 | 1.1675 | 4 | 1 |
| 13 | cgan | wasserstein | adamw | batch_size=16, lr=0.0001, weight_decay=0.001 | 0.9235 | 1.0908 | 4 | 1 |
| 14 | cgan | bce | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 0.9545 | 1.1752 | 5 | 2 |
| 15 | cgan | wasserstein | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 1.4412 | 0.6117 | 5 | 2 |
| 16 | cgan | wasserstein | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 2.1577 | 0.7450 | 7 | 4 |
| 17 | dcgan | bce | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 1.8807 | 0.7474 | 4 | 1 |
| 18 | vanilla_gan | wasserstein | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 2.0748 | 0.5215 | 8 | 5 |
| 19 | vanilla_gan | bce | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 2.3778 | 0.4074 | 7 | 4 |
| 20 | dcgan | wasserstein | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 2.9075 | 0.4402 | 4 | 1 |
| 21 | vanilla_gan | bce | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 2.4522 | 0.1126 | 4 | 1 |
| 22 | vanilla_gan | wasserstein | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 58.9403 | 0.4445 | 4 | 1 |
| 23 | dcgan | wasserstein | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 6.6587 | 0.0022 | 5 | 2 |
| 24 | dcgan | bce | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 7.2087 | 0.0014 | 5 | 2 |
| 25 | vanilla_gan | wasserstein | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 99.2941 | 100.0000 | 4 | 1 |
| 26 | dcgan | bce | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 42.7219 | 2.7940e-07 | 4 | 1 |
| 27 | dcgan | wasserstein | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 10.5607 | 0.0143 | 4 | 1 |
| 28 | vanilla_gan | bce | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 100.0000 | 0.0000 | 4 | 1 |

## Search Space

- Loss (classification): cross_entropy, label_smoothing, focal_loss
- Loss (autoencoder): mse, l1, bce
- Loss (GAN): bce, wasserstein (informational; GANs use fixed objectives)
- Optimizers: adam, sgd, adamw, rmsprop
- Hyperparameters: {"lr": [0.0001, 0.001, 0.01], "batch_size": [8, 16, 32], "weight_decay": [0.0, 0.0001, 0.001]}
