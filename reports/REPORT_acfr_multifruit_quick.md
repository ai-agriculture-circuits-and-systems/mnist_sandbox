# Acfr_Multifruit Regression Report

## Run configuration

| Setting | Value |
| --- | --- |
| Generated | 2026-06-03T03:46:03 |
| Dataset | acfr_multifruit |
| Classes | 3 |
| Class names | almond, apple, mangoe |
| Mode | quick-test |
| Max epochs | 10 |
| Early-stop patience | 3 |
| Min delta | 0.1 |
| NAS trials per config | 2 |
| Workers | 10 |
| Max batch size | 16 |
| Total wall time | 25217.2s |

Training stops when validation metric shows no significant improvement for 3 consecutive epochs.

## Classification — best per model

| Rank | Model | Loss | Optimizer | Hyperparameters | Test Acc (%) | Macro-F1 (%) | Test Loss | Epochs Run | Convergence Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | sknet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 90.00 | 79.89 | 0.3782 | 10 | 9 |
| 2 | convnext | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 88.33 | 78.10 | 0.9901 | 10 | 8 |
| 3 | repvgg | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 88.33 | 78.01 | 0.7770 | 10 | 7 |
| 4 | mobilenetv3 | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 86.67 | 73.24 | 0.3831 | 10 | 10 |
| 5 | alexnet | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 86.67 | 61.92 | 0.6831 | 10 | 7 |
| 6 | mlp | label_smoothing | adamw | batch_size=16, lr=0.0001, weight_decay=0.001 | 86.67 | 69.51 | 0.6493 | 9 | 6 |
| 7 | vit | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 86.67 | 61.92 | 0.5653 | 5 | 2 |
| 8 | lenet | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 86.67 | 69.20 | 0.6394 | 10 | 7 |
| 9 | vim_tiny | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 86.67 | 61.92 | 3.2838 | 7 | 4 |
| 10 | efficientnetv2 | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 86.67 | 76.29 | 0.2173 | 10 | 7 |
| 11 | deit | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 85.00 | 60.74 | 0.8167 | 4 | 1 |
| 12 | res2net | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 85.00 | 68.30 | 1.6353 | 10 | 10 |
| 13 | vgg | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 85.00 | 61.18 | 0.9485 | 10 | 9 |
| 14 | resnext | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 85.00 | 74.97 | 0.8814 | 9 | 6 |
| 15 | cbam_resnet | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 85.00 | 74.74 | 0.4720 | 10 | 10 |
| 16 | repghost | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 85.00 | 60.40 | 0.7816 | 9 | 6 |
| 17 | coord_resnet | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 85.00 | 60.30 | 0.3847 | 10 | 7 |
| 18 | googlenet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 85.00 | 61.18 | 0.6288 | 10 | 7 |
| 19 | eca_resnet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 85.00 | 75.01 | 0.1259 | 10 | 8 |
| 20 | squeezenet | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 85.00 | 67.73 | 0.7157 | 10 | 8 |
| 21 | lstm | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 85.00 | 60.30 | 0.2120 | 9 | 6 |
| 22 | bert | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 85.00 | 68.20 | 0.6589 | 10 | 7 |
| 23 | inception_resnet | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 85.00 | 60.30 | 6.6143 | 8 | 5 |
| 24 | shufflenet | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 83.33 | 71.69 | 0.2668 | 9 | 6 |
| 25 | xception | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 83.33 | 59.29 | 1.6002 | 6 | 3 |
| 26 | nin | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 83.33 | 59.18 | 0.6549 | 6 | 3 |
| 27 | ghostnet | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 83.33 | 70.56 | 0.9703 | 10 | 9 |
| 28 | gpt | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 83.33 | 60.17 | 0.7183 | 10 | 8 |
| 29 | lcnet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 83.33 | 75.69 | 0.6788 | 9 | 6 |
| 30 | coatnet | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 83.33 | 60.15 | 2.0683 | 6 | 3 |
| 31 | darknet | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 83.33 | 59.01 | 2.3668 | 5 | 2 |
| 32 | dpn | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 83.33 | 73.19 | 0.6852 | 9 | 6 |
| 33 | hrnet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 81.67 | 68.86 | 0.9591 | 5 | 2 |
| 34 | simple_cnn | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 81.67 | 57.88 | 3.8606 | 4 | 1 |
| 35 | efficientnet | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 81.67 | 58.32 | 1.4316 | 5 | 2 |
| 36 | gru | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 81.67 | 58.32 | 0.9661 | 9 | 6 |
| 37 | swin_tiny | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 80.00 | 64.20 | 0.5346 | 7 | 4 |
| 38 | densenet | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 78.33 | 57.00 | 0.3409 | 10 | 7 |
| 39 | mnasnet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 78.33 | 55.45 | 0.6898 | 9 | 6 |
| 40 | van | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 78.33 | 55.45 | 6.1443 | 7 | 4 |
| 41 | se_resnet | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 78.33 | 70.96 | 0.2052 | 10 | 10 |
| 42 | wide_resnet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 78.33 | 57.57 | 0.8273 | 5 | 2 |
| 43 | resnet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 76.67 | 70.76 | 0.4624 | 10 | 10 |
| 44 | hardnet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 75.00 | 63.77 | 1.3645 | 10 | 8 |
| 45 | cspnet | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 71.67 | 66.92 | 0.7577 | 10 | 10 |
| 46 | regnet | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 71.67 | 68.31 | 1.5035 | 10 | 8 |
| 47 | mobilenetv2 | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 65.00 | 63.15 | 1.3849 | 8 | 5 |
| 48 | poolformer | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 51.67 | 30.55 | 1.7815 | 7 | 4 |
| 49 | mobilenet | cross_entropy | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 48.33 | 21.72 | 1.0630 | 4 | 1 |
| 50 | capsnet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 48.33 | 21.72 | 1.0986 | 4 | 1 |

## Classification — all trials (600 rows)

| Rank | Model | Loss | Optimizer | Hyperparameters | Test Acc (%) | Macro-F1 (%) | Test Loss | Epochs Run | Convergence Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | sknet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 90.00 | 79.89 | 0.3782 | 10 | 9 |
| 2 | convnext | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 88.33 | 78.10 | 0.9901 | 10 | 8 |
| 3 | repvgg | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 88.33 | 78.01 | 0.7770 | 10 | 7 |
| 4 | mobilenetv3 | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 86.67 | 73.24 | 0.3831 | 10 | 10 |
| 5 | mobilenetv3 | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 86.67 | 69.20 | 0.2834 | 10 | 7 |
| 6 | alexnet | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 86.67 | 61.92 | 0.6831 | 10 | 7 |
| 7 | convnext | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 86.67 | 61.92 | 0.7588 | 6 | 3 |
| 8 | convnext | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 86.67 | 61.92 | 0.1732 | 10 | 10 |
| 9 | mlp | label_smoothing | adamw | batch_size=16, lr=0.0001, weight_decay=0.001 | 86.67 | 69.51 | 0.6493 | 9 | 6 |
| 10 | vit | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 86.67 | 61.92 | 0.5653 | 5 | 2 |
| 11 | lenet | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 86.67 | 69.20 | 0.6394 | 10 | 7 |
| 12 | vim_tiny | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 86.67 | 61.92 | 3.2838 | 7 | 4 |
| 13 | efficientnetv2 | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 86.67 | 76.29 | 0.2173 | 10 | 7 |
| 14 | deit | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 85.00 | 60.74 | 0.8167 | 4 | 1 |
| 15 | deit | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 85.00 | 71.77 | 0.9116 | 7 | 4 |
| 16 | mobilenetv3 | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 85.00 | 60.74 | 1.0538 | 10 | 10 |
| 17 | mobilenetv3 | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 85.00 | 68.20 | 0.7197 | 9 | 6 |
| 18 | alexnet | label_smoothing | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 85.00 | 67.86 | 0.6140 | 10 | 8 |
| 19 | alexnet | focal_loss | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0 | 85.00 | 61.11 | 0.4088 | 7 | 4 |
| 20 | convnext | focal_loss | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 85.00 | 61.18 | 0.4770 | 7 | 4 |
| 21 | mlp | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 85.00 | 71.77 | 0.1644 | 6 | 3 |
| 22 | res2net | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 85.00 | 68.30 | 1.6353 | 10 | 10 |
| 23 | vgg | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 85.00 | 61.18 | 0.9485 | 10 | 9 |
| 24 | vgg | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 85.00 | 68.03 | 0.8532 | 8 | 5 |
| 25 | vgg | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 85.00 | 61.34 | 0.7012 | 9 | 6 |
| 26 | vgg | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 85.00 | 67.72 | 0.3871 | 9 | 6 |
| 27 | resnext | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 85.00 | 74.97 | 0.8814 | 9 | 6 |
| 28 | vit | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 85.00 | 60.74 | 0.6040 | 5 | 2 |
| 29 | vit | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 85.00 | 61.18 | 0.2504 | 6 | 3 |
| 30 | vit | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 85.00 | 60.74 | 0.2716 | 5 | 2 |
| 31 | cbam_resnet | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 85.00 | 74.74 | 0.4720 | 10 | 10 |
| 32 | lenet | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 85.00 | 60.74 | 1.0774 | 8 | 5 |
| 33 | repghost | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 85.00 | 60.40 | 0.7816 | 9 | 6 |
| 34 | coord_resnet | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 85.00 | 60.30 | 0.3847 | 10 | 7 |
| 35 | googlenet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 85.00 | 61.18 | 0.6288 | 10 | 7 |
| 36 | googlenet | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 85.00 | 71.65 | 0.6428 | 7 | 4 |
| 37 | googlenet | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 85.00 | 67.52 | 0.6905 | 5 | 2 |
| 38 | eca_resnet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 85.00 | 75.01 | 0.1259 | 10 | 8 |
| 39 | squeezenet | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 85.00 | 67.73 | 0.7157 | 10 | 8 |
| 40 | squeezenet | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 85.00 | 67.73 | 0.7009 | 10 | 7 |
| 41 | lstm | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 85.00 | 60.30 | 0.2120 | 9 | 6 |
| 42 | bert | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 85.00 | 68.20 | 0.6589 | 10 | 7 |
| 43 | inception_resnet | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 85.00 | 60.30 | 6.6143 | 8 | 5 |
| 44 | inception_resnet | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 85.00 | 60.30 | 4.4265 | 4 | 1 |
| 45 | sknet | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 85.00 | 74.69 | 0.7488 | 10 | 10 |
| 46 | deit | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 83.33 | 70.30 | 0.4883 | 6 | 3 |
| 47 | deit | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 83.33 | 60.04 | 0.7277 | 8 | 5 |
| 48 | mobilenetv3 | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 83.33 | 59.24 | 1.3403 | 8 | 5 |
| 49 | shufflenet | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 83.33 | 71.69 | 0.2668 | 9 | 6 |
| 50 | xception | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 83.33 | 59.29 | 1.6002 | 6 | 3 |
| 51 | alexnet | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 83.33 | 59.86 | 0.4007 | 5 | 2 |
| 52 | nin | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 83.33 | 59.18 | 0.6549 | 6 | 3 |
| 53 | convnext | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 83.33 | 61.29 | 0.7083 | 9 | 6 |
| 54 | ghostnet | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 83.33 | 70.56 | 0.9703 | 10 | 9 |
| 55 | mlp | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.0001 | 83.33 | 70.66 | 0.5687 | 9 | 6 |
| 56 | mlp | cross_entropy | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 83.33 | 66.34 | 0.5189 | 8 | 5 |
| 57 | vgg | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.0001 | 83.33 | 66.34 | 0.5020 | 10 | 10 |
| 58 | vgg | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 83.33 | 66.24 | 0.8082 | 10 | 7 |
| 59 | gpt | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 83.33 | 60.17 | 0.7183 | 10 | 8 |
| 60 | vit | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 83.33 | 66.52 | 0.6312 | 4 | 1 |
| 61 | vit | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 83.33 | 66.52 | 0.8197 | 8 | 5 |
| 62 | lenet | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 83.33 | 59.54 | 0.8983 | 4 | 1 |
| 63 | lenet | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 83.33 | 60.40 | 0.1924 | 8 | 5 |
| 64 | googlenet | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 83.33 | 66.24 | 0.7734 | 8 | 5 |
| 65 | eca_resnet | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 83.33 | 59.54 | 1.3233 | 5 | 2 |
| 66 | lcnet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 83.33 | 75.69 | 0.6788 | 9 | 6 |
| 67 | squeezenet | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 83.33 | 66.49 | 0.6788 | 6 | 3 |
| 68 | coatnet | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 83.33 | 60.15 | 2.0683 | 6 | 3 |
| 69 | darknet | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 83.33 | 59.01 | 2.3668 | 5 | 2 |
| 70 | bert | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 83.33 | 70.40 | 1.0217 | 10 | 7 |
| 71 | dpn | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 83.33 | 73.19 | 0.6852 | 9 | 6 |
| 72 | xception | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 81.67 | 71.45 | 1.0200 | 10 | 9 |
| 73 | xception | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 81.67 | 58.02 | 2.6369 | 10 | 7 |
| 74 | alexnet | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 81.67 | 65.03 | 0.5399 | 10 | 7 |
| 75 | alexnet | label_smoothing | adam | batch_size=16, lr=0.0001, weight_decay=0.0001 | 81.67 | 71.61 | 0.7027 | 6 | 3 |
| 76 | hrnet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 81.67 | 68.86 | 0.9591 | 5 | 2 |
| 77 | nin | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 81.67 | 65.43 | 0.6702 | 5 | 2 |
| 78 | nin | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 81.67 | 57.88 | 0.9570 | 6 | 3 |
| 79 | nin | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 81.67 | 64.88 | 0.7772 | 7 | 4 |
| 80 | nin | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 81.67 | 69.02 | 0.1700 | 8 | 5 |
| 81 | nin | focal_loss | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 81.67 | 59.16 | 0.2183 | 10 | 8 |
| 82 | simple_cnn | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 81.67 | 57.88 | 3.8606 | 4 | 1 |
| 83 | simple_cnn | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 81.67 | 65.20 | 1.2649 | 10 | 9 |
| 84 | simple_cnn | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 81.67 | 59.58 | 2.2779 | 10 | 9 |
| 85 | convnext | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 81.67 | 57.97 | 8.2479 | 4 | 1 |
| 86 | convnext | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 81.67 | 58.07 | 1.8460 | 6 | 3 |
| 87 | convnext | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 81.67 | 59.16 | 0.3681 | 6 | 3 |
| 88 | mlp | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 81.67 | 65.25 | 0.6815 | 5 | 2 |
| 89 | mlp | label_smoothing | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 81.67 | 65.03 | 0.6542 | 6 | 3 |
| 90 | mlp | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 81.67 | 58.32 | 0.2087 | 4 | 1 |
| 91 | vgg | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 81.67 | 58.32 | 0.6142 | 9 | 6 |
| 92 | gpt | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 81.67 | 65.33 | 1.0462 | 7 | 4 |
| 93 | gpt | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 81.67 | 65.10 | 0.4525 | 10 | 9 |
| 94 | resnext | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 81.67 | 71.61 | 0.4036 | 10 | 9 |
| 95 | resnext | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 81.67 | 58.32 | 0.1943 | 8 | 5 |
| 96 | vit | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 81.67 | 57.97 | 0.6351 | 9 | 6 |
| 97 | vit | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 81.67 | 58.32 | 1.0628 | 7 | 4 |
| 98 | vit | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 81.67 | 58.32 | 0.1973 | 4 | 1 |
| 99 | cbam_resnet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 81.67 | 76.73 | 0.3806 | 10 | 10 |
| 100 | efficientnet | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 81.67 | 58.32 | 1.4316 | 5 | 2 |
| 101 | efficientnet | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 81.67 | 58.32 | 1.0685 | 7 | 4 |
| 102 | lenet | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 81.67 | 60.46 | 0.7552 | 5 | 2 |
| 103 | lenet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 81.67 | 57.88 | 0.1661 | 4 | 1 |
| 104 | googlenet | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 81.67 | 64.98 | 0.2149 | 10 | 8 |
| 105 | vim_tiny | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 81.67 | 71.72 | 0.6816 | 10 | 8 |
| 106 | vim_tiny | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 81.67 | 58.73 | 1.9529 | 10 | 9 |
| 107 | squeezenet | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 81.67 | 58.32 | 0.9766 | 5 | 2 |
| 108 | squeezenet | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 81.67 | 58.32 | 0.3486 | 9 | 6 |
| 109 | squeezenet | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 81.67 | 65.03 | 0.3757 | 10 | 7 |
| 110 | coatnet | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 81.67 | 58.95 | 1.4506 | 9 | 6 |
| 111 | gru | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 81.67 | 58.32 | 0.9661 | 9 | 6 |
| 112 | bert | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 81.67 | 58.32 | 1.1882 | 9 | 6 |
| 113 | bert | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 81.67 | 58.32 | 1.1024 | 5 | 2 |
| 114 | bert | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 81.67 | 58.32 | 1.0941 | 4 | 1 |
| 115 | bert | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 81.67 | 58.32 | 0.4847 | 5 | 2 |
| 116 | bert | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 81.67 | 58.32 | 0.2651 | 9 | 6 |
| 117 | deit | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 80.00 | 67.53 | 0.5834 | 7 | 4 |
| 118 | deit | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 80.00 | 59.02 | 0.3280 | 4 | 1 |
| 119 | alexnet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 80.00 | 67.61 | 0.7105 | 5 | 2 |
| 120 | nin | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0 | 80.00 | 59.02 | 0.6889 | 10 | 8 |
| 121 | nin | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 80.00 | 58.75 | 0.2055 | 10 | 7 |
| 122 | convnext | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 80.00 | 64.42 | 0.1736 | 7 | 4 |
| 123 | mlp | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 80.00 | 57.08 | 0.6102 | 5 | 2 |
| 124 | mlp | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 80.00 | 63.98 | 0.2373 | 5 | 2 |
| 125 | vgg | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 80.00 | 63.68 | 0.3676 | 6 | 3 |
| 126 | vit | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 80.00 | 57.48 | 0.5835 | 5 | 2 |
| 127 | vit | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 80.00 | 64.00 | 0.7252 | 5 | 2 |
| 128 | vit | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 80.00 | 64.00 | 0.1781 | 7 | 4 |
| 129 | lenet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 80.00 | 64.29 | 0.5892 | 9 | 6 |
| 130 | swin_tiny | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 80.00 | 64.20 | 0.5346 | 7 | 4 |
| 131 | swin_tiny | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 80.00 | 67.86 | 0.4436 | 5 | 2 |
| 132 | googlenet | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 80.00 | 70.89 | 0.6806 | 8 | 5 |
| 133 | vim_tiny | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 80.00 | 64.01 | 1.2367 | 7 | 4 |
| 134 | lstm | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 80.00 | 57.08 | 0.2811 | 7 | 4 |
| 135 | darknet | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 80.00 | 57.08 | 8.4895 | 10 | 7 |
| 136 | gru | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 80.00 | 57.89 | 0.8816 | 8 | 5 |
| 137 | deit | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 78.33 | 70.96 | 0.5815 | 5 | 2 |
| 138 | deit | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 78.33 | 55.47 | 0.2347 | 4 | 1 |
| 139 | densenet | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 78.33 | 57.00 | 0.3409 | 10 | 7 |
| 140 | simple_cnn | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 78.33 | 57.57 | 512.4632 | 8 | 5 |
| 141 | simple_cnn | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 78.33 | 58.14 | 1921.9164 | 6 | 3 |
| 142 | mlp | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 78.33 | 56.59 | 0.7049 | 4 | 1 |
| 143 | vgg | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 78.33 | 66.19 | 0.4655 | 9 | 6 |
| 144 | gpt | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 78.33 | 62.70 | 0.4503 | 10 | 10 |
| 145 | cbam_resnet | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 78.33 | 73.62 | 0.1460 | 10 | 10 |
| 146 | swin_tiny | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 78.33 | 56.59 | 0.6839 | 7 | 4 |
| 147 | mnasnet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 78.33 | 55.45 | 0.6898 | 9 | 6 |
| 148 | vim_tiny | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 78.33 | 63.03 | 0.8521 | 10 | 8 |
| 149 | squeezenet | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 78.33 | 66.39 | 0.9152 | 6 | 3 |
| 150 | squeezenet | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 78.33 | 63.13 | 0.3908 | 6 | 3 |
| 151 | van | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 78.33 | 55.45 | 6.1443 | 7 | 4 |
| 152 | darknet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 78.33 | 55.80 | 1.6501 | 6 | 3 |
| 153 | se_resnet | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 78.33 | 70.96 | 0.2052 | 10 | 10 |
| 154 | wide_resnet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 78.33 | 57.57 | 0.8273 | 5 | 2 |
| 155 | wide_resnet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 78.33 | 68.90 | 0.7787 | 10 | 8 |
| 156 | deit | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 76.67 | 67.63 | 0.7787 | 8 | 5 |
| 157 | simple_cnn | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 76.67 | 61.72 | 40.9771 | 9 | 6 |
| 158 | simple_cnn | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 76.67 | 67.81 | 0.1946 | 10 | 9 |
| 159 | mlp | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 76.67 | 61.59 | 0.7741 | 5 | 2 |
| 160 | repghost | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 76.67 | 61.59 | 0.8696 | 4 | 1 |
| 161 | mnasnet | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 76.67 | 56.07 | 0.4502 | 9 | 6 |
| 162 | resnet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 76.67 | 70.76 | 0.4624 | 10 | 10 |
| 163 | lstm | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 76.67 | 54.25 | 1.4586 | 8 | 5 |
| 164 | repvgg | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 76.67 | 54.25 | 0.3000 | 7 | 4 |
| 165 | darknet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 76.67 | 54.25 | 0.9708 | 7 | 4 |
| 166 | bert | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 76.67 | 67.46 | 0.4532 | 10 | 10 |
| 167 | dpn | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 76.67 | 67.68 | 0.7345 | 10 | 8 |
| 168 | sknet | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 76.67 | 65.50 | 0.4473 | 7 | 4 |
| 169 | hardnet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 75.00 | 63.77 | 1.3645 | 10 | 8 |
| 170 | hrnet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 75.00 | 66.57 | 0.4597 | 10 | 10 |
| 171 | simple_cnn | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 75.00 | 66.56 | 0.5970 | 10 | 7 |
| 172 | simple_cnn | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 75.00 | 68.22 | 6.5706 | 9 | 6 |
| 173 | mlp | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 75.00 | 60.70 | 0.2477 | 7 | 4 |
| 174 | repghost | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 75.00 | 60.50 | 0.6979 | 10 | 7 |
| 175 | vim_tiny | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 75.00 | 66.49 | 0.3377 | 9 | 6 |
| 176 | coatnet | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 75.00 | 71.44 | 0.6308 | 10 | 10 |
| 177 | efficientnetv2 | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 75.00 | 59.92 | 0.5233 | 9 | 6 |
| 178 | van | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 75.00 | 64.06 | — | 8 | 5 |
| 179 | gru | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 75.00 | 55.84 | 0.4237 | 10 | 10 |
| 180 | gru | focal_loss | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0 | 75.00 | 53.15 | 0.4525 | 9 | 6 |
| 181 | hardnet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 73.33 | 65.33 | 0.5137 | 10 | 7 |
| 182 | hrnet | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 73.33 | 69.36 | 0.2244 | 10 | 10 |
| 183 | convnext | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 73.33 | 63.23 | 0.5354 | 7 | 4 |
| 184 | ghostnet | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 73.33 | 62.27 | 0.7218 | 5 | 2 |
| 185 | cbam_resnet | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 73.33 | 51.95 | 1.0132 | 5 | 2 |
| 186 | coord_resnet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 73.33 | 67.97 | 0.7002 | 10 | 7 |
| 187 | vim_tiny | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 73.33 | 68.42 | 0.7509 | 10 | 10 |
| 188 | eca_resnet | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 73.33 | 69.51 | 1.1283 | 10 | 9 |
| 189 | efficientnetv2 | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 73.33 | 65.00 | 0.7237 | 10 | 10 |
| 190 | lstm | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 73.33 | 51.82 | 0.8106 | 8 | 5 |
| 191 | van | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 73.33 | 66.30 | 931238.3179 | 10 | 10 |
| 192 | van | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 73.33 | 67.00 | 0.6773 | 10 | 9 |
| 193 | se_resnet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 73.33 | 69.52 | 0.5995 | 10 | 10 |
| 194 | wide_resnet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 73.33 | 51.77 | 0.5025 | 5 | 2 |
| 195 | dpn | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 73.33 | 53.26 | 2.1379 | 7 | 4 |
| 196 | dpn | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 73.33 | 51.94 | 1.5973 | 4 | 1 |
| 197 | inception_resnet | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 73.33 | 66.34 | 0.9085 | 10 | 10 |
| 198 | hardnet | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 71.67 | 69.06 | 0.5801 | 10 | 7 |
| 199 | mobilenetv3 | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 71.67 | 57.43 | 1.8228 | 9 | 6 |
| 200 | alexnet | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 71.67 | 50.34 | 0.5947 | 7 | 4 |
| 201 | hrnet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 71.67 | 66.92 | 0.2485 | 10 | 9 |
| 202 | ghostnet | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 71.67 | 68.31 | 0.2913 | 10 | 10 |
| 203 | ghostnet | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 71.67 | 68.09 | 0.2181 | 10 | 8 |
| 204 | cspnet | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 71.67 | 66.92 | 0.7577 | 10 | 10 |
| 205 | resnext | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 71.67 | 68.89 | 0.7535 | 10 | 9 |
| 206 | cbam_resnet | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 71.67 | 67.28 | 0.7149 | 10 | 10 |
| 207 | cbam_resnet | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 71.67 | 52.18 | 0.2868 | 7 | 4 |
| 208 | repghost | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 71.67 | 68.64 | 0.6828 | 10 | 8 |
| 209 | repghost | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 71.67 | 66.71 | 0.2427 | 10 | 9 |
| 210 | swin_tiny | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 71.67 | 61.26 | 1.0975 | 4 | 1 |
| 211 | coord_resnet | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 71.67 | 68.21 | 0.7818 | 10 | 9 |
| 212 | vim_tiny | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 71.67 | 66.92 | 0.6861 | 10 | 10 |
| 213 | eca_resnet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 71.67 | 68.89 | 0.7608 | 10 | 10 |
| 214 | regnet | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 71.67 | 68.31 | 1.5035 | 10 | 8 |
| 215 | coatnet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 71.67 | 51.03 | 1.6162 | 7 | 4 |
| 216 | van | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 71.67 | 68.31 | 0.2057 | 6 | 3 |
| 217 | gru | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 71.67 | 51.19 | 0.8250 | 8 | 5 |
| 218 | se_resnet | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 71.67 | 68.21 | 0.8899 | 10 | 9 |
| 219 | wide_resnet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 71.67 | 63.32 | 0.4712 | 10 | 8 |
| 220 | sknet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 71.67 | 68.09 | 0.7319 | 10 | 10 |
| 221 | sknet | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 71.67 | 66.92 | 0.7330 | 10 | 10 |
| 222 | hardnet | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 70.00 | 66.82 | 0.6946 | 10 | 8 |
| 223 | hardnet | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 70.00 | 66.82 | 0.2346 | 10 | 8 |
| 224 | shufflenet | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 70.00 | 62.97 | 0.1688 | 10 | 10 |
| 225 | densenet | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 70.00 | 55.87 | 1.3935 | 5 | 2 |
| 226 | hrnet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 70.00 | 62.73 | 0.5930 | 9 | 6 |
| 227 | hrnet | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 70.00 | 62.88 | 0.8213 | 10 | 8 |
| 228 | hrnet | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 70.00 | 66.82 | 0.2316 | 10 | 9 |
| 229 | ghostnet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 70.00 | 65.53 | 0.5585 | 10 | 8 |
| 230 | cbam_resnet | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 70.00 | 66.43 | 0.7046 | 10 | 10 |
| 231 | cbam_resnet | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 70.00 | 65.71 | 0.6265 | 10 | 7 |
| 232 | cbam_resnet | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 70.00 | 64.36 | 0.2104 | 10 | 9 |
| 233 | efficientnet | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 70.00 | 62.88 | 0.7431 | 10 | 8 |
| 234 | repghost | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 70.00 | 66.67 | 0.5413 | 10 | 9 |
| 235 | repghost | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 70.00 | 64.55 | 0.2137 | 10 | 10 |
| 236 | coord_resnet | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 70.00 | 66.43 | 0.7309 | 10 | 8 |
| 237 | coord_resnet | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 70.00 | 62.97 | 0.4983 | 10 | 8 |
| 238 | vim_tiny | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 70.00 | 66.82 | 0.5339 | 10 | 9 |
| 239 | vim_tiny | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 70.00 | 65.53 | 0.3025 | 10 | 9 |
| 240 | eca_resnet | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 70.00 | 65.71 | 0.5839 | 10 | 10 |
| 241 | coatnet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 70.00 | 66.11 | 2.8563 | 10 | 8 |
| 242 | coatnet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 70.00 | 65.53 | 0.2356 | 10 | 9 |
| 243 | van | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 70.00 | 62.73 | 121844.6699 | 8 | 5 |
| 244 | darknet | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 70.00 | 63.63 | 0.8625 | 10 | 9 |
| 245 | wide_resnet | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 70.00 | 64.36 | 1.8655 | 9 | 6 |
| 246 | hardnet | cross_entropy | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 68.33 | 56.47 | 6.0866 | 4 | 1 |
| 247 | hrnet | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 68.33 | 65.23 | 0.7748 | 10 | 9 |
| 248 | convnext | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 68.33 | 64.57 | 0.9860 | 6 | 3 |
| 249 | ghostnet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 68.33 | 64.34 | 1.0648 | 10 | 7 |
| 250 | ghostnet | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 68.33 | 65.89 | 0.7297 | 10 | 10 |
| 251 | vgg | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 68.33 | 58.06 | 1.0969 | 6 | 3 |
| 252 | cspnet | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 68.33 | 56.82 | 0.8456 | 7 | 4 |
| 253 | resnext | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 68.33 | 64.49 | 0.6657 | 10 | 8 |
| 254 | coord_resnet | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 68.33 | 63.21 | 0.6330 | 10 | 9 |
| 255 | resnet | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 68.33 | 61.57 | 0.9220 | 10 | 10 |
| 256 | vim_tiny | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 68.33 | 61.83 | 1.5415 | 8 | 5 |
| 257 | vim_tiny | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 68.33 | 61.65 | 0.7109 | 10 | 9 |
| 258 | eca_resnet | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 68.33 | 66.33 | 0.7821 | 9 | 6 |
| 259 | van | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 68.33 | 47.33 | 0.8470 | 10 | 7 |
| 260 | gru | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 68.33 | 55.08 | 0.8601 | 9 | 6 |
| 261 | wide_resnet | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 68.33 | 66.15 | 0.7753 | 10 | 10 |
| 262 | sknet | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 68.33 | 61.65 | 0.5585 | 10 | 10 |
| 263 | simple_cnn | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 66.67 | 61.91 | 0.8329 | 10 | 7 |
| 264 | ghostnet | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 66.67 | 61.07 | 0.4962 | 4 | 1 |
| 265 | cspnet | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 66.67 | 46.18 | 6.8574 | 4 | 1 |
| 266 | eca_resnet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 66.67 | 64.66 | 0.2447 | 10 | 10 |
| 267 | repvgg | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 66.67 | 64.26 | 0.6858 | 8 | 5 |
| 268 | dpn | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 66.67 | 57.33 | 0.4139 | 9 | 6 |
| 269 | hardnet | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 65.00 | 59.63 | 0.3458 | 8 | 5 |
| 270 | hrnet | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 65.00 | 62.58 | 0.7917 | 10 | 10 |
| 271 | ghostnet | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 65.00 | 60.87 | 1.0337 | 10 | 9 |
| 272 | repghost | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 65.00 | 62.31 | 0.9343 | 9 | 6 |
| 273 | mobilenetv2 | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 65.00 | 63.15 | 1.3849 | 8 | 5 |
| 274 | wide_resnet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 65.00 | 60.78 | 0.8587 | 9 | 6 |
| 275 | wide_resnet | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 65.00 | 63.15 | 0.9078 | 8 | 5 |
| 276 | dpn | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 65.00 | 45.83 | 1.5095 | 7 | 4 |
| 277 | xception | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 63.33 | 58.38 | 2.6980 | 10 | 9 |
| 278 | resnet | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 63.33 | 43.62 | 0.7101 | 8 | 5 |
| 279 | resnet | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 63.33 | 44.10 | 0.4205 | 5 | 2 |
| 280 | coatnet | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 63.33 | 58.38 | 1.1754 | 6 | 3 |
| 281 | darknet | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 63.33 | 59.37 | 3.4327 | 10 | 7 |
| 282 | mobilenetv2 | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 63.33 | 57.23 | 3.3241 | 7 | 4 |
| 283 | se_resnet | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 63.33 | 60.64 | 0.5062 | 10 | 9 |
| 284 | densenet | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 61.67 | 43.90 | 0.3822 | 7 | 4 |
| 285 | resnext | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 61.67 | 57.19 | 3.0146 | 4 | 1 |
| 286 | lenet | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 61.67 | 48.73 | 0.5044 | 5 | 2 |
| 287 | repvgg | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 61.67 | 53.66 | 10.2717 | 9 | 6 |
| 288 | dpn | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 61.67 | 55.22 | 2.6331 | 8 | 5 |
| 289 | coord_resnet | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 60.00 | 56.00 | 0.2508 | 10 | 10 |
| 290 | repvgg | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 60.00 | 56.66 | 8.1029 | 7 | 4 |
| 291 | van | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 60.00 | 41.73 | 1.0155 | 6 | 3 |
| 292 | se_resnet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 60.00 | 51.96 | 0.7354 | 5 | 2 |
| 293 | se_resnet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 60.00 | 57.46 | 1.1625 | 7 | 4 |
| 294 | dpn | focal_loss | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 60.00 | 57.57 | 0.3169 | 10 | 10 |
| 295 | cspnet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 58.33 | 38.12 | 0.8452 | 5 | 2 |
| 296 | resnext | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 58.33 | 40.11 | 0.8965 | 4 | 1 |
| 297 | repvgg | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 58.33 | 54.39 | 2.9945 | 9 | 6 |
| 298 | van | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 58.33 | 39.86 | 0.9217 | 4 | 1 |
| 299 | repghost | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 56.67 | 51.58 | 0.5616 | 4 | 1 |
| 300 | van | cross_entropy | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 56.67 | 54.93 | 1.3243 | 4 | 1 |
| 301 | mobilenetv2 | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 56.67 | 34.51 | 1.0764 | 7 | 4 |
| 302 | wide_resnet | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 56.67 | 54.93 | 0.3278 | 10 | 9 |
| 303 | cspnet | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 53.33 | 49.63 | 1.7499 | 7 | 4 |
| 304 | resnext | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 53.33 | 51.85 | 1.0403 | 6 | 3 |
| 305 | resnext | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 53.33 | 43.49 | 0.5891 | 7 | 4 |
| 306 | coord_resnet | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 53.33 | 29.89 | 1.1413 | 6 | 3 |
| 307 | inception_resnet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 53.33 | 35.30 | 1.3435 | 6 | 3 |
| 308 | sknet | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 53.33 | 31.28 | 1.3429 | 5 | 2 |
| 309 | res2net | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 51.67 | 36.32 | 0.3862 | 6 | 3 |
| 310 | cspnet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 51.67 | 35.83 | 0.3905 | 5 | 2 |
| 311 | lenet | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 51.67 | 50.11 | 1.0424 | 5 | 2 |
| 312 | coatnet | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 51.67 | 33.74 | 10.0749 | 4 | 1 |
| 313 | darknet | label_smoothing | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 51.67 | 33.58 | 1.4763 | 5 | 2 |
| 314 | wide_resnet | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 51.67 | 36.32 | 0.5906 | 4 | 1 |
| 315 | dpn | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 51.67 | 36.32 | 1545.7439 | 4 | 1 |
| 316 | poolformer | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 51.67 | 30.55 | 1.7815 | 7 | 4 |
| 317 | res2net | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 50.00 | 38.10 | 12.8667 | 5 | 2 |
| 318 | gpt | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 50.00 | 40.71 | 1.1000 | 10 | 7 |
| 319 | gpt | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 50.00 | 43.23 | 1.0880 | 8 | 5 |
| 320 | gpt | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 50.00 | 42.80 | 1.0892 | 7 | 4 |
| 321 | lstm | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 50.00 | 38.65 | 1.0608 | 6 | 3 |
| 322 | repvgg | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 50.00 | 30.30 | 1.2196 | 5 | 2 |
| 323 | repvgg | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 50.00 | 42.18 | 6.5704 | 8 | 5 |
| 324 | repvgg | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 50.00 | 46.70 | 3.7952 | 10 | 10 |
| 325 | bert | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 50.00 | 43.23 | 0.4635 | 8 | 5 |
| 326 | poolformer | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 50.00 | 35.36 | 6.8588 | 8 | 5 |
| 327 | deit | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 48.33 | 21.72 | 1.0654 | 7 | 4 |
| 328 | deit | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 48.33 | 21.72 | 0.4817 | 6 | 3 |
| 329 | deit | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 48.33 | 21.72 | 0.4706 | 5 | 2 |
| 330 | hardnet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 48.33 | 21.72 | 1.8726 | 4 | 1 |
| 331 | shufflenet | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 48.33 | 21.72 | 1.0832 | 4 | 1 |
| 332 | shufflenet | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 48.33 | 37.34 | 1.7138 | 8 | 5 |
| 333 | shufflenet | label_smoothing | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 48.33 | 21.72 | 1.0803 | 4 | 1 |
| 334 | shufflenet | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 48.33 | 21.72 | 0.4827 | 4 | 1 |
| 335 | xception | cross_entropy | sgd | batch_size=16, lr=0.0001, weight_decay=0.0 | 48.33 | 21.72 | 1.0987 | 4 | 1 |
| 336 | xception | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 48.33 | 21.72 | 4697.6439 | 4 | 1 |
| 337 | xception | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 48.33 | 21.72 | 1.8208 | 4 | 1 |
| 338 | xception | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 48.33 | 21.72 | 8.4061 | 4 | 1 |
| 339 | xception | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 48.33 | 21.72 | 0.4922 | 4 | 1 |
| 340 | xception | focal_loss | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 48.33 | 21.72 | 0.5508 | 4 | 1 |
| 341 | alexnet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 48.33 | 21.72 | 62.0906 | 5 | 2 |
| 342 | alexnet | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 48.33 | 21.72 | 1.0978 | 4 | 1 |
| 343 | densenet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 48.33 | 21.72 | 1.8983 | 4 | 1 |
| 344 | densenet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 48.33 | 21.72 | 2.0730 | 4 | 1 |
| 345 | densenet | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 48.33 | 21.72 | 1.7477 | 4 | 1 |
| 346 | densenet | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 48.33 | 21.72 | 1.1786 | 4 | 1 |
| 347 | densenet | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 48.33 | 21.72 | 1.7159 | 4 | 1 |
| 348 | densenet | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 48.33 | 21.72 | 1.3018 | 4 | 1 |
| 349 | densenet | label_smoothing | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0 | 48.33 | 21.72 | 2.2811 | 4 | 1 |
| 350 | densenet | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 48.33 | 21.72 | 0.5140 | 4 | 1 |
| 351 | densenet | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.001 | 48.33 | 21.72 | 0.9713 | 4 | 1 |
| 352 | hrnet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 48.33 | 44.65 | 1.3395 | 4 | 1 |
| 353 | nin | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 48.33 | 21.72 | 0.4900 | 4 | 1 |
| 354 | simple_cnn | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 48.33 | 21.72 | 1.8244 | 7 | 4 |
| 355 | simple_cnn | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 48.33 | 21.72 | 6.3988 | 6 | 3 |
| 356 | convnext | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 48.33 | 22.48 | 2.2860 | 6 | 3 |
| 357 | ghostnet | focal_loss | sgd | batch_size=16, lr=0.0001, weight_decay=0.0 | 48.33 | 21.72 | 0.4995 | 6 | 3 |
| 358 | res2net | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 48.33 | 21.72 | 1.4776 | 4 | 1 |
| 359 | res2net | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 48.33 | 21.72 | 271.7902 | 5 | 2 |
| 360 | res2net | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 48.33 | 21.72 | 1.1250 | 4 | 1 |
| 361 | res2net | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 48.33 | 21.72 | 1.3965 | 6 | 3 |
| 362 | vgg | cross_entropy | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 48.33 | 21.72 | 2.9829 | 5 | 2 |
| 363 | cspnet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 48.33 | 21.72 | 23.4907 | 5 | 2 |
| 364 | cspnet | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 48.33 | 21.72 | 1.1313 | 4 | 1 |
| 365 | gpt | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 48.33 | 40.07 | 1.0991 | 4 | 1 |
| 366 | gpt | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 48.33 | 21.72 | 1.1500 | 6 | 3 |
| 367 | mobilenet | cross_entropy | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 48.33 | 21.72 | 1.0630 | 4 | 1 |
| 368 | mobilenet | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 48.33 | 21.72 | 1.2004 | 6 | 3 |
| 369 | mobilenet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 48.33 | 21.72 | 1.6024 | 5 | 2 |
| 370 | mobilenet | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 48.33 | 21.72 | 1.5076 | 4 | 1 |
| 371 | mobilenet | focal_loss | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 48.33 | 21.72 | 0.5511 | 4 | 1 |
| 372 | mobilenet | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 48.33 | 21.72 | 0.4843 | 4 | 1 |
| 373 | resnext | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 48.33 | 21.72 | 0.5803 | 4 | 1 |
| 374 | cbam_resnet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 48.33 | 21.72 | 2.2941 | 4 | 1 |
| 375 | cbam_resnet | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 48.33 | 21.72 | 1.4172 | 4 | 1 |
| 376 | efficientnet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 48.33 | 21.72 | 1.7820 | 8 | 5 |
| 377 | efficientnet | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 48.33 | 21.72 | 2.6087 | 6 | 3 |
| 378 | efficientnet | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 48.33 | 21.72 | 1.2177 | 5 | 2 |
| 379 | efficientnet | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 48.33 | 21.72 | 19.1350 | 4 | 1 |
| 380 | lenet | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 48.33 | 21.72 | 1.0749 | 4 | 1 |
| 381 | lenet | focal_loss | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 48.33 | 21.72 | 0.4832 | 4 | 1 |
| 382 | repghost | label_smoothing | adamw | batch_size=16, lr=0.0001, weight_decay=0.0 | 48.33 | 21.72 | 1.0843 | 4 | 1 |
| 383 | swin_tiny | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 48.33 | 21.72 | 1.3142 | 5 | 2 |
| 384 | swin_tiny | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 48.33 | 21.72 | 4.0501 | 6 | 3 |
| 385 | swin_tiny | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 48.33 | 21.72 | 1.1361 | 4 | 1 |
| 386 | swin_tiny | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 48.33 | 21.72 | 1.7871 | 4 | 1 |
| 387 | swin_tiny | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 48.33 | 21.72 | 1.4351 | 5 | 2 |
| 388 | swin_tiny | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 48.33 | 21.72 | 1.6610 | 7 | 4 |
| 389 | swin_tiny | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 48.33 | 21.72 | 0.4690 | 4 | 1 |
| 390 | swin_tiny | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 48.33 | 21.72 | 0.6961 | 4 | 1 |
| 391 | coord_resnet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 48.33 | 21.72 | 1.8339 | 4 | 1 |
| 392 | coord_resnet | cross_entropy | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 48.33 | 21.72 | 1.6635 | 4 | 1 |
| 393 | coord_resnet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 48.33 | 33.32 | 0.5028 | 4 | 1 |
| 394 | googlenet | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 48.33 | 21.72 | 1.1309 | 4 | 1 |
| 395 | mnasnet | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 48.33 | 21.72 | 28.1514 | 4 | 1 |
| 396 | mnasnet | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 48.33 | 21.72 | 1.1546 | 4 | 1 |
| 397 | mnasnet | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 48.33 | 21.72 | 6.0976 | 4 | 1 |
| 398 | mnasnet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 48.33 | 21.72 | 20.6019 | 4 | 1 |
| 399 | resnet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 48.33 | 21.72 | 26.0341 | 4 | 1 |
| 400 | resnet | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 48.33 | 21.72 | 1.9648 | 4 | 1 |
| 401 | resnet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 48.33 | 21.72 | 1.0707 | 5 | 2 |
| 402 | resnet | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 48.33 | 21.72 | 0.6355 | 4 | 1 |
| 403 | resnet | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 48.33 | 21.72 | 1.1277 | 5 | 2 |
| 404 | capsnet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 48.33 | 21.72 | 1.0986 | 4 | 1 |
| 405 | capsnet | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 48.33 | 21.72 | 1.0986 | 4 | 1 |
| 406 | capsnet | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 48.33 | 21.72 | 1.0986 | 4 | 1 |
| 407 | capsnet | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 48.33 | 21.72 | 1.0986 | 4 | 1 |
| 408 | capsnet | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 48.33 | 21.72 | 1.0986 | 4 | 1 |
| 409 | capsnet | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 48.33 | 21.72 | 0.4883 | 4 | 1 |
| 410 | capsnet | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 48.33 | 21.72 | 0.4883 | 4 | 1 |
| 411 | eca_resnet | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 48.33 | 21.72 | 1.9803 | 4 | 1 |
| 412 | eca_resnet | label_smoothing | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 48.33 | 21.72 | 1.2272 | 4 | 1 |
| 413 | eca_resnet | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 48.33 | 21.72 | 1.8655 | 4 | 1 |
| 414 | lcnet | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 48.33 | 21.72 | 1.1062 | 4 | 1 |
| 415 | lcnet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 48.33 | 21.72 | 1.1253 | 4 | 1 |
| 416 | lcnet | label_smoothing | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 48.33 | 21.72 | 1.1017 | 4 | 1 |
| 417 | lcnet | focal_loss | adamw | batch_size=16, lr=0.0001, weight_decay=0.0001 | 48.33 | 21.72 | 0.4763 | 4 | 1 |
| 418 | lcnet | focal_loss | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 48.33 | 21.72 | 0.4694 | 4 | 1 |
| 419 | regnet | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 48.33 | 21.72 | 1.6327 | 5 | 2 |
| 420 | regnet | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 48.33 | 21.72 | 0.6461 | 4 | 1 |
| 421 | regnet | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 48.33 | 21.72 | 0.5172 | 4 | 1 |
| 422 | regnet | focal_loss | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.001 | 48.33 | 37.54 | 0.9645 | 8 | 5 |
| 423 | squeezenet | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 48.33 | 21.72 | 0.9444 | 5 | 2 |
| 424 | squeezenet | cross_entropy | adamw | batch_size=16, lr=0.0001, weight_decay=0.0 | 48.33 | 21.72 | 1.0016 | 4 | 1 |
| 425 | squeezenet | label_smoothing | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 48.33 | 21.97 | 0.9623 | 5 | 2 |
| 426 | squeezenet | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 48.33 | 21.72 | 0.4115 | 5 | 2 |
| 427 | coatnet | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 48.33 | 21.72 | 1.3182 | 4 | 1 |
| 428 | efficientnetv2 | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 48.33 | 21.72 | 0.4909 | 4 | 1 |
| 429 | lstm | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 48.33 | 21.72 | 1.0802 | 4 | 1 |
| 430 | lstm | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 48.33 | 21.72 | — | 4 | 1 |
| 431 | lstm | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 48.33 | 21.72 | 0.5160 | 4 | 1 |
| 432 | lstm | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 48.33 | 21.72 | 0.4928 | 4 | 1 |
| 433 | repvgg | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 48.33 | 45.71 | 8.9220 | 9 | 6 |
| 434 | van | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 48.33 | 28.43 | 2.4140 | 5 | 2 |
| 435 | darknet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 48.33 | 21.72 | 3.5979 | 4 | 1 |
| 436 | darknet | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 48.33 | 21.72 | 0.5912 | 5 | 2 |
| 437 | gru | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 48.33 | 21.72 | 1.0977 | 4 | 1 |
| 438 | gru | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 48.33 | 38.99 | 0.4475 | 5 | 2 |
| 439 | mobilenetv2 | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 48.33 | 21.72 | 2.5871 | 4 | 1 |
| 440 | mobilenetv2 | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 48.33 | 21.72 | 1.7540 | 4 | 1 |
| 441 | mobilenetv2 | focal_loss | sgd | batch_size=16, lr=0.0001, weight_decay=0.0001 | 48.33 | 21.72 | 0.4605 | 4 | 1 |
| 442 | mobilenetv2 | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 48.33 | 21.72 | 1.9060 | 5 | 2 |
| 443 | mobilenetv2 | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 48.33 | 21.72 | 1.3294 | 4 | 1 |
| 444 | se_resnet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 48.33 | 22.48 | 3.0141 | 7 | 4 |
| 445 | se_resnet | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 48.33 | 21.72 | 1.8251 | 4 | 1 |
| 446 | se_resnet | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 48.33 | 21.72 | 2.8241 | 4 | 1 |
| 447 | se_resnet | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 48.33 | 21.72 | 5.6183 | 4 | 1 |
| 448 | wide_resnet | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 48.33 | 21.72 | 1.1488 | 4 | 1 |
| 449 | bert | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 48.33 | 38.10 | 1.0824 | 7 | 4 |
| 450 | bert | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 48.33 | 42.25 | 1.0638 | 5 | 2 |
| 451 | dpn | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 48.33 | 21.72 | 1.1449 | 4 | 1 |
| 452 | dpn | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 48.33 | 21.72 | 1.4439 | 4 | 1 |
| 453 | dpn | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 48.33 | 21.72 | 88.7051 | 5 | 2 |
| 454 | inception_resnet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 48.33 | 21.72 | 681508.6641 | 4 | 1 |
| 455 | inception_resnet | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 48.33 | 21.72 | 1.1989e+06 | 4 | 1 |
| 456 | poolformer | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 48.33 | 21.72 | 1.3659 | 5 | 2 |
| 457 | poolformer | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 48.33 | 21.72 | 1.0296 | 6 | 3 |
| 458 | poolformer | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 48.33 | 21.72 | 1.1377 | 5 | 2 |
| 459 | poolformer | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.001 | 48.33 | 21.72 | 1.3590 | 7 | 4 |
| 460 | poolformer | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 48.33 | 21.72 | 1.2120 | 6 | 3 |
| 461 | poolformer | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 48.33 | 21.72 | 145.5784 | 7 | 4 |
| 462 | poolformer | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 48.33 | 21.72 | 20.1588 | 4 | 1 |
| 463 | poolformer | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 48.33 | 21.72 | 0.5520 | 4 | 1 |
| 464 | poolformer | focal_loss | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 48.33 | 21.72 | 0.5722 | 9 | 6 |
| 465 | vgg | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 46.67 | 38.40 | 0.7974 | 4 | 1 |
| 466 | cbam_resnet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 46.67 | 21.46 | 0.4674 | 5 | 2 |
| 467 | capsnet | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 46.67 | 21.21 | 1.0986 | 4 | 1 |
| 468 | capsnet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 46.67 | 21.21 | 0.4883 | 7 | 4 |
| 469 | coatnet | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 46.67 | 35.45 | 1.6886 | 4 | 1 |
| 470 | coatnet | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 46.67 | 24.10 | 13.0283 | 6 | 3 |
| 471 | repvgg | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.0 | 46.67 | 34.91 | 1.4359 | 9 | 6 |
| 472 | inception_resnet | label_smoothing | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 46.67 | 35.45 | 1.4719 | 4 | 1 |
| 473 | hardnet | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 45.00 | 40.23 | 4.0905 | 4 | 1 |
| 474 | hardnet | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.001 | 45.00 | 40.23 | 1.2976 | 7 | 4 |
| 475 | repghost | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 45.00 | 40.23 | 1.4734 | 5 | 2 |
| 476 | resnet | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 45.00 | 20.69 | 1.1148 | 7 | 4 |
| 477 | capsnet | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 45.00 | 20.69 | 0.4883 | 4 | 1 |
| 478 | van | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 45.00 | 32.90 | 1.4965 | 4 | 1 |
| 479 | gru | label_smoothing | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 45.00 | 26.06 | 1.1018 | 4 | 1 |
| 480 | capsnet | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 43.33 | 20.16 | 1.0986 | 4 | 1 |
| 481 | hrnet | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 41.67 | 36.81 | 0.6485 | 5 | 2 |
| 482 | res2net | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.0 | 41.67 | 27.61 | 1.1250 | 4 | 1 |
| 483 | cspnet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 41.67 | 27.61 | 0.5005 | 7 | 4 |
| 484 | hardnet | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 40.00 | 19.05 | 0.7463 | 5 | 2 |
| 485 | hardnet | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 40.00 | 19.05 | 0.9301 | 4 | 1 |
| 486 | mobilenetv3 | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 40.00 | 19.05 | 1.1882 | 4 | 1 |
| 487 | mobilenetv3 | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 40.00 | 19.05 | 1.4258 | 4 | 1 |
| 488 | mobilenetv3 | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.001 | 40.00 | 19.05 | 1.0964 | 4 | 1 |
| 489 | mobilenetv3 | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 40.00 | 19.05 | 1.0956 | 4 | 1 |
| 490 | mobilenetv3 | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 40.00 | 19.05 | 82.8198 | 4 | 1 |
| 491 | mobilenetv3 | focal_loss | sgd | batch_size=16, lr=0.0001, weight_decay=0.0001 | 40.00 | 19.05 | 0.4849 | 4 | 1 |
| 492 | shufflenet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 40.00 | 19.05 | 1.9214 | 4 | 1 |
| 493 | shufflenet | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 40.00 | 19.05 | 1.6939 | 4 | 1 |
| 494 | shufflenet | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 40.00 | 19.05 | 1.1257 | 4 | 1 |
| 495 | shufflenet | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 40.00 | 19.05 | 1.7657 | 5 | 2 |
| 496 | shufflenet | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 40.00 | 19.05 | 1.6009 | 5 | 2 |
| 497 | xception | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 40.00 | 19.05 | 1.1033 | 4 | 1 |
| 498 | xception | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 40.00 | 19.05 | 1.0968 | 4 | 1 |
| 499 | alexnet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 40.00 | 19.05 | 0.4880 | 4 | 1 |
| 500 | nin | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 40.00 | 19.05 | 1.0995 | 4 | 1 |
| 501 | nin | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 40.00 | 19.05 | 1.0923 | 4 | 1 |
| 502 | ghostnet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 40.00 | 19.05 | 1.4337 | 4 | 1 |
| 503 | ghostnet | cross_entropy | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 40.00 | 19.05 | 1.0472 | 4 | 1 |
| 504 | res2net | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 40.00 | 19.05 | 2.1172 | 4 | 1 |
| 505 | res2net | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 40.00 | 19.05 | 1.7792 | 4 | 1 |
| 506 | res2net | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 40.00 | 19.05 | 2.4248 | 5 | 2 |
| 507 | res2net | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 40.00 | 19.05 | 1.2998 | 4 | 1 |
| 508 | cspnet | cross_entropy | adamw | batch_size=16, lr=0.0001, weight_decay=0.0001 | 40.00 | 19.05 | 1.3552 | 4 | 1 |
| 509 | cspnet | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 40.00 | 19.05 | 1.2858 | 4 | 1 |
| 510 | cspnet | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.001 | 40.00 | 19.05 | 0.8253 | 4 | 1 |
| 511 | gpt | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 40.00 | 19.05 | 1.1454 | 4 | 1 |
| 512 | gpt | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 40.00 | 19.05 | 0.5239 | 4 | 1 |
| 513 | gpt | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 40.00 | 19.05 | 0.5306 | 5 | 2 |
| 514 | mobilenet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 40.00 | 19.05 | 1.8519 | 4 | 1 |
| 515 | mobilenet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 40.00 | 19.05 | 1.1972 | 4 | 1 |
| 516 | mobilenet | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 40.00 | 19.05 | 1.1628 | 4 | 1 |
| 517 | mobilenet | label_smoothing | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 40.00 | 19.05 | 1.3487 | 4 | 1 |
| 518 | mobilenet | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 40.00 | 19.05 | 0.5875 | 4 | 1 |
| 519 | mobilenet | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 40.00 | 19.05 | 0.8044 | 4 | 1 |
| 520 | resnext | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 40.00 | 19.05 | 2.0938 | 4 | 1 |
| 521 | efficientnet | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 40.00 | 19.05 | 1.1536 | 5 | 2 |
| 522 | efficientnet | cross_entropy | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 40.00 | 19.05 | 1.8329 | 4 | 1 |
| 523 | efficientnet | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 40.00 | 19.05 | 1.1664 | 4 | 1 |
| 524 | efficientnet | focal_loss | sgd | batch_size=16, lr=0.0001, weight_decay=0.0 | 40.00 | 19.05 | 0.4827 | 4 | 1 |
| 525 | lenet | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 40.00 | 19.05 | 1.1001 | 4 | 1 |
| 526 | googlenet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 40.00 | 19.05 | 1.1345 | 4 | 1 |
| 527 | googlenet | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 40.00 | 19.05 | 1.1035 | 4 | 1 |
| 528 | googlenet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 40.00 | 19.05 | 0.5063 | 4 | 1 |
| 529 | googlenet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 40.00 | 19.05 | 0.4784 | 4 | 1 |
| 530 | googlenet | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 40.00 | 19.05 | 0.5719 | 5 | 2 |
| 531 | mnasnet | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 40.00 | 19.05 | 1.0964 | 4 | 1 |
| 532 | mnasnet | cross_entropy | adamw | batch_size=16, lr=0.0001, weight_decay=0.0001 | 40.00 | 19.05 | 1.7389 | 4 | 1 |
| 533 | mnasnet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 40.00 | 19.05 | 1.7100 | 4 | 1 |
| 534 | mnasnet | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 40.00 | 19.05 | 1.9223 | 4 | 1 |
| 535 | mnasnet | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 40.00 | 19.05 | 1.5605 | 4 | 1 |
| 536 | mnasnet | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.001 | 40.00 | 19.05 | 4.1863 | 4 | 1 |
| 537 | eca_resnet | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 40.00 | 19.05 | 1.5574 | 4 | 1 |
| 538 | lcnet | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 40.00 | 19.05 | 1.2902 | 4 | 1 |
| 539 | lcnet | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 40.00 | 19.05 | 1.6501 | 4 | 1 |
| 540 | lcnet | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 40.00 | 19.05 | 1.2316 | 4 | 1 |
| 541 | lcnet | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 40.00 | 19.05 | 0.5235 | 4 | 1 |
| 542 | lcnet | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 40.00 | 19.05 | 0.5374 | 4 | 1 |
| 543 | regnet | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 40.00 | 19.05 | 1.2554 | 4 | 1 |
| 544 | regnet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 40.00 | 19.05 | 4.0630 | 4 | 1 |
| 545 | regnet | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 40.00 | 19.05 | 1.3797 | 4 | 1 |
| 546 | regnet | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 40.00 | 19.05 | 2.1097 | 5 | 2 |
| 547 | regnet | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 40.00 | 19.05 | 1.2350 | 4 | 1 |
| 548 | regnet | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 40.00 | 19.05 | 0.6084 | 4 | 1 |
| 549 | coatnet | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 40.00 | 19.05 | 1.1748 | 4 | 1 |
| 550 | efficientnetv2 | cross_entropy | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 40.00 | 19.05 | 1.2897 | 4 | 1 |
| 551 | efficientnetv2 | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 40.00 | 19.05 | 1.7991 | 4 | 1 |
| 552 | efficientnetv2 | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0 | 40.00 | 19.05 | 1.9482 | 5 | 2 |
| 553 | efficientnetv2 | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 40.00 | 19.05 | 1.8507 | 4 | 1 |
| 554 | efficientnetv2 | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 40.00 | 19.05 | 1.2679 | 4 | 1 |
| 555 | efficientnetv2 | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 40.00 | 19.05 | 1.2236 | 4 | 1 |
| 556 | efficientnetv2 | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 40.00 | 19.05 | 1.5053 | 4 | 1 |
| 557 | efficientnetv2 | focal_loss | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 40.00 | 19.05 | 0.5522 | 4 | 1 |
| 558 | lstm | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 40.00 | 19.05 | 1.1053 | 4 | 1 |
| 559 | lstm | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 40.00 | 19.05 | 1.1039 | 4 | 1 |
| 560 | repvgg | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 40.00 | 19.05 | 2.1894 | 4 | 1 |
| 561 | darknet | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 40.00 | 19.05 | 2.2978 | 4 | 1 |
| 562 | darknet | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 40.00 | 19.05 | 1.7611 | 4 | 1 |
| 563 | darknet | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 40.00 | 19.05 | 1.1360 | 4 | 1 |
| 564 | gru | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 40.00 | 19.05 | 0.4833 | 4 | 1 |
| 565 | mobilenetv2 | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 40.00 | 19.05 | 1.7864 | 4 | 1 |
| 566 | mobilenetv2 | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 40.00 | 19.05 | 2.9897 | 4 | 1 |
| 567 | mobilenetv2 | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 40.00 | 19.05 | 3.2564 | 4 | 1 |
| 568 | mobilenetv2 | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.0 | 40.00 | 19.05 | 1.0983 | 4 | 1 |
| 569 | se_resnet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 40.00 | 19.05 | 1.2130 | 4 | 1 |
| 570 | se_resnet | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 40.00 | 19.05 | 0.7063 | 4 | 1 |
| 571 | bert | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 40.00 | 19.05 | 1.1327 | 4 | 1 |
| 572 | inception_resnet | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 40.00 | 19.05 | 1.2782 | 4 | 1 |
| 573 | inception_resnet | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 40.00 | 19.05 | 1.8390 | 4 | 1 |
| 574 | inception_resnet | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 40.00 | 19.05 | 1.5198 | 4 | 1 |
| 575 | inception_resnet | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 40.00 | 19.05 | 1.0969 | 4 | 1 |
| 576 | inception_resnet | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 40.00 | 19.05 | 1.1307 | 5 | 2 |
| 577 | poolformer | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 40.00 | 19.05 | 0.5241 | 4 | 1 |
| 578 | sknet | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 40.00 | 19.05 | 1.7045 | 5 | 2 |
| 579 | sknet | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 40.00 | 19.05 | 2.0721 | 4 | 1 |
| 580 | sknet | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 40.00 | 19.05 | 2.4821 | 4 | 1 |
| 581 | sknet | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 40.00 | 19.05 | 2.3687 | 5 | 2 |
| 582 | sknet | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 40.00 | 19.05 | 0.8252 | 4 | 1 |
| 583 | capsnet | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 36.67 | 17.89 | 1.0986 | 6 | 3 |
| 584 | resnext | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 31.67 | 26.02 | 2.7293 | 7 | 4 |
| 585 | repghost | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 31.67 | 30.71 | 0.7351 | 4 | 1 |
| 586 | coord_resnet | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 13.33 | 9.29 | 1.5566 | 4 | 1 |
| 587 | shufflenet | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 11.67 | 6.97 | 1.1103 | 4 | 1 |
| 588 | alexnet | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 11.67 | 6.97 | 1.1039 | 4 | 1 |
| 589 | hrnet | cross_entropy | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 11.67 | 6.97 | 1.1020 | 4 | 1 |
| 590 | nin | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 11.67 | 6.97 | 1.1200 | 4 | 1 |
| 591 | efficientnet | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 11.67 | 7.41 | 1.1091 | 4 | 1 |
| 592 | resnet | cross_entropy | sgd | batch_size=16, lr=0.0001, weight_decay=0.0001 | 11.67 | 9.72 | 1.0999 | 4 | 1 |
| 593 | resnet | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 11.67 | 6.97 | 2.1707 | 4 | 1 |
| 594 | eca_resnet | cross_entropy | sgd | batch_size=16, lr=0.0001, weight_decay=0.0 | 11.67 | 6.97 | 1.1603 | 4 | 1 |
| 595 | lcnet | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 11.67 | 6.97 | 1.1100 | 4 | 1 |
| 596 | regnet | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 11.67 | 6.97 | 1.7108 | 4 | 1 |
| 597 | lstm | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 11.67 | 6.97 | 1.1239 | 4 | 1 |
| 598 | gru | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 11.67 | 6.97 | 1.1422 | 4 | 1 |
| 599 | gru | label_smoothing | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 11.67 | 6.97 | 1.1184 | 4 | 1 |
| 600 | wide_resnet | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 11.67 | 6.97 | 2.0299 | 4 | 1 |

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
| 1 | dcgan | wasserstein | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 0.0000 | 100.0000 | 5 | 2 |
| 2 | vanilla_gan | wasserstein | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 100.0000 | 100.0000 | 5 | 2 |
| 3 | wgan | bce | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 0.0521 | -0.0571 | 4 | 1 |
| 4 | cgan | wasserstein | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 1.2878 | 1.9584 | 8 | 5 |

## GAN — all trials (28 rows)

| Rank | Model | Loss | Optimizer | Hyperparameters | G Loss | D Loss | Epochs Run | Convergence Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | dcgan | wasserstein | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 0.0000 | 100.0000 | 5 | 2 |
| 2 | vanilla_gan | wasserstein | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 100.0000 | 100.0000 | 5 | 2 |
| 3 | wgan | bce | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 0.0521 | -0.0571 | 4 | 1 |
| 4 | wgan | bce | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 0.1090 | -0.1187 | 4 | 1 |
| 5 | wgan | wasserstein | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 0.1535 | -0.1715 | 4 | 1 |
| 6 | wgan | wasserstein | adam | batch_size=16, lr=0.001, weight_decay=0.001 | 0.2442 | -0.2248 | 4 | 1 |
| 7 | cgan | wasserstein | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 1.2878 | 1.9584 | 8 | 5 |
| 8 | cgan | bce | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 0.7930 | 1.3642 | 5 | 2 |
| 9 | vanilla_gan | bce | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 0.7050 | 1.1225 | 4 | 1 |
| 10 | cgan | wasserstein | adamw | batch_size=16, lr=0.0001, weight_decay=0.001 | 0.9242 | 1.0407 | 4 | 1 |
| 11 | cgan | wasserstein | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 0.7169 | 1.3306 | 4 | 1 |
| 12 | cgan | bce | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 0.8593 | 1.1349 | 4 | 1 |
| 13 | vanilla_gan | wasserstein | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 2.9730 | 0.0997 | 4 | 1 |
| 14 | cgan | bce | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 0.8317 | 1.2372 | 7 | 4 |
| 15 | cgan | wasserstein | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 0.8623 | 1.2320 | 5 | 2 |
| 16 | vanilla_gan | bce | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 2.2218 | 0.1607 | 4 | 1 |
| 17 | cgan | bce | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 0.8983 | 1.2510 | 8 | 5 |
| 18 | dcgan | wasserstein | sgd | batch_size=16, lr=0.0001, weight_decay=0.0 | 0.9580 | 1.1872 | 4 | 1 |
| 19 | dcgan | bce | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 2.0286 | 0.6018 | 4 | 1 |
| 20 | vanilla_gan | wasserstein | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 5.1203 | 6.0930 | 4 | 1 |
| 21 | dcgan | wasserstein | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 3.1202 | 0.3878 | 4 | 1 |
| 22 | vanilla_gan | bce | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 1.5176 | 0.6085 | 4 | 1 |
| 23 | vanilla_gan | wasserstein | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 1.4803 | 0.7041 | 4 | 1 |
| 24 | dcgan | bce | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 5.9074 | 0.0041 | 5 | 2 |
| 25 | dcgan | wasserstein | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 6.5217 | 0.0052 | 5 | 2 |
| 26 | dcgan | bce | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 7.6023 | 0.0440 | 4 | 1 |
| 27 | dcgan | bce | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 26.8365 | 6.2339e-05 | 4 | 1 |
| 28 | vanilla_gan | bce | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 100.0000 | 0.0000 | 4 | 1 |

## Search Space

- Loss (classification): cross_entropy, label_smoothing, focal_loss
- Loss (autoencoder): mse, l1, bce
- Loss (GAN): bce, wasserstein (informational; GANs use fixed objectives)
- Optimizers: adam, sgd, adamw, rmsprop
- Hyperparameters: {"lr": [0.0001, 0.001, 0.01], "batch_size": [8, 16, 32], "weight_decay": [0.0, 0.0001, 0.001]}
