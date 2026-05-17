# Strawberry Regression Report

## Run configuration

| Setting | Value |
| --- | --- |
| Generated | 2026-05-17T19:25:08 |
| Dataset | strawberry |
| Classes | 6 |
| Class names | early-turning, green, late-turning, red, turning, white |
| Mode | quick-test |
| Max epochs | 1 |
| Early-stop patience | 3 |
| Min delta | 0.1 |
| NAS trials per config | 2 |
| Workers | 8 |
| Max batch size | 16 |
| Total wall time | 5943.2s |

Training stops when validation metric shows no significant improvement for 3 consecutive epochs.

## Classification — best per model

| Rank | Model | Loss | Optimizer | Hyperparameters | Test Acc (%) | Test Loss | Epochs Run | Convergence Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | convnext | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 38.18 | 2.7681 | 1 | 1 |
| 2 | mlp | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 38.18 | 1.6961 | 1 | 1 |
| 3 | repvgg | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 36.36 | 1.9173 | 1 | 1 |
| 4 | cspnet | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 1.7715 | 1 | 1 |
| 5 | ghostnet | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 1.7641 | 1 | 1 |
| 6 | lenet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 1.5204 | 1 | 1 |
| 7 | poolformer | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 15.4148 | 1 | 1 |
| 8 | shufflenet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 1.6672 | 1 | 1 |
| 9 | vit | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 1.7498 | 1 | 1 |
| 10 | coord_resnet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 1.7479 | 1 | 1 |
| 11 | efficientnetv2 | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 1.7786 | 1 | 1 |
| 12 | lcnet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 1.6889 | 1 | 1 |
| 13 | nin | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 1.5612 | 1 | 1 |
| 14 | se_resnet | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 34.55 | 1.7633 | 1 | 1 |
| 15 | vim_tiny | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 34.55 | 478.8486 | 1 | 1 |
| 16 | efficientnet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 1.9425 | 1 | 1 |
| 17 | inception_resnet | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 34.55 | 1.7399 | 1 | 1 |
| 18 | mobilenetv3 | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 1.8254 | 1 | 1 |
| 19 | resnext | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 1.7848 | 1 | 1 |
| 20 | vgg | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 1.5117 | 1 | 1 |
| 21 | coatnet | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 34.55 | 1.8939 | 1 | 1 |
| 22 | eca_resnet | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0 | 34.55 | 1667.0861 | 1 | 1 |
| 23 | hrnet | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.7535 | 1 | 1 |
| 24 | mobilenetv2 | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 4.1867 | 1 | 1 |
| 25 | resnet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 1.6638 | 1 | 1 |
| 26 | van | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 1.7803 | 1 | 1 |
| 27 | cbam_resnet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 1579.0772 | 1 | 1 |
| 28 | dpn | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 2.4536 | 1 | 1 |
| 29 | hardnet | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 34.55 | 1.7722 | 1 | 1 |
| 30 | mobilenet | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.001 | 34.55 | 1.7645 | 1 | 1 |
| 31 | res2net | cross_entropy | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 34.55 | 1.7463 | 1 | 1 |
| 32 | swin_tiny | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 34.55 | 1.5341 | 1 | 1 |
| 33 | alexnet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 13.3288 | 1 | 1 |
| 34 | darknet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 3.1016e+06 | 1 | 1 |
| 35 | googlenet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 34.55 | 1.7174 | 1 | 1 |
| 36 | lstm | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 1.5487 | 1 | 1 |
| 37 | regnet | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 34.55 | 1.6882 | 1 | 1 |
| 38 | simple_cnn | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.6773 | 1 | 1 |
| 39 | wide_resnet | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 34.55 | 6.9245 | 1 | 1 |
| 40 | densenet | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 34.55 | 1.7093 | 1 | 1 |
| 41 | gru | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 34.55 | 1.5497 | 1 | 1 |
| 42 | mnasnet | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 34.55 | 1.7766 | 1 | 1 |
| 43 | squeezenet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 34.55 | 1.7219 | 1 | 1 |
| 44 | bert | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 1.7565 | 1 | 1 |
| 45 | deit | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 1.9707 | 1 | 1 |
| 46 | gpt | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 1.6048 | 1 | 1 |
| 47 | repghost | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 34.55 | 1.7269 | 1 | 1 |
| 48 | sknet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 799.2956 | 1 | 1 |
| 49 | xception | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 1.7785 | 1 | 1 |

## Classification — all trials (588 rows)

| Rank | Model | Loss | Optimizer | Hyperparameters | Test Acc (%) | Test Loss | Epochs Run | Convergence Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | convnext | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 38.18 | 2.7681 | 1 | 1 |
| 2 | mlp | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 38.18 | 1.6961 | 1 | 1 |
| 3 | repvgg | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 36.36 | 1.9173 | 1 | 1 |
| 4 | cspnet | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 1.7715 | 1 | 1 |
| 5 | cspnet | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 1.1985 | 1 | 1 |
| 6 | ghostnet | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 1.7641 | 1 | 1 |
| 7 | ghostnet | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 34.55 | 1.7540 | 1 | 1 |
| 8 | ghostnet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 50.9159 | 1 | 1 |
| 9 | ghostnet | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 1.1435 | 1 | 1 |
| 10 | ghostnet | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 1.2117 | 1 | 1 |
| 11 | lenet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 1.5204 | 1 | 1 |
| 12 | lenet | cross_entropy | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 1.6290 | 1 | 1 |
| 13 | lenet | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 34.55 | 1.5071 | 1 | 1 |
| 14 | lenet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 1.6817 | 1 | 1 |
| 15 | lenet | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 1.6629 | 1 | 1 |
| 16 | lenet | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 34.55 | 1.1864 | 1 | 1 |
| 17 | lenet | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 1.0574 | 1 | 1 |
| 18 | poolformer | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 15.4148 | 1 | 1 |
| 19 | poolformer | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 34.55 | 4.0157 | 1 | 1 |
| 20 | poolformer | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 33.8823 | 1 | 1 |
| 21 | shufflenet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 1.6672 | 1 | 1 |
| 22 | shufflenet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 1.7182 | 1 | 1 |
| 23 | shufflenet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 1.7835 | 1 | 1 |
| 24 | shufflenet | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.7775 | 1 | 1 |
| 25 | shufflenet | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 1.7438 | 1 | 1 |
| 26 | shufflenet | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.001 | 34.55 | 1.7606 | 1 | 1 |
| 27 | shufflenet | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 1.7059 | 1 | 1 |
| 28 | shufflenet | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.001 | 34.55 | 1.2052 | 1 | 1 |
| 29 | shufflenet | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 1.2005 | 1 | 1 |
| 30 | shufflenet | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.2164 | 1 | 1 |
| 31 | vit | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 1.7498 | 1 | 1 |
| 32 | vit | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.4925 | 1 | 1 |
| 33 | vit | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.5030 | 1 | 1 |
| 34 | vit | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 1.7512 | 1 | 1 |
| 35 | vit | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 1.5668 | 1 | 1 |
| 36 | vit | label_smoothing | adamw | batch_size=16, lr=0.0001, weight_decay=0.001 | 34.55 | 1.5526 | 1 | 1 |
| 37 | vit | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 4.9045 | 1 | 1 |
| 38 | vit | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 34.55 | 0.9507 | 1 | 1 |
| 39 | vit | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 34.55 | 1.0500 | 1 | 1 |
| 40 | vit | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 34.55 | 1.3989 | 1 | 1 |
| 41 | vit | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 15.1434 | 1 | 1 |
| 42 | coord_resnet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 1.7479 | 1 | 1 |
| 43 | coord_resnet | cross_entropy | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 35132.4663 | 1 | 1 |
| 44 | coord_resnet | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 52702.4199 | 1 | 1 |
| 45 | coord_resnet | focal_loss | adam | batch_size=16, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.2145 | 1 | 1 |
| 46 | coord_resnet | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 1.1639 | 1 | 1 |
| 47 | coord_resnet | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 174341.3247 | 1 | 1 |
| 48 | coord_resnet | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 885.7380 | 1 | 1 |
| 49 | efficientnetv2 | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 1.7786 | 1 | 1 |
| 50 | efficientnetv2 | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.8065 | 1 | 1 |
| 51 | efficientnetv2 | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 2.0474 | 1 | 1 |
| 52 | efficientnetv2 | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 1.7192 | 1 | 1 |
| 53 | efficientnetv2 | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 1.2421 | 1 | 1 |
| 54 | lcnet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 1.6889 | 1 | 1 |
| 55 | lcnet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 1.7524 | 1 | 1 |
| 56 | lcnet | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.7908 | 1 | 1 |
| 57 | lcnet | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 1.7731 | 1 | 1 |
| 58 | lcnet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 2.4653 | 1 | 1 |
| 59 | lcnet | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.001 | 34.55 | 1.2499 | 1 | 1 |
| 60 | nin | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 1.5612 | 1 | 1 |
| 61 | nin | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 1.7866 | 1 | 1 |
| 62 | nin | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 34.55 | 1.7699 | 1 | 1 |
| 63 | nin | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 34.55 | 1.7439 | 1 | 1 |
| 64 | nin | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 1.7848 | 1 | 1 |
| 65 | nin | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 34.55 | 1.7918 | 1 | 1 |
| 66 | nin | label_smoothing | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 34.55 | 1.5759 | 1 | 1 |
| 67 | nin | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 34.55 | 1.0245 | 1 | 1 |
| 68 | nin | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 34.55 | 1.2178 | 1 | 1 |
| 69 | nin | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 1.0522 | 1 | 1 |
| 70 | nin | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 34.55 | 1.2185 | 1 | 1 |
| 71 | se_resnet | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 34.55 | 1.7633 | 1 | 1 |
| 72 | se_resnet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 7250.4355 | 1 | 1 |
| 73 | se_resnet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 34.55 | 8004.6237 | 1 | 1 |
| 74 | se_resnet | focal_loss | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 34.55 | 1.3581 | 1 | 1 |
| 75 | vim_tiny | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 34.55 | 478.8486 | 1 | 1 |
| 76 | vim_tiny | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 96471.6551 | 1 | 1 |
| 77 | convnext | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 2.6345 | 1 | 1 |
| 78 | convnext | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.6716 | 1 | 1 |
| 79 | convnext | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 34.55 | 1.6827 | 1 | 1 |
| 80 | convnext | label_smoothing | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 34.55 | 1.7554 | 1 | 1 |
| 81 | convnext | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 3.3659 | 1 | 1 |
| 82 | convnext | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 34.55 | 1.1752 | 1 | 1 |
| 83 | convnext | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 2.5052 | 1 | 1 |
| 84 | convnext | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.7555 | 1 | 1 |
| 85 | efficientnet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 1.9425 | 1 | 1 |
| 86 | efficientnet | cross_entropy | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 34.55 | 1.7954 | 1 | 1 |
| 87 | efficientnet | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 3.2766e+09 | 1 | 1 |
| 88 | efficientnet | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 5471.6027 | 1 | 1 |
| 89 | inception_resnet | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 34.55 | 1.7399 | 1 | 1 |
| 90 | inception_resnet | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 1.6047 | 1 | 1 |
| 91 | inception_resnet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 4.7721e+07 | 1 | 1 |
| 92 | inception_resnet | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 34.55 | 1.7770 | 1 | 1 |
| 93 | inception_resnet | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 1.7400 | 1 | 1 |
| 94 | inception_resnet | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 1.2177 | 1 | 1 |
| 95 | inception_resnet | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 34.55 | 4.0818e+06 | 1 | 1 |
| 96 | mobilenetv3 | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 1.8254 | 1 | 1 |
| 97 | mobilenetv3 | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 1.7957 | 1 | 1 |
| 98 | mobilenetv3 | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 34.55 | 1.7860 | 1 | 1 |
| 99 | mobilenetv3 | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 34.55 | 1.6973 | 1 | 1 |
| 100 | mobilenetv3 | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 2.0458 | 1 | 1 |
| 101 | mobilenetv3 | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 1.7684 | 1 | 1 |
| 102 | mobilenetv3 | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 34.55 | 1.7893 | 1 | 1 |
| 103 | mobilenetv3 | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 3.5548e+07 | 1 | 1 |
| 104 | mobilenetv3 | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 1.2468 | 1 | 1 |
| 105 | resnext | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 1.7848 | 1 | 1 |
| 106 | resnext | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0 | 34.55 | 1.7576 | 1 | 1 |
| 107 | resnext | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 34.55 | 1.7369 | 1 | 1 |
| 108 | resnext | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.2334 | 1 | 1 |
| 109 | resnext | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 1.1369 | 1 | 1 |
| 110 | resnext | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 465.6800 | 1 | 1 |
| 111 | vgg | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 1.5117 | 1 | 1 |
| 112 | vgg | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 34.55 | 1.7939 | 1 | 1 |
| 113 | vgg | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 4290.8370 | 1 | 1 |
| 114 | vgg | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 1.4610e+15 | 1 | 1 |
| 115 | vgg | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.001 | 34.55 | 1.7930 | 1 | 1 |
| 116 | vgg | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 1.5706 | 1 | 1 |
| 117 | vgg | label_smoothing | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 34.55 | 20121.3672 | 1 | 1 |
| 118 | vgg | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 34.55 | 0.9340 | 1 | 1 |
| 119 | vgg | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 17441.1636 | 1 | 1 |
| 120 | coatnet | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 34.55 | 1.8939 | 1 | 1 |
| 121 | coatnet | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 1.7818 | 1 | 1 |
| 122 | eca_resnet | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0 | 34.55 | 1667.0861 | 1 | 1 |
| 123 | eca_resnet | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 34.55 | 1.8328 | 1 | 1 |
| 124 | eca_resnet | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 2.5001 | 1 | 1 |
| 125 | eca_resnet | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 1.1562 | 1 | 1 |
| 126 | eca_resnet | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 3517.9250 | 1 | 1 |
| 127 | hrnet | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.7535 | 1 | 1 |
| 128 | hrnet | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 34.55 | 3.8226 | 1 | 1 |
| 129 | hrnet | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 1.6667 | 1 | 1 |
| 130 | hrnet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 1.5480 | 1 | 1 |
| 131 | hrnet | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 1.1811 | 1 | 1 |
| 132 | hrnet | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 5.2837 | 1 | 1 |
| 133 | mobilenetv2 | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 4.1867 | 1 | 1 |
| 134 | mobilenetv2 | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 1.2282 | 1 | 1 |
| 135 | mobilenetv2 | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 34.55 | 1.8225 | 1 | 1 |
| 136 | resnet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 1.6638 | 1 | 1 |
| 137 | resnet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 83.1461 | 1 | 1 |
| 138 | resnet | label_smoothing | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 34.55 | 1.7601 | 1 | 1 |
| 139 | resnet | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 1.1182 | 1 | 1 |
| 140 | resnet | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 617.3548 | 1 | 1 |
| 141 | van | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 1.7803 | 1 | 1 |
| 142 | van | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 1.7636 | 1 | 1 |
| 143 | van | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 1.9372 | 1 | 1 |
| 144 | van | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 1.6543 | 1 | 1 |
| 145 | van | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 1.6266 | 1 | 1 |
| 146 | van | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 34.55 | 1.6310 | 1 | 1 |
| 147 | van | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 1.2314 | 1 | 1 |
| 148 | van | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 1.2043 | 1 | 1 |
| 149 | van | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 1.3387 | 1 | 1 |
| 150 | cbam_resnet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 1579.0772 | 1 | 1 |
| 151 | cbam_resnet | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 34.55 | 1.7634 | 1 | 1 |
| 152 | cbam_resnet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 91.6260 | 1 | 1 |
| 153 | cbam_resnet | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 1.6741 | 1 | 1 |
| 154 | cbam_resnet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.2217 | 1 | 1 |
| 155 | cbam_resnet | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 1.1596 | 1 | 1 |
| 156 | dpn | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 2.4536 | 1 | 1 |
| 157 | dpn | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 1.7752 | 1 | 1 |
| 158 | dpn | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 34.55 | 1.6878 | 1 | 1 |
| 159 | dpn | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 1.0744 | 1 | 1 |
| 160 | dpn | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 34.55 | 1.2204 | 1 | 1 |
| 161 | dpn | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 1.0903e+07 | 1 | 1 |
| 162 | hardnet | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 34.55 | 1.7722 | 1 | 1 |
| 163 | hardnet | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 34.55 | 138031.7271 | 1 | 1 |
| 164 | hardnet | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 35.6249 | 1 | 1 |
| 165 | hardnet | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 1.7220 | 1 | 1 |
| 166 | hardnet | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 34.55 | 1.7786 | 1 | 1 |
| 167 | hardnet | focal_loss | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 34.55 | 1.2082 | 1 | 1 |
| 168 | mobilenet | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.001 | 34.55 | 1.7645 | 1 | 1 |
| 169 | mobilenet | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 34.55 | 1.7392 | 1 | 1 |
| 170 | mobilenet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 34.55 | 1.7190 | 1 | 1 |
| 171 | mobilenet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 34.55 | 1.6725 | 1 | 1 |
| 172 | mobilenet | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.7710 | 1 | 1 |
| 173 | mobilenet | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.7764 | 1 | 1 |
| 174 | mobilenet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 14733.7443 | 1 | 1 |
| 175 | mobilenet | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 34.55 | 1.0394 | 1 | 1 |
| 176 | mobilenet | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 34.55 | 1.1546 | 1 | 1 |
| 177 | res2net | cross_entropy | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 34.55 | 1.7463 | 1 | 1 |
| 178 | res2net | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.8338 | 1 | 1 |
| 179 | res2net | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 1.7002 | 1 | 1 |
| 180 | res2net | label_smoothing | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 34.55 | 1.7618 | 1 | 1 |
| 181 | res2net | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 231993.7188 | 1 | 1 |
| 182 | swin_tiny | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 34.55 | 1.5341 | 1 | 1 |
| 183 | swin_tiny | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 34.55 | 1.9787 | 1 | 1 |
| 184 | swin_tiny | focal_loss | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 34.55 | 1.0472 | 1 | 1 |
| 185 | swin_tiny | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 7.1832 | 1 | 1 |
| 186 | alexnet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 13.3288 | 1 | 1 |
| 187 | alexnet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 34.55 | 1.6049 | 1 | 1 |
| 188 | alexnet | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 34.55 | 1.9680 | 1 | 1 |
| 189 | alexnet | label_smoothing | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 34.55 | 1.7500 | 1 | 1 |
| 190 | alexnet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 34.55 | 1.2410 | 1 | 1 |
| 191 | alexnet | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 34.55 | 1.0104 | 1 | 1 |
| 192 | alexnet | focal_loss | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0 | 34.55 | 9.2481 | 1 | 1 |
| 193 | darknet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 3.1016e+06 | 1 | 1 |
| 194 | darknet | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 1507.3325 | 1 | 1 |
| 195 | darknet | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.8433 | 1 | 1 |
| 196 | darknet | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 1.5852 | 1 | 1 |
| 197 | darknet | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 7.0678e+09 | 1 | 1 |
| 198 | darknet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 1.8589e+10 | 1 | 1 |
| 199 | darknet | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 182.3828 | 1 | 1 |
| 200 | googlenet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 34.55 | 1.7174 | 1 | 1 |
| 201 | googlenet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 1.7027 | 1 | 1 |
| 202 | googlenet | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0 | 34.55 | 1.7204 | 1 | 1 |
| 203 | googlenet | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 34.55 | 1.7846 | 1 | 1 |
| 204 | googlenet | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 1.5996 | 1 | 1 |
| 205 | googlenet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 30.2755 | 1 | 1 |
| 206 | googlenet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 0.9433 | 1 | 1 |
| 207 | googlenet | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 1.1536 | 1 | 1 |
| 208 | googlenet | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 34.55 | 1.1915 | 1 | 1 |
| 209 | lstm | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 1.5487 | 1 | 1 |
| 210 | lstm | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0 | 34.55 | 1.5698 | 1 | 1 |
| 211 | lstm | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 34.55 | 1.7810 | 1 | 1 |
| 212 | lstm | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 34.55 | 1.7750 | 1 | 1 |
| 213 | lstm | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 34.55 | 1.6924 | 1 | 1 |
| 214 | lstm | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 1.1646 | 1 | 1 |
| 215 | lstm | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 34.55 | 1.2031 | 1 | 1 |
| 216 | lstm | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 1.1582 | 1 | 1 |
| 217 | lstm | focal_loss | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 34.55 | 1.0589 | 1 | 1 |
| 218 | regnet | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 34.55 | 1.6882 | 1 | 1 |
| 219 | regnet | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 1.6908 | 1 | 1 |
| 220 | regnet | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 34.55 | 1.6849 | 1 | 1 |
| 221 | regnet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 1.7950 | 1 | 1 |
| 222 | regnet | label_smoothing | adamw | batch_size=16, lr=0.0001, weight_decay=0.0 | 34.55 | 1.7570 | 1 | 1 |
| 223 | regnet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 2.4991e+06 | 1 | 1 |
| 224 | simple_cnn | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.6773 | 1 | 1 |
| 225 | simple_cnn | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 1.7387 | 1 | 1 |
| 226 | simple_cnn | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0 | 34.55 | 6.8467 | 1 | 1 |
| 227 | simple_cnn | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.7052 | 1 | 1 |
| 228 | simple_cnn | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 1.7772 | 1 | 1 |
| 229 | simple_cnn | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 34.55 | 2.7774 | 1 | 1 |
| 230 | simple_cnn | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 3.9773 | 1 | 1 |
| 231 | simple_cnn | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 34.55 | 1.2814 | 1 | 1 |
| 232 | wide_resnet | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 34.55 | 6.9245 | 1 | 1 |
| 233 | wide_resnet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 34.55 | 156.5378 | 1 | 1 |
| 234 | densenet | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 34.55 | 1.7093 | 1 | 1 |
| 235 | densenet | cross_entropy | sgd | batch_size=16, lr=0.0001, weight_decay=0.0 | 34.55 | 1.7550 | 1 | 1 |
| 236 | densenet | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 34.55 | 1.5980 | 1 | 1 |
| 237 | densenet | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 1.2514e+06 | 1 | 1 |
| 238 | densenet | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 1.7018 | 1 | 1 |
| 239 | densenet | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 34.55 | 9.0334 | 1 | 1 |
| 240 | densenet | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 1.7172 | 1 | 1 |
| 241 | densenet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 34.55 | 1.2585 | 1 | 1 |
| 242 | densenet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 34.55 | 1.2480 | 1 | 1 |
| 243 | densenet | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 42.8562 | 1 | 1 |
| 244 | densenet | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 34.55 | 1.1612 | 1 | 1 |
| 245 | gru | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 34.55 | 1.5497 | 1 | 1 |
| 246 | gru | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 34.55 | 1.7884 | 1 | 1 |
| 247 | gru | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 1.6350 | 1 | 1 |
| 248 | gru | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.7947 | 1 | 1 |
| 249 | gru | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 1.6985 | 1 | 1 |
| 250 | gru | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 1.6389 | 1 | 1 |
| 251 | gru | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 1.2092 | 1 | 1 |
| 252 | gru | focal_loss | adamw | batch_size=16, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.2284 | 1 | 1 |
| 253 | gru | focal_loss | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 34.55 | 1.1908 | 1 | 1 |
| 254 | mnasnet | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 34.55 | 1.7766 | 1 | 1 |
| 255 | mnasnet | cross_entropy | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 34.55 | 1.7968 | 1 | 1 |
| 256 | mnasnet | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 34.55 | 1.5955 | 1 | 1 |
| 257 | mnasnet | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 1.9856 | 1 | 1 |
| 258 | mnasnet | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 1.7519 | 1 | 1 |
| 259 | mnasnet | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 2.2737 | 1 | 1 |
| 260 | mnasnet | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 1.1964 | 1 | 1 |
| 261 | repvgg | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 1.7561 | 1 | 1 |
| 262 | repvgg | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 1.6180 | 1 | 1 |
| 263 | repvgg | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 34.55 | 1.9474 | 1 | 1 |
| 264 | repvgg | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 91312.7456 | 1 | 1 |
| 265 | repvgg | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 34.55 | 1.2435 | 1 | 1 |
| 266 | repvgg | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 2.3863 | 1 | 1 |
| 267 | squeezenet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 34.55 | 1.7219 | 1 | 1 |
| 268 | squeezenet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 1.6769 | 1 | 1 |
| 269 | squeezenet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.6022 | 1 | 1 |
| 270 | squeezenet | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 34.55 | 1.5197 | 1 | 1 |
| 271 | squeezenet | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 2.3942 | 1 | 1 |
| 272 | squeezenet | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.6661 | 1 | 1 |
| 273 | squeezenet | label_smoothing | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 34.55 | 1.7738 | 1 | 1 |
| 274 | squeezenet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 1.1951 | 1 | 1 |
| 275 | squeezenet | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 1.0832 | 1 | 1 |
| 276 | squeezenet | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 1.1423 | 1 | 1 |
| 277 | squeezenet | focal_loss | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 34.55 | 1.2096 | 1 | 1 |
| 278 | bert | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 1.7565 | 1 | 1 |
| 279 | bert | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 1.7587 | 1 | 1 |
| 280 | bert | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 1.7643 | 1 | 1 |
| 281 | bert | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 34.55 | 1.7719 | 1 | 1 |
| 282 | bert | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 34.55 | 1.7616 | 1 | 1 |
| 283 | bert | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.7851 | 1 | 1 |
| 284 | bert | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.7684 | 1 | 1 |
| 285 | bert | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 34.55 | 1.2322 | 1 | 1 |
| 286 | bert | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.2366 | 1 | 1 |
| 287 | bert | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 1.0795 | 1 | 1 |
| 288 | deit | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 1.9707 | 1 | 1 |
| 289 | deit | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 34.55 | 1.7221 | 1 | 1 |
| 290 | deit | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 2.0422 | 1 | 1 |
| 291 | deit | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 2.5058 | 1 | 1 |
| 292 | deit | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 34.55 | 1.5363 | 1 | 1 |
| 293 | deit | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 1.8795 | 1 | 1 |
| 294 | deit | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 9.9130 | 1 | 1 |
| 295 | deit | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 34.55 | 1.0299 | 1 | 1 |
| 296 | deit | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 0.9699 | 1 | 1 |
| 297 | deit | focal_loss | adamw | batch_size=16, lr=0.0001, weight_decay=0.0001 | 34.55 | 0.9253 | 1 | 1 |
| 298 | gpt | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 1.6048 | 1 | 1 |
| 299 | gpt | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 1.7827 | 1 | 1 |
| 300 | gpt | cross_entropy | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 1.6222 | 1 | 1 |
| 301 | gpt | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 1.4557 | 1 | 1 |
| 302 | gpt | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 34.55 | 1.7727 | 1 | 1 |
| 303 | gpt | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 1.7826 | 1 | 1 |
| 304 | gpt | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 34.55 | 1.5914 | 1 | 1 |
| 305 | gpt | label_smoothing | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 34.55 | 1.6702 | 1 | 1 |
| 306 | gpt | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.001 | 34.55 | 1.2210 | 1 | 1 |
| 307 | gpt | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 1.2193 | 1 | 1 |
| 308 | gpt | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 1.0793 | 1 | 1 |
| 309 | gpt | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 34.55 | 1.0469 | 1 | 1 |
| 310 | mlp | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 34.55 | 17.1708 | 1 | 1 |
| 311 | mlp | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 34.55 | 1.7918 | 1 | 1 |
| 312 | mlp | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.001 | 34.55 | 1.7499 | 1 | 1 |
| 313 | mlp | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 15.2149 | 1 | 1 |
| 314 | mlp | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 49.9367 | 1 | 1 |
| 315 | mlp | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.1856 | 1 | 1 |
| 316 | mlp | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 1.2343 | 1 | 1 |
| 317 | mlp | focal_loss | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 34.55 | 0.9802 | 1 | 1 |
| 318 | repghost | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 34.55 | 1.7269 | 1 | 1 |
| 319 | repghost | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 34.55 | 1.8220 | 1 | 1 |
| 320 | repghost | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 34.55 | 1.7998 | 1 | 1 |
| 321 | repghost | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 2.3839 | 1 | 1 |
| 322 | sknet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 799.2956 | 1 | 1 |
| 323 | sknet | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 34.55 | 1.7521 | 1 | 1 |
| 324 | sknet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 1482.1357 | 1 | 1 |
| 325 | sknet | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 34.55 | 1.7654 | 1 | 1 |
| 326 | sknet | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 2286.1573 | 1 | 1 |
| 327 | sknet | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 1.1015 | 1 | 1 |
| 328 | sknet | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 8909.4260 | 1 | 1 |
| 329 | xception | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 1.7785 | 1 | 1 |
| 330 | xception | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 1.7845 | 1 | 1 |
| 331 | xception | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.001 | 34.55 | 1.7817 | 1 | 1 |
| 332 | xception | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 34.55 | 1.7462 | 1 | 1 |
| 333 | xception | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 34.55 | 1.7437 | 1 | 1 |
| 334 | xception | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 1.2347 | 1 | 1 |
| 335 | xception | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 1.2064 | 1 | 1 |
| 336 | xception | focal_loss | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 34.55 | 1.2171 | 1 | 1 |
| 337 | vim_tiny | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 32.73 | 2.0969 | 1 | 1 |
| 338 | simple_cnn | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 32.73 | 4.2580 | 1 | 1 |
| 339 | mlp | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 32.73 | 13.2628 | 1 | 1 |
| 340 | mlp | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 32.73 | 1.2766 | 1 | 1 |
| 341 | cspnet | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.001 | 29.09 | 1.2657 | 1 | 1 |
| 342 | ghostnet | cross_entropy | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 29.09 | 1.7614 | 1 | 1 |
| 343 | ghostnet | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 29.09 | 1.7666 | 1 | 1 |
| 344 | ghostnet | label_smoothing | adam | batch_size=16, lr=0.0001, weight_decay=0.001 | 29.09 | 1.7874 | 1 | 1 |
| 345 | ghostnet | focal_loss | adamw | batch_size=16, lr=0.0001, weight_decay=0.0001 | 29.09 | 1.2254 | 1 | 1 |
| 346 | lenet | cross_entropy | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 29.09 | 1.7530 | 1 | 1 |
| 347 | lenet | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 29.09 | 1.7566 | 1 | 1 |
| 348 | lenet | label_smoothing | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 29.09 | 1.7713 | 1 | 1 |
| 349 | lenet | focal_loss | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 29.09 | 1.2235 | 1 | 1 |
| 350 | poolformer | label_smoothing | adam | batch_size=16, lr=0.0001, weight_decay=0.001 | 29.09 | 2.1590 | 1 | 1 |
| 351 | poolformer | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 29.09 | 1.6193 | 1 | 1 |
| 352 | poolformer | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 29.09 | 30.4500 | 1 | 1 |
| 353 | poolformer | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 29.09 | 2.2822 | 1 | 1 |
| 354 | shufflenet | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 29.09 | 1.7727 | 1 | 1 |
| 355 | shufflenet | focal_loss | adamw | batch_size=16, lr=0.0001, weight_decay=0.0 | 29.09 | 1.2150 | 1 | 1 |
| 356 | vit | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 29.09 | 1.5342 | 1 | 1 |
| 357 | coord_resnet | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 29.09 | 1.7448 | 1 | 1 |
| 358 | coord_resnet | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 29.09 | 1.7730 | 1 | 1 |
| 359 | coord_resnet | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 29.09 | 123.3857 | 1 | 1 |
| 360 | efficientnetv2 | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 29.09 | 1.2407 | 1 | 1 |
| 361 | efficientnetv2 | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 29.09 | 1.3516 | 1 | 1 |
| 362 | lcnet | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 29.09 | 1.2237 | 1 | 1 |
| 363 | lcnet | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 29.09 | 1.2470 | 1 | 1 |
| 364 | nin | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 29.09 | 1.7596 | 1 | 1 |
| 365 | se_resnet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 29.09 | 8.0434 | 1 | 1 |
| 366 | vim_tiny | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 29.09 | 20673.3074 | 1 | 1 |
| 367 | vim_tiny | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 29.09 | 1.7815 | 1 | 1 |
| 368 | convnext | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 29.09 | 3.8326 | 1 | 1 |
| 369 | efficientnet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 29.09 | 2.3195 | 1 | 1 |
| 370 | efficientnet | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.001 | 29.09 | 1.2248 | 1 | 1 |
| 371 | efficientnet | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 29.09 | 1.2314 | 1 | 1 |
| 372 | inception_resnet | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 29.09 | 8.4909e+06 | 1 | 1 |
| 373 | mobilenetv3 | focal_loss | adamw | batch_size=16, lr=0.0001, weight_decay=0.001 | 29.09 | 1.2413 | 1 | 1 |
| 374 | resnext | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.0001 | 29.09 | 1.7655 | 1 | 1 |
| 375 | resnext | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 29.09 | 6.9465 | 1 | 1 |
| 376 | resnext | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 29.09 | 1.7313 | 1 | 1 |
| 377 | resnext | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 29.09 | 1.7455 | 1 | 1 |
| 378 | vgg | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.0 | 29.09 | 1.7882 | 1 | 1 |
| 379 | coatnet | cross_entropy | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 29.09 | 1.7723 | 1 | 1 |
| 380 | coatnet | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 29.09 | 3.6147 | 1 | 1 |
| 381 | coatnet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 29.09 | 1.2414 | 1 | 1 |
| 382 | eca_resnet | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 29.09 | 5.9711 | 1 | 1 |
| 383 | eca_resnet | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 29.09 | 1.7893 | 1 | 1 |
| 384 | hrnet | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 29.09 | 1.7714 | 1 | 1 |
| 385 | hrnet | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 29.09 | 1.7686 | 1 | 1 |
| 386 | hrnet | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 29.09 | 1.2355 | 1 | 1 |
| 387 | resnet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 29.09 | 1.6156 | 1 | 1 |
| 388 | resnet | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 29.09 | 2.1189 | 1 | 1 |
| 389 | resnet | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 29.09 | 1156.7917 | 1 | 1 |
| 390 | van | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 29.09 | 1.5354 | 1 | 1 |
| 391 | cbam_resnet | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 29.09 | 1.6047 | 1 | 1 |
| 392 | dpn | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 29.09 | 1.7439 | 1 | 1 |
| 393 | dpn | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 29.09 | 1.7011 | 1 | 1 |
| 394 | dpn | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 29.09 | 1.0700 | 1 | 1 |
| 395 | hardnet | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 29.09 | 1.0245e+08 | 1 | 1 |
| 396 | mobilenet | label_smoothing | adamw | batch_size=16, lr=0.0001, weight_decay=0.0 | 29.09 | 1.7728 | 1 | 1 |
| 397 | mobilenet | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 29.09 | 2433.6621 | 1 | 1 |
| 398 | res2net | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 29.09 | 84388.5659 | 1 | 1 |
| 399 | res2net | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 29.09 | 16342.7433 | 1 | 1 |
| 400 | res2net | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 29.09 | 74.6094 | 1 | 1 |
| 401 | res2net | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 29.09 | 1.1525 | 1 | 1 |
| 402 | swin_tiny | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 29.09 | 5.0558 | 1 | 1 |
| 403 | swin_tiny | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 29.09 | 2.4703 | 1 | 1 |
| 404 | swin_tiny | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 29.09 | 3.4701 | 1 | 1 |
| 405 | swin_tiny | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 29.09 | 83.5586 | 1 | 1 |
| 406 | swin_tiny | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 29.09 | 4.4197 | 1 | 1 |
| 407 | alexnet | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 29.09 | 1.7924 | 1 | 1 |
| 408 | alexnet | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 29.09 | 2.3265 | 1 | 1 |
| 409 | alexnet | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 29.09 | 1.7898 | 1 | 1 |
| 410 | alexnet | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 29.09 | 1.6840 | 1 | 1 |
| 411 | alexnet | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 29.09 | 1017.9514 | 1 | 1 |
| 412 | darknet | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 29.09 | 497148.4643 | 1 | 1 |
| 413 | googlenet | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 29.09 | 1.2128 | 1 | 1 |
| 414 | lstm | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 29.09 | 1.6962 | 1 | 1 |
| 415 | lstm | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 29.09 | 1.8059 | 1 | 1 |
| 416 | regnet | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 29.09 | 1.6949 | 1 | 1 |
| 417 | regnet | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 29.09 | 1.7269 | 1 | 1 |
| 418 | regnet | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 29.09 | 1.1194 | 1 | 1 |
| 419 | regnet | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 29.09 | 1.2272 | 1 | 1 |
| 420 | simple_cnn | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 29.09 | 1.6932 | 1 | 1 |
| 421 | simple_cnn | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 29.09 | 2696.7951 | 1 | 1 |
| 422 | simple_cnn | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 29.09 | 1.2307 | 1 | 1 |
| 423 | wide_resnet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 29.09 | 2.0370 | 1 | 1 |
| 424 | wide_resnet | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 29.09 | 5.8921 | 1 | 1 |
| 425 | wide_resnet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 29.09 | 1.4136e+06 | 1 | 1 |
| 426 | wide_resnet | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 29.09 | 1.7864 | 1 | 1 |
| 427 | wide_resnet | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 29.09 | 2.0046 | 1 | 1 |
| 428 | wide_resnet | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 29.09 | 1.1815 | 1 | 1 |
| 429 | wide_resnet | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 29.09 | 1.0986 | 1 | 1 |
| 430 | gru | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 29.09 | 1.7498 | 1 | 1 |
| 431 | gru | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 29.09 | 1.1682 | 1 | 1 |
| 432 | mnasnet | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 29.09 | 32.0449 | 1 | 1 |
| 433 | bert | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 29.09 | 1.7894 | 1 | 1 |
| 434 | bert | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 29.09 | 1.2453 | 1 | 1 |
| 435 | deit | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 29.09 | 1.4899 | 1 | 1 |
| 436 | deit | focal_loss | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 29.09 | 0.8611 | 1 | 1 |
| 437 | mlp | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 29.09 | 1.7775 | 1 | 1 |
| 438 | repghost | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 29.09 | 1.7663 | 1 | 1 |
| 439 | repghost | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 29.09 | 1.8230 | 1 | 1 |
| 440 | repghost | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 29.09 | 1.7822 | 1 | 1 |
| 441 | repghost | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 29.09 | 1.1524 | 1 | 1 |
| 442 | sknet | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 29.09 | 1.7133 | 1 | 1 |
| 443 | sknet | label_smoothing | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 29.09 | 1.6644 | 1 | 1 |
| 444 | sknet | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 29.09 | 1.3984 | 1 | 1 |
| 445 | xception | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 29.09 | 137.8311 | 1 | 1 |
| 446 | xception | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 29.09 | 63305.2996 | 1 | 1 |
| 447 | coord_resnet | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 27.27 | 1.7295 | 1 | 1 |
| 448 | se_resnet | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 27.27 | 1.6364 | 1 | 1 |
| 449 | vim_tiny | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 27.27 | 2.3158 | 1 | 1 |
| 450 | inception_resnet | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 27.27 | 3.5736 | 1 | 1 |
| 451 | cspnet | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 25.45 | 1.6801 | 1 | 1 |
| 452 | vim_tiny | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 25.45 | 36255.3225 | 1 | 1 |
| 453 | cspnet | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 23.64 | 1.7803 | 1 | 1 |
| 454 | cspnet | cross_entropy | adamw | batch_size=16, lr=0.0001, weight_decay=0.0001 | 23.64 | 1.8181 | 1 | 1 |
| 455 | cspnet | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 23.64 | 1.4674e+07 | 1 | 1 |
| 456 | cspnet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 23.64 | 5816.3505 | 1 | 1 |
| 457 | cspnet | label_smoothing | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 23.64 | 1.7790 | 1 | 1 |
| 458 | cspnet | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 23.64 | 6.6274e+09 | 1 | 1 |
| 459 | ghostnet | cross_entropy | adamw | batch_size=16, lr=0.0001, weight_decay=0.001 | 23.64 | 1.7958 | 1 | 1 |
| 460 | ghostnet | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 23.64 | 1.7840 | 1 | 1 |
| 461 | ghostnet | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 23.64 | 1.2110 | 1 | 1 |
| 462 | poolformer | cross_entropy | sgd | batch_size=16, lr=0.0001, weight_decay=0.0001 | 23.64 | 1.6371 | 1 | 1 |
| 463 | poolformer | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 23.64 | 23.9480 | 1 | 1 |
| 464 | poolformer | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 23.64 | 223.6171 | 1 | 1 |
| 465 | coord_resnet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 23.64 | 2.1054 | 1 | 1 |
| 466 | efficientnetv2 | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 23.64 | 1.7877 | 1 | 1 |
| 467 | efficientnetv2 | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 23.64 | 1.8215 | 1 | 1 |
| 468 | efficientnetv2 | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 23.64 | 1.7779 | 1 | 1 |
| 469 | efficientnetv2 | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 23.64 | 145.4091 | 1 | 1 |
| 470 | lcnet | cross_entropy | adamw | batch_size=16, lr=0.0001, weight_decay=0.0001 | 23.64 | 1.7815 | 1 | 1 |
| 471 | lcnet | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 23.64 | 1.7789 | 1 | 1 |
| 472 | se_resnet | cross_entropy | adam | batch_size=16, lr=0.001, weight_decay=0.001 | 23.64 | 1.8742 | 1 | 1 |
| 473 | se_resnet | label_smoothing | adamw | batch_size=16, lr=0.0001, weight_decay=0.0001 | 23.64 | 1.7898 | 1 | 1 |
| 474 | se_resnet | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 23.64 | 1.8605 | 1 | 1 |
| 475 | se_resnet | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 23.64 | 74664.1766 | 1 | 1 |
| 476 | vim_tiny | label_smoothing | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.001 | 23.64 | 1.8130 | 1 | 1 |
| 477 | convnext | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 23.64 | 1.9513 | 1 | 1 |
| 478 | convnext | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 23.64 | 2.7446 | 1 | 1 |
| 479 | efficientnet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 23.64 | 1.7978 | 1 | 1 |
| 480 | efficientnet | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 23.64 | 1.7989 | 1 | 1 |
| 481 | efficientnet | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 23.64 | 7852.8424 | 1 | 1 |
| 482 | efficientnet | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 23.64 | 1.1710 | 1 | 1 |
| 483 | inception_resnet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 23.64 | 48.7242 | 1 | 1 |
| 484 | inception_resnet | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 23.64 | 1.1255 | 1 | 1 |
| 485 | mobilenetv3 | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 23.64 | 1.2514 | 1 | 1 |
| 486 | resnext | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 23.64 | 1.8048 | 1 | 1 |
| 487 | resnext | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 23.64 | 1.2201 | 1 | 1 |
| 488 | vgg | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 23.64 | 1.2408 | 1 | 1 |
| 489 | vgg | focal_loss | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.001 | 23.64 | 1.2266 | 1 | 1 |
| 490 | coatnet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 23.64 | 1.8892 | 1 | 1 |
| 491 | coatnet | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 23.64 | 1.9767 | 1 | 1 |
| 492 | coatnet | focal_loss | adam | batch_size=16, lr=0.0001, weight_decay=0.0001 | 23.64 | 1.3014 | 1 | 1 |
| 493 | coatnet | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 23.64 | 58.0408 | 1 | 1 |
| 494 | coatnet | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 23.64 | 2.1854 | 1 | 1 |
| 495 | eca_resnet | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 23.64 | 1.8958 | 1 | 1 |
| 496 | eca_resnet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 23.64 | 1.7569 | 1 | 1 |
| 497 | eca_resnet | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 23.64 | 1.9941 | 1 | 1 |
| 498 | eca_resnet | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 23.64 | 1.7484 | 1 | 1 |
| 499 | eca_resnet | focal_loss | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 23.64 | 1.2502 | 1 | 1 |
| 500 | hrnet | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 23.64 | 1.7242 | 1 | 1 |
| 501 | hrnet | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 23.64 | 1.9026 | 1 | 1 |
| 502 | hrnet | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 23.64 | 1.8017 | 1 | 1 |
| 503 | mobilenetv2 | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 23.64 | 2.1657 | 1 | 1 |
| 504 | mobilenetv2 | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 23.64 | 1.8089 | 1 | 1 |
| 505 | mobilenetv2 | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 23.64 | 3.2224 | 1 | 1 |
| 506 | mobilenetv2 | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 23.64 | 1.8593 | 1 | 1 |
| 507 | mobilenetv2 | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 23.64 | 1.2772 | 1 | 1 |
| 508 | mobilenetv2 | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 23.64 | 1.2473 | 1 | 1 |
| 509 | resnet | cross_entropy | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 23.64 | 4.3952e+07 | 1 | 1 |
| 510 | resnet | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 23.64 | 20848.9731 | 1 | 1 |
| 511 | resnet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 23.64 | 3.7552e+08 | 1 | 1 |
| 512 | resnet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 23.64 | 16455.9138 | 1 | 1 |
| 513 | van | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 23.64 | 1.7963 | 1 | 1 |
| 514 | van | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.001 | 23.64 | 1.2318 | 1 | 1 |
| 515 | cbam_resnet | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 23.64 | 1.9004 | 1 | 1 |
| 516 | cbam_resnet | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 23.64 | 1.8586 | 1 | 1 |
| 517 | cbam_resnet | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 23.64 | 1.7382 | 1 | 1 |
| 518 | cbam_resnet | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 23.64 | 1.1680 | 1 | 1 |
| 519 | dpn | cross_entropy | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 23.64 | 1.5757 | 1 | 1 |
| 520 | dpn | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 23.64 | 1.8032 | 1 | 1 |
| 521 | hardnet | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 23.64 | 2.9675e+09 | 1 | 1 |
| 522 | hardnet | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 23.64 | 1.2822 | 1 | 1 |
| 523 | mobilenet | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 23.64 | 1.3144 | 1 | 1 |
| 524 | res2net | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 23.64 | 1.7314 | 1 | 1 |
| 525 | res2net | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 23.64 | 9.2332e+14 | 1 | 1 |
| 526 | swin_tiny | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 23.64 | 4.3493 | 1 | 1 |
| 527 | swin_tiny | focal_loss | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 23.64 | 5.0212 | 1 | 1 |
| 528 | darknet | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 23.64 | 1.7670 | 1 | 1 |
| 529 | darknet | focal_loss | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 23.64 | 1.4637 | 1 | 1 |
| 530 | darknet | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 23.64 | 1.1333 | 1 | 1 |
| 531 | googlenet | cross_entropy | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 23.64 | 1.7871 | 1 | 1 |
| 532 | googlenet | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 23.64 | 1.7868 | 1 | 1 |
| 533 | regnet | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 23.64 | 1.1914 | 1 | 1 |
| 534 | regnet | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 23.64 | 5801.0588 | 1 | 1 |
| 535 | wide_resnet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 23.64 | 1.3794e+06 | 1 | 1 |
| 536 | wide_resnet | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 23.64 | 1.1383 | 1 | 1 |
| 537 | densenet | label_smoothing | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 23.64 | 1.7925 | 1 | 1 |
| 538 | gru | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 23.64 | 1.7699 | 1 | 1 |
| 539 | mnasnet | label_smoothing | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 23.64 | 1.7971 | 1 | 1 |
| 540 | mnasnet | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 23.64 | 1.2402 | 1 | 1 |
| 541 | mnasnet | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 23.64 | 4.4015 | 1 | 1 |
| 542 | repvgg | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 23.64 | 1.7483 | 1 | 1 |
| 543 | repvgg | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 23.64 | 1.8154 | 1 | 1 |
| 544 | repvgg | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.001 | 23.64 | 1.3061 | 1 | 1 |
| 545 | squeezenet | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 23.64 | 1.5922 | 1 | 1 |
| 546 | repghost | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 23.64 | 1.2881 | 1 | 1 |
| 547 | repghost | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 23.64 | 1.2571 | 1 | 1 |
| 548 | sknet | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 23.64 | 1.7151 | 1 | 1 |
| 549 | sknet | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 23.64 | 1.1896 | 1 | 1 |
| 550 | xception | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 23.64 | 1.7944 | 1 | 1 |
| 551 | xception | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 9.09 | 9.2134 | 1 | 1 |
| 552 | cspnet | label_smoothing | adamw | batch_size=16, lr=0.0001, weight_decay=0.001 | 5.45 | 1.7819 | 1 | 1 |
| 553 | lenet | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 5.45 | 1.2716 | 1 | 1 |
| 554 | poolformer | focal_loss | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 5.45 | 24.4974 | 1 | 1 |
| 555 | efficientnetv2 | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 5.45 | 1.8212 | 1 | 1 |
| 556 | lcnet | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 5.45 | 1.7642 | 1 | 1 |
| 557 | lcnet | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 5.45 | 1.2878 | 1 | 1 |
| 558 | se_resnet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 5.45 | 1.2506 | 1 | 1 |
| 559 | vim_tiny | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 5.45 | 1.9155 | 1 | 1 |
| 560 | vim_tiny | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 5.45 | 1.7841 | 1 | 1 |
| 561 | vim_tiny | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 5.45 | 1.2527 | 1 | 1 |
| 562 | vim_tiny | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 5.45 | 1.3723 | 1 | 1 |
| 563 | efficientnet | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 5.45 | 1.7885 | 1 | 1 |
| 564 | inception_resnet | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.0001 | 5.45 | 1.8043 | 1 | 1 |
| 565 | coatnet | cross_entropy | adam | batch_size=16, lr=0.001, weight_decay=0.001 | 5.45 | 1.9372 | 1 | 1 |
| 566 | coatnet | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 5.45 | 1.8642 | 1 | 1 |
| 567 | mobilenetv2 | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 5.45 | 1.8559 | 1 | 1 |
| 568 | mobilenetv2 | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 5.45 | 1.8014 | 1 | 1 |
| 569 | mobilenetv2 | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 5.45 | 2.0713 | 1 | 1 |
| 570 | cbam_resnet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 5.45 | 1.9287 | 1 | 1 |
| 571 | dpn | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 5.45 | 1.7885 | 1 | 1 |
| 572 | hardnet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 5.45 | 1.7809 | 1 | 1 |
| 573 | hardnet | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 5.45 | 1.7901 | 1 | 1 |
| 574 | hardnet | focal_loss | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 5.45 | 1.2704 | 1 | 1 |
| 575 | res2net | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 5.45 | 1.2372 | 1 | 1 |
| 576 | swin_tiny | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 5.45 | 70.7379 | 1 | 1 |
| 577 | darknet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 5.45 | 2.0381 | 1 | 1 |
| 578 | lstm | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 5.45 | 1.8144 | 1 | 1 |
| 579 | mnasnet | focal_loss | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.001 | 5.45 | 1.2827 | 1 | 1 |
| 580 | repvgg | label_smoothing | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 5.45 | 1.7939 | 1 | 1 |
| 581 | repvgg | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 5.45 | 1.2898 | 1 | 1 |
| 582 | repghost | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 5.45 | 1.8237 | 1 | 1 |
| 583 | repghost | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 5.45 | 2.0604 | 1 | 1 |
| 584 | cspnet | focal_loss | adam | batch_size=16, lr=0.0001, weight_decay=0.0001 | 3.64 | 1.2696 | 1 | 1 |
| 585 | poolformer | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 3.64 | 193.5898 | 1 | 1 |
| 586 | se_resnet | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.0001 | 3.64 | 1.8308 | 1 | 1 |
| 587 | mobilenetv3 | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 3.64 | 3.5264e+07 | 1 | 1 |
| 588 | wide_resnet | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 3.64 | 1.8051 | 1 | 1 |

## Autoencoder — best per model

| Rank | Model | Loss | Optimizer | Hyperparameters | Recon Loss | Epochs Run | Convergence Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | denoising_ae | mse | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 2 | vae | mse | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 3 | conv_ae | mse | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 4 | simple_ae | mse | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 0.0000 | 1 | 1 |

## Autoencoder — all trials (45 rows)

| Rank | Model | Loss | Optimizer | Hyperparameters | Recon Loss | Epochs Run | Convergence Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | denoising_ae | mse | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 2 | denoising_ae | mse | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 0.0000 | 1 | 1 |
| 3 | denoising_ae | mse | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 4 | denoising_ae | mse | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 5 | denoising_ae | l1 | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 6 | denoising_ae | l1 | sgd | batch_size=16, lr=0.0001, weight_decay=0.0 | 0.0000 | 1 | 1 |
| 7 | denoising_ae | l1 | adamw | batch_size=16, lr=0.0001, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 8 | denoising_ae | l1 | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 9 | denoising_ae | bce | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 0.0000 | 1 | 1 |
| 10 | denoising_ae | bce | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 0.0000 | 1 | 1 |
| 11 | denoising_ae | bce | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 12 | denoising_ae | bce | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 13 | vae | mse | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 14 | vae | mse | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 15 | vae | mse | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 0.0000 | 1 | 1 |
| 16 | vae | mse | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 17 | vae | l1 | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 18 | vae | l1 | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 19 | vae | l1 | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 20 | vae | l1 | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 21 | vae | bce | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 22 | conv_ae | mse | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 23 | conv_ae | mse | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 0.0000 | 1 | 1 |
| 24 | conv_ae | mse | adamw | batch_size=16, lr=0.0001, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 25 | conv_ae | mse | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 26 | conv_ae | l1 | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 27 | conv_ae | l1 | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 28 | conv_ae | l1 | adamw | batch_size=16, lr=0.001, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 29 | conv_ae | l1 | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 0.0000 | 1 | 1 |
| 30 | conv_ae | bce | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 0.0000 | 1 | 1 |
| 31 | conv_ae | bce | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 32 | conv_ae | bce | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 33 | conv_ae | bce | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 34 | simple_ae | mse | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 35 | simple_ae | mse | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 36 | simple_ae | mse | adamw | batch_size=16, lr=0.0001, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 37 | simple_ae | mse | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 0.0000 | 1 | 1 |
| 38 | simple_ae | l1 | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 39 | simple_ae | l1 | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 40 | simple_ae | l1 | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 41 | simple_ae | l1 | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 42 | simple_ae | bce | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 43 | simple_ae | bce | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 0.0000 | 1 | 1 |
| 44 | simple_ae | bce | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 45 | simple_ae | bce | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 0.0000 | 1 | 1 |

## GAN — best per model

| Rank | Model | Loss | Optimizer | Hyperparameters | G Loss | D Loss | Epochs Run | Convergence Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | wgan | wasserstein | adam | batch_size=16, lr=0.0001, weight_decay=0.001 | 0.0050 | -0.0052 | 1 | 1 |
| 2 | cgan | wasserstein | sgd | batch_size=16, lr=0.0001, weight_decay=0.0001 | 0.4992 | 1.4491 | 1 | 1 |
| 3 | vanilla_gan | bce | sgd | batch_size=16, lr=0.0001, weight_decay=0.0001 | 0.6913 | 1.3748 | 1 | 1 |
| 4 | dcgan | wasserstein | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 0.6958 | 1.3284 | 1 | 1 |

## GAN — all trials (28 rows)

| Rank | Model | Loss | Optimizer | Hyperparameters | G Loss | D Loss | Epochs Run | Convergence Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | wgan | wasserstein | adam | batch_size=16, lr=0.0001, weight_decay=0.001 | 0.0050 | -0.0052 | 1 | 1 |
| 2 | wgan | wasserstein | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 0.1159 | -0.0717 | 1 | 1 |
| 3 | wgan | bce | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 0.1841 | -0.0773 | 1 | 1 |
| 4 | wgan | bce | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 0.2321 | -0.1375 | 1 | 1 |
| 5 | cgan | wasserstein | sgd | batch_size=16, lr=0.0001, weight_decay=0.0001 | 0.4992 | 1.4491 | 1 | 1 |
| 6 | cgan | wasserstein | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 0.6020 | 2.0864 | 1 | 1 |
| 7 | cgan | bce | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 0.6888 | 1.4013 | 1 | 1 |
| 8 | vanilla_gan | bce | sgd | batch_size=16, lr=0.0001, weight_decay=0.0001 | 0.6913 | 1.3748 | 1 | 1 |
| 9 | dcgan | wasserstein | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 0.6958 | 1.3284 | 1 | 1 |
| 10 | cgan | bce | adam | batch_size=16, lr=0.0001, weight_decay=0.0001 | 0.7048 | 1.3931 | 1 | 1 |
| 11 | vanilla_gan | wasserstein | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 0.7299 | 1.0626 | 1 | 1 |
| 12 | dcgan | bce | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 0.7812 | 1.3580 | 1 | 1 |
| 13 | vanilla_gan | bce | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 0.8193 | 0.8279 | 1 | 1 |
| 14 | cgan | bce | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 0.8820 | 1.8764 | 1 | 1 |
| 15 | vanilla_gan | wasserstein | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 0.8845 | 8.5827 | 1 | 1 |
| 16 | cgan | bce | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 0.9382 | 2.0048 | 1 | 1 |
| 17 | vanilla_gan | bce | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 1.0009 | 0.6262 | 1 | 1 |
| 18 | vanilla_gan | wasserstein | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 1.0787 | 1.9498 | 1 | 1 |
| 19 | vanilla_gan | wasserstein | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 1.0925 | 14.4404 | 1 | 1 |
| 20 | dcgan | bce | adamw | batch_size=16, lr=0.0001, weight_decay=0.0001 | 1.2086 | 1.3083 | 1 | 1 |
| 21 | dcgan | wasserstein | adam | batch_size=16, lr=0.0001, weight_decay=0.001 | 1.2390 | 1.2869 | 1 | 1 |
| 22 | cgan | wasserstein | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 1.7859 | 26.7168 | 1 | 1 |
| 23 | cgan | wasserstein | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 3.0163 | 7.1859 | 1 | 1 |
| 24 | dcgan | bce | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 5.7180 | 0.6340 | 1 | 1 |
| 25 | dcgan | bce | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 5.7528 | 0.5247 | 1 | 1 |
| 26 | dcgan | wasserstein | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 6.0092 | 1.1281 | 1 | 1 |
| 27 | dcgan | wasserstein | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 10.3424 | 0.5221 | 1 | 1 |
| 28 | vanilla_gan | bce | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 38.8306 | 47.4145 | 1 | 1 |

## Failed Configurations

- **vae** / bce / sgd: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

- **vae** / bce / adamw: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

- **vae** / bce / rmsprop: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

- **capsnet** / cross_entropy / adam: CUDA out of memory. Tried to allocate 3.96 GiB. GPU 2 has a total capacity of 31.73 GiB of which 3.05 GiB is free. Process 1725961 has 1.32 GiB memory in use. Process 47377 has 14.96 GiB memory in use. Including non-PyTorch memory, this process has 12.39 GiB memory in use. Of the allocated memory 11.94 GiB is allocated by PyTorch, and 76.32 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
- **capsnet** / cross_entropy / sgd: CUDA out of memory. Tried to allocate 3.96 GiB. GPU 2 has a total capacity of 31.73 GiB of which 2.99 GiB is free. Process 1725961 has 1.32 GiB memory in use. Process 47377 has 14.96 GiB memory in use. Including non-PyTorch memory, this process has 12.45 GiB memory in use. Of the allocated memory 11.96 GiB is allocated by PyTorch, and 115.99 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
- **capsnet** / cross_entropy / adamw: CUDA out of memory. Tried to allocate 3.96 GiB. GPU 2 has a total capacity of 31.73 GiB of which 3.05 GiB is free. Process 1725961 has 1.32 GiB memory in use. Process 47377 has 14.96 GiB memory in use. Including non-PyTorch memory, this process has 12.39 GiB memory in use. Of the allocated memory 11.94 GiB is allocated by PyTorch, and 76.32 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
- **capsnet** / cross_entropy / rmsprop: CUDA out of memory. Tried to allocate 3.96 GiB. GPU 2 has a total capacity of 31.73 GiB of which 3.03 GiB is free. Process 1725961 has 1.32 GiB memory in use. Process 47377 has 14.96 GiB memory in use. Including non-PyTorch memory, this process has 12.41 GiB memory in use. Of the allocated memory 11.96 GiB is allocated by PyTorch, and 75.99 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
- **capsnet** / label_smoothing / adam: CUDA out of memory. Tried to allocate 3.96 GiB. GPU 2 has a total capacity of 31.73 GiB of which 3.01 GiB is free. Process 1725961 has 1.32 GiB memory in use. Process 47377 has 14.96 GiB memory in use. Including non-PyTorch memory, this process has 12.43 GiB memory in use. Of the allocated memory 11.94 GiB is allocated by PyTorch, and 116.32 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
- **capsnet** / label_smoothing / sgd: CUDA out of memory. Tried to allocate 1.98 GiB. GPU 2 has a total capacity of 31.73 GiB of which 1.84 GiB is free. Process 1725961 has 1.32 GiB memory in use. Process 47377 has 14.96 GiB memory in use. Including non-PyTorch memory, this process has 13.60 GiB memory in use. Of the allocated memory 13.07 GiB is allocated by PyTorch, and 162.31 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
- **capsnet** / label_smoothing / adamw: CUDA out of memory. Tried to allocate 3.96 GiB. GPU 2 has a total capacity of 31.73 GiB of which 3.03 GiB is free. Process 1725961 has 1.32 GiB memory in use. Process 47377 has 14.96 GiB memory in use. Including non-PyTorch memory, this process has 12.41 GiB memory in use. Of the allocated memory 11.94 GiB is allocated by PyTorch, and 96.32 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
- **capsnet** / label_smoothing / rmsprop: CUDA out of memory. Tried to allocate 3.96 GiB. GPU 2 has a total capacity of 31.73 GiB of which 3.01 GiB is free. Process 1725961 has 1.32 GiB memory in use. Process 47377 has 14.96 GiB memory in use. Including non-PyTorch memory, this process has 12.43 GiB memory in use. Of the allocated memory 11.96 GiB is allocated by PyTorch, and 95.99 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
- **capsnet** / focal_loss / adam: CUDA out of memory. Tried to allocate 3.96 GiB. GPU 2 has a total capacity of 31.73 GiB of which 3.05 GiB is free. Process 1725961 has 1.32 GiB memory in use. Process 47377 has 14.96 GiB memory in use. Including non-PyTorch memory, this process has 12.39 GiB memory in use. Of the allocated memory 11.94 GiB is allocated by PyTorch, and 76.32 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
- **capsnet** / focal_loss / sgd: CUDA out of memory. Tried to allocate 3.96 GiB. GPU 2 has a total capacity of 31.73 GiB of which 2.97 GiB is free. Process 1725961 has 1.32 GiB memory in use. Process 47377 has 14.96 GiB memory in use. Including non-PyTorch memory, this process has 12.47 GiB memory in use. Of the allocated memory 11.96 GiB is allocated by PyTorch, and 135.99 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
- **capsnet** / focal_loss / adamw: CUDA out of memory. Tried to allocate 3.96 GiB. GPU 2 has a total capacity of 31.73 GiB of which 3.05 GiB is free. Process 1725961 has 1.32 GiB memory in use. Process 47377 has 14.96 GiB memory in use. Including non-PyTorch memory, this process has 12.39 GiB memory in use. Of the allocated memory 11.94 GiB is allocated by PyTorch, and 76.32 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
- **capsnet** / focal_loss / rmsprop: CUDA out of memory. Tried to allocate 3.96 GiB. GPU 2 has a total capacity of 31.73 GiB of which 2.99 GiB is free. Process 1725961 has 1.32 GiB memory in use. Process 47377 has 14.96 GiB memory in use. Including non-PyTorch memory, this process has 12.45 GiB memory in use. Of the allocated memory 11.96 GiB is allocated by PyTorch, and 115.99 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

## Search Space

- Loss (classification): cross_entropy, label_smoothing, focal_loss
- Loss (autoencoder): mse, l1, bce
- Loss (GAN): bce, wasserstein (informational; GANs use fixed objectives)
- Optimizers: adam, sgd, adamw, rmsprop
- Hyperparameters: {"lr": [0.0001, 0.001, 0.01], "batch_size": [8, 16, 32], "weight_decay": [0.0, 0.0001, 0.001]}
