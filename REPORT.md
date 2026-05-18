# Strawberry Regression Report

## Run configuration

| Setting | Value |
| --- | --- |
| Generated | 2026-05-18T00:37:08 |
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
| Total wall time | 5874.9s |

Training stops when validation metric shows no significant improvement for 3 consecutive epochs.

## Classification — best per model

| Rank | Model | Loss | Optimizer | Hyperparameters | Test Acc (%) | Test Loss | Epochs Run | Convergence Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | mlp | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 40.00 | 8.7714 | 1 | 1 |
| 2 | eca_resnet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 36.36 | 6.5102 | 1 | 1 |
| 3 | wide_resnet | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 36.36 | 1.7549 | 1 | 1 |
| 4 | cspnet | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 34.55 | 1.7852 | 1 | 1 |
| 5 | ghostnet | cross_entropy | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 34.55 | 1.7425 | 1 | 1 |
| 6 | lenet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 34.55 | 1.7379 | 1 | 1 |
| 7 | poolformer | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 34.55 | 3.0592 | 1 | 1 |
| 8 | shufflenet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 1.6820 | 1 | 1 |
| 9 | vit | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 1.5998 | 1 | 1 |
| 10 | coord_resnet | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 1.7279 | 1 | 1 |
| 11 | efficientnetv2 | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 34.55 | 1.8242 | 1 | 1 |
| 12 | lcnet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 1.7805 | 1 | 1 |
| 13 | nin | cross_entropy | sgd | batch_size=16, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.7871 | 1 | 1 |
| 14 | se_resnet | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 25.6013 | 1 | 1 |
| 15 | vim_tiny | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 34.55 | 6.4589e+06 | 1 | 1 |
| 16 | convnext | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 34.55 | 1.7693 | 1 | 1 |
| 17 | efficientnet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 1.5929 | 1 | 1 |
| 18 | inception_resnet | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 1.7499 | 1 | 1 |
| 19 | mobilenetv3 | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 1.7647 | 1 | 1 |
| 20 | resnext | cross_entropy | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 1.6730 | 1 | 1 |
| 21 | vgg | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 17908.2737 | 1 | 1 |
| 22 | cbam_resnet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 1.6620 | 1 | 1 |
| 23 | dpn | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 1.7459 | 1 | 1 |
| 24 | hardnet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 1.7237 | 1 | 1 |
| 25 | mobilenet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 1.5923 | 1 | 1 |
| 26 | res2net | cross_entropy | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 34.55 | 1.7287 | 1 | 1 |
| 27 | swin_tiny | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 5.4352 | 1 | 1 |
| 28 | densenet | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 34.55 | 1.7415 | 1 | 1 |
| 29 | gru | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 16.4148 | 1 | 1 |
| 30 | mnasnet | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.7694 | 1 | 1 |
| 31 | repvgg | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 951831.1641 | 1 | 1 |
| 32 | squeezenet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 34.55 | 1.5775 | 1 | 1 |
| 33 | coatnet | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 34.55 | 4.6579e+06 | 1 | 1 |
| 34 | hrnet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 2.9494 | 1 | 1 |
| 35 | mobilenetv2 | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 34.55 | 1.8457 | 1 | 1 |
| 36 | resnet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 1.8337 | 1 | 1 |
| 37 | van | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 1.7167 | 1 | 1 |
| 38 | alexnet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 1.5533 | 1 | 1 |
| 39 | darknet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 1.6288 | 1 | 1 |
| 40 | googlenet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 34.55 | 1.7178 | 1 | 1 |
| 41 | lstm | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 1.6382 | 1 | 1 |
| 42 | regnet | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 34.55 | 1.7909 | 1 | 1 |
| 43 | simple_cnn | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.6950 | 1 | 1 |
| 44 | bert | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 34.55 | 1.7854 | 1 | 1 |
| 45 | deit | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 2.0203 | 1 | 1 |
| 46 | gpt | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 1.5944 | 1 | 1 |
| 47 | repghost | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 22.4339 | 1 | 1 |
| 48 | sknet | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 34.55 | 1.8116 | 1 | 1 |
| 49 | xception | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 1.6489 | 1 | 1 |
| 50 | capsnet | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 5.45 | 1.7918 | 1 | 1 |

## Classification — all trials (600 rows)

| Rank | Model | Loss | Optimizer | Hyperparameters | Test Acc (%) | Test Loss | Epochs Run | Convergence Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | mlp | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 40.00 | 8.7714 | 1 | 1 |
| 2 | eca_resnet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 36.36 | 6.5102 | 1 | 1 |
| 3 | wide_resnet | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 36.36 | 1.7549 | 1 | 1 |
| 4 | mlp | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 36.36 | 1.1994 | 1 | 1 |
| 5 | cspnet | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 34.55 | 1.7852 | 1 | 1 |
| 6 | cspnet | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.001 | 34.55 | 1.2333 | 1 | 1 |
| 7 | ghostnet | cross_entropy | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 34.55 | 1.7425 | 1 | 1 |
| 8 | ghostnet | cross_entropy | adamw | batch_size=16, lr=0.0001, weight_decay=0.001 | 34.55 | 1.7763 | 1 | 1 |
| 9 | ghostnet | label_smoothing | adam | batch_size=16, lr=0.0001, weight_decay=0.001 | 34.55 | 1.7865 | 1 | 1 |
| 10 | ghostnet | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 1.1626 | 1 | 1 |
| 11 | ghostnet | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 1.1883 | 1 | 1 |
| 12 | lenet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 34.55 | 1.7379 | 1 | 1 |
| 13 | lenet | cross_entropy | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 1.6087 | 1 | 1 |
| 14 | lenet | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 34.55 | 1.7224 | 1 | 1 |
| 15 | lenet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 1.6958 | 1 | 1 |
| 16 | lenet | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 1.6590 | 1 | 1 |
| 17 | lenet | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 1.0132 | 1 | 1 |
| 18 | poolformer | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 34.55 | 3.0592 | 1 | 1 |
| 19 | poolformer | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.7507 | 1 | 1 |
| 20 | poolformer | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 34.55 | 25.7141 | 1 | 1 |
| 21 | poolformer | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 34.55 | 117.3861 | 1 | 1 |
| 22 | poolformer | label_smoothing | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 34.55 | 5.1726 | 1 | 1 |
| 23 | poolformer | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 35.5984 | 1 | 1 |
| 24 | poolformer | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 3.1258 | 1 | 1 |
| 25 | shufflenet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 1.6820 | 1 | 1 |
| 26 | shufflenet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 1.7359 | 1 | 1 |
| 27 | shufflenet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 1.7601 | 1 | 1 |
| 28 | shufflenet | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 1.7514 | 1 | 1 |
| 29 | shufflenet | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 1.7462 | 1 | 1 |
| 30 | shufflenet | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 1.7136 | 1 | 1 |
| 31 | shufflenet | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 1.1155 | 1 | 1 |
| 32 | shufflenet | focal_loss | adamw | batch_size=16, lr=0.0001, weight_decay=0.0 | 34.55 | 1.2115 | 1 | 1 |
| 33 | shufflenet | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.2150 | 1 | 1 |
| 34 | vit | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 1.5998 | 1 | 1 |
| 35 | vit | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 34.55 | 1.5117 | 1 | 1 |
| 36 | vit | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 34.55 | 1.5279 | 1 | 1 |
| 37 | vit | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.5282 | 1 | 1 |
| 38 | vit | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 34.55 | 1.8208 | 1 | 1 |
| 39 | vit | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 1.5584 | 1 | 1 |
| 40 | vit | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 34.55 | 1.5518 | 1 | 1 |
| 41 | vit | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.6077 | 1 | 1 |
| 42 | vit | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 34.55 | 0.9506 | 1 | 1 |
| 43 | vit | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 0.9515 | 1 | 1 |
| 44 | vit | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 34.55 | 1.0025 | 1 | 1 |
| 45 | vit | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 3.2838 | 1 | 1 |
| 46 | coord_resnet | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 1.7279 | 1 | 1 |
| 47 | coord_resnet | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 34.55 | 1.7994 | 1 | 1 |
| 48 | coord_resnet | label_smoothing | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 34.55 | 1.7692 | 1 | 1 |
| 49 | coord_resnet | focal_loss | adam | batch_size=16, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.2526 | 1 | 1 |
| 50 | coord_resnet | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 154.5579 | 1 | 1 |
| 51 | efficientnetv2 | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 34.55 | 1.8242 | 1 | 1 |
| 52 | efficientnetv2 | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 34.55 | 1.7930 | 1 | 1 |
| 53 | efficientnetv2 | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 1.2801 | 1 | 1 |
| 54 | efficientnetv2 | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 34.55 | 1.2238 | 1 | 1 |
| 55 | lcnet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 1.7805 | 1 | 1 |
| 56 | lcnet | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 34.55 | 1.7868 | 1 | 1 |
| 57 | lcnet | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.7798 | 1 | 1 |
| 58 | lcnet | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 1.7691 | 1 | 1 |
| 59 | lcnet | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 1.7816 | 1 | 1 |
| 60 | lcnet | label_smoothing | adamw | batch_size=16, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.7966 | 1 | 1 |
| 61 | lcnet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 14.3592 | 1 | 1 |
| 62 | lcnet | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 1.2502 | 1 | 1 |
| 63 | lcnet | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.2483 | 1 | 1 |
| 64 | nin | cross_entropy | sgd | batch_size=16, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.7871 | 1 | 1 |
| 65 | nin | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 34.55 | 1.7795 | 1 | 1 |
| 66 | nin | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 34.55 | 1.7593 | 1 | 1 |
| 67 | nin | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 34.55 | 1.7538 | 1 | 1 |
| 68 | nin | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 1.7770 | 1 | 1 |
| 69 | nin | label_smoothing | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 34.55 | 1.6299 | 1 | 1 |
| 70 | nin | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 34.55 | 0.9626 | 1 | 1 |
| 71 | nin | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 34.55 | 1.2156 | 1 | 1 |
| 72 | nin | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 0.9143 | 1 | 1 |
| 73 | nin | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 34.55 | 1.7231 | 1 | 1 |
| 74 | se_resnet | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 25.6013 | 1 | 1 |
| 75 | se_resnet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 222.9397 | 1 | 1 |
| 76 | se_resnet | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 34.55 | 1.7807 | 1 | 1 |
| 77 | se_resnet | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 34.55 | 1.2324 | 1 | 1 |
| 78 | vim_tiny | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 34.55 | 6.4589e+06 | 1 | 1 |
| 79 | vim_tiny | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 34.55 | 6.3425 | 1 | 1 |
| 80 | convnext | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 34.55 | 1.7693 | 1 | 1 |
| 81 | convnext | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 34.55 | 2.5226 | 1 | 1 |
| 82 | convnext | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 34.55 | 2.3216 | 1 | 1 |
| 83 | convnext | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 34.55 | 15.0856 | 1 | 1 |
| 84 | efficientnet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 1.5929 | 1 | 1 |
| 85 | efficientnet | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 34.55 | 1.4775 | 1 | 1 |
| 86 | efficientnet | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 1.7295 | 1 | 1 |
| 87 | efficientnet | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 1.6067 | 1 | 1 |
| 88 | efficientnet | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.001 | 34.55 | 1.2859 | 1 | 1 |
| 89 | efficientnet | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 34.55 | 1.2437 | 1 | 1 |
| 90 | efficientnet | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 1.1738 | 1 | 1 |
| 91 | inception_resnet | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 1.7499 | 1 | 1 |
| 92 | inception_resnet | label_smoothing | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 34.55 | 38.8827 | 1 | 1 |
| 93 | inception_resnet | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 1.2150 | 1 | 1 |
| 94 | inception_resnet | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 34.55 | 1.0306 | 1 | 1 |
| 95 | mobilenetv3 | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 1.7647 | 1 | 1 |
| 96 | mobilenetv3 | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 1.9002 | 1 | 1 |
| 97 | mobilenetv3 | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 34.55 | 1.2083 | 1 | 1 |
| 98 | resnext | cross_entropy | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 1.6730 | 1 | 1 |
| 99 | resnext | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 1.7320 | 1 | 1 |
| 100 | resnext | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 1.2507 | 1 | 1 |
| 101 | resnext | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 1.1849 | 1 | 1 |
| 102 | vgg | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 17908.2737 | 1 | 1 |
| 103 | vgg | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.001 | 34.55 | 1.7690 | 1 | 1 |
| 104 | vgg | label_smoothing | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 34.55 | 38294.3127 | 1 | 1 |
| 105 | vgg | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 34.55 | 1.1049 | 1 | 1 |
| 106 | vgg | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 10128.3095 | 1 | 1 |
| 107 | vgg | focal_loss | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0 | 34.55 | 13193.0589 | 1 | 1 |
| 108 | cbam_resnet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 1.6620 | 1 | 1 |
| 109 | cbam_resnet | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 54.7377 | 1 | 1 |
| 110 | cbam_resnet | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 34.55 | 1.7395 | 1 | 1 |
| 111 | cbam_resnet | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 1.7047 | 1 | 1 |
| 112 | cbam_resnet | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 34.55 | 1.1867 | 1 | 1 |
| 113 | cbam_resnet | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 36163.4500 | 1 | 1 |
| 114 | dpn | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 1.7459 | 1 | 1 |
| 115 | dpn | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 9.8267e+07 | 1 | 1 |
| 116 | dpn | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 7.4747 | 1 | 1 |
| 117 | dpn | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 50.4368 | 1 | 1 |
| 118 | dpn | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 34.55 | 1.1829 | 1 | 1 |
| 119 | hardnet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 1.7237 | 1 | 1 |
| 120 | hardnet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 34.55 | 1.7927 | 1 | 1 |
| 121 | hardnet | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 7.5252e+11 | 1 | 1 |
| 122 | hardnet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 4.3561e+10 | 1 | 1 |
| 123 | hardnet | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 34.55 | 1.2446 | 1 | 1 |
| 124 | mobilenet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 1.5923 | 1 | 1 |
| 125 | mobilenet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 34.55 | 1.7295 | 1 | 1 |
| 126 | mobilenet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 34.55 | 1.6220 | 1 | 1 |
| 127 | mobilenet | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 1.7167 | 1 | 1 |
| 128 | mobilenet | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 34.55 | 1.1534 | 1 | 1 |
| 129 | mobilenet | focal_loss | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 34.55 | 1.2477 | 1 | 1 |
| 130 | mobilenet | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 1.0375 | 1 | 1 |
| 131 | mobilenet | focal_loss | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 34.55 | 1.1930 | 1 | 1 |
| 132 | res2net | cross_entropy | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 34.55 | 1.7287 | 1 | 1 |
| 133 | res2net | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 34.55 | 1.8354 | 1 | 1 |
| 134 | res2net | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 1.6872e+10 | 1 | 1 |
| 135 | res2net | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 1.7130 | 1 | 1 |
| 136 | swin_tiny | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 5.4352 | 1 | 1 |
| 137 | swin_tiny | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 1.6751 | 1 | 1 |
| 138 | swin_tiny | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 34.55 | 2.0352 | 1 | 1 |
| 139 | swin_tiny | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 4.2468 | 1 | 1 |
| 140 | swin_tiny | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 34.55 | 1.7460 | 1 | 1 |
| 141 | swin_tiny | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 15.9278 | 1 | 1 |
| 142 | swin_tiny | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 2.7105 | 1 | 1 |
| 143 | swin_tiny | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 7.7163 | 1 | 1 |
| 144 | densenet | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 34.55 | 1.7415 | 1 | 1 |
| 145 | densenet | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 1.5153 | 1 | 1 |
| 146 | densenet | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 34.55 | 1.6574 | 1 | 1 |
| 147 | densenet | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 1.6521 | 1 | 1 |
| 148 | densenet | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 34.55 | 1.6630 | 1 | 1 |
| 149 | densenet | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 1.7175 | 1 | 1 |
| 150 | densenet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 34.55 | 1.2290 | 1 | 1 |
| 151 | densenet | focal_loss | adamw | batch_size=16, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.2345 | 1 | 1 |
| 152 | gru | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 16.4148 | 1 | 1 |
| 153 | gru | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 1.6186 | 1 | 1 |
| 154 | gru | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 1.6507 | 1 | 1 |
| 155 | gru | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 34.55 | 1.1965 | 1 | 1 |
| 156 | gru | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 1.2231 | 1 | 1 |
| 157 | gru | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 1.0914 | 1 | 1 |
| 158 | gru | focal_loss | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 34.55 | 1.1920 | 1 | 1 |
| 159 | mnasnet | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.7694 | 1 | 1 |
| 160 | mnasnet | cross_entropy | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 34.55 | 1.7901 | 1 | 1 |
| 161 | mnasnet | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 34.55 | 4.3712 | 1 | 1 |
| 162 | mnasnet | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 34.55 | 1.9539 | 1 | 1 |
| 163 | mnasnet | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 6.3301 | 1 | 1 |
| 164 | mnasnet | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 1.7547 | 1 | 1 |
| 165 | mnasnet | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 3.0956 | 1 | 1 |
| 166 | mnasnet | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 95.4696 | 1 | 1 |
| 167 | mnasnet | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 1.2148 | 1 | 1 |
| 168 | repvgg | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 951831.1641 | 1 | 1 |
| 169 | repvgg | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 34.55 | 1.1062 | 1 | 1 |
| 170 | repvgg | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 1.2406 | 1 | 1 |
| 171 | squeezenet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 34.55 | 1.5775 | 1 | 1 |
| 172 | squeezenet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 1.5130 | 1 | 1 |
| 173 | squeezenet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.5364 | 1 | 1 |
| 174 | squeezenet | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 34.55 | 1.5480 | 1 | 1 |
| 175 | squeezenet | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 1.7732 | 1 | 1 |
| 176 | squeezenet | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.7333 | 1 | 1 |
| 177 | squeezenet | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 34.55 | 0.8797 | 1 | 1 |
| 178 | squeezenet | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 1.0192 | 1 | 1 |
| 179 | coatnet | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 34.55 | 4.6579e+06 | 1 | 1 |
| 180 | eca_resnet | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 30.0910 | 1 | 1 |
| 181 | eca_resnet | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 34.55 | 1.4309 | 1 | 1 |
| 182 | eca_resnet | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 1.1822 | 1 | 1 |
| 183 | eca_resnet | focal_loss | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 34.55 | 1.2089 | 1 | 1 |
| 184 | hrnet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 2.9494 | 1 | 1 |
| 185 | hrnet | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 1.7756 | 1 | 1 |
| 186 | hrnet | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 34.55 | 4.2718 | 1 | 1 |
| 187 | hrnet | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.001 | 34.55 | 1.7234 | 1 | 1 |
| 188 | hrnet | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 4.0997 | 1 | 1 |
| 189 | hrnet | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 1.7804 | 1 | 1 |
| 190 | hrnet | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 1.7326 | 1 | 1 |
| 191 | hrnet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 3.2474 | 1 | 1 |
| 192 | hrnet | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 19.3373 | 1 | 1 |
| 193 | mobilenetv2 | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 34.55 | 1.8457 | 1 | 1 |
| 194 | mobilenetv2 | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 34.55 | 1.7934 | 1 | 1 |
| 195 | mobilenetv2 | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 58.3433 | 1 | 1 |
| 196 | resnet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 1.8337 | 1 | 1 |
| 197 | resnet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 6.3239 | 1 | 1 |
| 198 | resnet | label_smoothing | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 34.55 | 1.7498 | 1 | 1 |
| 199 | resnet | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 34.55 | 2.1410 | 1 | 1 |
| 200 | resnet | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 34.55 | 1.1765 | 1 | 1 |
| 201 | resnet | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 73.8032 | 1 | 1 |
| 202 | van | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 1.7167 | 1 | 1 |
| 203 | van | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 34.55 | 1.7731 | 1 | 1 |
| 204 | van | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 1.5550 | 1 | 1 |
| 205 | van | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 2.6527 | 1 | 1 |
| 206 | van | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 1.6713 | 1 | 1 |
| 207 | van | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 1.6379 | 1 | 1 |
| 208 | van | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 34.55 | 1.6150 | 1 | 1 |
| 209 | van | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 34.55 | 1.2243 | 1 | 1 |
| 210 | van | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 1.2404 | 1 | 1 |
| 211 | van | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 1.2115 | 1 | 1 |
| 212 | van | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 1.0741 | 1 | 1 |
| 213 | alexnet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 1.5533 | 1 | 1 |
| 214 | alexnet | label_smoothing | adam | batch_size=16, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.7240 | 1 | 1 |
| 215 | alexnet | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 1.7904 | 1 | 1 |
| 216 | alexnet | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 34.55 | 1.7604 | 1 | 1 |
| 217 | alexnet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 1.4029e+10 | 1 | 1 |
| 218 | alexnet | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 34.55 | 1.1958 | 1 | 1 |
| 219 | alexnet | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 34.55 | 1.0536 | 1 | 1 |
| 220 | darknet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 1.6288 | 1 | 1 |
| 221 | darknet | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 1.6366 | 1 | 1 |
| 222 | darknet | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 2.0774 | 1 | 1 |
| 223 | darknet | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 1.4872e+07 | 1 | 1 |
| 224 | darknet | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 1.5176 | 1 | 1 |
| 225 | darknet | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 128548.7143 | 1 | 1 |
| 226 | googlenet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 34.55 | 1.7178 | 1 | 1 |
| 227 | googlenet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 1.6830 | 1 | 1 |
| 228 | googlenet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 1.7388 | 1 | 1 |
| 229 | googlenet | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 2.9507 | 1 | 1 |
| 230 | googlenet | label_smoothing | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.001 | 34.55 | 1.7614 | 1 | 1 |
| 231 | googlenet | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.001 | 34.55 | 1.2029 | 1 | 1 |
| 232 | googlenet | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 1.1794 | 1 | 1 |
| 233 | googlenet | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 34.55 | 1.1900 | 1 | 1 |
| 234 | lstm | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 1.6382 | 1 | 1 |
| 235 | lstm | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 1.5472 | 1 | 1 |
| 236 | lstm | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 34.55 | 1.7672 | 1 | 1 |
| 237 | lstm | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 1.7519 | 1 | 1 |
| 238 | lstm | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 34.55 | 1.6381 | 1 | 1 |
| 239 | lstm | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 1.2258 | 1 | 1 |
| 240 | lstm | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 1.1601 | 1 | 1 |
| 241 | lstm | focal_loss | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 34.55 | 1.1236 | 1 | 1 |
| 242 | regnet | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 34.55 | 1.7909 | 1 | 1 |
| 243 | regnet | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 1.7344 | 1 | 1 |
| 244 | regnet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 10625.5603 | 1 | 1 |
| 245 | regnet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.1791 | 1 | 1 |
| 246 | regnet | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 1.2104 | 1 | 1 |
| 247 | regnet | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 1.0648 | 1 | 1 |
| 248 | simple_cnn | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.6950 | 1 | 1 |
| 249 | simple_cnn | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 1.7076 | 1 | 1 |
| 250 | simple_cnn | label_smoothing | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 34.55 | 1.7582 | 1 | 1 |
| 251 | simple_cnn | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 1.7618 | 1 | 1 |
| 252 | simple_cnn | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 34.55 | 1.6964 | 1 | 1 |
| 253 | simple_cnn | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 34.55 | 1.2157 | 1 | 1 |
| 254 | simple_cnn | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.1899 | 1 | 1 |
| 255 | wide_resnet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 28.5812 | 1 | 1 |
| 256 | wide_resnet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 11803.9856 | 1 | 1 |
| 257 | wide_resnet | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 1.6979 | 1 | 1 |
| 258 | wide_resnet | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 34.55 | 31.5893 | 1 | 1 |
| 259 | wide_resnet | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 34.55 | 1.1270 | 1 | 1 |
| 260 | wide_resnet | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 1.1961 | 1 | 1 |
| 261 | wide_resnet | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 34.55 | 1.2483 | 1 | 1 |
| 262 | bert | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 34.55 | 1.7854 | 1 | 1 |
| 263 | bert | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 1.7580 | 1 | 1 |
| 264 | bert | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 1.7659 | 1 | 1 |
| 265 | bert | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 34.55 | 1.7700 | 1 | 1 |
| 266 | bert | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 34.55 | 1.7676 | 1 | 1 |
| 267 | bert | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 34.55 | 1.7894 | 1 | 1 |
| 268 | bert | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.7862 | 1 | 1 |
| 269 | bert | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.7751 | 1 | 1 |
| 270 | bert | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 34.55 | 1.2369 | 1 | 1 |
| 271 | bert | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.2427 | 1 | 1 |
| 272 | bert | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 34.55 | 1.2381 | 1 | 1 |
| 273 | bert | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 1.0661 | 1 | 1 |
| 274 | deit | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 2.0203 | 1 | 1 |
| 275 | deit | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 34.55 | 1.5554 | 1 | 1 |
| 276 | deit | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 1.7865 | 1 | 1 |
| 277 | deit | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.6803 | 1 | 1 |
| 278 | deit | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 1.6821 | 1 | 1 |
| 279 | deit | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 34.55 | 1.5893 | 1 | 1 |
| 280 | deit | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 34.55 | 1.6074 | 1 | 1 |
| 281 | deit | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 34.55 | 0.9366 | 1 | 1 |
| 282 | deit | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 34.55 | 0.9887 | 1 | 1 |
| 283 | gpt | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 1.5944 | 1 | 1 |
| 284 | gpt | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 1.7805 | 1 | 1 |
| 285 | gpt | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 34.55 | 1.5556 | 1 | 1 |
| 286 | gpt | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.7743 | 1 | 1 |
| 287 | gpt | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 34.55 | 1.7739 | 1 | 1 |
| 288 | gpt | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 34.55 | 1.7632 | 1 | 1 |
| 289 | gpt | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 34.55 | 1.5579 | 1 | 1 |
| 290 | gpt | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 1.6613 | 1 | 1 |
| 291 | gpt | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 34.55 | 1.2092 | 1 | 1 |
| 292 | gpt | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 1.1941 | 1 | 1 |
| 293 | gpt | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 0.9805 | 1 | 1 |
| 294 | gpt | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 34.55 | 1.0928 | 1 | 1 |
| 295 | mlp | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 9.7439 | 1 | 1 |
| 296 | mlp | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.001 | 34.55 | 2.0035 | 1 | 1 |
| 297 | mlp | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 18.6430 | 1 | 1 |
| 298 | mlp | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 15.7472 | 1 | 1 |
| 299 | mlp | focal_loss | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.0968 | 1 | 1 |
| 300 | repghost | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 22.4339 | 1 | 1 |
| 301 | repghost | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 34.55 | 1.7710 | 1 | 1 |
| 302 | repghost | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 1.7048 | 1 | 1 |
| 303 | repghost | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 1.7935 | 1 | 1 |
| 304 | sknet | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 34.55 | 1.8116 | 1 | 1 |
| 305 | sknet | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 1.7075 | 1 | 1 |
| 306 | sknet | cross_entropy | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 34.55 | 68503.3029 | 1 | 1 |
| 307 | sknet | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 34.55 | 2.6335 | 1 | 1 |
| 308 | sknet | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 34.55 | 1.7172 | 1 | 1 |
| 309 | sknet | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 34.55 | 1.8193 | 1 | 1 |
| 310 | sknet | label_smoothing | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 34.55 | 1.8918 | 1 | 1 |
| 311 | sknet | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 34.55 | 1.1977 | 1 | 1 |
| 312 | sknet | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 63.0320 | 1 | 1 |
| 313 | sknet | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.2437 | 1 | 1 |
| 314 | xception | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 34.55 | 1.6489 | 1 | 1 |
| 315 | xception | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 1.7840 | 1 | 1 |
| 316 | xception | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.001 | 34.55 | 1.7875 | 1 | 1 |
| 317 | xception | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 34.55 | 1.7894 | 1 | 1 |
| 318 | xception | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 1.7890 | 1 | 1 |
| 319 | xception | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 34.55 | 1.7854 | 1 | 1 |
| 320 | xception | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 34.55 | 172605.1244 | 1 | 1 |
| 321 | xception | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 34.55 | 2.1275 | 1 | 1 |
| 322 | xception | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 34.55 | 1.2321 | 1 | 1 |
| 323 | xception | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 34.55 | 1.0043 | 1 | 1 |
| 324 | xception | focal_loss | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.001 | 34.55 | 1.2408 | 1 | 1 |
| 325 | coord_resnet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 32.73 | 1.6785 | 1 | 1 |
| 326 | wide_resnet | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 32.73 | 1.2173 | 1 | 1 |
| 327 | mlp | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 32.73 | 1.7580 | 1 | 1 |
| 328 | cbam_resnet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 30.91 | 1.5759 | 1 | 1 |
| 329 | mlp | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 30.91 | 1.7977 | 1 | 1 |
| 330 | cspnet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 29.09 | 4588.2034 | 1 | 1 |
| 331 | cspnet | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 29.09 | 13.6282 | 1 | 1 |
| 332 | cspnet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 29.09 | 351.3337 | 1 | 1 |
| 333 | cspnet | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 29.09 | 1.2054 | 1 | 1 |
| 334 | cspnet | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 29.09 | 6.1013e+08 | 1 | 1 |
| 335 | ghostnet | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 29.09 | 1.7785 | 1 | 1 |
| 336 | ghostnet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 29.09 | 50.9300 | 1 | 1 |
| 337 | lenet | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 29.09 | 1.7631 | 1 | 1 |
| 338 | lenet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 29.09 | 1.2433 | 1 | 1 |
| 339 | lenet | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 29.09 | 1.2214 | 1 | 1 |
| 340 | poolformer | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 29.09 | 42.4962 | 1 | 1 |
| 341 | poolformer | focal_loss | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 29.09 | 39.9282 | 1 | 1 |
| 342 | shufflenet | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 29.09 | 1.7671 | 1 | 1 |
| 343 | shufflenet | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.001 | 29.09 | 1.7553 | 1 | 1 |
| 344 | coord_resnet | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 29.09 | 1.8554 | 1 | 1 |
| 345 | coord_resnet | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 29.09 | 946.1701 | 1 | 1 |
| 346 | coord_resnet | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 29.09 | 65981.5139 | 1 | 1 |
| 347 | coord_resnet | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 29.09 | 1.1980 | 1 | 1 |
| 348 | efficientnetv2 | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 29.09 | 1.7749 | 1 | 1 |
| 349 | efficientnetv2 | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 29.09 | 1.2871 | 1 | 1 |
| 350 | lcnet | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.001 | 29.09 | 1.2140 | 1 | 1 |
| 351 | nin | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 29.09 | 1.6348 | 1 | 1 |
| 352 | se_resnet | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 29.09 | 1.7791 | 1 | 1 |
| 353 | se_resnet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 29.09 | 2.0152 | 1 | 1 |
| 354 | se_resnet | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 29.09 | 2.5627 | 1 | 1 |
| 355 | se_resnet | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 29.09 | 1209.7911 | 1 | 1 |
| 356 | vim_tiny | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 29.09 | 1.7791 | 1 | 1 |
| 357 | vim_tiny | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 29.09 | 1.7988 | 1 | 1 |
| 358 | convnext | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 29.09 | 1.9755 | 1 | 1 |
| 359 | convnext | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 29.09 | 2.5539 | 1 | 1 |
| 360 | efficientnet | cross_entropy | sgd | batch_size=16, lr=0.0001, weight_decay=0.0 | 29.09 | 1.7822 | 1 | 1 |
| 361 | inception_resnet | cross_entropy | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 29.09 | 413.5407 | 1 | 1 |
| 362 | inception_resnet | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 29.09 | 1.7455 | 1 | 1 |
| 363 | inception_resnet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 29.09 | 336012.1077 | 1 | 1 |
| 364 | mobilenetv3 | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 29.09 | 1.7791 | 1 | 1 |
| 365 | mobilenetv3 | cross_entropy | sgd | batch_size=16, lr=0.0001, weight_decay=0.0 | 29.09 | 1.7826 | 1 | 1 |
| 366 | mobilenetv3 | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 29.09 | 1.6331 | 1 | 1 |
| 367 | mobilenetv3 | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 29.09 | 1.7807 | 1 | 1 |
| 368 | mobilenetv3 | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 29.09 | 1.2311 | 1 | 1 |
| 369 | resnext | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 29.09 | 1.7207 | 1 | 1 |
| 370 | resnext | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 29.09 | 1.7456 | 1 | 1 |
| 371 | resnext | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 29.09 | 1.2719 | 1 | 1 |
| 372 | resnext | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 29.09 | 1.1613 | 1 | 1 |
| 373 | vgg | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 29.09 | 141.1934 | 1 | 1 |
| 374 | vgg | label_smoothing | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 29.09 | 1.7877 | 1 | 1 |
| 375 | vgg | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 29.09 | 1.6730 | 1 | 1 |
| 376 | cbam_resnet | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 29.09 | 1.7691 | 1 | 1 |
| 377 | cbam_resnet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 29.09 | 1.9723 | 1 | 1 |
| 378 | cbam_resnet | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 29.09 | 1.7823 | 1 | 1 |
| 379 | dpn | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 29.09 | 1.4864 | 1 | 1 |
| 380 | dpn | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 29.09 | 1.7782 | 1 | 1 |
| 381 | dpn | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 29.09 | 3156.2321 | 1 | 1 |
| 382 | dpn | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 29.09 | 1.1686 | 1 | 1 |
| 383 | hardnet | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 29.09 | 1.7534 | 1 | 1 |
| 384 | hardnet | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 29.09 | 1.8090 | 1 | 1 |
| 385 | hardnet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 29.09 | 1.0132 | 1 | 1 |
| 386 | hardnet | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 29.09 | 1.2374 | 1 | 1 |
| 387 | hardnet | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 29.09 | 3.2867e+10 | 1 | 1 |
| 388 | mobilenet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 29.09 | 1.7235 | 1 | 1 |
| 389 | res2net | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 29.09 | 1.1180e+06 | 1 | 1 |
| 390 | res2net | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 29.09 | 8.8378e+14 | 1 | 1 |
| 391 | res2net | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 29.09 | 845.5398 | 1 | 1 |
| 392 | res2net | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 29.09 | 1.1990 | 1 | 1 |
| 393 | res2net | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 29.09 | 1.2499 | 1 | 1 |
| 394 | swin_tiny | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 29.09 | 7.5173 | 1 | 1 |
| 395 | swin_tiny | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 29.09 | 0.9163 | 1 | 1 |
| 396 | densenet | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 29.09 | 1.5586 | 1 | 1 |
| 397 | densenet | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 29.09 | 6.4893e+12 | 1 | 1 |
| 398 | gru | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 29.09 | 1.6959 | 1 | 1 |
| 399 | gru | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 29.09 | 1.7765 | 1 | 1 |
| 400 | gru | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 29.09 | 1.6656 | 1 | 1 |
| 401 | gru | label_smoothing | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 29.09 | 1.7770 | 1 | 1 |
| 402 | repvgg | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 29.09 | 195.1715 | 1 | 1 |
| 403 | squeezenet | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 29.09 | 1.7729 | 1 | 1 |
| 404 | squeezenet | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 29.09 | 1.1578 | 1 | 1 |
| 405 | squeezenet | focal_loss | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 29.09 | 1.2242 | 1 | 1 |
| 406 | coatnet | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 29.09 | 1.7871 | 1 | 1 |
| 407 | eca_resnet | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 29.09 | 1.7304 | 1 | 1 |
| 408 | eca_resnet | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 29.09 | 1.7475 | 1 | 1 |
| 409 | eca_resnet | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 29.09 | 1.5815 | 1 | 1 |
| 410 | eca_resnet | focal_loss | adamw | batch_size=16, lr=0.0001, weight_decay=0.0001 | 29.09 | 1.2093 | 1 | 1 |
| 411 | hrnet | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 29.09 | 1.2171 | 1 | 1 |
| 412 | resnet | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 29.09 | 13.7119 | 1 | 1 |
| 413 | resnet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 29.09 | 13832.1590 | 1 | 1 |
| 414 | van | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.0001 | 29.09 | 1.7967 | 1 | 1 |
| 415 | alexnet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 29.09 | 26.4015 | 1 | 1 |
| 416 | alexnet | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 29.09 | 1.5415 | 1 | 1 |
| 417 | alexnet | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 29.09 | 98.4539 | 1 | 1 |
| 418 | alexnet | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 29.09 | 2.9076e+09 | 1 | 1 |
| 419 | darknet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 29.09 | 1.1999e+11 | 1 | 1 |
| 420 | darknet | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 29.09 | 1.5008 | 1 | 1 |
| 421 | lstm | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 29.09 | 1.7909 | 1 | 1 |
| 422 | lstm | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0 | 29.09 | 1.5382 | 1 | 1 |
| 423 | regnet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 29.09 | 1.6309 | 1 | 1 |
| 424 | regnet | label_smoothing | adamw | batch_size=16, lr=0.0001, weight_decay=0.0 | 29.09 | 1.7337 | 1 | 1 |
| 425 | simple_cnn | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 29.09 | 3.1817 | 1 | 1 |
| 426 | wide_resnet | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 29.09 | 62112.4538 | 1 | 1 |
| 427 | deit | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 29.09 | 1.5629 | 1 | 1 |
| 428 | deit | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 29.09 | 1.6586 | 1 | 1 |
| 429 | mlp | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 29.09 | 1.1619 | 1 | 1 |
| 430 | repghost | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 29.09 | 1.7859 | 1 | 1 |
| 431 | repghost | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 29.09 | 1.7260 | 1 | 1 |
| 432 | repghost | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 29.09 | 1.7801 | 1 | 1 |
| 433 | repghost | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 29.09 | 1.2048 | 1 | 1 |
| 434 | repghost | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 29.09 | 1.1776 | 1 | 1 |
| 435 | xception | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 29.09 | 2.2017 | 1 | 1 |
| 436 | se_resnet | cross_entropy | adam | batch_size=16, lr=0.001, weight_decay=0.001 | 27.27 | 1.7903 | 1 | 1 |
| 437 | resnext | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 27.27 | 1.6380 | 1 | 1 |
| 438 | cbam_resnet | focal_loss | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 27.27 | 1.2706 | 1 | 1 |
| 439 | mobilenet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 27.27 | 3.5790 | 1 | 1 |
| 440 | mobilenetv2 | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 27.27 | 29.9875 | 1 | 1 |
| 441 | darknet | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 27.27 | 1.0868e+06 | 1 | 1 |
| 442 | mlp | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 27.27 | 1.2104 | 1 | 1 |
| 443 | vim_tiny | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 25.45 | 1.5899 | 1 | 1 |
| 444 | mlp | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 25.45 | 1.7633 | 1 | 1 |
| 445 | cspnet | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 23.64 | 1.7967 | 1 | 1 |
| 446 | cspnet | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 23.64 | 3.7031e+07 | 1 | 1 |
| 447 | cspnet | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 23.64 | 1.7961 | 1 | 1 |
| 448 | cspnet | label_smoothing | adamw | batch_size=16, lr=0.0001, weight_decay=0.001 | 23.64 | 1.7880 | 1 | 1 |
| 449 | ghostnet | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 23.64 | 1.8079 | 1 | 1 |
| 450 | ghostnet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 23.64 | 4.2306 | 1 | 1 |
| 451 | ghostnet | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 23.64 | 1.7623 | 1 | 1 |
| 452 | ghostnet | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 23.64 | 1.2602 | 1 | 1 |
| 453 | lenet | cross_entropy | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 23.64 | 1.7973 | 1 | 1 |
| 454 | poolformer | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 23.64 | 77.3183 | 1 | 1 |
| 455 | poolformer | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 23.64 | 3.2572 | 1 | 1 |
| 456 | shufflenet | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 23.64 | 1.2471 | 1 | 1 |
| 457 | coord_resnet | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 23.64 | 44133.6487 | 1 | 1 |
| 458 | efficientnetv2 | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 23.64 | 1.7922 | 1 | 1 |
| 459 | efficientnetv2 | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 23.64 | 1.8141 | 1 | 1 |
| 460 | efficientnetv2 | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 23.64 | 1.7751 | 1 | 1 |
| 461 | efficientnetv2 | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 23.64 | 1.7245 | 1 | 1 |
| 462 | efficientnetv2 | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 23.64 | 1.7204 | 1 | 1 |
| 463 | lcnet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 23.64 | 1.7958 | 1 | 1 |
| 464 | nin | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 23.64 | 1.7473 | 1 | 1 |
| 465 | se_resnet | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 23.64 | 1.7226 | 1 | 1 |
| 466 | se_resnet | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 23.64 | 1.3409 | 1 | 1 |
| 467 | se_resnet | focal_loss | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 23.64 | 1.2273 | 1 | 1 |
| 468 | vim_tiny | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 23.64 | 108579.9269 | 1 | 1 |
| 469 | vim_tiny | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 23.64 | 346218.2868 | 1 | 1 |
| 470 | vim_tiny | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 23.64 | 1.8036 | 1 | 1 |
| 471 | vim_tiny | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 23.64 | 2.4006 | 1 | 1 |
| 472 | vim_tiny | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 23.64 | 1.3011 | 1 | 1 |
| 473 | convnext | label_smoothing | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 23.64 | 1.5947 | 1 | 1 |
| 474 | convnext | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 23.64 | 2.6912 | 1 | 1 |
| 475 | convnext | focal_loss | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 23.64 | 1.6106 | 1 | 1 |
| 476 | convnext | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 23.64 | 2.5166 | 1 | 1 |
| 477 | inception_resnet | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.0001 | 23.64 | 1.8036 | 1 | 1 |
| 478 | inception_resnet | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 23.64 | 1.7545 | 1 | 1 |
| 479 | inception_resnet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 23.64 | 3237.4437 | 1 | 1 |
| 480 | inception_resnet | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 23.64 | 391646.0324 | 1 | 1 |
| 481 | mobilenetv3 | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 23.64 | 1.7862 | 1 | 1 |
| 482 | mobilenetv3 | label_smoothing | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 23.64 | 1.7832 | 1 | 1 |
| 483 | mobilenetv3 | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 23.64 | 9.6405e+07 | 1 | 1 |
| 484 | resnext | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0 | 23.64 | 1.7427 | 1 | 1 |
| 485 | resnext | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 23.64 | 1.7734 | 1 | 1 |
| 486 | resnext | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 23.64 | 1.7970 | 1 | 1 |
| 487 | vgg | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 23.64 | 1.7864 | 1 | 1 |
| 488 | vgg | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 23.64 | 3.7056e+11 | 1 | 1 |
| 489 | vgg | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 23.64 | 1.2403 | 1 | 1 |
| 490 | cbam_resnet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 23.64 | 1.2282 | 1 | 1 |
| 491 | dpn | cross_entropy | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 23.64 | 16.4828 | 1 | 1 |
| 492 | dpn | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 23.64 | 1.2349 | 1 | 1 |
| 493 | hardnet | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 23.64 | 1.7729 | 1 | 1 |
| 494 | mobilenet | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 23.64 | 1.7642 | 1 | 1 |
| 495 | mobilenet | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 23.64 | 1.7283 | 1 | 1 |
| 496 | res2net | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.001 | 23.64 | 1.7816 | 1 | 1 |
| 497 | swin_tiny | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 23.64 | 4.2633 | 1 | 1 |
| 498 | densenet | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.0001 | 23.64 | 1.7871 | 1 | 1 |
| 499 | densenet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 23.64 | 6.1713 | 1 | 1 |
| 500 | mnasnet | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 23.64 | 4.3131 | 1 | 1 |
| 501 | mnasnet | focal_loss | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.001 | 23.64 | 1.3025 | 1 | 1 |
| 502 | repvgg | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.0001 | 23.64 | 1.7637 | 1 | 1 |
| 503 | repvgg | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 23.64 | 1.7522 | 1 | 1 |
| 504 | repvgg | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 23.64 | 1.7331 | 1 | 1 |
| 505 | repvgg | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 23.64 | 1.6511 | 1 | 1 |
| 506 | repvgg | label_smoothing | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 23.64 | 1.7926 | 1 | 1 |
| 507 | repvgg | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 23.64 | 1.2419 | 1 | 1 |
| 508 | repvgg | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 23.64 | 1.2970 | 1 | 1 |
| 509 | squeezenet | label_smoothing | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 23.64 | 1.7418 | 1 | 1 |
| 510 | coatnet | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 23.64 | 12.0655 | 1 | 1 |
| 511 | coatnet | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 23.64 | 1.2987 | 1 | 1 |
| 512 | eca_resnet | cross_entropy | sgd | batch_size=16, lr=0.0001, weight_decay=0.0001 | 23.64 | 1.7606 | 1 | 1 |
| 513 | eca_resnet | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 23.64 | 1.9648 | 1 | 1 |
| 514 | mobilenetv2 | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 23.64 | 1.8214 | 1 | 1 |
| 515 | mobilenetv2 | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 23.64 | 1.7865 | 1 | 1 |
| 516 | mobilenetv2 | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 23.64 | 1.7928 | 1 | 1 |
| 517 | mobilenetv2 | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 23.64 | 2.6261 | 1 | 1 |
| 518 | mobilenetv2 | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 23.64 | 1.2874 | 1 | 1 |
| 519 | resnet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 23.64 | 2.0897 | 1 | 1 |
| 520 | resnet | cross_entropy | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 23.64 | 5.8333e+06 | 1 | 1 |
| 521 | resnet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 23.64 | 1.8852e+08 | 1 | 1 |
| 522 | resnet | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 23.64 | 41176.2087 | 1 | 1 |
| 523 | alexnet | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 23.64 | 1.7874 | 1 | 1 |
| 524 | darknet | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 23.64 | 1.3400 | 1 | 1 |
| 525 | darknet | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 23.64 | 1.3759 | 1 | 1 |
| 526 | googlenet | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 23.64 | 1.7766 | 1 | 1 |
| 527 | googlenet | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 23.64 | 1.7576 | 1 | 1 |
| 528 | googlenet | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 23.64 | 1.2390 | 1 | 1 |
| 529 | regnet | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 23.64 | 1.8010 | 1 | 1 |
| 530 | regnet | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 23.64 | 1.6775 | 1 | 1 |
| 531 | regnet | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 23.64 | 1.7959 | 1 | 1 |
| 532 | regnet | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 23.64 | 1.1693 | 1 | 1 |
| 533 | simple_cnn | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 23.64 | 1.1788 | 1 | 1 |
| 534 | simple_cnn | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 23.64 | 2.3248 | 1 | 1 |
| 535 | wide_resnet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 23.64 | 13052.0170 | 1 | 1 |
| 536 | deit | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 23.64 | 0.9386 | 1 | 1 |
| 537 | repghost | cross_entropy | adam | batch_size=16, lr=0.001, weight_decay=0.001 | 23.64 | 1.8473 | 1 | 1 |
| 538 | repghost | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 23.64 | 1.7703 | 1 | 1 |
| 539 | repghost | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 23.64 | 1.3311 | 1 | 1 |
| 540 | sknet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 23.64 | 40.9212 | 1 | 1 |
| 541 | coord_resnet | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 21.82 | 1.0704 | 1 | 1 |
| 542 | eca_resnet | cross_entropy | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 20.00 | 1.8560 | 1 | 1 |
| 543 | efficientnet | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 14.55 | 834.3833 | 1 | 1 |
| 544 | cspnet | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.001 | 5.45 | 1.7921 | 1 | 1 |
| 545 | ghostnet | focal_loss | adamw | batch_size=16, lr=0.0001, weight_decay=0.0001 | 5.45 | 1.2770 | 1 | 1 |
| 546 | lenet | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 5.45 | 1.7767 | 1 | 1 |
| 547 | lenet | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 5.45 | 1.2688 | 1 | 1 |
| 548 | efficientnetv2 | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 5.45 | 1.2890 | 1 | 1 |
| 549 | vim_tiny | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 5.45 | 2.0548 | 1 | 1 |
| 550 | vim_tiny | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 5.45 | 1.4187 | 1 | 1 |
| 551 | convnext | focal_loss | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 5.45 | 2.5478 | 1 | 1 |
| 552 | efficientnet | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 5.45 | 1.8067 | 1 | 1 |
| 553 | efficientnet | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 5.45 | 1.8058 | 1 | 1 |
| 554 | inception_resnet | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 5.45 | 1.9367 | 1 | 1 |
| 555 | hardnet | label_smoothing | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 5.45 | 1.8209 | 1 | 1 |
| 556 | res2net | cross_entropy | adamw | batch_size=16, lr=0.0001, weight_decay=0.001 | 5.45 | 1.7986 | 1 | 1 |
| 557 | res2net | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 5.45 | 1.2604 | 1 | 1 |
| 558 | capsnet | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 5.45 | 1.7918 | 1 | 1 |
| 559 | mnasnet | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 5.45 | 1.2614 | 1 | 1 |
| 560 | repvgg | label_smoothing | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 5.45 | 2.0484 | 1 | 1 |
| 561 | coatnet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 5.45 | 2.1445 | 1 | 1 |
| 562 | coatnet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 5.45 | 1.8715 | 1 | 1 |
| 563 | coatnet | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 5.45 | 1.9457 | 1 | 1 |
| 564 | coatnet | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 5.45 | 2.0148 | 1 | 1 |
| 565 | coatnet | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 5.45 | 1.8545 | 1 | 1 |
| 566 | coatnet | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 5.45 | 1.9588 | 1 | 1 |
| 567 | coatnet | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 5.45 | 1.2265 | 1 | 1 |
| 568 | coatnet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 5.45 | 1.2354 | 1 | 1 |
| 569 | hrnet | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 5.45 | 1.7678 | 1 | 1 |
| 570 | hrnet | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 5.45 | 1.2137 | 1 | 1 |
| 571 | mobilenetv2 | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 5.45 | 1.8219 | 1 | 1 |
| 572 | mobilenetv2 | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 5.45 | 1.5316 | 1 | 1 |
| 573 | mobilenetv2 | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 5.45 | 1.2850 | 1 | 1 |
| 574 | googlenet | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 5.45 | 1.7967 | 1 | 1 |
| 575 | lstm | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 5.45 | 1.2349 | 1 | 1 |
| 576 | wide_resnet | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 5.45 | 1.7977 | 1 | 1 |
| 577 | sknet | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 5.45 | 1.2698 | 1 | 1 |
| 578 | poolformer | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 3.64 | 14.5833 | 1 | 1 |
| 579 | lcnet | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 3.64 | 1.2824 | 1 | 1 |
| 580 | convnext | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 3.64 | 1.4820 | 1 | 1 |
| 581 | efficientnet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 3.64 | 1.7964 | 1 | 1 |
| 582 | mobilenetv3 | focal_loss | adamw | batch_size=16, lr=0.0001, weight_decay=0.001 | 3.64 | 1.2568 | 1 | 1 |
| 583 | dpn | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 3.64 | 1.7971 | 1 | 1 |
| 584 | swin_tiny | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 3.64 | 4.0784 | 1 | 1 |
| 585 | capsnet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 3.64 | 1.7918 | 1 | 1 |
| 586 | capsnet | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 3.64 | 1.7918 | 1 | 1 |
| 587 | capsnet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 3.64 | 1.7918 | 1 | 1 |
| 588 | capsnet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 3.64 | 1.7918 | 1 | 1 |
| 589 | capsnet | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 3.64 | 1.7918 | 1 | 1 |
| 590 | capsnet | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 3.64 | 1.7918 | 1 | 1 |
| 591 | capsnet | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 3.64 | 1.7918 | 1 | 1 |
| 592 | capsnet | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 3.64 | 1.2443 | 1 | 1 |
| 593 | capsnet | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 3.64 | 1.2443 | 1 | 1 |
| 594 | capsnet | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 3.64 | 1.2443 | 1 | 1 |
| 595 | capsnet | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 3.64 | 1.2443 | 1 | 1 |
| 596 | gru | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 3.64 | 1.8180 | 1 | 1 |
| 597 | darknet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 3.64 | 2.4550e+13 | 1 | 1 |
| 598 | lstm | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 3.64 | 1.7887 | 1 | 1 |
| 599 | simple_cnn | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0 | 3.64 | 66.9182 | 1 | 1 |
| 600 | simple_cnn | label_smoothing | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.001 | 3.64 | 2.4095 | 1 | 1 |

## Autoencoder — best per model

| Rank | Model | Loss | Optimizer | Hyperparameters | Recon Loss | Epochs Run | Convergence Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | denoising_ae | mse | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 2 | conv_ae | mse | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 3 | simple_ae | mse | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 4 | vae | mse | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 0.0000 | 1 | 1 |

## Autoencoder — all trials (48 rows)

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
| 13 | conv_ae | mse | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 14 | conv_ae | mse | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 0.0000 | 1 | 1 |
| 15 | conv_ae | mse | adamw | batch_size=16, lr=0.0001, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 16 | conv_ae | mse | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 17 | conv_ae | l1 | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 18 | conv_ae | l1 | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 19 | conv_ae | l1 | adamw | batch_size=16, lr=0.001, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 20 | conv_ae | l1 | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 0.0000 | 1 | 1 |
| 21 | conv_ae | bce | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 0.0000 | 1 | 1 |
| 22 | conv_ae | bce | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 23 | conv_ae | bce | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 24 | conv_ae | bce | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 25 | simple_ae | mse | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 26 | simple_ae | mse | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 27 | simple_ae | mse | adamw | batch_size=16, lr=0.0001, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 28 | simple_ae | mse | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 0.0000 | 1 | 1 |
| 29 | simple_ae | l1 | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 30 | simple_ae | l1 | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 31 | simple_ae | l1 | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 32 | simple_ae | l1 | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 33 | simple_ae | bce | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 34 | simple_ae | bce | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 0.0000 | 1 | 1 |
| 35 | simple_ae | bce | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 36 | simple_ae | bce | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 37 | vae | mse | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 38 | vae | mse | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 39 | vae | mse | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 0.0000 | 1 | 1 |
| 40 | vae | mse | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 41 | vae | l1 | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 42 | vae | l1 | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 43 | vae | l1 | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 44 | vae | l1 | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 45 | vae | bce | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 46 | vae | bce | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 47 | vae | bce | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 48 | vae | bce | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.001 | 0.0000 | 1 | 1 |

## GAN — best per model

| Rank | Model | Loss | Optimizer | Hyperparameters | G Loss | D Loss | Epochs Run | Convergence Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | wgan | wasserstein | adam | batch_size=16, lr=0.0001, weight_decay=0.001 | 0.0050 | -0.0018 | 1 | 1 |
| 2 | cgan | bce | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 0.6896 | 1.3966 | 1 | 1 |
| 3 | vanilla_gan | bce | sgd | batch_size=16, lr=0.0001, weight_decay=0.0001 | 0.7033 | 1.3840 | 1 | 1 |
| 4 | dcgan | bce | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 0.7772 | 1.3194 | 1 | 1 |

## GAN — all trials (28 rows)

| Rank | Model | Loss | Optimizer | Hyperparameters | G Loss | D Loss | Epochs Run | Convergence Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | wgan | wasserstein | adam | batch_size=16, lr=0.0001, weight_decay=0.001 | 0.0050 | -0.0018 | 1 | 1 |
| 2 | wgan | bce | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 0.1490 | -0.1267 | 1 | 1 |
| 3 | wgan | wasserstein | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 0.2185 | -0.1480 | 1 | 1 |
| 4 | wgan | bce | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 0.2297 | -0.1421 | 1 | 1 |
| 5 | cgan | bce | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 0.6896 | 1.3966 | 1 | 1 |
| 6 | cgan | wasserstein | sgd | batch_size=16, lr=0.0001, weight_decay=0.0001 | 0.6951 | 1.4235 | 1 | 1 |
| 7 | vanilla_gan | bce | sgd | batch_size=16, lr=0.0001, weight_decay=0.0001 | 0.7033 | 1.3840 | 1 | 1 |
| 8 | vanilla_gan | wasserstein | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 0.7171 | 1.1140 | 1 | 1 |
| 9 | cgan | bce | adam | batch_size=16, lr=0.0001, weight_decay=0.0001 | 0.7174 | 1.3938 | 1 | 1 |
| 10 | cgan | wasserstein | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 0.7500 | 1.4104 | 1 | 1 |
| 11 | dcgan | bce | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 0.7772 | 1.3194 | 1 | 1 |
| 12 | vanilla_gan | bce | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 0.8415 | 0.8256 | 1 | 1 |
| 13 | dcgan | wasserstein | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 0.8433 | 1.3962 | 1 | 1 |
| 14 | vanilla_gan | wasserstein | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 0.9037 | 6.5144 | 1 | 1 |
| 15 | cgan | bce | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 0.9373 | 1.8599 | 1 | 1 |
| 16 | vanilla_gan | wasserstein | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 0.9857 | 6.6303 | 1 | 1 |
| 17 | vanilla_gan | bce | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 1.0276 | 0.6367 | 1 | 1 |
| 18 | cgan | bce | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 1.2620 | 2.0084 | 1 | 1 |
| 19 | cgan | wasserstein | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 1.2666 | 21.2528 | 1 | 1 |
| 20 | dcgan | bce | adamw | batch_size=16, lr=0.0001, weight_decay=0.0001 | 1.2973 | 1.2011 | 1 | 1 |
| 21 | vanilla_gan | wasserstein | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 1.3566 | 2.0477 | 1 | 1 |
| 22 | dcgan | wasserstein | adam | batch_size=16, lr=0.0001, weight_decay=0.001 | 1.4805 | 1.2073 | 1 | 1 |
| 23 | vanilla_gan | bce | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 4.3356 | 87.6696 | 1 | 1 |
| 24 | cgan | wasserstein | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 5.1011 | 9.9451 | 1 | 1 |
| 25 | dcgan | bce | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 5.3363 | 0.6425 | 1 | 1 |
| 26 | dcgan | bce | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 5.7036 | 0.5683 | 1 | 1 |
| 27 | dcgan | wasserstein | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 6.4538 | 0.9820 | 1 | 1 |
| 28 | dcgan | wasserstein | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 11.8365 | 0.6246 | 1 | 1 |

## Search Space

- Loss (classification): cross_entropy, label_smoothing, focal_loss
- Loss (autoencoder): mse, l1, bce
- Loss (GAN): bce, wasserstein (informational; GANs use fixed objectives)
- Optimizers: adam, sgd, adamw, rmsprop
- Hyperparameters: {"lr": [0.0001, 0.001, 0.01], "batch_size": [8, 16, 32], "weight_decay": [0.0, 0.0001, 0.001]}
