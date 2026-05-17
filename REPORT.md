# MNIST Regression Report

Generated: 2026-05-17T14:44:30
Mode: quick-test
Max epochs: 1 | Early-stop patience: 3 | Min delta: 0.1 | NAS trials per config: 2 | Workers: 8 | Max batch: 16
Total wall time: 372.3s

Training stops when validation metric shows no significant improvement for 3 consecutive epochs.

## Classification Models (ranked by test accuracy)

| Rank | Model | Loss | Optimizer | Hyperparameters | Test Acc (%) | Test Loss | Epochs Run | Convergence Epoch |
|------|-------|------|-----------|-----------------|--------------|-----------|------------|-------------------|
| 1 | ghostnet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 45.00 | 1.7306 | 1 | 1 |
| 2 | cbam_resnet | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 41.00 | 2.1137 | 1 | 1 |
| 3 | coord_resnet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 35.00 | 2.1523 | 1 | 1 |
| 4 | cbam_resnet | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 33.00 | 2.0661 | 1 | 1 |
| 5 | cbam_resnet | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 32.00 | 1.6441 | 1 | 1 |
| 6 | hrnet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 32.00 | 1.6827 | 1 | 1 |
| 7 | vgg | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 31.00 | 1.8485 | 1 | 1 |
| 8 | vgg | focal_loss | adamw | batch_size=16, lr=0.0001, weight_decay=0.0001 | 31.00 | 1.8435 | 1 | 1 |
| 9 | se_resnet | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 27.00 | 2.3265 | 1 | 1 |
| 10 | convnext | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 27.00 | 2.1464 | 1 | 1 |
| 11 | convnext | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 26.00 | 2.0978 | 1 | 1 |
| 12 | squeezenet | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 25.00 | 1.8459 | 1 | 1 |
| 13 | se_resnet | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 25.00 | 2.2178 | 1 | 1 |
| 14 | cbam_resnet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 25.00 | 2.1789 | 1 | 1 |
| 15 | convnext | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 25.00 | 1.5297 | 1 | 1 |
| 16 | repghost | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 24.00 | 1.8049 | 1 | 1 |
| 17 | simple_cnn | label_smoothing | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 24.00 | 2.3005 | 1 | 1 |
| 18 | bert | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 23.00 | 2.3023 | 1 | 1 |
| 19 | repghost | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 23.00 | 2.2404 | 1 | 1 |
| 20 | coord_resnet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 23.00 | 2.6989 | 1 | 1 |
| 21 | dpn | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 23.00 | 63.2937 | 1 | 1 |
| 22 | simple_cnn | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 23.00 | 1.8588 | 1 | 1 |
| 23 | lenet | cross_entropy | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 22.00 | 2.2996 | 1 | 1 |
| 24 | convnext | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 22.00 | 3.3253 | 1 | 1 |
| 25 | convnext | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 22.00 | 4.5517 | 1 | 1 |
| 26 | convnext | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 22.00 | 2.6208 | 1 | 1 |
| 27 | vgg | label_smoothing | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 22.00 | 2.2941 | 1 | 1 |
| 28 | repghost | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 21.00 | 1.8537 | 1 | 1 |
| 29 | repghost | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 21.00 | 2.5523 | 1 | 1 |
| 30 | vit | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 21.00 | 2.2688 | 1 | 1 |
| 31 | dpn | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 21.00 | 64.8902 | 1 | 1 |
| 32 | simple_cnn | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 21.00 | 2.2935 | 1 | 1 |
| 33 | deit | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 20.00 | 2.4262 | 1 | 1 |
| 34 | poolformer | cross_entropy | sgd | batch_size=16, lr=0.0001, weight_decay=0.0001 | 20.00 | 2.3193 | 1 | 1 |
| 35 | eca_resnet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 20.00 | 2.4471 | 1 | 1 |
| 36 | hrnet | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 20.00 | 1.8986 | 1 | 1 |
| 37 | densenet | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 19.00 | 2.3070 | 1 | 1 |
| 38 | bert | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 19.00 | 1.8647 | 1 | 1 |
| 39 | gpt | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 19.00 | 2.3023 | 1 | 1 |
| 40 | coord_resnet | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 19.00 | 1.8605 | 1 | 1 |
| 41 | cbam_resnet | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 19.00 | 2.2594 | 1 | 1 |
| 42 | dpn | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 19.00 | 214.3030 | 1 | 1 |
| 43 | hrnet | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 19.00 | 1.8635 | 1 | 1 |
| 44 | van | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 19.00 | 20.0900 | 1 | 1 |
| 45 | convnext | focal_loss | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 19.00 | 2.2341 | 1 | 1 |
| 46 | simple_cnn | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 19.00 | 1.8554 | 1 | 1 |
| 47 | deit | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 18.00 | 2.4022 | 1 | 1 |
| 48 | ghostnet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 18.00 | 2.4271 | 1 | 1 |
| 49 | lenet | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 18.00 | 1.8463 | 1 | 1 |
| 50 | coord_resnet | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 18.00 | 275.4136 | 1 | 1 |
| 51 | eca_resnet | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 18.00 | 2.4897 | 1 | 1 |
| 52 | resnet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 18.00 | 2.4813 | 1 | 1 |
| 53 | van | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 18.00 | nan | 1 | 1 |
| 54 | regnet | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 18.00 | 4.5508 | 1 | 1 |
| 55 | simple_cnn | label_smoothing | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.001 | 18.00 | 2.2885 | 1 | 1 |
| 56 | deit | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 17.00 | 2.2624 | 1 | 1 |
| 57 | cspnet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 17.00 | 15.8413 | 1 | 1 |
| 58 | ghostnet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 17.00 | 1.8422 | 1 | 1 |
| 59 | lenet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 17.00 | 2.2851 | 1 | 1 |
| 60 | eca_resnet | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 17.00 | 2.7092 | 1 | 1 |
| 61 | hrnet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 17.00 | 2.1836 | 1 | 1 |
| 62 | hrnet | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 17.00 | 2.1953 | 1 | 1 |
| 63 | hrnet | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 17.00 | 2.2082 | 1 | 1 |
| 64 | resnext | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 17.00 | 2.4000 | 1 | 1 |
| 65 | vgg | cross_entropy | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 17.00 | 2.3467 | 1 | 1 |
| 66 | darknet | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 17.00 | 21922.9826 | 1 | 1 |
| 67 | simple_cnn | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 17.00 | 2.2486 | 1 | 1 |
| 68 | wide_resnet | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 17.00 | 2.2991 | 1 | 1 |
| 69 | deit | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 16.00 | 1.8274 | 1 | 1 |
| 70 | cspnet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 16.00 | 152.6011 | 1 | 1 |
| 71 | lenet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 16.00 | 2.2859 | 1 | 1 |
| 72 | lenet | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 16.00 | 2.2984 | 1 | 1 |
| 73 | vim_tiny | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 16.00 | 2.2909 | 1 | 1 |
| 74 | hardnet | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 16.00 | 3.9232 | 1 | 1 |
| 75 | hrnet | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 16.00 | 2.3457 | 1 | 1 |
| 76 | van | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 16.00 | nan | 1 | 1 |
| 77 | convnext | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 16.00 | 6.2692 | 1 | 1 |
| 78 | convnext | label_smoothing | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 16.00 | 2.3112 | 1 | 1 |
| 79 | convnext | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 16.00 | 4.1292 | 1 | 1 |
| 80 | densenet | cross_entropy | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.2872 | 1 | 1 |
| 81 | densenet | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 2.2835 | 1 | 1 |
| 82 | densenet | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.2841 | 1 | 1 |
| 83 | densenet | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.2892 | 1 | 1 |
| 84 | densenet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 15.00 | 5.9790 | 1 | 1 |
| 85 | densenet | focal_loss | adamw | batch_size=16, lr=0.0001, weight_decay=0.0001 | 15.00 | 1.8705 | 1 | 1 |
| 86 | densenet | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 1.8523 | 1 | 1 |
| 87 | gru | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 2.2952 | 1 | 1 |
| 88 | gru | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 2.2933 | 1 | 1 |
| 89 | gru | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 15.00 | 2.2915 | 1 | 1 |
| 90 | gru | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 2.2895 | 1 | 1 |
| 91 | gru | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 1.8649 | 1 | 1 |
| 92 | mnasnet | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.2989 | 1 | 1 |
| 93 | mnasnet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 2.3404 | 1 | 1 |
| 94 | mnasnet | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 2.5680 | 1 | 1 |
| 95 | mnasnet | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 15.00 | 4.2222 | 1 | 1 |
| 96 | mnasnet | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 2.2931 | 1 | 1 |
| 97 | mnasnet | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 2.2987 | 1 | 1 |
| 98 | mnasnet | label_smoothing | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 15.00 | 2.3054 | 1 | 1 |
| 99 | mnasnet | focal_loss | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.001 | 15.00 | 1.8545 | 1 | 1 |
| 100 | repvgg | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 2.3062 | 1 | 1 |
| 101 | repvgg | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 15.00 | 2.2973 | 1 | 1 |
| 102 | repvgg | cross_entropy | adamw | batch_size=16, lr=0.0001, weight_decay=0.0 | 15.00 | 2.2961 | 1 | 1 |
| 103 | repvgg | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.3987 | 1 | 1 |
| 104 | repvgg | label_smoothing | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 15.00 | 2.3007 | 1 | 1 |
| 105 | repvgg | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.3754 | 1 | 1 |
| 106 | repvgg | label_smoothing | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.3448 | 1 | 1 |
| 107 | repvgg | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 1.8865 | 1 | 1 |
| 108 | repvgg | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 1.8581 | 1 | 1 |
| 109 | repvgg | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 1.8818 | 1 | 1 |
| 110 | repvgg | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.2178 | 1 | 1 |
| 111 | squeezenet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 2.2984 | 1 | 1 |
| 112 | squeezenet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.2806 | 1 | 1 |
| 113 | squeezenet | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 2.2965 | 1 | 1 |
| 114 | squeezenet | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 2.3010 | 1 | 1 |
| 115 | squeezenet | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.3233 | 1 | 1 |
| 116 | squeezenet | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 2.2790 | 1 | 1 |
| 117 | squeezenet | label_smoothing | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 15.00 | 2.2969 | 1 | 1 |
| 118 | squeezenet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 15.00 | 1.8569 | 1 | 1 |
| 119 | bert | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.001 | 15.00 | 2.3007 | 1 | 1 |
| 120 | bert | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 15.00 | 2.2989 | 1 | 1 |
| 121 | bert | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.001 | 15.00 | 2.2996 | 1 | 1 |
| 122 | bert | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 2.3001 | 1 | 1 |
| 123 | bert | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.2996 | 1 | 1 |
| 124 | bert | label_smoothing | adamw | batch_size=16, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.3019 | 1 | 1 |
| 125 | bert | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 15.00 | 2.3022 | 1 | 1 |
| 126 | bert | focal_loss | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 15.00 | 1.8631 | 1 | 1 |
| 127 | bert | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 1.8625 | 1 | 1 |
| 128 | bert | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 1.8562 | 1 | 1 |
| 129 | deit | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 15.00 | 2.4562 | 1 | 1 |
| 130 | deit | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.2739 | 1 | 1 |
| 131 | deit | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 15.00 | 2.5361 | 1 | 1 |
| 132 | deit | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 2.3244 | 1 | 1 |
| 133 | deit | focal_loss | adamw | batch_size=16, lr=0.0001, weight_decay=0.0001 | 15.00 | 1.8192 | 1 | 1 |
| 134 | gpt | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 2.2947 | 1 | 1 |
| 135 | gpt | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.3006 | 1 | 1 |
| 136 | gpt | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.2989 | 1 | 1 |
| 137 | gpt | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.3005 | 1 | 1 |
| 138 | gpt | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.3005 | 1 | 1 |
| 139 | gpt | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 15.00 | 2.2934 | 1 | 1 |
| 140 | gpt | label_smoothing | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 15.00 | 2.2981 | 1 | 1 |
| 141 | gpt | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 1.8609 | 1 | 1 |
| 142 | gpt | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 15.00 | 1.8607 | 1 | 1 |
| 143 | gpt | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 15.00 | 1.8547 | 1 | 1 |
| 144 | gpt | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 1.8578 | 1 | 1 |
| 145 | repghost | cross_entropy | adam | batch_size=16, lr=0.001, weight_decay=0.001 | 15.00 | 2.3083 | 1 | 1 |
| 146 | repghost | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.2693 | 1 | 1 |
| 147 | repghost | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 15.00 | 2.2757 | 1 | 1 |
| 148 | repghost | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.2859 | 1 | 1 |
| 149 | repghost | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 1.8414 | 1 | 1 |
| 150 | sknet | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 2.3607 | 1 | 1 |
| 151 | sknet | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.3563 | 1 | 1 |
| 152 | sknet | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 2.5139 | 1 | 1 |
| 153 | sknet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 15.00 | 2.3584 | 1 | 1 |
| 154 | sknet | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 2.2992 | 1 | 1 |
| 155 | sknet | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 2.3210 | 1 | 1 |
| 156 | sknet | label_smoothing | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 15.00 | 2.3567 | 1 | 1 |
| 157 | sknet | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 1.9399 | 1 | 1 |
| 158 | sknet | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 15.00 | 1.8926 | 1 | 1 |
| 159 | sknet | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.2205 | 1 | 1 |
| 160 | cspnet | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 15.00 | 2.2979 | 1 | 1 |
| 161 | cspnet | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.4078 | 1 | 1 |
| 162 | cspnet | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 15.00 | 2.3301 | 1 | 1 |
| 163 | cspnet | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.001 | 15.00 | 2.4071 | 1 | 1 |
| 164 | cspnet | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 2.3050 | 1 | 1 |
| 165 | cspnet | label_smoothing | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.3148 | 1 | 1 |
| 166 | cspnet | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 15.00 | 1.8518 | 1 | 1 |
| 167 | cspnet | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.001 | 15.00 | 1.9017 | 1 | 1 |
| 168 | ghostnet | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 2.3600 | 1 | 1 |
| 169 | ghostnet | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 15.00 | 1.8402 | 1 | 1 |
| 170 | ghostnet | focal_loss | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.001 | 15.00 | 1.8770 | 1 | 1 |
| 171 | lenet | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 2.3051 | 1 | 1 |
| 172 | lenet | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 2.2737 | 1 | 1 |
| 173 | lenet | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 15.00 | 2.2996 | 1 | 1 |
| 174 | lenet | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 1.8486 | 1 | 1 |
| 175 | poolformer | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 7.3117 | 1 | 1 |
| 176 | poolformer | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.7640 | 1 | 1 |
| 177 | shufflenet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 15.00 | 2.3169 | 1 | 1 |
| 178 | shufflenet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 15.00 | 2.2985 | 1 | 1 |
| 179 | shufflenet | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 15.00 | 2.2998 | 1 | 1 |
| 180 | shufflenet | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.2911 | 1 | 1 |
| 181 | shufflenet | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 15.00 | 1.8636 | 1 | 1 |
| 182 | shufflenet | focal_loss | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 15.00 | 1.8702 | 1 | 1 |
| 183 | vit | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 15.00 | 2.6521 | 1 | 1 |
| 184 | vit | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.3187 | 1 | 1 |
| 185 | vit | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.3423 | 1 | 1 |
| 186 | vit | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 15.00 | 2.3811 | 1 | 1 |
| 187 | vit | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.3048 | 1 | 1 |
| 188 | vit | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 15.00 | 2.4783 | 1 | 1 |
| 189 | vit | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 1.8466 | 1 | 1 |
| 190 | vit | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 1.8609 | 1 | 1 |
| 191 | coord_resnet | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 2.2948 | 1 | 1 |
| 192 | coord_resnet | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.6032 | 1 | 1 |
| 193 | coord_resnet | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 2.6719 | 1 | 1 |
| 194 | coord_resnet | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 2.9948 | 1 | 1 |
| 195 | coord_resnet | label_smoothing | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 15.00 | 2.4379 | 1 | 1 |
| 196 | coord_resnet | focal_loss | adam | batch_size=16, lr=0.0001, weight_decay=0.0001 | 15.00 | 1.8565 | 1 | 1 |
| 197 | coord_resnet | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 1.8617 | 1 | 1 |
| 198 | efficientnetv2 | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 15.00 | 2.3599 | 1 | 1 |
| 199 | efficientnetv2 | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 15.00 | 2.3042 | 1 | 1 |
| 200 | efficientnetv2 | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 2.4584 | 1 | 1 |
| 201 | efficientnetv2 | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.3018 | 1 | 1 |
| 202 | efficientnetv2 | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.3028 | 1 | 1 |
| 203 | efficientnetv2 | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 2.3111 | 1 | 1 |
| 204 | efficientnetv2 | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.2999 | 1 | 1 |
| 205 | efficientnetv2 | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.5451 | 1 | 1 |
| 206 | efficientnetv2 | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 1.9006 | 1 | 1 |
| 207 | efficientnetv2 | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 15.00 | 2.5025 | 1 | 1 |
| 208 | efficientnetv2 | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 1.8634 | 1 | 1 |
| 209 | lcnet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 2.2976 | 1 | 1 |
| 210 | lcnet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 4.1957 | 1 | 1 |
| 211 | lcnet | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 2.3953 | 1 | 1 |
| 212 | lcnet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 98.3482 | 1 | 1 |
| 213 | lcnet | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 15.00 | 1.8708 | 1 | 1 |
| 214 | lcnet | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 1.8556 | 1 | 1 |
| 215 | nin | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 2.2917 | 1 | 1 |
| 216 | nin | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.2991 | 1 | 1 |
| 217 | nin | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.2938 | 1 | 1 |
| 218 | nin | label_smoothing | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.2966 | 1 | 1 |
| 219 | nin | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 15.00 | 1.8454 | 1 | 1 |
| 220 | nin | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 1.8457 | 1 | 1 |
| 221 | nin | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 15.00 | 1.8841 | 1 | 1 |
| 222 | se_resnet | cross_entropy | adam | batch_size=16, lr=0.001, weight_decay=0.001 | 15.00 | 2.6676 | 1 | 1 |
| 223 | se_resnet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 3.5364 | 1 | 1 |
| 224 | se_resnet | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 2.4164 | 1 | 1 |
| 225 | se_resnet | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 2.3586 | 1 | 1 |
| 226 | se_resnet | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.0489 | 1 | 1 |
| 227 | se_resnet | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 2.4934 | 1 | 1 |
| 228 | vim_tiny | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 15.00 | 2.3077 | 1 | 1 |
| 229 | vim_tiny | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.3651 | 1 | 1 |
| 230 | vim_tiny | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.9774 | 1 | 1 |
| 231 | vim_tiny | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 2.3192 | 1 | 1 |
| 232 | vim_tiny | label_smoothing | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.001 | 15.00 | 2.3255 | 1 | 1 |
| 233 | vim_tiny | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.001 | 15.00 | 2.3139 | 1 | 1 |
| 234 | vim_tiny | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 15.00 | 1.8603 | 1 | 1 |
| 235 | vim_tiny | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.0190 | 1 | 1 |
| 236 | cbam_resnet | cross_entropy | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.5714 | 1 | 1 |
| 237 | cbam_resnet | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.2947 | 1 | 1 |
| 238 | cbam_resnet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.3064 | 1 | 1 |
| 239 | cbam_resnet | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 2.2966 | 1 | 1 |
| 240 | cbam_resnet | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.0925 | 1 | 1 |
| 241 | cbam_resnet | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 2.0440 | 1 | 1 |
| 242 | dpn | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 2.2954 | 1 | 1 |
| 243 | dpn | cross_entropy | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 2.4149 | 1 | 1 |
| 244 | dpn | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.2961 | 1 | 1 |
| 245 | dpn | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 1.8648 | 1 | 1 |
| 246 | dpn | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 1.8637 | 1 | 1 |
| 247 | hardnet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.3020 | 1 | 1 |
| 248 | hardnet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 2.3025 | 1 | 1 |
| 249 | hardnet | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 2.2958 | 1 | 1 |
| 250 | hardnet | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 15.00 | 2.7158 | 1 | 1 |
| 251 | hardnet | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 2.3020 | 1 | 1 |
| 252 | hardnet | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 15.00 | 2.3963 | 1 | 1 |
| 253 | hardnet | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 3.0354 | 1 | 1 |
| 254 | mobilenet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 2.3735 | 1 | 1 |
| 255 | mobilenet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.3100 | 1 | 1 |
| 256 | mobilenet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 15.00 | 2.3122 | 1 | 1 |
| 257 | mobilenet | label_smoothing | adamw | batch_size=16, lr=0.0001, weight_decay=0.0 | 15.00 | 2.2969 | 1 | 1 |
| 258 | mobilenet | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 1.8526 | 1 | 1 |
| 259 | mobilenet | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 15.00 | 1.9728 | 1 | 1 |
| 260 | mobilenet | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 1.8576 | 1 | 1 |
| 261 | mobilenet | focal_loss | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 15.00 | 1.8502 | 1 | 1 |
| 262 | res2net | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.8581 | 1 | 1 |
| 263 | res2net | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 15.00 | 2.7509 | 1 | 1 |
| 264 | res2net | cross_entropy | adamw | batch_size=16, lr=0.0001, weight_decay=0.001 | 15.00 | 2.3042 | 1 | 1 |
| 265 | res2net | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 2.8396 | 1 | 1 |
| 266 | res2net | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.3078 | 1 | 1 |
| 267 | res2net | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.001 | 15.00 | 2.4637 | 1 | 1 |
| 268 | res2net | focal_loss | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 15.00 | 1.8667 | 1 | 1 |
| 269 | res2net | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.1458 | 1 | 1 |
| 270 | res2net | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 2.4192 | 1 | 1 |
| 271 | swin_tiny | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 15.00 | 2.4621 | 1 | 1 |
| 272 | swin_tiny | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 4.2252 | 1 | 1 |
| 273 | swin_tiny | focal_loss | adam | batch_size=16, lr=0.0001, weight_decay=0.001 | 15.00 | 2.2268 | 1 | 1 |
| 274 | swin_tiny | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 1.9395 | 1 | 1 |
| 275 | swin_tiny | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 62.8065 | 1 | 1 |
| 276 | eca_resnet | cross_entropy | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.5489 | 1 | 1 |
| 277 | eca_resnet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 15.00 | 2.3606 | 1 | 1 |
| 278 | eca_resnet | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.5606 | 1 | 1 |
| 279 | eca_resnet | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 2.3003 | 1 | 1 |
| 280 | eca_resnet | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.7619 | 1 | 1 |
| 281 | eca_resnet | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 3.1363 | 1 | 1 |
| 282 | eca_resnet | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 1.8767 | 1 | 1 |
| 283 | eca_resnet | focal_loss | adamw | batch_size=16, lr=0.0001, weight_decay=0.0001 | 15.00 | 1.8635 | 1 | 1 |
| 284 | eca_resnet | focal_loss | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 15.00 | 2.0663 | 1 | 1 |
| 285 | hrnet | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.3027 | 1 | 1 |
| 286 | hrnet | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.2966 | 1 | 1 |
| 287 | hrnet | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 1.8449 | 1 | 1 |
| 288 | mobilenetv2 | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.3837 | 1 | 1 |
| 289 | mobilenetv2 | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 15.00 | 2.2995 | 1 | 1 |
| 290 | mobilenetv2 | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 2.3041 | 1 | 1 |
| 291 | mobilenetv2 | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.3180 | 1 | 1 |
| 292 | mobilenetv2 | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 3.0093 | 1 | 1 |
| 293 | mobilenetv2 | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 15.00 | 15.8836 | 1 | 1 |
| 294 | mobilenetv2 | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 15.00 | 2.2905 | 1 | 1 |
| 295 | mobilenetv2 | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 15.00 | 1.8538 | 1 | 1 |
| 296 | mobilenetv2 | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 1.8806 | 1 | 1 |
| 297 | mobilenetv2 | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 2.2469 | 1 | 1 |
| 298 | resnet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 6.3645 | 1 | 1 |
| 299 | resnet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 15.00 | 2.8111 | 1 | 1 |
| 300 | resnet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.2973 | 1 | 1 |
| 301 | resnet | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 3.6839 | 1 | 1 |
| 302 | resnet | label_smoothing | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 15.00 | 2.2940 | 1 | 1 |
| 303 | resnet | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.9090 | 1 | 1 |
| 304 | resnet | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 2.8363 | 1 | 1 |
| 305 | resnet | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.6776 | 1 | 1 |
| 306 | resnet | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 1.8548 | 1 | 1 |
| 307 | resnet | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 15.00 | 189.6048 | 1 | 1 |
| 308 | resnet | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 15.00 | 4.2125 | 1 | 1 |
| 309 | van | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 2.3612 | 1 | 1 |
| 310 | van | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 2.2964 | 1 | 1 |
| 311 | van | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 2.2959 | 1 | 1 |
| 312 | van | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 15.00 | 2.3004 | 1 | 1 |
| 313 | van | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 15.00 | 2.7130 | 1 | 1 |
| 314 | van | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 1.8636 | 1 | 1 |
| 315 | van | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 15.00 | 1.8570 | 1 | 1 |
| 316 | van | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 1.8703 | 1 | 1 |
| 317 | convnext | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 2.2057 | 1 | 1 |
| 318 | inception_resnet | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.2954 | 1 | 1 |
| 319 | inception_resnet | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 2.2903 | 1 | 1 |
| 320 | inception_resnet | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 2.4117 | 1 | 1 |
| 321 | inception_resnet | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.2980 | 1 | 1 |
| 322 | inception_resnet | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 2.2946 | 1 | 1 |
| 323 | inception_resnet | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.4370 | 1 | 1 |
| 324 | inception_resnet | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.4748 | 1 | 1 |
| 325 | inception_resnet | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.0330 | 1 | 1 |
| 326 | inception_resnet | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 1.9752 | 1 | 1 |
| 327 | inception_resnet | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 2.2245 | 1 | 1 |
| 328 | mobilenetv3 | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 15.00 | 2.3007 | 1 | 1 |
| 329 | mobilenetv3 | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.3003 | 1 | 1 |
| 330 | mobilenetv3 | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 2.8796 | 1 | 1 |
| 331 | mobilenetv3 | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 2.3012 | 1 | 1 |
| 332 | mobilenetv3 | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 2.2958 | 1 | 1 |
| 333 | mobilenetv3 | label_smoothing | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.3013 | 1 | 1 |
| 334 | mobilenetv3 | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 1.8566 | 1 | 1 |
| 335 | mobilenetv3 | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 15.00 | 1.8630 | 1 | 1 |
| 336 | resnext | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.3142 | 1 | 1 |
| 337 | resnext | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 2.3023 | 1 | 1 |
| 338 | resnext | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 15.00 | 2.8293 | 1 | 1 |
| 339 | resnext | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.5099 | 1 | 1 |
| 340 | resnext | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 15.00 | 2.3855 | 1 | 1 |
| 341 | resnext | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.3052 | 1 | 1 |
| 342 | resnext | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.0507 | 1 | 1 |
| 343 | resnext | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 15.00 | 2.0621 | 1 | 1 |
| 344 | resnext | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 1.9753 | 1 | 1 |
| 345 | resnext | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 2.0391 | 1 | 1 |
| 346 | vgg | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.2989 | 1 | 1 |
| 347 | vgg | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 2.2974 | 1 | 1 |
| 348 | vgg | focal_loss | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.001 | 15.00 | 1.8592 | 1 | 1 |
| 349 | alexnet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 15.00 | 2.3426 | 1 | 1 |
| 350 | alexnet | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 2.3017 | 1 | 1 |
| 351 | alexnet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.2951 | 1 | 1 |
| 352 | alexnet | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 15.00 | 2.2979 | 1 | 1 |
| 353 | alexnet | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.2932 | 1 | 1 |
| 354 | alexnet | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.3016 | 1 | 1 |
| 355 | alexnet | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 2.2848 | 1 | 1 |
| 356 | alexnet | label_smoothing | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 15.00 | 2.2947 | 1 | 1 |
| 357 | alexnet | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 15.00 | 1.8582 | 1 | 1 |
| 358 | alexnet | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 1.8511 | 1 | 1 |
| 359 | darknet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 15.00 | 1426951.2679 | 1 | 1 |
| 360 | darknet | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 1.8922 | 1 | 1 |
| 361 | darknet | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.001 | 15.00 | 1.8948 | 1 | 1 |
| 362 | googlenet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 2.2960 | 1 | 1 |
| 363 | googlenet | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 2.3031 | 1 | 1 |
| 364 | googlenet | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.2938 | 1 | 1 |
| 365 | googlenet | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 15.00 | 2.3024 | 1 | 1 |
| 366 | googlenet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 1.8562 | 1 | 1 |
| 367 | lstm | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 15.00 | 2.3617 | 1 | 1 |
| 368 | lstm | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 2.2958 | 1 | 1 |
| 369 | lstm | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.2908 | 1 | 1 |
| 370 | lstm | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.2996 | 1 | 1 |
| 371 | lstm | label_smoothing | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.3024 | 1 | 1 |
| 372 | lstm | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 2.2921 | 1 | 1 |
| 373 | lstm | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 2.2956 | 1 | 1 |
| 374 | lstm | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 1.8561 | 1 | 1 |
| 375 | lstm | focal_loss | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 1.8497 | 1 | 1 |
| 376 | regnet | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 2.3081 | 1 | 1 |
| 377 | regnet | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 15.00 | 2.2847 | 1 | 1 |
| 378 | regnet | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 15.00 | 2.7573 | 1 | 1 |
| 379 | regnet | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 15.00 | 2.3162 | 1 | 1 |
| 380 | regnet | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.3616 | 1 | 1 |
| 381 | regnet | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 1.8763 | 1 | 1 |
| 382 | regnet | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 1.8573 | 1 | 1 |
| 383 | simple_cnn | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.2926 | 1 | 1 |
| 384 | simple_cnn | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 15.00 | 2.2765 | 1 | 1 |
| 385 | simple_cnn | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 2.4895 | 1 | 1 |
| 386 | simple_cnn | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 1.8253 | 1 | 1 |
| 387 | wide_resnet | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 3.1171 | 1 | 1 |
| 388 | wide_resnet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.8440 | 1 | 1 |
| 389 | wide_resnet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 395.8911 | 1 | 1 |
| 390 | wide_resnet | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 15.00 | 2.3730 | 1 | 1 |
| 391 | wide_resnet | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.8184 | 1 | 1 |
| 392 | wide_resnet | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 2.3049 | 1 | 1 |
| 393 | wide_resnet | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 1.8554 | 1 | 1 |
| 394 | wide_resnet | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 2.2917 | 1 | 1 |
| 395 | wide_resnet | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 2.0968 | 1 | 1 |
| 396 | densenet | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 14.00 | 3.7723 | 1 | 1 |
| 397 | resnext | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 14.00 | 2.2943 | 1 | 1 |
| 398 | vgg | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 13.00 | 1.8652 | 1 | 1 |
| 399 | simple_cnn | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0 | 13.00 | 2.2510 | 1 | 1 |
| 400 | densenet | cross_entropy | sgd | batch_size=16, lr=0.0001, weight_decay=0.0 | 12.00 | 2.3008 | 1 | 1 |
| 401 | gru | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 12.00 | 2.3016 | 1 | 1 |
| 402 | gru | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 12.00 | 1.8563 | 1 | 1 |
| 403 | squeezenet | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.001 | 12.00 | 2.2928 | 1 | 1 |
| 404 | squeezenet | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 12.00 | 1.8555 | 1 | 1 |
| 405 | squeezenet | focal_loss | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 12.00 | 1.8607 | 1 | 1 |
| 406 | lenet | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 12.00 | 1.8472 | 1 | 1 |
| 407 | poolformer | label_smoothing | adam | batch_size=16, lr=0.0001, weight_decay=0.001 | 12.00 | 2.6168 | 1 | 1 |
| 408 | poolformer | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 12.00 | 2.9845 | 1 | 1 |
| 409 | shufflenet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 12.00 | 2.3011 | 1 | 1 |
| 410 | shufflenet | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 12.00 | 2.2981 | 1 | 1 |
| 411 | shufflenet | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.001 | 12.00 | 2.2999 | 1 | 1 |
| 412 | shufflenet | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.001 | 12.00 | 1.8546 | 1 | 1 |
| 413 | vit | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 12.00 | 9.3133 | 1 | 1 |
| 414 | coord_resnet | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 12.00 | 2.2991 | 1 | 1 |
| 415 | efficientnetv2 | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 12.00 | 1.8685 | 1 | 1 |
| 416 | nin | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 12.00 | 2.2996 | 1 | 1 |
| 417 | nin | cross_entropy | adamw | batch_size=16, lr=0.0001, weight_decay=0.001 | 12.00 | 2.3005 | 1 | 1 |
| 418 | nin | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 12.00 | 1.8596 | 1 | 1 |
| 419 | se_resnet | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 12.00 | 2.2955 | 1 | 1 |
| 420 | se_resnet | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 12.00 | 38.3875 | 1 | 1 |
| 421 | hardnet | cross_entropy | sgd | batch_size=16, lr=0.0001, weight_decay=0.0001 | 12.00 | 2.2920 | 1 | 1 |
| 422 | mobilenet | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 12.00 | 2.2981 | 1 | 1 |
| 423 | mobilenet | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 12.00 | 2.2979 | 1 | 1 |
| 424 | res2net | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 12.00 | 250438.1618 | 1 | 1 |
| 425 | swin_tiny | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 12.00 | 2.6947 | 1 | 1 |
| 426 | swin_tiny | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 12.00 | 6.6449 | 1 | 1 |
| 427 | swin_tiny | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 12.00 | 3.5926 | 1 | 1 |
| 428 | inception_resnet | cross_entropy | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 12.00 | 5743190.7143 | 1 | 1 |
| 429 | mobilenetv3 | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 12.00 | 24117.4725 | 1 | 1 |
| 430 | vgg | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 12.00 | 2.3031 | 1 | 1 |
| 431 | vgg | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 12.00 | 14529136786.2857 | 1 | 1 |
| 432 | alexnet | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 12.00 | 5.3466 | 1 | 1 |
| 433 | googlenet | label_smoothing | adam | batch_size=16, lr=0.0001, weight_decay=0.0001 | 12.00 | 2.3001 | 1 | 1 |
| 434 | googlenet | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 12.00 | 2.2995 | 1 | 1 |
| 435 | googlenet | label_smoothing | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.001 | 12.00 | 2.2982 | 1 | 1 |
| 436 | lstm | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 12.00 | 2.3001 | 1 | 1 |
| 437 | simple_cnn | focal_loss | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 12.00 | 1.8580 | 1 | 1 |
| 438 | wide_resnet | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 12.00 | 2.3090 | 1 | 1 |
| 439 | densenet | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.0001 | 11.00 | 2.3037 | 1 | 1 |
| 440 | densenet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 11.00 | 1.8632 | 1 | 1 |
| 441 | gru | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 11.00 | 1.8697 | 1 | 1 |
| 442 | gru | focal_loss | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0 | 11.00 | 1.8551 | 1 | 1 |
| 443 | mnasnet | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 11.00 | 1.8681 | 1 | 1 |
| 444 | repvgg | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 11.00 | 2.7413 | 1 | 1 |
| 445 | deit | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 11.00 | 2.3840 | 1 | 1 |
| 446 | repghost | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 11.00 | 2.3185 | 1 | 1 |
| 447 | repghost | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 11.00 | 2.2990 | 1 | 1 |
| 448 | cspnet | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 11.00 | 48594.3867 | 1 | 1 |
| 449 | ghostnet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 11.00 | 2.3003 | 1 | 1 |
| 450 | ghostnet | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 11.00 | 2.3015 | 1 | 1 |
| 451 | poolformer | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 11.00 | 18.6743 | 1 | 1 |
| 452 | poolformer | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 11.00 | 2.5020 | 1 | 1 |
| 453 | poolformer | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 11.00 | 18.3496 | 1 | 1 |
| 454 | vit | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 11.00 | 2.1554 | 1 | 1 |
| 455 | lcnet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 11.00 | 2.3077 | 1 | 1 |
| 456 | lcnet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 11.00 | 2.3011 | 1 | 1 |
| 457 | lcnet | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 11.00 | 2.3013 | 1 | 1 |
| 458 | lcnet | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 11.00 | 2.3052 | 1 | 1 |
| 459 | lcnet | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.001 | 11.00 | 1.8600 | 1 | 1 |
| 460 | se_resnet | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 11.00 | 1.8651 | 1 | 1 |
| 461 | vim_tiny | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 11.00 | 2.3020 | 1 | 1 |
| 462 | mobilenet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 11.00 | 2.6915 | 1 | 1 |
| 463 | mobilenet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 11.00 | 2.3264 | 1 | 1 |
| 464 | swin_tiny | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 11.00 | 2.7232 | 1 | 1 |
| 465 | swin_tiny | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 11.00 | 64.5480 | 1 | 1 |
| 466 | van | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 11.00 | 2.3868 | 1 | 1 |
| 467 | inception_resnet | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 11.00 | 1.8619 | 1 | 1 |
| 468 | mobilenetv3 | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 11.00 | 2.2190 | 1 | 1 |
| 469 | vgg | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.0 | 11.00 | 2.3028 | 1 | 1 |
| 470 | darknet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 11.00 | 54289.6749 | 1 | 1 |
| 471 | darknet | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 11.00 | 194113.6373 | 1 | 1 |
| 472 | googlenet | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 11.00 | 1.8625 | 1 | 1 |
| 473 | googlenet | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 11.00 | 1.8593 | 1 | 1 |
| 474 | regnet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 11.00 | 2.5157 | 1 | 1 |
| 475 | regnet | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 11.00 | 1.8576 | 1 | 1 |
| 476 | capsnet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 10.00 | 2.3026 | 1 | 1 |
| 477 | capsnet | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 10.00 | 2.3026 | 1 | 1 |
| 478 | capsnet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 10.00 | 2.3026 | 1 | 1 |
| 479 | capsnet | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 10.00 | 2.3026 | 1 | 1 |
| 480 | capsnet | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 10.00 | 2.3026 | 1 | 1 |
| 481 | capsnet | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 10.00 | 2.3026 | 1 | 1 |
| 482 | capsnet | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 10.00 | 2.3026 | 1 | 1 |
| 483 | capsnet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 10.00 | 2.3026 | 1 | 1 |
| 484 | capsnet | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 10.00 | 1.8651 | 1 | 1 |
| 485 | capsnet | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 10.00 | 1.8651 | 1 | 1 |
| 486 | capsnet | focal_loss | adamw | batch_size=16, lr=0.0001, weight_decay=0.0001 | 10.00 | 1.8651 | 1 | 1 |
| 487 | capsnet | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 10.00 | 1.8651 | 1 | 1 |
| 488 | gru | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 10.00 | 2.3032 | 1 | 1 |
| 489 | gru | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 10.00 | nan | 1 | 1 |
| 490 | gru | label_smoothing | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 10.00 | 2.2980 | 1 | 1 |
| 491 | mnasnet | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 10.00 | 1.8780 | 1 | 1 |
| 492 | mnasnet | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 10.00 | 1.8707 | 1 | 1 |
| 493 | deit | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 10.00 | 2.1061 | 1 | 1 |
| 494 | deit | focal_loss | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 10.00 | 2.4023 | 1 | 1 |
| 495 | sknet | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 10.00 | 1.8792 | 1 | 1 |
| 496 | ghostnet | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 10.00 | 2.3178 | 1 | 1 |
| 497 | ghostnet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 10.00 | 2.3089 | 1 | 1 |
| 498 | ghostnet | label_smoothing | adam | batch_size=16, lr=0.0001, weight_decay=0.001 | 10.00 | 2.3047 | 1 | 1 |
| 499 | ghostnet | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 10.00 | 1.8841 | 1 | 1 |
| 500 | lenet | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 10.00 | 2.2994 | 1 | 1 |
| 501 | lenet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 10.00 | 1.8622 | 1 | 1 |
| 502 | poolformer | cross_entropy | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 10.00 | 37.1048 | 1 | 1 |
| 503 | poolformer | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 10.00 | 2.8822 | 1 | 1 |
| 504 | shufflenet | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 10.00 | 2.2993 | 1 | 1 |
| 505 | shufflenet | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 10.00 | 1.8684 | 1 | 1 |
| 506 | vit | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 10.00 | 4.1251 | 1 | 1 |
| 507 | lcnet | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 10.00 | 1.8643 | 1 | 1 |
| 508 | nin | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 10.00 | 2.3014 | 1 | 1 |
| 509 | se_resnet | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.0001 | 10.00 | 2.2970 | 1 | 1 |
| 510 | vim_tiny | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 10.00 | 427909.5312 | 1 | 1 |
| 511 | cbam_resnet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 10.00 | 1.8718 | 1 | 1 |
| 512 | dpn | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 10.00 | 2.3842 | 1 | 1 |
| 513 | dpn | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 10.00 | 5438.4987 | 1 | 1 |
| 514 | hardnet | focal_loss | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 10.00 | 1.8658 | 1 | 1 |
| 515 | hardnet | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 10.00 | 1.8675 | 1 | 1 |
| 516 | hardnet | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 10.00 | 22570.5222 | 1 | 1 |
| 517 | res2net | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 10.00 | 14649960.0000 | 1 | 1 |
| 518 | swin_tiny | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 10.00 | 3.0776 | 1 | 1 |
| 519 | swin_tiny | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 10.00 | 42.9792 | 1 | 1 |
| 520 | hrnet | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 10.00 | 2.3040 | 1 | 1 |
| 521 | mobilenetv2 | label_smoothing | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 10.00 | 2.3038 | 1 | 1 |
| 522 | convnext | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 10.00 | 6.9946 | 1 | 1 |
| 523 | mobilenetv3 | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 10.00 | 2.3612 | 1 | 1 |
| 524 | mobilenetv3 | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 10.00 | 2.3405 | 1 | 1 |
| 525 | vgg | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 10.00 | 288.8771 | 1 | 1 |
| 526 | alexnet | focal_loss | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0 | 10.00 | 3.1974 | 1 | 1 |
| 527 | darknet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 10.00 | 102880.5150 | 1 | 1 |
| 528 | darknet | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 10.00 | 92609.2022 | 1 | 1 |
| 529 | darknet | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 10.00 | 2.3165 | 1 | 1 |
| 530 | darknet | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 10.00 | 2.5643 | 1 | 1 |
| 531 | googlenet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 10.00 | 2.3007 | 1 | 1 |
| 532 | lstm | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 10.00 | 1.8589 | 1 | 1 |
| 533 | lstm | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 10.00 | 1.8656 | 1 | 1 |
| 534 | regnet | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 10.00 | 2.2995 | 1 | 1 |
| 535 | regnet | label_smoothing | adamw | batch_size=16, lr=0.0001, weight_decay=0.0 | 10.00 | 2.3060 | 1 | 1 |
| 536 | wide_resnet | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 10.00 | 2.3189 | 1 | 1 |
| 537 | mnasnet | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 9.00 | 2.9750 | 1 | 1 |
| 538 | sknet | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 9.00 | 2.3036 | 1 | 1 |
| 539 | cspnet | label_smoothing | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 9.00 | 2.3019 | 1 | 1 |
| 540 | poolformer | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 9.00 | 44.2580 | 1 | 1 |
| 541 | nin | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 9.00 | 2.3066 | 1 | 1 |
| 542 | dpn | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 9.00 | 2.3071 | 1 | 1 |
| 543 | dpn | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 9.00 | 1.8624 | 1 | 1 |
| 544 | res2net | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 9.00 | 1.8619 | 1 | 1 |
| 545 | hrnet | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 9.00 | 2.3097 | 1 | 1 |
| 546 | mobilenetv2 | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 9.00 | 4.1018 | 1 | 1 |
| 547 | googlenet | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 9.00 | 1.8636 | 1 | 1 |
| 548 | poolformer | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 8.00 | 3963.1505 | 1 | 1 |
| 549 | vim_tiny | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 8.00 | 9.1755 | 1 | 1 |
| 550 | darknet | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 8.00 | 13.3833 | 1 | 1 |
| 551 | darknet | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 8.00 | 625825.1683 | 1 | 1 |
| 552 | repghost | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 6.00 | 2.3142 | 1 | 1 |

## Autoencoder Models (ranked by reconstruction loss, lower is better)

| Rank | Model | Loss | Optimizer | Hyperparameters | Recon Loss | Epochs Run | Convergence Epoch |
|------|-------|------|-----------|-----------------|------------|------------|-------------------|
| 1 | simple_ae | mse | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 2 | simple_ae | mse | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 3 | simple_ae | mse | adamw | batch_size=16, lr=0.0001, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 4 | simple_ae | mse | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 0.0000 | 1 | 1 |
| 5 | simple_ae | l1 | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 6 | simple_ae | l1 | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 7 | simple_ae | l1 | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 8 | simple_ae | l1 | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 9 | simple_ae | bce | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 10 | simple_ae | bce | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 0.0000 | 1 | 1 |
| 11 | simple_ae | bce | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 12 | simple_ae | bce | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 0.0000 | 1 | 1 |
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
| 25 | vae | mse | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 26 | vae | mse | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 27 | vae | mse | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 0.0000 | 1 | 1 |
| 28 | vae | mse | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 29 | vae | l1 | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 30 | vae | l1 | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 31 | vae | l1 | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 32 | vae | l1 | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 33 | vae | bce | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 34 | vae | bce | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 35 | vae | bce | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 36 | vae | bce | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 37 | denoising_ae | mse | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 38 | denoising_ae | mse | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 0.0000 | 1 | 1 |
| 39 | denoising_ae | mse | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 40 | denoising_ae | mse | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 41 | denoising_ae | l1 | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 42 | denoising_ae | l1 | sgd | batch_size=16, lr=0.0001, weight_decay=0.0 | 0.0000 | 1 | 1 |
| 43 | denoising_ae | l1 | adamw | batch_size=16, lr=0.0001, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 44 | denoising_ae | l1 | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 45 | denoising_ae | bce | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 0.0000 | 1 | 1 |
| 46 | denoising_ae | bce | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 0.0000 | 1 | 1 |
| 47 | denoising_ae | bce | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 48 | denoising_ae | bce | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 0.0000 | 1 | 1 |

## GAN Models (ranked by generator loss, lower is better)

| Rank | Model | Loss | Optimizer | Hyperparameters | G Loss | D Loss | Epochs Run | Convergence Epoch |
|------|-------|------|-----------|-----------------|--------|--------|------------|-------------------|
| 1 | wgan | wasserstein | adam | batch_size=16, lr=0.0001, weight_decay=0.001 | -0.0020 | -0.0027 | 1 | 1 |
| 2 | wgan | bce | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 0.0413 | -0.2161 | 1 | 1 |
| 3 | wgan | wasserstein | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 0.0621 | -0.1996 | 1 | 1 |
| 4 | wgan | bce | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 0.1806 | -0.3169 | 1 | 1 |
| 5 | dcgan | wasserstein | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 0.7008 | 1.4175 | 1 | 1 |
| 6 | dcgan | bce | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 0.7365 | 1.3012 | 1 | 1 |
| 7 | cgan | wasserstein | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 0.7636 | 1.3636 | 1 | 1 |
| 8 | cgan | wasserstein | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 0.8978 | 85.6092 | 1 | 1 |
| 9 | cgan | bce | adam | batch_size=16, lr=0.0001, weight_decay=0.001 | 0.9190 | 1.3381 | 1 | 1 |
| 10 | cgan | bce | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 0.9198 | 1.2146 | 1 | 1 |
| 11 | dcgan | bce | adamw | batch_size=16, lr=0.0001, weight_decay=0.0001 | 0.9807 | 1.0789 | 1 | 1 |
| 12 | dcgan | wasserstein | adam | batch_size=16, lr=0.0001, weight_decay=0.001 | 1.0462 | 1.0045 | 1 | 1 |
| 13 | cgan | wasserstein | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 1.1153 | 1.2552 | 1 | 1 |
| 14 | cgan | wasserstein | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 1.9294 | 21.5144 | 1 | 1 |
| 15 | cgan | bce | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 2.1446 | 1.9029 | 1 | 1 |
| 16 | cgan | bce | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 2.8004 | 1.5839 | 1 | 1 |
| 17 | dcgan | bce | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 5.7850 | 0.4567 | 1 | 1 |
| 18 | dcgan | wasserstein | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 5.8994 | 0.8758 | 1 | 1 |
| 19 | dcgan | wasserstein | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 6.0024 | 75.1930 | 1 | 1 |
| 20 | dcgan | bce | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 6.0576 | 0.7920 | 1 | 1 |

## Failed Configurations

- **mlp** / cross_entropy / adam: All trials failed
- **mlp** / cross_entropy / sgd: All trials failed
- **mlp** / cross_entropy / adamw: All trials failed
- **mlp** / cross_entropy / rmsprop: All trials failed
- **mlp** / label_smoothing / adam: All trials failed
- **mlp** / label_smoothing / sgd: All trials failed
- **mlp** / label_smoothing / adamw: All trials failed
- **mlp** / label_smoothing / rmsprop: All trials failed
- **mlp** / focal_loss / adam: All trials failed
- **mlp** / focal_loss / sgd: All trials failed
- **mlp** / focal_loss / adamw: All trials failed
- **mlp** / focal_loss / rmsprop: All trials failed
- **xception** / cross_entropy / adam: All trials failed
- **xception** / cross_entropy / sgd: All trials failed
- **xception** / cross_entropy / adamw: All trials failed
- **xception** / cross_entropy / rmsprop: All trials failed
- **xception** / label_smoothing / adam: All trials failed
- **xception** / label_smoothing / sgd: All trials failed
- **xception** / label_smoothing / adamw: All trials failed
- **xception** / label_smoothing / rmsprop: All trials failed
- **xception** / focal_loss / adam: All trials failed
- **xception** / focal_loss / sgd: All trials failed
- **xception** / focal_loss / adamw: All trials failed
- **xception** / focal_loss / rmsprop: All trials failed
- **vanilla_gan** / bce / adam: All trials failed
- **vanilla_gan** / bce / sgd: All trials failed
- **vanilla_gan** / bce / adamw: All trials failed
- **vanilla_gan** / bce / rmsprop: All trials failed
- **vanilla_gan** / wasserstein / adam: All trials failed
- **vanilla_gan** / wasserstein / sgd: All trials failed
- **vanilla_gan** / wasserstein / adamw: All trials failed
- **vanilla_gan** / wasserstein / rmsprop: All trials failed
- **coatnet** / cross_entropy / adam: All trials failed
- **coatnet** / cross_entropy / sgd: All trials failed
- **coatnet** / cross_entropy / adamw: All trials failed
- **coatnet** / cross_entropy / rmsprop: All trials failed
- **coatnet** / label_smoothing / adam: All trials failed
- **coatnet** / label_smoothing / sgd: All trials failed
- **coatnet** / label_smoothing / adamw: All trials failed
- **coatnet** / label_smoothing / rmsprop: All trials failed
- **coatnet** / focal_loss / adam: All trials failed
- **coatnet** / focal_loss / sgd: All trials failed
- **coatnet** / focal_loss / adamw: All trials failed
- **coatnet** / focal_loss / rmsprop: All trials failed
- **efficientnet** / cross_entropy / adam: All trials failed
- **efficientnet** / cross_entropy / sgd: All trials failed
- **efficientnet** / cross_entropy / adamw: All trials failed
- **efficientnet** / cross_entropy / rmsprop: All trials failed
- **efficientnet** / label_smoothing / adam: All trials failed
- **efficientnet** / label_smoothing / sgd: All trials failed
- **efficientnet** / label_smoothing / adamw: All trials failed
- **efficientnet** / label_smoothing / rmsprop: All trials failed
- **efficientnet** / focal_loss / adam: All trials failed
- **efficientnet** / focal_loss / sgd: All trials failed
- **efficientnet** / focal_loss / adamw: All trials failed
- **efficientnet** / focal_loss / rmsprop: All trials failed

## Search Space

- Loss (classification): cross_entropy, label_smoothing, focal_loss
- Loss (autoencoder): mse, l1, bce
- Loss (GAN): bce, wasserstein (informational; GANs use fixed objectives)
- Optimizers: adam, sgd, adamw, rmsprop
- Hyperparameters: {"lr": [0.0001, 0.001, 0.01], "batch_size": [8, 16, 32], "weight_decay": [0.0, 0.0001, 0.001]}
