# MNIST Regression Report

Generated: 2026-05-17T14:31:48
Mode: quick-test
Max epochs: 1 | Early-stop patience: 3 | Min delta: 0.1 | NAS trials per config: 2 | Workers: 8 | Max batch: 16
Total wall time: 370.1s

Training stops when validation metric shows no significant improvement for 3 consecutive epochs.

## Classification Models (ranked by test accuracy)

| Rank | Model | Loss | Optimizer | Hyperparameters | Test Acc (%) | Test Loss | Epochs Run | Convergence Epoch |
|------|-------|------|-----------|-----------------|--------------|-----------|------------|-------------------|
| 1 | repghost | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 37.00 | 1.5888 | 1 | 1 |
| 2 | vgg | focal_loss | adamw | batch_size=16, lr=0.0001, weight_decay=0.0001 | 31.00 | 1.8349 | 1 | 1 |
| 3 | coord_resnet | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 28.00 | 11.2426 | 1 | 1 |
| 4 | se_resnet | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 28.00 | 2.1642 | 1 | 1 |
| 5 | cbam_resnet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 28.00 | 2.3435 | 1 | 1 |
| 6 | convnext | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 28.00 | 1.7797 | 1 | 1 |
| 7 | ghostnet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 27.00 | 2.0088 | 1 | 1 |
| 8 | vgg | label_smoothing | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 27.00 | 2.2893 | 1 | 1 |
| 9 | simple_cnn | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 27.00 | 1.7865 | 1 | 1 |
| 10 | coord_resnet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 26.00 | 2.8135 | 1 | 1 |
| 11 | eca_resnet | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0 | 26.00 | 2.1926 | 1 | 1 |
| 12 | hrnet | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 26.00 | 2.1363 | 1 | 1 |
| 13 | convnext | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 26.00 | 1.9518 | 1 | 1 |
| 14 | convnext | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 26.00 | 2.0734 | 1 | 1 |
| 15 | deit | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 25.00 | 2.2550 | 1 | 1 |
| 16 | coord_resnet | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 25.00 | 5.9360 | 1 | 1 |
| 17 | hrnet | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 25.00 | 2.3012 | 1 | 1 |
| 18 | deit | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 24.00 | 2.2349 | 1 | 1 |
| 19 | poolformer | label_smoothing | adam | batch_size=16, lr=0.0001, weight_decay=0.001 | 24.00 | 2.4336 | 1 | 1 |
| 20 | hardnet | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 24.00 | 5.7596 | 1 | 1 |
| 21 | hrnet | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 24.00 | 1.7803 | 1 | 1 |
| 22 | repghost | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 23.00 | 2.2833 | 1 | 1 |
| 23 | cbam_resnet | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 23.00 | 3.7057 | 1 | 1 |
| 24 | convnext | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 23.00 | 2.2302 | 1 | 1 |
| 25 | simple_cnn | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 23.00 | 2.2922 | 1 | 1 |
| 26 | ghostnet | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 22.00 | 2.2490 | 1 | 1 |
| 27 | lenet | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 22.00 | 2.2882 | 1 | 1 |
| 28 | se_resnet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 22.00 | 16.6061 | 1 | 1 |
| 29 | cbam_resnet | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 22.00 | 2.4358 | 1 | 1 |
| 30 | swin_tiny | focal_loss | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 22.00 | 1.8317 | 1 | 1 |
| 31 | van | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 22.00 | nan | 1 | 1 |
| 32 | van | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 22.00 | nan | 1 | 1 |
| 33 | convnext | label_smoothing | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 22.00 | 2.2406 | 1 | 1 |
| 34 | simple_cnn | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 22.00 | 1.8133 | 1 | 1 |
| 35 | squeezenet | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 21.00 | 2.2982 | 1 | 1 |
| 36 | vit | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 21.00 | 2.3081 | 1 | 1 |
| 37 | hardnet | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 21.00 | 3.1940 | 1 | 1 |
| 38 | swin_tiny | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 21.00 | 2.3244 | 1 | 1 |
| 39 | convnext | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 21.00 | 3.1268 | 1 | 1 |
| 40 | densenet | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 20.00 | 1.8467 | 1 | 1 |
| 41 | repvgg | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 20.00 | 1.9450 | 1 | 1 |
| 42 | deit | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 20.00 | 2.2631 | 1 | 1 |
| 43 | ghostnet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 20.00 | 2.1578 | 1 | 1 |
| 44 | cbam_resnet | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 20.00 | 2.3408 | 1 | 1 |
| 45 | simple_cnn | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 20.00 | 2.2988 | 1 | 1 |
| 46 | deit | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 19.00 | 1.8091 | 1 | 1 |
| 47 | repghost | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 19.00 | 1.8478 | 1 | 1 |
| 48 | lenet | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 19.00 | 2.3007 | 1 | 1 |
| 49 | hrnet | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 19.00 | 2.2720 | 1 | 1 |
| 50 | convnext | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 19.00 | 2.4258 | 1 | 1 |
| 51 | convnext | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 19.00 | 3.9629 | 1 | 1 |
| 52 | repghost | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 18.00 | 3.6888 | 1 | 1 |
| 53 | sknet | cross_entropy | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 18.00 | 214.2205 | 1 | 1 |
| 54 | hrnet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 18.00 | 1.7341 | 1 | 1 |
| 55 | simple_cnn | label_smoothing | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.001 | 18.00 | 2.2917 | 1 | 1 |
| 56 | repvgg | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 17.00 | 15.8947 | 1 | 1 |
| 57 | bert | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 17.00 | 2.3019 | 1 | 1 |
| 58 | cspnet | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 17.00 | 2981.9787 | 1 | 1 |
| 59 | coord_resnet | label_smoothing | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 17.00 | 2.4641 | 1 | 1 |
| 60 | se_resnet | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 17.00 | 8.1556 | 1 | 1 |
| 61 | hardnet | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 17.00 | 1.9003 | 1 | 1 |
| 62 | simple_cnn | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 17.00 | 2.2942 | 1 | 1 |
| 63 | simple_cnn | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 17.00 | 2.2810 | 1 | 1 |
| 64 | squeezenet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 16.00 | 2.2926 | 1 | 1 |
| 65 | swin_tiny | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 16.00 | 2.4659 | 1 | 1 |
| 66 | densenet | cross_entropy | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.2918 | 1 | 1 |
| 67 | densenet | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 2.2848 | 1 | 1 |
| 68 | densenet | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.2922 | 1 | 1 |
| 69 | densenet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 15.00 | 3.0608 | 1 | 1 |
| 70 | densenet | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 15.00 | 77.7933 | 1 | 1 |
| 71 | densenet | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.2958 | 1 | 1 |
| 72 | densenet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 15.00 | 2.2330 | 1 | 1 |
| 73 | densenet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 1.8633 | 1 | 1 |
| 74 | densenet | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 42.1124 | 1 | 1 |
| 75 | gru | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.2983 | 1 | 1 |
| 76 | gru | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 2.2901 | 1 | 1 |
| 77 | gru | label_smoothing | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 15.00 | 2.3013 | 1 | 1 |
| 78 | gru | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 15.00 | 2.3022 | 1 | 1 |
| 79 | gru | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 2.2924 | 1 | 1 |
| 80 | gru | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 15.00 | 1.8569 | 1 | 1 |
| 81 | gru | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 1.8509 | 1 | 1 |
| 82 | mnasnet | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 2.2865 | 1 | 1 |
| 83 | mnasnet | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.4099 | 1 | 1 |
| 84 | mnasnet | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 15.00 | 2.2990 | 1 | 1 |
| 85 | mnasnet | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 2.2996 | 1 | 1 |
| 86 | mnasnet | label_smoothing | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 15.00 | 2.2996 | 1 | 1 |
| 87 | mnasnet | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 1.8588 | 1 | 1 |
| 88 | mnasnet | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 1.8616 | 1 | 1 |
| 89 | mnasnet | focal_loss | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.001 | 15.00 | 1.8594 | 1 | 1 |
| 90 | repvgg | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 2.3123 | 1 | 1 |
| 91 | repvgg | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 2.3021 | 1 | 1 |
| 92 | repvgg | cross_entropy | adamw | batch_size=16, lr=0.0001, weight_decay=0.0 | 15.00 | 2.2970 | 1 | 1 |
| 93 | repvgg | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 3.4565 | 1 | 1 |
| 94 | repvgg | label_smoothing | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.3068 | 1 | 1 |
| 95 | repvgg | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.4947 | 1 | 1 |
| 96 | repvgg | label_smoothing | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.3726 | 1 | 1 |
| 97 | repvgg | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 1.8499 | 1 | 1 |
| 98 | repvgg | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 1.8998 | 1 | 1 |
| 99 | squeezenet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 2.2978 | 1 | 1 |
| 100 | squeezenet | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 2.2913 | 1 | 1 |
| 101 | squeezenet | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 15.00 | 2.2993 | 1 | 1 |
| 102 | squeezenet | label_smoothing | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 15.00 | 2.3010 | 1 | 1 |
| 103 | squeezenet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 15.00 | 1.8484 | 1 | 1 |
| 104 | squeezenet | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 15.00 | 1.8356 | 1 | 1 |
| 105 | squeezenet | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 1.8538 | 1 | 1 |
| 106 | bert | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.001 | 15.00 | 2.3009 | 1 | 1 |
| 107 | bert | cross_entropy | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 2.2946 | 1 | 1 |
| 108 | bert | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 2.2993 | 1 | 1 |
| 109 | bert | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.2991 | 1 | 1 |
| 110 | bert | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 15.00 | 2.3029 | 1 | 1 |
| 111 | bert | focal_loss | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 15.00 | 1.8631 | 1 | 1 |
| 112 | bert | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 1.8648 | 1 | 1 |
| 113 | bert | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 1.8618 | 1 | 1 |
| 114 | bert | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 1.8588 | 1 | 1 |
| 115 | deit | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 15.00 | 2.6795 | 1 | 1 |
| 116 | deit | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.2914 | 1 | 1 |
| 117 | deit | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.4078 | 1 | 1 |
| 118 | deit | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.3443 | 1 | 1 |
| 119 | deit | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 15.00 | 2.0356 | 1 | 1 |
| 120 | gpt | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 2.2972 | 1 | 1 |
| 121 | gpt | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.3013 | 1 | 1 |
| 122 | gpt | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.2987 | 1 | 1 |
| 123 | gpt | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.3006 | 1 | 1 |
| 124 | gpt | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.3004 | 1 | 1 |
| 125 | gpt | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 2.3025 | 1 | 1 |
| 126 | gpt | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 15.00 | 2.2984 | 1 | 1 |
| 127 | gpt | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 2.3010 | 1 | 1 |
| 128 | gpt | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 1.8610 | 1 | 1 |
| 129 | gpt | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 15.00 | 1.8627 | 1 | 1 |
| 130 | gpt | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 15.00 | 1.8510 | 1 | 1 |
| 131 | gpt | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 1.8492 | 1 | 1 |
| 132 | repghost | cross_entropy | adam | batch_size=16, lr=0.001, weight_decay=0.001 | 15.00 | 2.3173 | 1 | 1 |
| 133 | repghost | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 15.00 | 2.2934 | 1 | 1 |
| 134 | repghost | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 1.8176 | 1 | 1 |
| 135 | repghost | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 1.8710 | 1 | 1 |
| 136 | sknet | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 2.3013 | 1 | 1 |
| 137 | sknet | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 2.3024 | 1 | 1 |
| 138 | sknet | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 2.5141 | 1 | 1 |
| 139 | sknet | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.2964 | 1 | 1 |
| 140 | sknet | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 2.3249 | 1 | 1 |
| 141 | sknet | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 2.4154 | 1 | 1 |
| 142 | sknet | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 1.9161 | 1 | 1 |
| 143 | sknet | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 1.8553 | 1 | 1 |
| 144 | sknet | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 15.00 | 1624.5293 | 1 | 1 |
| 145 | sknet | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.0068 | 1 | 1 |
| 146 | cspnet | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 15.00 | 2.3001 | 1 | 1 |
| 147 | cspnet | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 15.00 | 2.2982 | 1 | 1 |
| 148 | cspnet | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.3251 | 1 | 1 |
| 149 | cspnet | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 15.00 | 2.3134 | 1 | 1 |
| 150 | cspnet | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.001 | 15.00 | 2.3852 | 1 | 1 |
| 151 | cspnet | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 2.3180 | 1 | 1 |
| 152 | cspnet | label_smoothing | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.3156 | 1 | 1 |
| 153 | cspnet | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 15.00 | 1.8571 | 1 | 1 |
| 154 | cspnet | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.001 | 15.00 | 1.9014 | 1 | 1 |
| 155 | ghostnet | cross_entropy | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.2980 | 1 | 1 |
| 156 | ghostnet | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 2.3064 | 1 | 1 |
| 157 | ghostnet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.3044 | 1 | 1 |
| 158 | ghostnet | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 1.8334 | 1 | 1 |
| 159 | ghostnet | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 15.00 | 1.8505 | 1 | 1 |
| 160 | lenet | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 2.2895 | 1 | 1 |
| 161 | lenet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 15.00 | 2.2894 | 1 | 1 |
| 162 | lenet | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 15.00 | 1.8499 | 1 | 1 |
| 163 | lenet | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 1.8509 | 1 | 1 |
| 164 | poolformer | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 15.00 | 2.9967 | 1 | 1 |
| 165 | poolformer | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.5451 | 1 | 1 |
| 166 | poolformer | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 3.4573 | 1 | 1 |
| 167 | shufflenet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 15.00 | 2.2940 | 1 | 1 |
| 168 | shufflenet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 15.00 | 2.2957 | 1 | 1 |
| 169 | shufflenet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 2.2984 | 1 | 1 |
| 170 | shufflenet | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.2970 | 1 | 1 |
| 171 | shufflenet | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.2977 | 1 | 1 |
| 172 | shufflenet | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 15.00 | 1.8544 | 1 | 1 |
| 173 | shufflenet | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 1.8634 | 1 | 1 |
| 174 | vit | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.2781 | 1 | 1 |
| 175 | vit | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.2878 | 1 | 1 |
| 176 | vit | label_smoothing | adamw | batch_size=16, lr=0.0001, weight_decay=0.001 | 15.00 | 2.2871 | 1 | 1 |
| 177 | vit | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 1.8473 | 1 | 1 |
| 178 | vit | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 1.8548 | 1 | 1 |
| 179 | vit | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 15.00 | 2.0723 | 1 | 1 |
| 180 | coord_resnet | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 2.2946 | 1 | 1 |
| 181 | coord_resnet | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.4814 | 1 | 1 |
| 182 | coord_resnet | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 2.4812 | 1 | 1 |
| 183 | coord_resnet | focal_loss | adam | batch_size=16, lr=0.0001, weight_decay=0.0001 | 15.00 | 1.8586 | 1 | 1 |
| 184 | coord_resnet | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 1.8443 | 1 | 1 |
| 185 | coord_resnet | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 1.9650 | 1 | 1 |
| 186 | coord_resnet | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.1458 | 1 | 1 |
| 187 | efficientnetv2 | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 2.3195 | 1 | 1 |
| 188 | efficientnetv2 | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 15.00 | 2.3109 | 1 | 1 |
| 189 | efficientnetv2 | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 2.3085 | 1 | 1 |
| 190 | efficientnetv2 | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.3056 | 1 | 1 |
| 191 | efficientnetv2 | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.3164 | 1 | 1 |
| 192 | efficientnetv2 | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 2.3109 | 1 | 1 |
| 193 | efficientnetv2 | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.3030 | 1 | 1 |
| 194 | efficientnetv2 | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.4472 | 1 | 1 |
| 195 | efficientnetv2 | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 1.8807 | 1 | 1 |
| 196 | efficientnetv2 | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 1.8619 | 1 | 1 |
| 197 | efficientnetv2 | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 1.8539 | 1 | 1 |
| 198 | lcnet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 2.3006 | 1 | 1 |
| 199 | lcnet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 2.2957 | 1 | 1 |
| 200 | lcnet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 2.3002 | 1 | 1 |
| 201 | lcnet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 5.4765 | 1 | 1 |
| 202 | lcnet | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.2992 | 1 | 1 |
| 203 | lcnet | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 2.3048 | 1 | 1 |
| 204 | lcnet | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 2.4485 | 1 | 1 |
| 205 | lcnet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 2.8895 | 1 | 1 |
| 206 | lcnet | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 15.00 | 1.8864 | 1 | 1 |
| 207 | lcnet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 1.8624 | 1 | 1 |
| 208 | lcnet | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.001 | 15.00 | 1.8615 | 1 | 1 |
| 209 | lcnet | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 1.8611 | 1 | 1 |
| 210 | nin | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 2.2910 | 1 | 1 |
| 211 | nin | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.2988 | 1 | 1 |
| 212 | nin | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.2969 | 1 | 1 |
| 213 | nin | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 15.00 | 1.8493 | 1 | 1 |
| 214 | nin | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 1.8546 | 1 | 1 |
| 215 | se_resnet | cross_entropy | adam | batch_size=16, lr=0.001, weight_decay=0.001 | 15.00 | 3.1289 | 1 | 1 |
| 216 | se_resnet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 3.6738 | 1 | 1 |
| 217 | se_resnet | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 2.3536 | 1 | 1 |
| 218 | se_resnet | label_smoothing | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.001 | 15.00 | 2.4657 | 1 | 1 |
| 219 | se_resnet | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.3036 | 1 | 1 |
| 220 | se_resnet | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 2.8503 | 1 | 1 |
| 221 | vim_tiny | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 15.00 | 2.2959 | 1 | 1 |
| 222 | vim_tiny | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 2.2943 | 1 | 1 |
| 223 | vim_tiny | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.3396 | 1 | 1 |
| 224 | vim_tiny | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.4302 | 1 | 1 |
| 225 | vim_tiny | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 2.3644 | 1 | 1 |
| 226 | vim_tiny | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 2.5926 | 1 | 1 |
| 227 | vim_tiny | label_smoothing | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.001 | 15.00 | 2.3962 | 1 | 1 |
| 228 | vim_tiny | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.001 | 15.00 | 2.0884 | 1 | 1 |
| 229 | vim_tiny | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 15.00 | 1.8618 | 1 | 1 |
| 230 | vim_tiny | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 1.9991 | 1 | 1 |
| 231 | cbam_resnet | cross_entropy | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.4452 | 1 | 1 |
| 232 | cbam_resnet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.3179 | 1 | 1 |
| 233 | cbam_resnet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 15.00 | 2.1768 | 1 | 1 |
| 234 | cbam_resnet | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 2.3079 | 1 | 1 |
| 235 | cbam_resnet | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.0230 | 1 | 1 |
| 236 | cbam_resnet | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 2.1407 | 1 | 1 |
| 237 | dpn | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 15.00 | 2.3280 | 1 | 1 |
| 238 | dpn | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 2.2879 | 1 | 1 |
| 239 | dpn | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 5.0513 | 1 | 1 |
| 240 | dpn | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.5508 | 1 | 1 |
| 241 | dpn | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 2.2998 | 1 | 1 |
| 242 | dpn | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 2.2972 | 1 | 1 |
| 243 | dpn | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 1.8582 | 1 | 1 |
| 244 | dpn | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 1.8547 | 1 | 1 |
| 245 | dpn | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.8355 | 1 | 1 |
| 246 | hardnet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.3231 | 1 | 1 |
| 247 | hardnet | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 15.00 | 10494066.2857 | 1 | 1 |
| 248 | hardnet | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 2.4895 | 1 | 1 |
| 249 | hardnet | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 948.6214 | 1 | 1 |
| 250 | mobilenet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 2.2862 | 1 | 1 |
| 251 | mobilenet | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.2929 | 1 | 1 |
| 252 | mobilenet | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.2903 | 1 | 1 |
| 253 | mobilenet | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 1.8599 | 1 | 1 |
| 254 | mobilenet | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 1.8397 | 1 | 1 |
| 255 | mobilenet | focal_loss | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 15.00 | 1.8490 | 1 | 1 |
| 256 | res2net | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 3.5698 | 1 | 1 |
| 257 | res2net | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 15.00 | 2.7503 | 1 | 1 |
| 258 | res2net | cross_entropy | adamw | batch_size=16, lr=0.0001, weight_decay=0.001 | 15.00 | 2.2916 | 1 | 1 |
| 259 | res2net | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 2.5910 | 1 | 1 |
| 260 | res2net | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.001 | 15.00 | 2.4914 | 1 | 1 |
| 261 | res2net | focal_loss | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 15.00 | 1.8569 | 1 | 1 |
| 262 | res2net | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 15.00 | 1.8726 | 1 | 1 |
| 263 | res2net | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.0700 | 1 | 1 |
| 264 | res2net | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 1.9777 | 1 | 1 |
| 265 | swin_tiny | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 9.7795 | 1 | 1 |
| 266 | swin_tiny | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.4577 | 1 | 1 |
| 267 | swin_tiny | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 15.00 | 5.7692 | 1 | 1 |
| 268 | eca_resnet | cross_entropy | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.6801 | 1 | 1 |
| 269 | eca_resnet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 15.00 | 2.4610 | 1 | 1 |
| 270 | eca_resnet | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 4.2169 | 1 | 1 |
| 271 | eca_resnet | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 3.5981 | 1 | 1 |
| 272 | eca_resnet | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 2.2990 | 1 | 1 |
| 273 | eca_resnet | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.7166 | 1 | 1 |
| 274 | eca_resnet | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 2.6473 | 1 | 1 |
| 275 | eca_resnet | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 4.1743 | 1 | 1 |
| 276 | eca_resnet | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 1.8597 | 1 | 1 |
| 277 | eca_resnet | focal_loss | adamw | batch_size=16, lr=0.0001, weight_decay=0.0001 | 15.00 | 1.8829 | 1 | 1 |
| 278 | eca_resnet | focal_loss | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 15.00 | 1.9860 | 1 | 1 |
| 279 | hrnet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 2.3678 | 1 | 1 |
| 280 | hrnet | focal_loss | sgd | batch_size=16, lr=0.0001, weight_decay=0.0 | 15.00 | 1.8646 | 1 | 1 |
| 281 | hrnet | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 1.8314 | 1 | 1 |
| 282 | mobilenetv2 | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 2.8076 | 1 | 1 |
| 283 | mobilenetv2 | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 15.00 | 2.3085 | 1 | 1 |
| 284 | mobilenetv2 | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 2.2952 | 1 | 1 |
| 285 | mobilenetv2 | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 2.3321 | 1 | 1 |
| 286 | mobilenetv2 | label_smoothing | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.3051 | 1 | 1 |
| 287 | mobilenetv2 | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 2.7392 | 1 | 1 |
| 288 | mobilenetv2 | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.4903 | 1 | 1 |
| 289 | mobilenetv2 | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 2.1646 | 1 | 1 |
| 290 | mobilenetv2 | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 15.00 | 1.8532 | 1 | 1 |
| 291 | mobilenetv2 | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 1.8719 | 1 | 1 |
| 292 | resnet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 5.9574 | 1 | 1 |
| 293 | resnet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.3167 | 1 | 1 |
| 294 | resnet | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 3.5970 | 1 | 1 |
| 295 | resnet | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 2.2899 | 1 | 1 |
| 296 | resnet | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.6167 | 1 | 1 |
| 297 | resnet | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 2.9467 | 1 | 1 |
| 298 | resnet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 383.2679 | 1 | 1 |
| 299 | resnet | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 1.8571 | 1 | 1 |
| 300 | resnet | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 1.8952 | 1 | 1 |
| 301 | resnet | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 15.00 | 140.0029 | 1 | 1 |
| 302 | van | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 2.3424 | 1 | 1 |
| 303 | van | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 2.2958 | 1 | 1 |
| 304 | van | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 2.3270 | 1 | 1 |
| 305 | van | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 15.00 | 2.3047 | 1 | 1 |
| 306 | van | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.3035 | 1 | 1 |
| 307 | van | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 1.8665 | 1 | 1 |
| 308 | van | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 15.00 | 1.8627 | 1 | 1 |
| 309 | van | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 1.9896 | 1 | 1 |
| 310 | convnext | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 15.00 | 28.3902 | 1 | 1 |
| 311 | convnext | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 15.00 | 5.3236 | 1 | 1 |
| 312 | inception_resnet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.4038 | 1 | 1 |
| 313 | inception_resnet | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 15.00 | 2.2978 | 1 | 1 |
| 314 | inception_resnet | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.3476 | 1 | 1 |
| 315 | inception_resnet | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.2989 | 1 | 1 |
| 316 | inception_resnet | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 2.2973 | 1 | 1 |
| 317 | inception_resnet | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.3395 | 1 | 1 |
| 318 | inception_resnet | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 1.9374 | 1 | 1 |
| 319 | inception_resnet | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 1.8652 | 1 | 1 |
| 320 | inception_resnet | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 1.9671 | 1 | 1 |
| 321 | inception_resnet | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 1.9015 | 1 | 1 |
| 322 | mobilenetv3 | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.2973 | 1 | 1 |
| 323 | mobilenetv3 | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 15.00 | 2.4169 | 1 | 1 |
| 324 | mobilenetv3 | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 2.2995 | 1 | 1 |
| 325 | mobilenetv3 | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 2.3017 | 1 | 1 |
| 326 | mobilenetv3 | label_smoothing | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.3010 | 1 | 1 |
| 327 | mobilenetv3 | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 1.8593 | 1 | 1 |
| 328 | mobilenetv3 | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 15.00 | 1.8623 | 1 | 1 |
| 329 | resnext | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 2.2976 | 1 | 1 |
| 330 | resnext | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 15.00 | 3.8988 | 1 | 1 |
| 331 | resnext | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.5666 | 1 | 1 |
| 332 | resnext | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.3024 | 1 | 1 |
| 333 | resnext | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 2.2947 | 1 | 1 |
| 334 | resnext | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.2973 | 1 | 1 |
| 335 | resnext | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 1.9458 | 1 | 1 |
| 336 | resnext | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 15.00 | 2.2408 | 1 | 1 |
| 337 | resnext | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 2.4457 | 1 | 1 |
| 338 | vgg | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.2968 | 1 | 1 |
| 339 | vgg | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 2.3270 | 1 | 1 |
| 340 | vgg | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 2.2957 | 1 | 1 |
| 341 | vgg | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 1.8707 | 1 | 1 |
| 342 | vgg | focal_loss | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.001 | 15.00 | 1.8608 | 1 | 1 |
| 343 | alexnet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 2.2950 | 1 | 1 |
| 344 | alexnet | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.2951 | 1 | 1 |
| 345 | alexnet | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 2.2922 | 1 | 1 |
| 346 | alexnet | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 1.8447 | 1 | 1 |
| 347 | alexnet | focal_loss | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 1.8762 | 1 | 1 |
| 348 | darknet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 15.00 | 124791.1551 | 1 | 1 |
| 349 | darknet | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 11403.9047 | 1 | 1 |
| 350 | darknet | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 970779.9911 | 1 | 1 |
| 351 | darknet | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 15.00 | 547.4644 | 1 | 1 |
| 352 | darknet | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 46056.0580 | 1 | 1 |
| 353 | darknet | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.001 | 15.00 | 1.8733 | 1 | 1 |
| 354 | googlenet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 2.2950 | 1 | 1 |
| 355 | googlenet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.2910 | 1 | 1 |
| 356 | googlenet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 2.2951 | 1 | 1 |
| 357 | googlenet | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 15.00 | 2.2932 | 1 | 1 |
| 358 | googlenet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 1.8452 | 1 | 1 |
| 359 | googlenet | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 1.8583 | 1 | 1 |
| 360 | lstm | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.001 | 15.00 | 2.3024 | 1 | 1 |
| 361 | lstm | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 2.2888 | 1 | 1 |
| 362 | lstm | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 2.2916 | 1 | 1 |
| 363 | lstm | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 1.8567 | 1 | 1 |
| 364 | lstm | focal_loss | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 1.8539 | 1 | 1 |
| 365 | regnet | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 2.3123 | 1 | 1 |
| 366 | regnet | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 15.00 | 3.1398 | 1 | 1 |
| 367 | regnet | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 2.3155 | 1 | 1 |
| 368 | regnet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 2.7927 | 1 | 1 |
| 369 | regnet | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 15.00 | 2.3033 | 1 | 1 |
| 370 | regnet | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.5566 | 1 | 1 |
| 371 | regnet | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 1.8721 | 1 | 1 |
| 372 | regnet | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 1.8724 | 1 | 1 |
| 373 | simple_cnn | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 2.2511 | 1 | 1 |
| 374 | simple_cnn | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.2915 | 1 | 1 |
| 375 | simple_cnn | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 1.8232 | 1 | 1 |
| 376 | wide_resnet | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.5773 | 1 | 1 |
| 377 | wide_resnet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.5074 | 1 | 1 |
| 378 | wide_resnet | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 2.2988 | 1 | 1 |
| 379 | wide_resnet | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.5801 | 1 | 1 |
| 380 | wide_resnet | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 2.2312 | 1 | 1 |
| 381 | wide_resnet | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 1.8452 | 1 | 1 |
| 382 | wide_resnet | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 2.1656 | 1 | 1 |
| 383 | wide_resnet | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.1137 | 1 | 1 |
| 384 | sknet | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 14.00 | 2.3016 | 1 | 1 |
| 385 | repvgg | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 13.00 | 2.1188 | 1 | 1 |
| 386 | repghost | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 13.00 | 2.3694 | 1 | 1 |
| 387 | resnet | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 13.00 | 2.3032 | 1 | 1 |
| 388 | convnext | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 13.00 | 2.5914 | 1 | 1 |
| 389 | gru | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 12.00 | 2.2880 | 1 | 1 |
| 390 | gru | focal_loss | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0 | 12.00 | 1.8415 | 1 | 1 |
| 391 | mnasnet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 12.00 | 2.2974 | 1 | 1 |
| 392 | squeezenet | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 12.00 | 2.2857 | 1 | 1 |
| 393 | squeezenet | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 12.00 | 2.2970 | 1 | 1 |
| 394 | squeezenet | focal_loss | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 12.00 | 1.8618 | 1 | 1 |
| 395 | bert | label_smoothing | adamw | batch_size=16, lr=0.0001, weight_decay=0.0001 | 12.00 | 2.3009 | 1 | 1 |
| 396 | deit | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 12.00 | 10.7555 | 1 | 1 |
| 397 | deit | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 12.00 | 2.2286 | 1 | 1 |
| 398 | repghost | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 12.00 | 2.2885 | 1 | 1 |
| 399 | ghostnet | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 12.00 | 1.8636 | 1 | 1 |
| 400 | ghostnet | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 12.00 | 1.8485 | 1 | 1 |
| 401 | lenet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 12.00 | 2.2919 | 1 | 1 |
| 402 | lenet | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 12.00 | 2.2945 | 1 | 1 |
| 403 | lenet | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 12.00 | 2.2919 | 1 | 1 |
| 404 | poolformer | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 12.00 | 32.6416 | 1 | 1 |
| 405 | shufflenet | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 12.00 | 1.8599 | 1 | 1 |
| 406 | vit | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 12.00 | 2.3205 | 1 | 1 |
| 407 | vit | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 12.00 | 11.9198 | 1 | 1 |
| 408 | nin | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 12.00 | 2.2984 | 1 | 1 |
| 409 | nin | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 12.00 | 1.8599 | 1 | 1 |
| 410 | vim_tiny | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 12.00 | 3102.1999 | 1 | 1 |
| 411 | cbam_resnet | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 12.00 | 2.2979 | 1 | 1 |
| 412 | hardnet | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 12.00 | 61.0705 | 1 | 1 |
| 413 | hardnet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 12.00 | 6.0950 | 1 | 1 |
| 414 | hardnet | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 12.00 | 1.8586 | 1 | 1 |
| 415 | mobilenet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 12.00 | 2.3643 | 1 | 1 |
| 416 | mobilenet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 12.00 | 2.3391 | 1 | 1 |
| 417 | mobilenet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 12.00 | 2.5398 | 1 | 1 |
| 418 | res2net | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 12.00 | 152812041.1429 | 1 | 1 |
| 419 | swin_tiny | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 12.00 | 78.9143 | 1 | 1 |
| 420 | hrnet | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 12.00 | 2.2894 | 1 | 1 |
| 421 | inception_resnet | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 12.00 | 9619038.4615 | 1 | 1 |
| 422 | vgg | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 12.00 | 22.3630 | 1 | 1 |
| 423 | alexnet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 12.00 | 4.8067 | 1 | 1 |
| 424 | alexnet | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 12.00 | 2.2995 | 1 | 1 |
| 425 | alexnet | label_smoothing | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 12.00 | 2.2932 | 1 | 1 |
| 426 | alexnet | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 12.00 | 1.8579 | 1 | 1 |
| 427 | darknet | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 12.00 | 1.9078 | 1 | 1 |
| 428 | googlenet | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 12.00 | 2.2993 | 1 | 1 |
| 429 | googlenet | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 12.00 | 2.3014 | 1 | 1 |
| 430 | lstm | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 12.00 | 2.2899 | 1 | 1 |
| 431 | lstm | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 12.00 | 2.3005 | 1 | 1 |
| 432 | lstm | cross_entropy | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 12.00 | 2.2877 | 1 | 1 |
| 433 | lstm | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 12.00 | 2.2989 | 1 | 1 |
| 434 | regnet | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 12.00 | 2.0780 | 1 | 1 |
| 435 | wide_resnet | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 12.00 | 2.2919 | 1 | 1 |
| 436 | wide_resnet | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 12.00 | 5.2696 | 1 | 1 |
| 437 | wide_resnet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 12.00 | 27142.6420 | 1 | 1 |
| 438 | gru | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 11.00 | 2.3031 | 1 | 1 |
| 439 | gru | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 11.00 | 2.2973 | 1 | 1 |
| 440 | gru | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 11.00 | 1.8592 | 1 | 1 |
| 441 | mnasnet | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 11.00 | 3.7669 | 1 | 1 |
| 442 | mnasnet | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 11.00 | 1.8612 | 1 | 1 |
| 443 | ghostnet | label_smoothing | adam | batch_size=16, lr=0.0001, weight_decay=0.001 | 11.00 | 2.3028 | 1 | 1 |
| 444 | poolformer | focal_loss | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 11.00 | 41.8652 | 1 | 1 |
| 445 | shufflenet | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 11.00 | 2.3025 | 1 | 1 |
| 446 | shufflenet | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.001 | 11.00 | 2.3021 | 1 | 1 |
| 447 | vit | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 11.00 | 2.6783 | 1 | 1 |
| 448 | coord_resnet | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 11.00 | 2.3055 | 1 | 1 |
| 449 | nin | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 11.00 | 2.3043 | 1 | 1 |
| 450 | nin | label_smoothing | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 11.00 | 2.3015 | 1 | 1 |
| 451 | se_resnet | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 11.00 | 2.3120 | 1 | 1 |
| 452 | se_resnet | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 11.00 | 1.8584 | 1 | 1 |
| 453 | dpn | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 11.00 | 2.3335 | 1 | 1 |
| 454 | mobilenet | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 11.00 | 2.1232 | 1 | 1 |
| 455 | res2net | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 11.00 | 742371.7589 | 1 | 1 |
| 456 | swin_tiny | focal_loss | adam | batch_size=16, lr=0.0001, weight_decay=0.001 | 11.00 | 2.5340 | 1 | 1 |
| 457 | hrnet | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.001 | 11.00 | 2.2999 | 1 | 1 |
| 458 | hrnet | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 11.00 | 2.3080 | 1 | 1 |
| 459 | mobilenetv2 | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 11.00 | 2.2896 | 1 | 1 |
| 460 | inception_resnet | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 11.00 | 222.0819 | 1 | 1 |
| 461 | mobilenetv3 | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 11.00 | 2.4572 | 1 | 1 |
| 462 | mobilenetv3 | focal_loss | adamw | batch_size=16, lr=0.0001, weight_decay=0.001 | 11.00 | 1.8657 | 1 | 1 |
| 463 | mobilenetv3 | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 11.00 | 517.7137 | 1 | 1 |
| 464 | resnext | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 11.00 | 2.3253 | 1 | 1 |
| 465 | vgg | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 11.00 | 101518147584.0000 | 1 | 1 |
| 466 | vgg | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.0 | 11.00 | 2.3021 | 1 | 1 |
| 467 | darknet | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 11.00 | 2.7274 | 1 | 1 |
| 468 | googlenet | cross_entropy | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 11.00 | 2.3059 | 1 | 1 |
| 469 | googlenet | label_smoothing | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.001 | 11.00 | 2.3017 | 1 | 1 |
| 470 | googlenet | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 11.00 | 1.8688 | 1 | 1 |
| 471 | regnet | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 11.00 | 2.2921 | 1 | 1 |
| 472 | regnet | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 11.00 | 2.3117 | 1 | 1 |
| 473 | capsnet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 10.00 | 2.3026 | 1 | 1 |
| 474 | capsnet | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 10.00 | 2.3026 | 1 | 1 |
| 475 | capsnet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 10.00 | 2.3026 | 1 | 1 |
| 476 | capsnet | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 10.00 | 2.3026 | 1 | 1 |
| 477 | capsnet | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 10.00 | 2.3026 | 1 | 1 |
| 478 | capsnet | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 10.00 | 2.3026 | 1 | 1 |
| 479 | capsnet | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 10.00 | 2.3026 | 1 | 1 |
| 480 | capsnet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 10.00 | 2.3026 | 1 | 1 |
| 481 | capsnet | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 10.00 | 1.8651 | 1 | 1 |
| 482 | capsnet | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 10.00 | 1.8651 | 1 | 1 |
| 483 | capsnet | focal_loss | adamw | batch_size=16, lr=0.0001, weight_decay=0.0001 | 10.00 | 1.8651 | 1 | 1 |
| 484 | capsnet | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 10.00 | 1.8651 | 1 | 1 |
| 485 | densenet | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 10.00 | 83868.5469 | 1 | 1 |
| 486 | densenet | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.0001 | 10.00 | 2.2974 | 1 | 1 |
| 487 | mnasnet | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 10.00 | 7.3984 | 1 | 1 |
| 488 | bert | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 10.00 | 2.3024 | 1 | 1 |
| 489 | deit | focal_loss | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 10.00 | 2.0387 | 1 | 1 |
| 490 | repghost | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 10.00 | 2.3190 | 1 | 1 |
| 491 | repghost | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 10.00 | 2.3146 | 1 | 1 |
| 492 | cspnet | label_smoothing | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 10.00 | 2.2984 | 1 | 1 |
| 493 | cspnet | focal_loss | adam | batch_size=16, lr=0.0001, weight_decay=0.0001 | 10.00 | 1.8661 | 1 | 1 |
| 494 | lenet | cross_entropy | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 10.00 | 2.2977 | 1 | 1 |
| 495 | lenet | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 10.00 | 1.8555 | 1 | 1 |
| 496 | lenet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 10.00 | 1.8660 | 1 | 1 |
| 497 | poolformer | cross_entropy | sgd | batch_size=16, lr=0.0001, weight_decay=0.0001 | 10.00 | 2.4269 | 1 | 1 |
| 498 | poolformer | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 10.00 | 4.3791 | 1 | 1 |
| 499 | poolformer | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 10.00 | 29.2418 | 1 | 1 |
| 500 | poolformer | label_smoothing | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 10.00 | 32.2482 | 1 | 1 |
| 501 | poolformer | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 10.00 | 85.8970 | 1 | 1 |
| 502 | shufflenet | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 10.00 | 2.2965 | 1 | 1 |
| 503 | shufflenet | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 10.00 | 1.8632 | 1 | 1 |
| 504 | vit | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 10.00 | 2.5555 | 1 | 1 |
| 505 | vit | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 10.00 | 2.6931 | 1 | 1 |
| 506 | efficientnetv2 | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 10.00 | 2.1958 | 1 | 1 |
| 507 | nin | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 10.00 | 1.9067 | 1 | 1 |
| 508 | se_resnet | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 10.00 | 2.3072 | 1 | 1 |
| 509 | vim_tiny | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 10.00 | 1382538.0000 | 1 | 1 |
| 510 | cbam_resnet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 10.00 | 1.8623 | 1 | 1 |
| 511 | dpn | cross_entropy | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 10.00 | 2.4482 | 1 | 1 |
| 512 | hardnet | cross_entropy | sgd | batch_size=16, lr=0.0001, weight_decay=0.0001 | 10.00 | 2.3114 | 1 | 1 |
| 513 | hardnet | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 10.00 | 2.3058 | 1 | 1 |
| 514 | mobilenet | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 10.00 | 2.2997 | 1 | 1 |
| 515 | mobilenet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 10.00 | 2.3253 | 1 | 1 |
| 516 | res2net | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 10.00 | 2.2998 | 1 | 1 |
| 517 | swin_tiny | label_smoothing | adam | batch_size=16, lr=0.0001, weight_decay=0.001 | 10.00 | 2.6777 | 1 | 1 |
| 518 | swin_tiny | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 10.00 | 10.8261 | 1 | 1 |
| 519 | swin_tiny | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 10.00 | 50.2507 | 1 | 1 |
| 520 | mobilenetv2 | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 10.00 | 2.3026 | 1 | 1 |
| 521 | resnet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 10.00 | 2.3053 | 1 | 1 |
| 522 | van | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 10.00 | 5.4478 | 1 | 1 |
| 523 | van | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 10.00 | nan | 1 | 1 |
| 524 | convnext | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 10.00 | 2.1126 | 1 | 1 |
| 525 | mobilenetv3 | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 10.00 | 2.4109 | 1 | 1 |
| 526 | mobilenetv3 | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 10.00 | 2.3009 | 1 | 1 |
| 527 | resnext | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 10.00 | 2.5222 | 1 | 1 |
| 528 | resnext | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 10.00 | 2.1325 | 1 | 1 |
| 529 | alexnet | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 10.00 | 2.3030 | 1 | 1 |
| 530 | alexnet | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 10.00 | 40.4074 | 1 | 1 |
| 531 | darknet | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 10.00 | 4577951.1071 | 1 | 1 |
| 532 | googlenet | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 10.00 | 1.8688 | 1 | 1 |
| 533 | lstm | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 10.00 | 1.8758 | 1 | 1 |
| 534 | lstm | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 10.00 | 1.8625 | 1 | 1 |
| 535 | regnet | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 10.00 | 1.8840 | 1 | 1 |
| 536 | simple_cnn | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 10.00 | 2.2949 | 1 | 1 |
| 537 | wide_resnet | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 10.00 | 2.3031 | 1 | 1 |
| 538 | ghostnet | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 9.00 | 2.3037 | 1 | 1 |
| 539 | poolformer | label_smoothing | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 9.00 | 2.6022 | 1 | 1 |
| 540 | nin | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 9.00 | 2.3037 | 1 | 1 |
| 541 | nin | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 9.00 | 2.3024 | 1 | 1 |
| 542 | dpn | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 9.00 | 1.8663 | 1 | 1 |
| 543 | swin_tiny | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 9.00 | 6.7846 | 1 | 1 |
| 544 | hrnet | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 9.00 | 2.3058 | 1 | 1 |
| 545 | darknet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 9.00 | 8103.5980 | 1 | 1 |
| 546 | darknet | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 9.00 | 15113.8825 | 1 | 1 |
| 547 | lstm | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 9.00 | 2.3116 | 1 | 1 |
| 548 | vgg | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 8.00 | 2.3037 | 1 | 1 |
| 549 | alexnet | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 8.00 | 2.3028 | 1 | 1 |
| 550 | vgg | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 6.00 | 1.8659 | 1 | 1 |
| 551 | darknet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 6.00 | 866877677.7143 | 1 | 1 |
| 552 | simple_cnn | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 6.00 | 1.8716 | 1 | 1 |

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
| 1 | wgan | wasserstein | adam | batch_size=16, lr=0.0001, weight_decay=0.001 | -0.0023 | -0.0086 | 1 | 1 |
| 2 | wgan | bce | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 0.0692 | -0.1914 | 1 | 1 |
| 3 | wgan | wasserstein | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 0.0948 | -0.2443 | 1 | 1 |
| 4 | wgan | bce | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 0.1861 | -0.3481 | 1 | 1 |
| 5 | cgan | wasserstein | sgd | batch_size=16, lr=0.0001, weight_decay=0.0001 | 0.6487 | 1.4086 | 1 | 1 |
| 6 | cgan | bce | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 0.7743 | 1.3160 | 1 | 1 |
| 7 | dcgan | bce | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 0.7784 | 1.2320 | 1 | 1 |
| 8 | dcgan | wasserstein | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 0.8005 | 1.3900 | 1 | 1 |
| 9 | cgan | bce | adam | batch_size=16, lr=0.0001, weight_decay=0.001 | 0.8492 | 1.3427 | 1 | 1 |
| 10 | cgan | wasserstein | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 0.8991 | 1.2932 | 1 | 1 |
| 11 | dcgan | wasserstein | adam | batch_size=16, lr=0.0001, weight_decay=0.001 | 1.0781 | 0.9915 | 1 | 1 |
| 12 | dcgan | bce | adamw | batch_size=16, lr=0.0001, weight_decay=0.0001 | 1.0809 | 0.9711 | 1 | 1 |
| 13 | dcgan | wasserstein | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 2.2396 | 71.6814 | 1 | 1 |
| 14 | cgan | wasserstein | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 2.4516 | 4.2144 | 1 | 1 |
| 15 | cgan | bce | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 3.2156 | 1.4820 | 1 | 1 |
| 16 | cgan | bce | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 3.3194 | 1.2085 | 1 | 1 |
| 17 | cgan | wasserstein | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 4.6290 | 7.1197 | 1 | 1 |
| 18 | dcgan | bce | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 5.2191 | 0.4549 | 1 | 1 |
| 19 | dcgan | wasserstein | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 5.3744 | 1.0035 | 1 | 1 |
| 20 | dcgan | bce | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 5.4860 | 0.7620 | 1 | 1 |

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
