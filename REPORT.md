# MNIST Regression Report

Generated: 2026-05-17T15:29:42
Mode: quick-test
Max epochs: 1 | Early-stop patience: 3 | Min delta: 0.1 | NAS trials per config: 2 | Workers: 8 | Max batch: 16
Total wall time: 370.2s

Training stops when validation metric shows no significant improvement for 3 consecutive epochs.

## Classification Models (ranked by test accuracy)

| Rank | Model | Loss | Optimizer | Hyperparameters | Test Acc (%) | Test Loss | Epochs Run | Convergence Epoch |
|------|-------|------|-----------|-----------------|--------------|-----------|------------|-------------------|
| 1 | mlp | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 75.00 | 1.3886 | 1 | 1 |
| 2 | mlp | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 63.00 | 1.6474 | 1 | 1 |
| 3 | mlp | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 62.00 | 2.1196 | 1 | 1 |
| 4 | mlp | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 58.00 | 1.8839 | 1 | 1 |
| 5 | mlp | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 55.00 | 2.2281 | 1 | 1 |
| 6 | mlp | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 50.00 | 2.2150 | 1 | 1 |
| 7 | mlp | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 47.00 | 3.5874 | 1 | 1 |
| 8 | mlp | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 45.00 | 1.7836 | 1 | 1 |
| 9 | ghostnet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 42.00 | 1.8447 | 1 | 1 |
| 10 | cbam_resnet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 38.00 | 2.1600 | 1 | 1 |
| 11 | hrnet | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 35.00 | 2.1944 | 1 | 1 |
| 12 | se_resnet | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 35.00 | 2.0623 | 1 | 1 |
| 13 | mlp | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 33.00 | 1.7978 | 1 | 1 |
| 14 | convnext | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 33.00 | 2.0599 | 1 | 1 |
| 15 | vgg | focal_loss | adamw | batch_size=16, lr=0.0001, weight_decay=0.0001 | 32.00 | 1.7979 | 1 | 1 |
| 16 | repghost | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 31.00 | 2.1433 | 1 | 1 |
| 17 | convnext | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 30.00 | 1.8497 | 1 | 1 |
| 18 | convnext | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 28.00 | 2.3057 | 1 | 1 |
| 19 | convnext | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 26.00 | 2.0174 | 1 | 1 |
| 20 | convnext | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 26.00 | 2.8575 | 1 | 1 |
| 21 | convnext | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 26.00 | 2.3158 | 1 | 1 |
| 22 | lenet | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 25.00 | 1.8460 | 1 | 1 |
| 23 | hrnet | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 25.00 | 2.0838 | 1 | 1 |
| 24 | convnext | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 25.00 | 2.1879 | 1 | 1 |
| 25 | convnext | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 25.00 | 3.4535 | 1 | 1 |
| 26 | convnext | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 25.00 | 1.8308 | 1 | 1 |
| 27 | hrnet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 24.00 | 1.6941 | 1 | 1 |
| 28 | coord_resnet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 24.00 | 2.6276 | 1 | 1 |
| 29 | vim_tiny | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 24.00 | 2.1850 | 1 | 1 |
| 30 | repghost | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 24.00 | 2.2956 | 1 | 1 |
| 31 | ghostnet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 23.00 | 2.2673 | 1 | 1 |
| 32 | poolformer | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 23.00 | 2.9675 | 1 | 1 |
| 33 | bert | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 23.00 | 2.3020 | 1 | 1 |
| 34 | lenet | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 22.00 | 1.8624 | 1 | 1 |
| 35 | hrnet | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 22.00 | 1.8433 | 1 | 1 |
| 36 | vim_tiny | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 22.00 | 2.3810 | 1 | 1 |
| 37 | bert | focal_loss | adam | batch_size=16, lr=0.0001, weight_decay=0.001 | 22.00 | 1.8634 | 1 | 1 |
| 38 | cbam_resnet | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 22.00 | 3.6154 | 1 | 1 |
| 39 | repvgg | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 21.00 | 2.2310 | 1 | 1 |
| 40 | coord_resnet | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 21.00 | 3.6418 | 1 | 1 |
| 41 | deit | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 21.00 | 2.6050 | 1 | 1 |
| 42 | hardnet | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 21.00 | 2.4884 | 1 | 1 |
| 43 | densenet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 20.00 | 1.8615 | 1 | 1 |
| 44 | mlp | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 20.00 | 2.2887 | 1 | 1 |
| 45 | cbam_resnet | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 20.00 | 8.1537 | 1 | 1 |
| 46 | simple_cnn | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 20.00 | 2.2960 | 1 | 1 |
| 47 | repvgg | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 19.00 | 1.9331 | 1 | 1 |
| 48 | van | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 19.00 | nan | 1 | 1 |
| 49 | repghost | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 19.00 | 2.8007 | 1 | 1 |
| 50 | inception_resnet | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 19.00 | 5.9486 | 1 | 1 |
| 51 | simple_cnn | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 19.00 | 2.2688 | 1 | 1 |
| 52 | repvgg | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 18.00 | 1.9533 | 1 | 1 |
| 53 | squeezenet | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 18.00 | 1.8499 | 1 | 1 |
| 54 | lcnet | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 18.00 | 2.5466 | 1 | 1 |
| 55 | deit | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 18.00 | 2.2664 | 1 | 1 |
| 56 | sknet | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 18.00 | 115.8828 | 1 | 1 |
| 57 | hardnet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 18.00 | 13.0246 | 1 | 1 |
| 58 | swin_tiny | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 18.00 | 2.7120 | 1 | 1 |
| 59 | resnext | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 18.00 | 1.8373 | 1 | 1 |
| 60 | vgg | label_smoothing | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 18.00 | 2.2784 | 1 | 1 |
| 61 | vgg | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 18.00 | 1.8439 | 1 | 1 |
| 62 | alexnet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 18.00 | 2.2954 | 1 | 1 |
| 63 | squeezenet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 17.00 | 2.2895 | 1 | 1 |
| 64 | lenet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 17.00 | 2.2866 | 1 | 1 |
| 65 | coord_resnet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 17.00 | 5.3274 | 1 | 1 |
| 66 | repghost | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 17.00 | 2.5822 | 1 | 1 |
| 67 | cbam_resnet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 17.00 | 2.3496 | 1 | 1 |
| 68 | hardnet | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 17.00 | 6.9668 | 1 | 1 |
| 69 | convnext | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 17.00 | 3.0612 | 1 | 1 |
| 70 | eca_resnet | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 16.00 | 2.6612 | 1 | 1 |
| 71 | resnet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 16.00 | 8.5390 | 1 | 1 |
| 72 | resnet | label_smoothing | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 16.00 | 2.2952 | 1 | 1 |
| 73 | coord_resnet | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 16.00 | 1.9153 | 1 | 1 |
| 74 | se_resnet | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 16.00 | 3.3646 | 1 | 1 |
| 75 | deit | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 16.00 | 2.2455 | 1 | 1 |
| 76 | repghost | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 16.00 | 1.8651 | 1 | 1 |
| 77 | cbam_resnet | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 16.00 | 2.1708 | 1 | 1 |
| 78 | simple_cnn | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 16.00 | 1.8560 | 1 | 1 |
| 79 | densenet | cross_entropy | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.2922 | 1 | 1 |
| 80 | densenet | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 2.2882 | 1 | 1 |
| 81 | densenet | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.2978 | 1 | 1 |
| 82 | densenet | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 2.2889 | 1 | 1 |
| 83 | densenet | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.2891 | 1 | 1 |
| 84 | densenet | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 15.00 | 5.5238 | 1 | 1 |
| 85 | densenet | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 2.3849 | 1 | 1 |
| 86 | densenet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 15.00 | 20.6916 | 1 | 1 |
| 87 | densenet | focal_loss | adamw | batch_size=16, lr=0.0001, weight_decay=0.0001 | 15.00 | 1.8659 | 1 | 1 |
| 88 | gru | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 15.00 | 2.3075 | 1 | 1 |
| 89 | gru | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 2.2846 | 1 | 1 |
| 90 | gru | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 2.2887 | 1 | 1 |
| 91 | gru | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 1.8563 | 1 | 1 |
| 92 | gru | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 1.8595 | 1 | 1 |
| 93 | gru | focal_loss | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 15.00 | 1.8612 | 1 | 1 |
| 94 | mnasnet | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 2.3001 | 1 | 1 |
| 95 | mnasnet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 2.3061 | 1 | 1 |
| 96 | mnasnet | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 15.00 | 3.2132 | 1 | 1 |
| 97 | mnasnet | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 15.00 | 2.8579 | 1 | 1 |
| 98 | mnasnet | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 15.00 | 2.3007 | 1 | 1 |
| 99 | mnasnet | label_smoothing | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 15.00 | 2.3023 | 1 | 1 |
| 100 | mnasnet | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 1.8496 | 1 | 1 |
| 101 | mnasnet | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 1.8623 | 1 | 1 |
| 102 | mnasnet | focal_loss | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.001 | 15.00 | 1.8589 | 1 | 1 |
| 103 | repvgg | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 2.3086 | 1 | 1 |
| 104 | repvgg | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 2.3092 | 1 | 1 |
| 105 | repvgg | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 2.3525 | 1 | 1 |
| 106 | repvgg | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.4563 | 1 | 1 |
| 107 | repvgg | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.5253 | 1 | 1 |
| 108 | repvgg | label_smoothing | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.3851 | 1 | 1 |
| 109 | repvgg | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 1.8533 | 1 | 1 |
| 110 | repvgg | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 1.9094 | 1 | 1 |
| 111 | squeezenet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.2991 | 1 | 1 |
| 112 | squeezenet | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 2.2931 | 1 | 1 |
| 113 | squeezenet | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 15.00 | 2.2973 | 1 | 1 |
| 114 | squeezenet | label_smoothing | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 15.00 | 2.3022 | 1 | 1 |
| 115 | squeezenet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 15.00 | 1.8505 | 1 | 1 |
| 116 | squeezenet | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 1.8544 | 1 | 1 |
| 117 | cspnet | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 15.00 | 2.2971 | 1 | 1 |
| 118 | cspnet | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 15.00 | 2.2990 | 1 | 1 |
| 119 | cspnet | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.3705 | 1 | 1 |
| 120 | cspnet | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 15.00 | 2.3121 | 1 | 1 |
| 121 | cspnet | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.001 | 15.00 | 2.3409 | 1 | 1 |
| 122 | cspnet | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 2.3059 | 1 | 1 |
| 123 | cspnet | label_smoothing | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.3115 | 1 | 1 |
| 124 | cspnet | focal_loss | adam | batch_size=16, lr=0.0001, weight_decay=0.0001 | 15.00 | 1.8647 | 1 | 1 |
| 125 | cspnet | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 15.00 | 1.8582 | 1 | 1 |
| 126 | cspnet | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.001 | 15.00 | 1.9538 | 1 | 1 |
| 127 | ghostnet | cross_entropy | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.2940 | 1 | 1 |
| 128 | ghostnet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.2856 | 1 | 1 |
| 129 | ghostnet | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 2.2635 | 1 | 1 |
| 130 | ghostnet | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 1.8512 | 1 | 1 |
| 131 | ghostnet | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 15.00 | 1.8694 | 1 | 1 |
| 132 | ghostnet | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 1.8642 | 1 | 1 |
| 133 | lenet | cross_entropy | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 15.00 | 2.2898 | 1 | 1 |
| 134 | lenet | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 2.2895 | 1 | 1 |
| 135 | lenet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 15.00 | 2.2893 | 1 | 1 |
| 136 | lenet | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 15.00 | 2.2938 | 1 | 1 |
| 137 | lenet | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 15.00 | 2.2990 | 1 | 1 |
| 138 | lenet | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 1.8527 | 1 | 1 |
| 139 | poolformer | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 15.00 | 42.8946 | 1 | 1 |
| 140 | poolformer | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 2.3959 | 1 | 1 |
| 141 | poolformer | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.6058 | 1 | 1 |
| 142 | shufflenet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 15.00 | 2.2988 | 1 | 1 |
| 143 | shufflenet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 15.00 | 2.2938 | 1 | 1 |
| 144 | shufflenet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 2.2961 | 1 | 1 |
| 145 | shufflenet | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.3097 | 1 | 1 |
| 146 | shufflenet | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.3041 | 1 | 1 |
| 147 | shufflenet | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 15.00 | 2.2970 | 1 | 1 |
| 148 | shufflenet | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.2923 | 1 | 1 |
| 149 | shufflenet | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.001 | 15.00 | 1.8615 | 1 | 1 |
| 150 | shufflenet | focal_loss | adamw | batch_size=16, lr=0.0001, weight_decay=0.0 | 15.00 | 1.8602 | 1 | 1 |
| 151 | vit | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 15.00 | 2.3090 | 1 | 1 |
| 152 | vit | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.2684 | 1 | 1 |
| 153 | vit | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.2735 | 1 | 1 |
| 154 | vit | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.2951 | 1 | 1 |
| 155 | vit | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.2911 | 1 | 1 |
| 156 | vit | label_smoothing | adamw | batch_size=16, lr=0.0001, weight_decay=0.001 | 15.00 | 2.3249 | 1 | 1 |
| 157 | vit | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 1.8411 | 1 | 1 |
| 158 | vit | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 1.8551 | 1 | 1 |
| 159 | coatnet | cross_entropy | adam | batch_size=16, lr=0.001, weight_decay=0.001 | 15.00 | 2.7900 | 1 | 1 |
| 160 | coatnet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 3.1524 | 1 | 1 |
| 161 | coatnet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 2.5202 | 1 | 1 |
| 162 | coatnet | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 3.0215 | 1 | 1 |
| 163 | coatnet | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.8516 | 1 | 1 |
| 164 | coatnet | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 15.00 | 2.3110 | 1 | 1 |
| 165 | coatnet | label_smoothing | adamw | batch_size=16, lr=0.0001, weight_decay=0.0 | 15.00 | 2.3205 | 1 | 1 |
| 166 | coatnet | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 2.7614 | 1 | 1 |
| 167 | coatnet | focal_loss | adam | batch_size=16, lr=0.0001, weight_decay=0.0001 | 15.00 | 1.8617 | 1 | 1 |
| 168 | coatnet | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 1.8780 | 1 | 1 |
| 169 | coatnet | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.6229 | 1 | 1 |
| 170 | eca_resnet | cross_entropy | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.8686 | 1 | 1 |
| 171 | eca_resnet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 15.00 | 2.4424 | 1 | 1 |
| 172 | eca_resnet | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.8182 | 1 | 1 |
| 173 | eca_resnet | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 18.8567 | 1 | 1 |
| 174 | eca_resnet | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 2.8419 | 1 | 1 |
| 175 | eca_resnet | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 2.2884 | 1 | 1 |
| 176 | eca_resnet | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.7327 | 1 | 1 |
| 177 | eca_resnet | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 2.8542 | 1 | 1 |
| 178 | eca_resnet | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 1.8496 | 1 | 1 |
| 179 | eca_resnet | focal_loss | adamw | batch_size=16, lr=0.0001, weight_decay=0.0001 | 15.00 | 1.8640 | 1 | 1 |
| 180 | eca_resnet | focal_loss | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 15.00 | 2.0562 | 1 | 1 |
| 181 | hrnet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 2.2463 | 1 | 1 |
| 182 | hrnet | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.2896 | 1 | 1 |
| 183 | hrnet | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.2884 | 1 | 1 |
| 184 | hrnet | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 2.5203 | 1 | 1 |
| 185 | hrnet | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 1.8611 | 1 | 1 |
| 186 | mobilenetv2 | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.4010 | 1 | 1 |
| 187 | mobilenetv2 | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 15.00 | 2.3869 | 1 | 1 |
| 188 | mobilenetv2 | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 15.00 | 8.7447 | 1 | 1 |
| 189 | mobilenetv2 | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 2.3046 | 1 | 1 |
| 190 | mobilenetv2 | label_smoothing | adam | batch_size=16, lr=0.0001, weight_decay=0.001 | 15.00 | 2.3144 | 1 | 1 |
| 191 | mobilenetv2 | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 3.5062 | 1 | 1 |
| 192 | mobilenetv2 | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 2.1385 | 1 | 1 |
| 193 | mobilenetv2 | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 15.00 | 1.8499 | 1 | 1 |
| 194 | mobilenetv2 | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 1.9029 | 1 | 1 |
| 195 | resnet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 3.9261 | 1 | 1 |
| 196 | resnet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 15.00 | 2.3253 | 1 | 1 |
| 197 | resnet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.3244 | 1 | 1 |
| 198 | resnet | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 3.4760 | 1 | 1 |
| 199 | resnet | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.5477 | 1 | 1 |
| 200 | resnet | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 3.9267 | 1 | 1 |
| 201 | resnet | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.7889 | 1 | 1 |
| 202 | resnet | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 1.8514 | 1 | 1 |
| 203 | resnet | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 1.8900 | 1 | 1 |
| 204 | van | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 2.3470 | 1 | 1 |
| 205 | van | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 15.00 | 17.3974 | 1 | 1 |
| 206 | van | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 637.1320 | 1 | 1 |
| 207 | van | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 15.00 | 19.6076 | 1 | 1 |
| 208 | van | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 1.8622 | 1 | 1 |
| 209 | van | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 1.9429 | 1 | 1 |
| 210 | coord_resnet | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 2.2921 | 1 | 1 |
| 211 | coord_resnet | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.5382 | 1 | 1 |
| 212 | coord_resnet | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 2.4904 | 1 | 1 |
| 213 | coord_resnet | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 3.5408 | 1 | 1 |
| 214 | coord_resnet | focal_loss | adam | batch_size=16, lr=0.0001, weight_decay=0.0001 | 15.00 | 1.8678 | 1 | 1 |
| 215 | coord_resnet | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 1.8543 | 1 | 1 |
| 216 | coord_resnet | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.0087 | 1 | 1 |
| 217 | efficientnetv2 | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 15.00 | 2.6107 | 1 | 1 |
| 218 | efficientnetv2 | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 15.00 | 2.2993 | 1 | 1 |
| 219 | efficientnetv2 | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.2949 | 1 | 1 |
| 220 | efficientnetv2 | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.3061 | 1 | 1 |
| 221 | efficientnetv2 | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 2.3055 | 1 | 1 |
| 222 | efficientnetv2 | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.3038 | 1 | 1 |
| 223 | efficientnetv2 | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.3897 | 1 | 1 |
| 224 | efficientnetv2 | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 1.8861 | 1 | 1 |
| 225 | efficientnetv2 | focal_loss | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 15.00 | 1.8647 | 1 | 1 |
| 226 | efficientnetv2 | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 15.00 | 2.0471 | 1 | 1 |
| 227 | efficientnetv2 | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 15.00 | 5.5988 | 1 | 1 |
| 228 | lcnet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 2.2980 | 1 | 1 |
| 229 | lcnet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 2.3028 | 1 | 1 |
| 230 | lcnet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 2.3015 | 1 | 1 |
| 231 | lcnet | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 2.3221 | 1 | 1 |
| 232 | lcnet | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 15.00 | 1.8698 | 1 | 1 |
| 233 | lcnet | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.001 | 15.00 | 1.8581 | 1 | 1 |
| 234 | lcnet | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 1.8635 | 1 | 1 |
| 235 | nin | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 15.00 | 2.3007 | 1 | 1 |
| 236 | nin | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.2978 | 1 | 1 |
| 237 | nin | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 2.3015 | 1 | 1 |
| 238 | nin | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 2.3022 | 1 | 1 |
| 239 | nin | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 15.00 | 1.8502 | 1 | 1 |
| 240 | nin | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 15.00 | 1.8524 | 1 | 1 |
| 241 | nin | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 1.8583 | 1 | 1 |
| 242 | nin | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 1.8855 | 1 | 1 |
| 243 | se_resnet | cross_entropy | adam | batch_size=16, lr=0.001, weight_decay=0.001 | 15.00 | 2.6315 | 1 | 1 |
| 244 | se_resnet | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.3006 | 1 | 1 |
| 245 | se_resnet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 5.5761 | 1 | 1 |
| 246 | se_resnet | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 2.3285 | 1 | 1 |
| 247 | se_resnet | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 2.4302 | 1 | 1 |
| 248 | se_resnet | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.2682 | 1 | 1 |
| 249 | se_resnet | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 2.2105 | 1 | 1 |
| 250 | vim_tiny | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 15.00 | 2.3126 | 1 | 1 |
| 251 | vim_tiny | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 3.3334 | 1 | 1 |
| 252 | vim_tiny | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 2.3371 | 1 | 1 |
| 253 | vim_tiny | label_smoothing | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.001 | 15.00 | 2.4010 | 1 | 1 |
| 254 | vim_tiny | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.001 | 15.00 | 2.0336 | 1 | 1 |
| 255 | vim_tiny | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 15.00 | 1.8653 | 1 | 1 |
| 256 | vim_tiny | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 1.9855 | 1 | 1 |
| 257 | bert | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.001 | 15.00 | 2.3010 | 1 | 1 |
| 258 | bert | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.3009 | 1 | 1 |
| 259 | bert | cross_entropy | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 2.2944 | 1 | 1 |
| 260 | bert | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 2.3000 | 1 | 1 |
| 261 | bert | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 15.00 | 2.2910 | 1 | 1 |
| 262 | bert | label_smoothing | adamw | batch_size=16, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.3013 | 1 | 1 |
| 263 | bert | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 1.8650 | 1 | 1 |
| 264 | bert | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 1.8637 | 1 | 1 |
| 265 | bert | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 1.8484 | 1 | 1 |
| 266 | deit | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 2.2900 | 1 | 1 |
| 267 | deit | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 1.7952 | 1 | 1 |
| 268 | deit | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 2.0226 | 1 | 1 |
| 269 | deit | focal_loss | adamw | batch_size=16, lr=0.0001, weight_decay=0.0001 | 15.00 | 1.8151 | 1 | 1 |
| 270 | gpt | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 2.2869 | 1 | 1 |
| 271 | gpt | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.3007 | 1 | 1 |
| 272 | gpt | cross_entropy | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 15.00 | 2.2886 | 1 | 1 |
| 273 | gpt | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.2996 | 1 | 1 |
| 274 | gpt | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 15.00 | 2.3003 | 1 | 1 |
| 275 | gpt | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 15.00 | 2.2902 | 1 | 1 |
| 276 | gpt | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 2.2997 | 1 | 1 |
| 277 | gpt | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.001 | 15.00 | 1.8624 | 1 | 1 |
| 278 | gpt | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 15.00 | 1.8628 | 1 | 1 |
| 279 | gpt | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 15.00 | 1.8591 | 1 | 1 |
| 280 | mlp | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.2942 | 1 | 1 |
| 281 | repghost | cross_entropy | adam | batch_size=16, lr=0.001, weight_decay=0.001 | 15.00 | 2.2948 | 1 | 1 |
| 282 | repghost | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 15.00 | 2.2943 | 1 | 1 |
| 283 | repghost | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 1.8205 | 1 | 1 |
| 284 | repghost | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 1.8699 | 1 | 1 |
| 285 | sknet | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 2.3194 | 1 | 1 |
| 286 | sknet | cross_entropy | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 3569.2610 | 1 | 1 |
| 287 | sknet | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 2.5214 | 1 | 1 |
| 288 | sknet | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.3129 | 1 | 1 |
| 289 | sknet | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 15.00 | 2.6062 | 1 | 1 |
| 290 | sknet | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 2.4615 | 1 | 1 |
| 291 | sknet | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 1.9632 | 1 | 1 |
| 292 | sknet | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 1.8592 | 1 | 1 |
| 293 | sknet | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 1.9867 | 1 | 1 |
| 294 | xception | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 15.00 | 2.3009 | 1 | 1 |
| 295 | xception | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.001 | 15.00 | 2.3017 | 1 | 1 |
| 296 | xception | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.3000 | 1 | 1 |
| 297 | xception | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 15.00 | 2.2975 | 1 | 1 |
| 298 | xception | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.2984 | 1 | 1 |
| 299 | xception | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 2.2986 | 1 | 1 |
| 300 | xception | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 1.8630 | 1 | 1 |
| 301 | xception | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 15.00 | 1.9528 | 1 | 1 |
| 302 | xception | focal_loss | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.001 | 15.00 | 1.8635 | 1 | 1 |
| 303 | cbam_resnet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.3034 | 1 | 1 |
| 304 | cbam_resnet | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 2.5633 | 1 | 1 |
| 305 | cbam_resnet | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 2.2952 | 1 | 1 |
| 306 | cbam_resnet | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.2019 | 1 | 1 |
| 307 | cbam_resnet | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 2.5066 | 1 | 1 |
| 308 | dpn | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 2.2969 | 1 | 1 |
| 309 | dpn | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.3729 | 1 | 1 |
| 310 | dpn | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.5927 | 1 | 1 |
| 311 | dpn | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.4333 | 1 | 1 |
| 312 | dpn | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 1.8627 | 1 | 1 |
| 313 | dpn | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 1.8629 | 1 | 1 |
| 314 | dpn | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 1.8545 | 1 | 1 |
| 315 | hardnet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.3245 | 1 | 1 |
| 316 | hardnet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 2.3124 | 1 | 1 |
| 317 | hardnet | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 2.2912 | 1 | 1 |
| 318 | hardnet | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 2.2987 | 1 | 1 |
| 319 | hardnet | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 2.6483 | 1 | 1 |
| 320 | hardnet | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 1.8647 | 1 | 1 |
| 321 | mobilenet | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.001 | 15.00 | 2.3091 | 1 | 1 |
| 322 | mobilenet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.3026 | 1 | 1 |
| 323 | mobilenet | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 15.00 | 2.3054 | 1 | 1 |
| 324 | mobilenet | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.2994 | 1 | 1 |
| 325 | mobilenet | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.2925 | 1 | 1 |
| 326 | mobilenet | label_smoothing | adamw | batch_size=16, lr=0.0001, weight_decay=0.0 | 15.00 | 2.2911 | 1 | 1 |
| 327 | mobilenet | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 2.0650 | 1 | 1 |
| 328 | mobilenet | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 1.8549 | 1 | 1 |
| 329 | res2net | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 3.2277 | 1 | 1 |
| 330 | res2net | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 15.00 | 2.7699 | 1 | 1 |
| 331 | res2net | cross_entropy | adamw | batch_size=16, lr=0.0001, weight_decay=0.001 | 15.00 | 2.3022 | 1 | 1 |
| 332 | res2net | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 2.7674 | 1 | 1 |
| 333 | res2net | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 15.00 | 247150.3772 | 1 | 1 |
| 334 | res2net | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.2866 | 1 | 1 |
| 335 | res2net | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.001 | 15.00 | 2.4265 | 1 | 1 |
| 336 | res2net | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 15.00 | 2087317.3214 | 1 | 1 |
| 337 | res2net | focal_loss | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 15.00 | 1.8807 | 1 | 1 |
| 338 | res2net | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 15.00 | 1.8718 | 1 | 1 |
| 339 | res2net | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.0652 | 1 | 1 |
| 340 | res2net | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 2.3073 | 1 | 1 |
| 341 | swin_tiny | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.7456 | 1 | 1 |
| 342 | swin_tiny | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 6.0554 | 1 | 1 |
| 343 | swin_tiny | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 15.00 | 4.1980 | 1 | 1 |
| 344 | swin_tiny | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 3.0581 | 1 | 1 |
| 345 | swin_tiny | focal_loss | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 15.00 | 1.8970 | 1 | 1 |
| 346 | convnext | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 2.8557 | 1 | 1 |
| 347 | efficientnet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 15.00 | 2.7115 | 1 | 1 |
| 348 | efficientnet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 2.3011 | 1 | 1 |
| 349 | efficientnet | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.3962 | 1 | 1 |
| 350 | efficientnet | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.2930 | 1 | 1 |
| 351 | efficientnet | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.3036 | 1 | 1 |
| 352 | efficientnet | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 15.00 | 4.7493 | 1 | 1 |
| 353 | efficientnet | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.001 | 15.00 | 1.8691 | 1 | 1 |
| 354 | efficientnet | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 1.8601 | 1 | 1 |
| 355 | efficientnet | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 15.00 | 165.7079 | 1 | 1 |
| 356 | inception_resnet | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.2955 | 1 | 1 |
| 357 | inception_resnet | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 15.00 | 2.2988 | 1 | 1 |
| 358 | inception_resnet | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.2948 | 1 | 1 |
| 359 | inception_resnet | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 15.00 | 2.3025 | 1 | 1 |
| 360 | inception_resnet | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.4212 | 1 | 1 |
| 361 | inception_resnet | label_smoothing | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 18.4077 | 1 | 1 |
| 362 | inception_resnet | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.0324 | 1 | 1 |
| 363 | inception_resnet | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 1.8520 | 1 | 1 |
| 364 | inception_resnet | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 1.9220 | 1 | 1 |
| 365 | inception_resnet | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 2.0638 | 1 | 1 |
| 366 | mobilenetv3 | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 15.00 | 2.6537 | 1 | 1 |
| 367 | mobilenetv3 | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 15.00 | 2.3008 | 1 | 1 |
| 368 | mobilenetv3 | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.3000 | 1 | 1 |
| 369 | mobilenetv3 | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 2.4531 | 1 | 1 |
| 370 | mobilenetv3 | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 2.3000 | 1 | 1 |
| 371 | mobilenetv3 | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 2.2991 | 1 | 1 |
| 372 | mobilenetv3 | label_smoothing | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.3012 | 1 | 1 |
| 373 | mobilenetv3 | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 1.8575 | 1 | 1 |
| 374 | resnext | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.2916 | 1 | 1 |
| 375 | resnext | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 15.00 | 2.4563 | 1 | 1 |
| 376 | resnext | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 15.00 | 2.9454 | 1 | 1 |
| 377 | resnext | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.5198 | 1 | 1 |
| 378 | resnext | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.3008 | 1 | 1 |
| 379 | resnext | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 15.00 | 2.6021 | 1 | 1 |
| 380 | resnext | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.3035 | 1 | 1 |
| 381 | resnext | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 1.9790 | 1 | 1 |
| 382 | resnext | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 15.00 | 2.0672 | 1 | 1 |
| 383 | resnext | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 15.00 | 1.8838 | 1 | 1 |
| 384 | vgg | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.2941 | 1 | 1 |
| 385 | vgg | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 2.2958 | 1 | 1 |
| 386 | vgg | focal_loss | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.001 | 15.00 | 1.8631 | 1 | 1 |
| 387 | alexnet | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 15.00 | 2.2976 | 1 | 1 |
| 388 | alexnet | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.2936 | 1 | 1 |
| 389 | alexnet | label_smoothing | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.2882 | 1 | 1 |
| 390 | alexnet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 15.00 | 9933450.1429 | 1 | 1 |
| 391 | alexnet | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 15.00 | 1.8599 | 1 | 1 |
| 392 | alexnet | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 1.8551 | 1 | 1 |
| 393 | darknet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 15.00 | 52941.1565 | 1 | 1 |
| 394 | darknet | cross_entropy | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 15.00 | 2.3787 | 1 | 1 |
| 395 | darknet | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 1.9177 | 1 | 1 |
| 396 | googlenet | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 2.3014 | 1 | 1 |
| 397 | googlenet | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 2.2980 | 1 | 1 |
| 398 | googlenet | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 2.3002 | 1 | 1 |
| 399 | googlenet | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 1.8540 | 1 | 1 |
| 400 | googlenet | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 1.8629 | 1 | 1 |
| 401 | lstm | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 15.00 | 2.2928 | 1 | 1 |
| 402 | lstm | cross_entropy | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 2.2947 | 1 | 1 |
| 403 | lstm | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 2.3041 | 1 | 1 |
| 404 | lstm | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0 | 15.00 | 2.2866 | 1 | 1 |
| 405 | lstm | label_smoothing | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 2.2924 | 1 | 1 |
| 406 | lstm | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 1.8535 | 1 | 1 |
| 407 | lstm | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 1.8578 | 1 | 1 |
| 408 | lstm | focal_loss | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0 | 15.00 | 1.8553 | 1 | 1 |
| 409 | regnet | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 15.00 | 2.6826 | 1 | 1 |
| 410 | regnet | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 2.3230 | 1 | 1 |
| 411 | regnet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 15.00 | 2.4319 | 1 | 1 |
| 412 | regnet | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 15.00 | 2.3095 | 1 | 1 |
| 413 | regnet | focal_loss | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 1.9074 | 1 | 1 |
| 414 | regnet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 1.8675 | 1 | 1 |
| 415 | regnet | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 15.00 | 1.8562 | 1 | 1 |
| 416 | simple_cnn | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 15.00 | 2.2630 | 1 | 1 |
| 417 | simple_cnn | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 2.2526 | 1 | 1 |
| 418 | simple_cnn | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.2910 | 1 | 1 |
| 419 | simple_cnn | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 15.00 | 2.3016 | 1 | 1 |
| 420 | simple_cnn | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 2.2947 | 1 | 1 |
| 421 | simple_cnn | label_smoothing | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.001 | 15.00 | 2.2893 | 1 | 1 |
| 422 | simple_cnn | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 1.8598 | 1 | 1 |
| 423 | simple_cnn | focal_loss | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 1.8115 | 1 | 1 |
| 424 | simple_cnn | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 1.8281 | 1 | 1 |
| 425 | wide_resnet | cross_entropy | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.7160 | 1 | 1 |
| 426 | wide_resnet | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.5712 | 1 | 1 |
| 427 | wide_resnet | cross_entropy | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 15.00 | 2.3188 | 1 | 1 |
| 428 | wide_resnet | label_smoothing | sgd | batch_size=8, lr=0.01, weight_decay=0.0 | 15.00 | 2.3556 | 1 | 1 |
| 429 | wide_resnet | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 15.00 | 2.9850 | 1 | 1 |
| 430 | wide_resnet | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 15.00 | 2.4939 | 1 | 1 |
| 431 | wide_resnet | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 1.8506 | 1 | 1 |
| 432 | wide_resnet | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 15.00 | 2.2724 | 1 | 1 |
| 433 | wide_resnet | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 15.00 | 2.0883 | 1 | 1 |
| 434 | squeezenet | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.001 | 14.00 | 2.2944 | 1 | 1 |
| 435 | cspnet | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 14.00 | 34096.7436 | 1 | 1 |
| 436 | resnet | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 14.00 | 2.1725 | 1 | 1 |
| 437 | se_resnet | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 14.00 | 174.9745 | 1 | 1 |
| 438 | vim_tiny | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 14.00 | 1.9537 | 1 | 1 |
| 439 | lenet | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 13.00 | 2.2975 | 1 | 1 |
| 440 | vit | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 13.00 | 2.4120 | 1 | 1 |
| 441 | vit | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 13.00 | 2.2977 | 1 | 1 |
| 442 | vgg | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 13.00 | 2.3024 | 1 | 1 |
| 443 | densenet | label_smoothing | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 12.00 | 2.2950 | 1 | 1 |
| 444 | gru | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.0001 | 12.00 | 2.3009 | 1 | 1 |
| 445 | gru | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 12.00 | 2.2923 | 1 | 1 |
| 446 | mnasnet | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 12.00 | 2.2995 | 1 | 1 |
| 447 | mnasnet | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 12.00 | 1.8616 | 1 | 1 |
| 448 | repvgg | label_smoothing | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 12.00 | 2.3018 | 1 | 1 |
| 449 | squeezenet | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.0001 | 12.00 | 2.2976 | 1 | 1 |
| 450 | lenet | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 12.00 | 1.8645 | 1 | 1 |
| 451 | poolformer | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 12.00 | 2.4774 | 1 | 1 |
| 452 | poolformer | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 12.00 | 36.3875 | 1 | 1 |
| 453 | poolformer | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.001 | 12.00 | 2.8307 | 1 | 1 |
| 454 | shufflenet | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 12.00 | 2.2988 | 1 | 1 |
| 455 | mobilenetv2 | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.0 | 12.00 | 2.3026 | 1 | 1 |
| 456 | van | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 12.00 | 2.2992 | 1 | 1 |
| 457 | coord_resnet | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 12.00 | 2.2939 | 1 | 1 |
| 458 | nin | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.0001 | 12.00 | 2.3004 | 1 | 1 |
| 459 | nin | cross_entropy | adamw | batch_size=16, lr=0.0001, weight_decay=0.001 | 12.00 | 2.2979 | 1 | 1 |
| 460 | nin | label_smoothing | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 12.00 | 2.2888 | 1 | 1 |
| 461 | se_resnet | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 12.00 | 2.2994 | 1 | 1 |
| 462 | bert | label_smoothing | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 12.00 | 2.3010 | 1 | 1 |
| 463 | deit | cross_entropy | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 12.00 | 2.3521 | 1 | 1 |
| 464 | deit | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 12.00 | 2.2842 | 1 | 1 |
| 465 | deit | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 12.00 | 2.4015 | 1 | 1 |
| 466 | gpt | cross_entropy | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0001 | 12.00 | 2.3006 | 1 | 1 |
| 467 | gpt | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 12.00 | 1.8565 | 1 | 1 |
| 468 | xception | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0001 | 12.00 | 3135.6217 | 1 | 1 |
| 469 | cbam_resnet | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 12.00 | 1.8577 | 1 | 1 |
| 470 | dpn | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 12.00 | 2.3002 | 1 | 1 |
| 471 | dpn | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0 | 12.00 | 1.9857 | 1 | 1 |
| 472 | hardnet | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 12.00 | 3197.9848 | 1 | 1 |
| 473 | efficientnet | cross_entropy | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 12.00 | 2.2998 | 1 | 1 |
| 474 | efficientnet | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 12.00 | 1.8605 | 1 | 1 |
| 475 | inception_resnet | cross_entropy | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 12.00 | 25968.3806 | 1 | 1 |
| 476 | mobilenetv3 | label_smoothing | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 12.00 | 2.3017 | 1 | 1 |
| 477 | mobilenetv3 | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 12.00 | 176519.7388 | 1 | 1 |
| 478 | alexnet | label_smoothing | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 12.00 | 2.3027 | 1 | 1 |
| 479 | alexnet | focal_loss | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 12.00 | 4.2057 | 1 | 1 |
| 480 | darknet | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 12.00 | 56130.5291 | 1 | 1 |
| 481 | darknet | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.001 | 12.00 | 1.8577 | 1 | 1 |
| 482 | googlenet | label_smoothing | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 12.00 | 2.2895 | 1 | 1 |
| 483 | googlenet | label_smoothing | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.001 | 12.00 | 2.2971 | 1 | 1 |
| 484 | googlenet | focal_loss | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 12.00 | 1.8669 | 1 | 1 |
| 485 | lstm | label_smoothing | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 12.00 | 2.2976 | 1 | 1 |
| 486 | lstm | label_smoothing | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 12.00 | 2.3032 | 1 | 1 |
| 487 | regnet | cross_entropy | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 12.00 | 2.3023 | 1 | 1 |
| 488 | regnet | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 12.00 | 672.8967 | 1 | 1 |
| 489 | wide_resnet | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 12.00 | 5.7723 | 1 | 1 |
| 490 | wide_resnet | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 12.00 | 7.8350 | 1 | 1 |
| 491 | gru | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 11.00 | 2.3053 | 1 | 1 |
| 492 | squeezenet | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.0001 | 11.00 | 2.3107 | 1 | 1 |
| 493 | squeezenet | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 11.00 | 2.0528 | 1 | 1 |
| 494 | ghostnet | label_smoothing | adam | batch_size=16, lr=0.0001, weight_decay=0.001 | 11.00 | 2.2998 | 1 | 1 |
| 495 | ghostnet | focal_loss | adamw | batch_size=16, lr=0.0001, weight_decay=0.0001 | 11.00 | 1.8591 | 1 | 1 |
| 496 | lenet | cross_entropy | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 11.00 | 2.3041 | 1 | 1 |
| 497 | poolformer | label_smoothing | adam | batch_size=16, lr=0.0001, weight_decay=0.001 | 11.00 | 2.7314 | 1 | 1 |
| 498 | vit | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 11.00 | 1.8901 | 1 | 1 |
| 499 | coatnet | focal_loss | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 11.00 | 12.8315 | 1 | 1 |
| 500 | hrnet | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 11.00 | 1.8618 | 1 | 1 |
| 501 | mobilenetv2 | label_smoothing | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0 | 11.00 | 3.8986 | 1 | 1 |
| 502 | mobilenetv2 | focal_loss | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 11.00 | 2.2998 | 1 | 1 |
| 503 | van | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 11.00 | 1.8669 | 1 | 1 |
| 504 | nin | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 11.00 | 2.3021 | 1 | 1 |
| 505 | se_resnet | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.0001 | 11.00 | 1.8698 | 1 | 1 |
| 506 | deit | focal_loss | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 11.00 | 2.3165 | 1 | 1 |
| 507 | mlp | focal_loss | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 11.00 | 1.8454 | 1 | 1 |
| 508 | repghost | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 11.00 | 2.3322 | 1 | 1 |
| 509 | repghost | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.0 | 11.00 | 2.2959 | 1 | 1 |
| 510 | sknet | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.0001 | 11.00 | 13160.4746 | 1 | 1 |
| 511 | xception | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 11.00 | 1270.7223 | 1 | 1 |
| 512 | cbam_resnet | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 11.00 | 2.3025 | 1 | 1 |
| 513 | dpn | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 11.00 | 2.3676 | 1 | 1 |
| 514 | dpn | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 11.00 | 2.3090 | 1 | 1 |
| 515 | mobilenet | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.0001 | 11.00 | 1.9772 | 1 | 1 |
| 516 | mobilenet | focal_loss | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 11.00 | 1.8578 | 1 | 1 |
| 517 | swin_tiny | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 11.00 | 5.1674 | 1 | 1 |
| 518 | swin_tiny | focal_loss | adamw | batch_size=16, lr=0.001, weight_decay=0.001 | 11.00 | 4.6209 | 1 | 1 |
| 519 | swin_tiny | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 11.00 | 54.5430 | 1 | 1 |
| 520 | efficientnet | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 11.00 | 2.3028 | 1 | 1 |
| 521 | vgg | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 11.00 | 581562048.0000 | 1 | 1 |
| 522 | alexnet | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 11.00 | 2.3020 | 1 | 1 |
| 523 | darknet | label_smoothing | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0 | 11.00 | 2.5841 | 1 | 1 |
| 524 | darknet | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 11.00 | 4552.1877 | 1 | 1 |
| 525 | googlenet | cross_entropy | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 11.00 | 2.3012 | 1 | 1 |
| 526 | googlenet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 11.00 | 2.2928 | 1 | 1 |
| 527 | googlenet | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 11.00 | 1.8626 | 1 | 1 |
| 528 | regnet | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 11.00 | 2.3233 | 1 | 1 |
| 529 | regnet | label_smoothing | adamw | batch_size=16, lr=0.0001, weight_decay=0.0 | 11.00 | 2.3115 | 1 | 1 |
| 530 | capsnet | cross_entropy | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 10.00 | 2.3026 | 1 | 1 |
| 531 | capsnet | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 10.00 | 2.3026 | 1 | 1 |
| 532 | capsnet | cross_entropy | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 10.00 | 2.3026 | 1 | 1 |
| 533 | capsnet | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 10.00 | 2.3026 | 1 | 1 |
| 534 | capsnet | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 10.00 | 2.3026 | 1 | 1 |
| 535 | capsnet | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 10.00 | 2.3026 | 1 | 1 |
| 536 | capsnet | label_smoothing | adamw | batch_size=8, lr=0.001, weight_decay=0.001 | 10.00 | 2.3026 | 1 | 1 |
| 537 | capsnet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 10.00 | 2.3026 | 1 | 1 |
| 538 | capsnet | focal_loss | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 10.00 | 1.8651 | 1 | 1 |
| 539 | capsnet | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 10.00 | 1.8651 | 1 | 1 |
| 540 | capsnet | focal_loss | adamw | batch_size=16, lr=0.0001, weight_decay=0.0001 | 10.00 | 1.8651 | 1 | 1 |
| 541 | capsnet | focal_loss | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 10.00 | 1.8651 | 1 | 1 |
| 542 | gru | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 10.00 | 2.2978 | 1 | 1 |
| 543 | gru | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 10.00 | 2.3041 | 1 | 1 |
| 544 | gru | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 10.00 | 1.8621 | 1 | 1 |
| 545 | cspnet | label_smoothing | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 10.00 | 2.3062 | 1 | 1 |
| 546 | ghostnet | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 10.00 | 2.3072 | 1 | 1 |
| 547 | ghostnet | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 10.00 | 2.3051 | 1 | 1 |
| 548 | poolformer | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 10.00 | 17.9410 | 1 | 1 |
| 549 | poolformer | focal_loss | adam | batch_size=8, lr=0.01, weight_decay=0.001 | 10.00 | 22.0974 | 1 | 1 |
| 550 | poolformer | focal_loss | rmsprop | batch_size=8, lr=0.01, weight_decay=0.0001 | 10.00 | 1792.9611 | 1 | 1 |
| 551 | shufflenet | focal_loss | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 10.00 | 1.8631 | 1 | 1 |
| 552 | vit | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 10.00 | 5.5646 | 1 | 1 |
| 553 | hrnet | cross_entropy | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 10.00 | 2.3001 | 1 | 1 |
| 554 | hrnet | label_smoothing | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 10.00 | 2.3033 | 1 | 1 |
| 555 | van | cross_entropy | rmsprop | batch_size=8, lr=0.01, weight_decay=0.001 | 10.00 | 2.7028 | 1 | 1 |
| 556 | efficientnetv2 | cross_entropy | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 10.00 | 2.3023 | 1 | 1 |
| 557 | lcnet | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 10.00 | 2.3008 | 1 | 1 |
| 558 | lcnet | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 10.00 | 2.3006 | 1 | 1 |
| 559 | lcnet | label_smoothing | adamw | batch_size=16, lr=0.0001, weight_decay=0.0001 | 10.00 | 2.3063 | 1 | 1 |
| 560 | lcnet | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 10.00 | 1.8693 | 1 | 1 |
| 561 | vim_tiny | cross_entropy | sgd | batch_size=16, lr=0.0001, weight_decay=0.0 | 10.00 | 2.3055 | 1 | 1 |
| 562 | vim_tiny | cross_entropy | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 10.00 | 995.2924 | 1 | 1 |
| 563 | deit | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 10.00 | 2.4725 | 1 | 1 |
| 564 | repghost | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.001 | 10.00 | 2.3036 | 1 | 1 |
| 565 | sknet | cross_entropy | sgd | batch_size=16, lr=0.0001, weight_decay=0.0 | 10.00 | 2.3176 | 1 | 1 |
| 566 | xception | label_smoothing | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 10.00 | 739.7124 | 1 | 1 |
| 567 | dpn | label_smoothing | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 10.00 | 2.3001 | 1 | 1 |
| 568 | hardnet | cross_entropy | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 10.00 | 2.3073 | 1 | 1 |
| 569 | hardnet | focal_loss | sgd | batch_size=8, lr=0.001, weight_decay=0.0001 | 10.00 | 1.8846 | 1 | 1 |
| 570 | mobilenet | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 10.00 | 2.3046 | 1 | 1 |
| 571 | mobilenet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 10.00 | 2.3437 | 1 | 1 |
| 572 | swin_tiny | cross_entropy | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 10.00 | 48.7943 | 1 | 1 |
| 573 | swin_tiny | label_smoothing | adamw | batch_size=16, lr=0.0001, weight_decay=0.001 | 10.00 | 3.0557 | 1 | 1 |
| 574 | convnext | label_smoothing | adam | batch_size=8, lr=0.0001, weight_decay=0.001 | 10.00 | 2.4194 | 1 | 1 |
| 575 | mobilenetv3 | focal_loss | sgd | batch_size=16, lr=0.01, weight_decay=0.001 | 10.00 | 1.8650 | 1 | 1 |
| 576 | vgg | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 10.00 | 2.3160 | 1 | 1 |
| 577 | vgg | label_smoothing | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 10.00 | 2.3019 | 1 | 1 |
| 578 | vgg | label_smoothing | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 10.00 | 3.4879 | 1 | 1 |
| 579 | alexnet | cross_entropy | adam | batch_size=8, lr=0.01, weight_decay=0.0 | 10.00 | 2.3286 | 1 | 1 |
| 580 | darknet | cross_entropy | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0001 | 10.00 | 2.3153 | 1 | 1 |
| 581 | darknet | label_smoothing | sgd | batch_size=16, lr=0.01, weight_decay=0.0001 | 10.00 | 118.6468 | 1 | 1 |
| 582 | darknet | label_smoothing | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 10.00 | 2711315.2143 | 1 | 1 |
| 583 | googlenet | label_smoothing | adam | batch_size=16, lr=0.0001, weight_decay=0.0001 | 10.00 | 2.3000 | 1 | 1 |
| 584 | lstm | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 10.00 | 1.8593 | 1 | 1 |
| 585 | regnet | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 10.00 | 8.3562 | 1 | 1 |
| 586 | wide_resnet | cross_entropy | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 10.00 | 2.3089 | 1 | 1 |
| 587 | densenet | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 9.00 | 1093406.2321 | 1 | 1 |
| 588 | mnasnet | label_smoothing | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 9.00 | 3.0621 | 1 | 1 |
| 589 | shufflenet | focal_loss | sgd | batch_size=16, lr=0.001, weight_decay=0.0 | 9.00 | 1.8690 | 1 | 1 |
| 590 | van | cross_entropy | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 9.00 | 2.3040 | 1 | 1 |
| 591 | swin_tiny | label_smoothing | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 9.00 | 53.0188 | 1 | 1 |
| 592 | mobilenetv3 | focal_loss | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 9.00 | 2.3507 | 1 | 1 |
| 593 | resnext | label_smoothing | sgd | batch_size=8, lr=0.0001, weight_decay=0.0 | 9.00 | 2.3148 | 1 | 1 |
| 594 | alexnet | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 9.00 | 298085.2098 | 1 | 1 |
| 595 | darknet | label_smoothing | adam | batch_size=8, lr=0.001, weight_decay=0.0 | 9.00 | 2.4871 | 1 | 1 |
| 596 | darknet | focal_loss | sgd | batch_size=8, lr=0.01, weight_decay=0.001 | 9.00 | 567850.6587 | 1 | 1 |
| 597 | lstm | cross_entropy | sgd | batch_size=16, lr=0.01, weight_decay=0.0 | 9.00 | 2.3052 | 1 | 1 |
| 598 | poolformer | cross_entropy | adamw | batch_size=8, lr=0.01, weight_decay=0.001 | 8.00 | 45.9023 | 1 | 1 |
| 599 | vgg | focal_loss | sgd | batch_size=8, lr=0.0001, weight_decay=0.001 | 8.00 | 1.8664 | 1 | 1 |
| 600 | van | focal_loss | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0 | 7.00 | 1800476789432248938124604866560.0000 | 1 | 1 |

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
| 13 | vae | mse | adam | batch_size=16, lr=0.01, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 14 | vae | mse | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 15 | vae | mse | adamw | batch_size=8, lr=0.001, weight_decay=0.0 | 0.0000 | 1 | 1 |
| 16 | vae | mse | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 17 | vae | l1 | adam | batch_size=8, lr=0.0001, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 18 | vae | l1 | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 19 | vae | l1 | adamw | batch_size=16, lr=0.01, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 20 | vae | l1 | rmsprop | batch_size=16, lr=0.001, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 21 | vae | bce | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 22 | vae | bce | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 23 | vae | bce | adamw | batch_size=8, lr=0.01, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 24 | vae | bce | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 25 | conv_ae | mse | adam | batch_size=8, lr=0.001, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 26 | conv_ae | mse | sgd | batch_size=8, lr=0.001, weight_decay=0.0 | 0.0000 | 1 | 1 |
| 27 | conv_ae | mse | adamw | batch_size=16, lr=0.0001, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 28 | conv_ae | mse | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 29 | conv_ae | l1 | adam | batch_size=16, lr=0.001, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 30 | conv_ae | l1 | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 31 | conv_ae | l1 | adamw | batch_size=16, lr=0.001, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 32 | conv_ae | l1 | rmsprop | batch_size=8, lr=0.0001, weight_decay=0.0 | 0.0000 | 1 | 1 |
| 33 | conv_ae | bce | adam | batch_size=16, lr=0.01, weight_decay=0.0 | 0.0000 | 1 | 1 |
| 34 | conv_ae | bce | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 0.0000 | 1 | 1 |
| 35 | conv_ae | bce | adamw | batch_size=16, lr=0.001, weight_decay=0.0001 | 0.0000 | 1 | 1 |
| 36 | conv_ae | bce | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 0.0000 | 1 | 1 |
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
| 1 | wgan | wasserstein | adam | batch_size=16, lr=0.0001, weight_decay=0.001 | -0.0018 | -0.0058 | 1 | 1 |
| 2 | dcgan | bce | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 0.0000 | 85.9213 | 1 | 1 |
| 3 | wgan | wasserstein | rmsprop | batch_size=16, lr=0.01, weight_decay=0.0001 | 0.0625 | -0.1603 | 1 | 1 |
| 4 | wgan | bce | rmsprop | batch_size=16, lr=0.01, weight_decay=0.001 | 0.0800 | -0.1822 | 1 | 1 |
| 5 | wgan | bce | adam | batch_size=16, lr=0.01, weight_decay=0.0001 | 0.1981 | -0.3513 | 1 | 1 |
| 6 | dcgan | wasserstein | sgd | batch_size=16, lr=0.0001, weight_decay=0.001 | 0.5537 | 1.4533 | 1 | 1 |
| 7 | vanilla_gan | wasserstein | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 0.5993 | 1.3056 | 1 | 1 |
| 8 | vanilla_gan | bce | adamw | batch_size=8, lr=0.0001, weight_decay=0.0 | 0.6470 | 1.3053 | 1 | 1 |
| 9 | vanilla_gan | bce | sgd | batch_size=16, lr=0.0001, weight_decay=0.0001 | 0.6774 | 1.3996 | 1 | 1 |
| 10 | vanilla_gan | wasserstein | sgd | batch_size=8, lr=0.001, weight_decay=0.001 | 0.6909 | 1.3646 | 1 | 1 |
| 11 | vanilla_gan | bce | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 0.7006 | 1.3573 | 1 | 1 |
| 12 | vanilla_gan | wasserstein | adamw | batch_size=8, lr=0.0001, weight_decay=0.0001 | 0.7012 | 1.2805 | 1 | 1 |
| 13 | cgan | wasserstein | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 0.7093 | 1.3100 | 1 | 1 |
| 14 | dcgan | bce | sgd | batch_size=8, lr=0.0001, weight_decay=0.0001 | 0.7308 | 1.2906 | 1 | 1 |
| 15 | cgan | bce | sgd | batch_size=16, lr=0.001, weight_decay=0.001 | 0.8272 | 1.2934 | 1 | 1 |
| 16 | cgan | wasserstein | adam | batch_size=16, lr=0.0001, weight_decay=0.0 | 0.8419 | 1.3331 | 1 | 1 |
| 17 | cgan | bce | adam | batch_size=16, lr=0.0001, weight_decay=0.001 | 0.9274 | 1.3444 | 1 | 1 |
| 18 | vanilla_gan | wasserstein | adam | batch_size=8, lr=0.001, weight_decay=0.0001 | 0.9301 | 1.2810 | 1 | 1 |
| 19 | dcgan | bce | adamw | batch_size=16, lr=0.0001, weight_decay=0.0001 | 0.9761 | 1.0717 | 1 | 1 |
| 20 | dcgan | wasserstein | adam | batch_size=16, lr=0.0001, weight_decay=0.001 | 1.0066 | 1.0437 | 1 | 1 |
| 21 | cgan | wasserstein | adamw | batch_size=8, lr=0.01, weight_decay=0.0 | 2.1920 | 7.7410 | 1 | 1 |
| 22 | cgan | bce | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 2.9830 | 1.5244 | 1 | 1 |
| 23 | cgan | bce | rmsprop | batch_size=16, lr=0.0001, weight_decay=0.0 | 3.6289 | 1.0982 | 1 | 1 |
| 24 | vanilla_gan | bce | rmsprop | batch_size=8, lr=0.001, weight_decay=0.001 | 4.2303 | 16.4034 | 1 | 1 |
| 25 | cgan | wasserstein | rmsprop | batch_size=16, lr=0.001, weight_decay=0.001 | 4.6306 | 8.4237 | 1 | 1 |
| 26 | dcgan | bce | adam | batch_size=16, lr=0.001, weight_decay=0.0 | 5.3970 | 0.9829 | 1 | 1 |
| 27 | dcgan | wasserstein | adamw | batch_size=16, lr=0.001, weight_decay=0.0 | 6.3773 | 0.7937 | 1 | 1 |
| 28 | dcgan | wasserstein | rmsprop | batch_size=8, lr=0.001, weight_decay=0.0001 | 7.0302 | 2.9908 | 1 | 1 |

## Search Space

- Loss (classification): cross_entropy, label_smoothing, focal_loss
- Loss (autoencoder): mse, l1, bce
- Loss (GAN): bce, wasserstein (informational; GANs use fixed objectives)
- Optimizers: adam, sgd, adamw, rmsprop
- Hyperparameters: {"lr": [0.0001, 0.001, 0.01], "batch_size": [8, 16, 32], "weight_decay": [0.0, 0.0001, 0.001]}
