
python cifar10_conv_viz.py --epochs 300 --batch_size 64 --alpha 0.0002 --gpu 2 --dfa 0 --sparse 0 --init sqrt_fan_in --imgs 1 --opt adam &
python cifar10_conv_viz.py --epochs 300 --batch_size 64 --alpha 0.0002 --gpu 3 --dfa 1 --sparse 0 --init zero        --imgs 1 --opt adam &
