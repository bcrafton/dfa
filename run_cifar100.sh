
python cifar100_conv.py --epochs 300 --alpha 0.01 --gpu 0 --dfa 0 --sparse 0 --init sqrt_fan_in --opt adam --save 1 --name cifar100_conv_0.01 &
python cifar100_conv.py --epochs 300 --alpha 0.005 --gpu 1 --dfa 0 --sparse 0 --init sqrt_fan_in --opt adam --save 1 --name cifar100_conv_0.005 &

python cifar100_conv.py --epochs 300 --alpha 0.01 --gpu 2 --dfa 0 --sparse 0 --init sqrt_fan_in --opt rms --save 1 --name cifar100_conv_0.01 &
python cifar100_conv.py --epochs 300 --alpha 0.005 --gpu 3 --dfa 0 --sparse 0 --init sqrt_fan_in --opt rms --save 1 --name cifar100_conv_0.005 &

python cifar100_conv.py --epochs 300 --alpha 0.01 --gpu 0 --dfa 0 --sparse 0 --init sqrt_fan_in --opt decay --save 1 --name cifar100_conv_0.01 &
python cifar100_conv.py --epochs 300 --alpha 0.005 --gpu 1 --dfa 0 --sparse 0 --init sqrt_fan_in --opt decay --save 1 --name cifar100_conv_0.005 &

python cifar100_conv.py --epochs 300 --alpha 0.01 --gpu 2 --dfa 0 --sparse 0 --init sqrt_fan_in --opt gd --save 1 --name cifar100_conv_0.01 &
python cifar100_conv.py --epochs 300 --alpha 0.005 --gpu 3 --dfa 0 --sparse 0 --init sqrt_fan_in --opt gd --save 1 --name cifar100_conv_0.005 &
