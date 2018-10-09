
#python cifar100_conv.py --epochs 300 --alpha 0.01 --gpu 0 --dfa 0 --sparse 0 --init sqrt_fan_in --opt adam --save 1 --name cifar100_conv_0.01 &
#python cifar100_conv.py --epochs 300 --alpha 0.005 --gpu 1 --dfa 0 --sparse 0 --init sqrt_fan_in --opt adam --save 1 --name cifar100_conv_0.005 &

#python cifar100_conv.py --epochs 300 --alpha 0.01 --gpu 2 --dfa 0 --sparse 0 --init sqrt_fan_in --opt rms --save 1 --name cifar100_conv_0.01 &
#python cifar100_conv.py --epochs 300 --alpha 0.005 --gpu 3 --dfa 0 --sparse 0 --init sqrt_fan_in --opt rms --save 1 --name cifar100_conv_0.005 &

#python cifar100_conv.py --epochs 300 --alpha 0.01 --gpu 0 --dfa 0 --sparse 0 --init sqrt_fan_in --opt decay --save 1 --name cifar100_conv_0.01 &
#python cifar100_conv.py --epochs 300 --alpha 0.005 --gpu 1 --dfa 0 --sparse 0 --init sqrt_fan_in --opt decay --save 1 --name cifar100_conv_0.005 &

#python cifar100_conv.py --epochs 300 --alpha 0.01 --gpu 2 --dfa 0 --sparse 0 --init sqrt_fan_in --opt gd --save 1 --name cifar100_conv_0.01 &
#python cifar100_conv.py --epochs 300 --alpha 0.005 --gpu 3 --dfa 0 --sparse 0 --init sqrt_fan_in --opt gd --save 1 --name cifar100_conv_0.005 &

###

#python cifar100_conv.py --epochs 500 --alpha 0.01 --decay 0.995 --gpu 0 --dfa 0 --sparse 0 --init sqrt_fan_in --opt decay --save 1 --name cifar100_conv_0.01_0.995 &
#python cifar100_conv.py --epochs 500 --alpha 0.005 --decay 0.995 --gpu 1 --dfa 0 --sparse 0 --init sqrt_fan_in --opt decay --save 1 --name cifar100_conv_0.005_0.995 &
#python cifar100_conv.py --epochs 500 --alpha 0.01 --decay 0.99 --gpu 2 --dfa 0 --sparse 0 --init sqrt_fan_in --opt decay --save 1 --name cifar100_conv_0.01_0.99 &

###

python cifar100_conv.py --epochs 300 --alpha 0.01 --decay 0.99 --gpu 0 --dfa 1 --sparse 1 --init zero --opt decay --save 0 &
python cifar100_conv.py --epochs 300 --alpha 0.01 --decay 0.99 --gpu 0 --dfa 1 --sparse 5 --init zero --opt decay --save 0 &
python cifar100_conv.py --epochs 300 --alpha 0.01 --decay 0.99 --gpu 0 --dfa 1 --sparse 10 --init zero --opt decay --save 0 &
python cifar100_conv.py --epochs 300 --alpha 0.01 --decay 0.99 --gpu 0 --dfa 1 --sparse 25 --init zero --opt decay --save 0 &
