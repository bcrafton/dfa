
python mnist_conv.py --epochs 300 --batch_size 64 --alpha 0.01 --dfa 0 --sparse 0 --init sqrt_fan_in --opt gd &
python mnist_conv.py --epochs 300 --batch_size 64 --alpha 0.01 --dfa 1 --sparse 0 --init zero        --opt gd &
python mnist_conv.py --epochs 300 --batch_size 64 --alpha 0.01 --dfa 1 --sparse 1 --init zero        --opt gd &

python cifar10_conv.py --epochs 300 --batch_size 64 --alpha 0.005 --dfa 0 --sparse 0 --init sqrt_fan_in --opt adam &
python cifar10_conv.py --epochs 300 --batch_size 64 --alpha 0.005 --dfa 1 --sparse 0 --init zero        --opt adam &
python cifar10_conv.py --epochs 300 --batch_size 64 --alpha 0.005 --dfa 1 --sparse 1 --init zero        --opt adam &

python cifar100_conv.py --epochs 300 --batch_size 64 --alpha 0.005 --dfa 0 --sparse 0 --init sqrt_fan_in --opt adam &
python cifar100_conv.py --epochs 300 --batch_size 64 --alpha 0.005 --dfa 1 --sparse 0 --init zero        --opt adam &
python cifar100_conv.py --epochs 300 --batch_size 64 --alpha 0.005 --dfa 1 --sparse 1 --init zero        --opt adam &

# we need vgg or alexnet here.
