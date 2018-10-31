
python mnist_fc.py --epochs 100 --batch_size 32 --alpha 0.01 --dfa 0 --sparse 0 --init sqrt_fan_in --opt gd &
python mnist_fc.py --epochs 100 --batch_size 32 --alpha 0.01 --dfa 1 --sparse 0 --init zero        --opt gd &
python mnist_fc.py --epochs 100 --batch_size 32 --alpha 0.01 --dfa 1 --sparse 1 --init zero        --opt gd &

python cifar10_fc.py --epochs 200 --batch_size 64 --alpha 0.005 --dfa 0 --sparse 0 --init sqrt_fan_in --opt adam &
python cifar10_fc.py --epochs 200 --batch_size 64 --alpha 0.005 --dfa 1 --sparse 0 --init zero        --opt adam &
python cifar10_fc.py --epochs 200 --batch_size 64 --alpha 0.005 --dfa 1 --sparse 1 --init zero        --opt adam &

python cifar100_fc.py --epochs 300 --batch_size 64 --alpha 0.005 --dfa 0 --sparse 0 --init sqrt_fan_in --opt adam &
python cifar100_fc.py --epochs 300 --batch_size 64 --alpha 0.005 --dfa 1 --sparse 0 --init zero        --opt adam &
python cifar100_fc.py --epochs 300 --batch_size 64 --alpha 0.005 --dfa 1 --sparse 1 --init zero        --opt adam &
