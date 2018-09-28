
python cifar10_conv.py --epochs 2000 --batch_size 32 --alpha 0.00005 --gpu 2 --dfa 0 --sparse 0 --init sqrt_fan_in --opt adam --save 1 &
python cifar10_conv.py --epochs 2000 --batch_size 32 --alpha 0.00005 --gpu 3 --dfa 1 --sparse 0 --init zero        --opt adam --save 1 &
        
