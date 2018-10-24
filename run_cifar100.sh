
#python cifar100_conv.py --epochs 500 --alpha 0.005 --decay 0.995 --gpu 0 --dfa 1 --sparse 1 --init zero --opt adam --save 0 &
#python cifar100_conv.py --epochs 500 --alpha 0.005 --decay 0.995 --gpu 1 --dfa 1 --sparse 5 --init zero --opt adam --save 0 &
#python cifar100_conv.py --epochs 500 --alpha 0.005 --decay 0.995 --gpu 2 --dfa 1 --sparse 10 --init zero --opt adam --save 0 &
#python cifar100_conv.py --epochs 500 --alpha 0.005 --decay 0.995 --gpu 3 --dfa 1 --sparse 25 --init zero --opt adam --save 0 &

#wait

#python cifar100_conv.py --epochs 500 --alpha 0.005 --decay 0.995 --gpu 0 --dfa 1 --sparse 50 --init zero --opt adam --save 0 &
#python cifar100_conv.py --epochs 500 --alpha 0.005 --decay 0.995 --gpu 1 --dfa 1 --sparse 0 --init zero --opt adam --save 0 &

###

python cifar100_conv.py --epochs 1000 --alpha 0.005 --decay 0.995 --gpu 3 --dfa 0 --sparse 0   --init sqrt_fan_in --opt adam --save 1 --name cifar100_bp        > cifar100_bp
python cifar100_conv.py --epochs 1000 --alpha 0.005 --decay 0.995 --gpu 3 --dfa 1 --sparse 0   --init zero        --opt adam --save 1 --name cifar100_dfa       > cifar100_dfa
python cifar100_conv.py --epochs 1000 --alpha 0.005 --decay 0.995 --gpu 3 --dfa 1 --sparse 1   --init zero        --opt adam --save 1 --name cifar100_sparse1   > cifar100_sparse1
python cifar100_conv.py --epochs 1000 --alpha 0.005 --decay 0.995 --gpu 3 --dfa 1 --sparse 5   --init zero        --opt adam --save 1 --name cifar100_sparse5   > cifar100_sparse5
python cifar100_conv.py --epochs 1000 --alpha 0.005 --decay 0.995 --gpu 3 --dfa 1 --sparse 10  --init zero        --opt adam --save 1 --name cifar100_sparse10  > cifar100_sparse10
python cifar100_conv.py --epochs 1000 --alpha 0.005 --decay 0.995 --gpu 3 --dfa 1 --sparse 25  --init zero        --opt adam --save 1 --name cifar100_sparse25  > cifar100_sparse25
python cifar100_conv.py --epochs 1000 --alpha 0.005 --decay 0.995 --gpu 3 --dfa 1 --sparse 50  --init zero        --opt adam --save 1 --name cifar100_sparse50  > cifar100_sparse50
python cifar100_conv.py --epochs 1000 --alpha 0.005 --decay 0.995 --gpu 3 --dfa 1 --sparse 100 --init zero        --opt adam --save 1 --name cifar100_sparse100 > cifar100_sparse100
