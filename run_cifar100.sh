
#python cifar100_conv.py --epochs 500 --alpha 0.005 --decay 0.995 --gpu 0 --dfa 1 --sparse 1 --init zero --opt adam --save 0 &
#python cifar100_conv.py --epochs 500 --alpha 0.005 --decay 0.995 --gpu 1 --dfa 1 --sparse 5 --init zero --opt adam --save 0 &
#python cifar100_conv.py --epochs 500 --alpha 0.005 --decay 0.995 --gpu 2 --dfa 1 --sparse 10 --init zero --opt adam --save 0 &
#python cifar100_conv.py --epochs 500 --alpha 0.005 --decay 0.995 --gpu 3 --dfa 1 --sparse 25 --init zero --opt adam --save 0 &

#wait

#python cifar100_conv.py --epochs 500 --alpha 0.005 --decay 0.995 --gpu 0 --dfa 1 --sparse 50 --init zero --opt adam --save 0 &
#python cifar100_conv.py --epochs 500 --alpha 0.005 --decay 0.995 --gpu 1 --dfa 1 --sparse 0 --init zero --opt adam --save 0 &

###

python cifar100_conv.py --epochs 500 --alpha 0.005 --decay 0.995 --gpu 0 --dfa 1 --sparse 1 --init zero --opt adam --save 0 &
python cifar100_conv.py --epochs 500 --alpha 0.005 --decay 0.995 --gpu 1 --dfa 1 --sparse 5 --init zero --opt adam --save 0 &
python cifar100_conv.py --epochs 500 --alpha 0.005 --decay 0.995 --gpu 2 --dfa 1 --sparse 10 --init zero --opt adam --save 0 &
python cifar100_conv.py --epochs 500 --alpha 0.005 --decay 0.995 --gpu 3 --dfa 1 --sparse 25 --init zero --opt adam --save 0 &

wait

python cifar100_conv.py --epochs 500 --alpha 0.005 --decay 0.995 --gpu 0 --dfa 1 --sparse 50 --init zero --opt adam --save 0 &
python cifar100_conv.py --epochs 500 --alpha 0.005 --decay 0.995 --gpu 1 --dfa 1 --sparse 0 --init zero --opt adam --save 0 &
