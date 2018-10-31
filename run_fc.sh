
python mnist_fc.py --gpu 0 --epochs 300 --batch_size 32 --alpha 0.01 --dfa 0 --sparse 0 --init sqrt_fan_in --opt gd --save 1 --name mnist_fc_bp & 
python mnist_fc.py --gpu 1 --epochs 300 --batch_size 32 --alpha 0.01 --dfa 1 --sparse 0 --init zero        --opt gd --save 1 --name mnist_fc_dfa &
python mnist_fc.py --gpu 2 --epochs 300 --batch_size 32 --alpha 0.01 --dfa 1 --sparse 1 --init zero        --opt gd --save 1 --name mnist_fc_sparse &
wait
python cifar10_fc.py --gpu 0 --epochs 300 --batch_size 64 --alpha 0.005 --dfa 0 --sparse 0 --init sqrt_fan_in --opt adam --save 1 --name cifar10_fc_bp & 
python cifar10_fc.py --gpu 1 --epochs 300 --batch_size 64 --alpha 0.005 --dfa 1 --sparse 0 --init zero        --opt adam --save 1 --name cifar10_fc_dfa & 
python cifar10_fc.py --gpu 2 --epochs 300 --batch_size 64 --alpha 0.005 --dfa 1 --sparse 1 --init zero        --opt adam --save 1 --name cifar10_fc_sparse & 
wait
python cifar100_fc.py --gpu 0 --epochs 300 --batch_size 64 --alpha 0.005 --dfa 0 --sparse 0 --init sqrt_fan_in --opt adam --save 1 --name cifar100_fc_bp & 
python cifar100_fc.py --gpu 1 --epochs 300 --batch_size 64 --alpha 0.005 --dfa 1 --sparse 0 --init zero        --opt adam --save 1 --name cifar100_fc_dfa & 
python cifar100_fc.py --gpu 2 --epochs 300 --batch_size 64 --alpha 0.005 --dfa 1 --sparse 1 --init zero        --opt adam --save 1 --name cifar100_fc_sparse & 
