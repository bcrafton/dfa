
python mnist_conv.py --gpu 0 --epochs 300 --batch_size 64 --alpha 0.01 --dfa 0 --sparse 0 --init sqrt_fan_in --opt gd --save 1 --name mnist_conv_bp & 
python mnist_conv.py --gpu 1 --epochs 300 --batch_size 64 --alpha 0.01 --dfa 1 --sparse 0 --init zero        --opt gd --save 1 --name mnist_conv_dfa & 
python mnist_conv.py --gpu 2 --epochs 300 --batch_size 64 --alpha 0.01 --dfa 1 --sparse 1 --init zero        --opt gd --save 1 --name mnist_conv_sparse & 
wait
python cifar10_conv.py --gpu 0 --epochs 300 --batch_size 64 --alpha 0.005 --dfa 0 --sparse 0 --init sqrt_fan_in --opt adam --save 1 --name cifar10_conv_bp & 
python cifar10_conv.py --gpu 1 --epochs 300 --batch_size 64 --alpha 0.005 --dfa 1 --sparse 0 --init zero        --opt adam --save 1 --name cifar10_conv_dfa & 
python cifar10_conv.py --gpu 2 --epochs 300 --batch_size 64 --alpha 0.005 --dfa 1 --sparse 1 --init zero        --opt adam --save 1 --name cifar10_conv_sparse & 
wait
python cifar100_conv.py --gpu 0 --epochs 300 --batch_size 64 --alpha 0.005 --dfa 0 --sparse 0 --init sqrt_fan_in --opt adam --save 1 --name cifar100_conv_bp & 
python cifar100_conv.py --gpu 1 --epochs 300 --batch_size 64 --alpha 0.005 --dfa 1 --sparse 0 --init zero        --opt adam --save 1 --name cifar100_conv_dfa & 
python cifar100_conv.py --gpu 2 --epochs 300 --batch_size 64 --alpha 0.005 --dfa 1 --sparse 1 --init zero        --opt adam --save 1 --name cifar100_conv_sparse & 

# we need vgg or alexnet here.
