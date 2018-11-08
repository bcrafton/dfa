
python cifar10_conv1.py --epochs 300 --dfa 1 --alpha 0.001 --gpu 0   --name cifar10_conv_dfa_0.001   &
python cifar10_conv1.py --epochs 300 --dfa 1 --alpha 0.0005 --gpu 1  --name cifar10_conv_dfa_0.0005  &
python cifar10_conv1.py --epochs 300 --dfa 1 --alpha 0.0001 --gpu 2  --name cifar10_conv_dfa_0.0001  &
python cifar10_conv1.py --epochs 300 --dfa 1 --alpha 0.00005 --gpu 3 --name cifar10_conv_dfa_0.00005 &

wait 

python cifar10_conv1.py --epochs 300 --dfa 0 --alpha 0.001 --gpu 0   --name cifar10_conv_0.001   &
python cifar10_conv1.py --epochs 300 --dfa 0 --alpha 0.0005 --gpu 1  --name cifar10_conv_0.0005  &
python cifar10_conv1.py --epochs 300 --dfa 0 --alpha 0.0001 --gpu 2  --name cifar10_conv_0.0001  &
python cifar10_conv1.py --epochs 300 --dfa 0 --alpha 0.00005 --gpu 3 --name cifar10_conv_0.00005 &
