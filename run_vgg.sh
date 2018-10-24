
#python imagenet_vgg.py --epochs 100 --batch_size 32 --alpha 0.01 --gpu 0 --dfa 0 --sparse 0 --rank 0 --init sqrt_fan_in --save 1 --name imagenet_vgg_bp      > vgg_bp_results      &
#python imagenet_vgg.py --epochs 100 --batch_size 32 --alpha 0.01 --gpu 1 --dfa 1 --sparse 0 --rank 0 --init zero        --save 1 --name imagenet_vgg_dfa     > vgg_dfa_results     &
#python imagenet_vgg.py --epochs 100 --batch_size 64 --alpha 0.01 --gpu 2 --dfa 1 --sparse 1 --rank 0 --init zero        --save 1 --name imagenet_vgg_sparse  > vgg_sparse_results  &
#python imagenet_vgg.py --epochs 100 --batch_size 32 --alpha 0.01 --gpu 2 --dfa 1 --sparse 5 --rank 0 --init zero        --save 1 --name imagenet_vgg_sparse5 > vgg_sparse5_results &
#python imagenet_vgg.py --epochs 100 --batch_size 32 --alpha 0.01 --gpu 1 --dfa 1 --sparse 25 --rank 0 --init zero       --save 1 --name imagenet_vgg_sparse25 > vgg_sparse25_results &

python imagenet_vgg.py --epochs 25 --batch_size 64 --alpha 0.005 --decay 0.9 --gpu 0 --dfa 0 --sparse 0   --rank 0 --init sqrt_fan_in --opt gd --save 1 --name vgg_bp  > vgg_bp &
python imagenet_vgg.py --epochs 25 --batch_size 64 --alpha 0.005 --decay 0.9 --gpu 2 --dfa 1 --sparse 0   --rank 0 --init zero        --opt gd --save 1 --name vgg_dfa > vgg_dfa &
wait
python imagenet_vgg.py --epochs 25 --batch_size 64 --alpha 0.005 --decay 0.9 --gpu 0 --dfa 1 --sparse 1   --rank 0 --init zero        --opt gd --save 1 --name vgg_sparse1 > vgg_sparse1  &
python imagenet_vgg.py --epochs 25 --batch_size 64 --alpha 0.005 --decay 0.9 --gpu 2 --dfa 1 --sparse 5   --rank 0 --init zero        --opt gd --save 1 --name vgg_sparse5 > vgg_sparse5 &
wait
python imagenet_vgg.py --epochs 25 --batch_size 64 --alpha 0.005 --decay 0.9 --gpu 0 --dfa 1 --sparse 10  --rank 0 --init zero        --opt gd --save 1 --name vgg_sparse10 > vgg_sparse10  &
python imagenet_vgg.py --epochs 25 --batch_size 64 --alpha 0.005 --decay 0.9 --gpu 2 --dfa 1 --sparse 25  --rank 0 --init zero        --opt gd --save 1 --name vgg_sparse25 > vgg_sparse25 &
wait
python imagenet_vgg.py --epochs 25 --batch_size 64 --alpha 0.005 --decay 0.9 --gpu 0 --dfa 1 --sparse 50  --rank 0 --init zero        --opt gd --save 1 --name vgg_sparse50  > vgg_sparse50  &
python imagenet_vgg.py --epochs 25 --batch_size 64 --alpha 0.005 --decay 0.9 --gpu 2 --dfa 1 --sparse 100 --rank 0 --init zero        --opt gd --save 1 --name vgg_sparse100 > vgg_sparse100 &
