
#python imagenet_vgg.py --epochs 100 --batch_size 32 --alpha 0.01 --gpu 0 --dfa 0 --sparse 0 --rank 0 --init sqrt_fan_in --save 1 --name imagenet_vgg_bp      > vgg_bp_results      &
#python imagenet_vgg.py --epochs 100 --batch_size 32 --alpha 0.01 --gpu 1 --dfa 1 --sparse 0 --rank 0 --init zero        --save 1 --name imagenet_vgg_dfa     > vgg_dfa_results     &
#python imagenet_vgg.py --epochs 100 --batch_size 32 --alpha 0.01 --gpu 2 --dfa 1 --sparse 1 --rank 0 --init zero        --save 1 --name imagenet_vgg_sparse  > vgg_sparse_results  &
#python imagenet_vgg.py --epochs 100 --batch_size 32 --alpha 0.01 --gpu 2 --dfa 1 --sparse 5 --rank 0 --init zero        --save 1 --name imagenet_vgg_sparse5 > vgg_sparse5_results &
python imagenet_vgg.py --epochs 100 --batch_size 32 --alpha 0.01 --gpu 1 --dfa 1 --sparse 25 --rank 0 --init zero        --save 1 --name imagenet_vgg_sparse25 > vgg_sparse25_results &
