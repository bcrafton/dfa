python imagenet_grads.py --gpu 2 --dfa 1 --sparse 0 --save 1 > imagenet_dfa_grads &
python imagenet_grads.py --gpu 3 --dfa 0 --sparse 0 --save 1 > imagenet_bp_grads &
