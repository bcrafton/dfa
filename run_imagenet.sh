python imagenet.py --gpu 0 --dfa 1 --sparse 1 > imagenet_sparse_results &
python imagenet.py --gpu 1 --dfa 1 --sparse 0 > imagenet_dfa_results &
python imagenet.py --gpu 2 --dfa 0 --sparse 0 > imagenet_bp_results &
