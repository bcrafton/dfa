python imagenet_viz.py --gpu 0 --dfa 1 --sparse 0 --imgs 1 --init zero > imagenet_dfa_viz &
python imagenet_viz.py --gpu 1 --dfa 0 --sparse 0 --imgs 1 > imagenet_bp_viz &
