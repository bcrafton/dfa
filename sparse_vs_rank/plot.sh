
python plot_rank.py --benchmark mnist --itrs 10 --rank 5
python plot_rank.py --benchmark mnist --itrs 10 --rank 10
python plot_sparse_vs_rank.py --benchmark mnist --itrs 10 

python plot_rank.py --benchmark cifar10_1 --itrs 10 --rank 5
python plot_rank.py --benchmark cifar10_1 --itrs 10 --rank 10
python plot_sparse_vs_rank.py --benchmark cifar10_1 --itrs 10

python plot4.py --benchmark mnist --itrs 10 --fix_key rank --fix_val 5 --x acc --y angle --group_key sparse
python plot4.py --benchmark mnist --itrs 10 --fix_key rank --fix_val 5 --x angle --y acc --group_key sparse
