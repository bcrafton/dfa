
python plot4.py --benchmark mnist --itrs 10 --x sparse --y angle --color_key acc --fix_key rank --fix_val 5 & 
python plot4.py --benchmark mnist --itrs 10 --x sparse --y angle --color_key acc --fix_key rank --fix_val 10 &
python plot4.py --benchmark mnist --itrs 10 --fix_key rank --fix_val 5  --x sparse --y acc &
python plot4.py --benchmark mnist --itrs 10 --fix_key rank --fix_val 10 --x sparse --y acc &
python plot4.py --benchmark mnist --itrs 10 --fix_key rank --fix_val 5  --x sparse --y angle &
python plot4.py --benchmark mnist --itrs 10 --fix_key rank --fix_val 10 --x sparse --y angle &

wait

python plot4.py --benchmark mnist --itrs 10 --x rank --y acc   --group_key sparse &
python plot4.py --benchmark mnist --itrs 10 --x rank --y angle --group_key sparse &
python plot4.py --benchmark mnist --itrs 10 --fix_key rank --fix_val 5  --x acc   --y angle --group_key sparse &
python plot4.py --benchmark mnist --itrs 10 --fix_key rank --fix_val 10 --x acc   --y angle --group_key sparse &
python plot4.py --benchmark mnist --itrs 10 --fix_key rank --fix_val 5  --x angle --y acc   --group_key sparse &
python plot4.py --benchmark mnist --itrs 10 --fix_key rank --fix_val 10 --x angle --y acc   --group_key sparse &

wait 

python plot4.py --benchmark cifar10 --itrs 10 --x sparse --y angle --color_key acc --fix_key rank --fix_val 5 &
python plot4.py --benchmark cifar10 --itrs 10 --x sparse --y angle --color_key acc --fix_key rank --fix_val 10 &
python plot4.py --benchmark cifar10 --itrs 10 --fix_key rank --fix_val 5  --x sparse --y acc &
python plot4.py --benchmark cifar10 --itrs 10 --fix_key rank --fix_val 10 --x sparse --y acc &
python plot4.py --benchmark cifar10 --itrs 10 --fix_key rank --fix_val 5  --x sparse --y angle &
python plot4.py --benchmark cifar10 --itrs 10 --fix_key rank --fix_val 10 --x sparse --y angle &

wait

python plot4.py --benchmark cifar10 --itrs 10 --x rank --y acc   --group_key sparse &
python plot4.py --benchmark cifar10 --itrs 10 --x rank --y angle --group_key sparse & 
python plot4.py --benchmark cifar10 --itrs 10 --fix_key rank --fix_val 5  --x acc   --y angle --group_key sparse &
python plot4.py --benchmark cifar10 --itrs 10 --fix_key rank --fix_val 10 --x acc   --y angle --group_key sparse &
python plot4.py --benchmark cifar10 --itrs 10 --fix_key rank --fix_val 5  --x angle --y acc   --group_key sparse &
python plot4.py --benchmark cifar10 --itrs 10 --fix_key rank --fix_val 10 --x angle --y acc   --group_key sparse &


