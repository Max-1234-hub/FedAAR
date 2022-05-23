nvidia-smi

python federated/fed_aar.py --percent 1 --mode 'fedavg' --log --weight_d 0.15 --wk_iters 1 --seed 3 --beta=5e-2 --temp=1 --data_path 'Datasets_Fed_SL'
python federated/fed_aar.py --percent 1 --mode 'fedavg' --log --weight_d 0.15 --wk_iters 1 --seed 3 --beta=5e-2 --temp=1 --data_path 'Datasets_Fed_SL-1'
python federated/fed_aar.py --percent 1 --mode 'fedavg' --log --weight_d 0.15 --wk_iters 1 --seed 3 --beta=5e-2 --temp=1 --data_path 'Datasets_Fed_SL-2'