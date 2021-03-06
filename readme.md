## FedAAR

This repository is an official PyTorch implementation of the paper "__FedAAR: A Novel Federated Learning Framework for Animal Activity Recognition with Wearable Sensors__".

## Requirements

This is my experiment eviroument
- python3.7
- pytorch+cuda11.3

## Details
### 1. Original dataset
I used a public dataset (i.e., data from six horses and activities) that are avaliable at
https://doi.org/10.4121/uuid:2e08745c-4178-4183-8551-f248c992cb14. 
The reference is _Kamminga, J. W., Janßen, L. M., Meratnia, N., & Havinga, P. J. (2019). Horsing Around—A Dataset Comprising Horse Movement. Data, 4(4), 131._.

### 2. Processed data:
The folder `data` contains the three processed datasets that are used in our experiment, i.e., `Datasets_Fed_SL`, `Datasets_Fed_SL-1`, and `Datasets_Fed_SL-2`. Each one has split the original data into training and test data, where training data are from five horses and test data are from the remaining horse.
In particular, since the data in subfolders `./Datasets_Fed_SL/Test/`,  `./Datasets_Fed_SL-1/Test/`, and `./Datasets_Fed_SL-2/Test/` are too big to be uploaded. Herein you can also click [Data_FedAAR](https://drive.google.com/drive/folders/1ckWI1HaswGVwvcRoKbrMR7cE9AMbiobY?usp=sharing) to find the `Test` subfolders and even the whole used datasets.

### 3. train the model

```ruby
python federated/fed_aar.py --percent 1 --mode 'fedavg' --log --weight_d 0.15 --wk_iters 1 --seed 3 --beta=5e-2 --temp=1 --data_path 'Datasets_Fed_SL'
python federated/fed_aar.py --percent 1 --mode 'fedavg' --log --weight_d 0.15 --wk_iters 1 --seed 3 --beta=5e-2 --temp=1 --data_path 'Datasets_Fed_SL-1'
python federated/fed_aar.py --percent 1 --mode 'fedavg' --log --weight_d 0.15 --wk_iters 1 --seed 3 --beta=5e-2 --temp=1 --data_path 'Datasets_Fed_SL-2'
```

