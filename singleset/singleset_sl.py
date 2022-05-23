"""
Singleset on benchmark exp.
"""
import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import torch
import torch.nn as nn
import torch.optim as optim
import time
from nets.CMI_Net import CaNet
import argparse
from utils import data_utils
from datetime import datetime
import numpy as np
import random

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from Regularization import Regularization

def prepare_data():
    # Prepare data 
    # Client1
    C1_trainset     = data_utils.DigitsDataset(data_path="/home/axmao2/data/Datasets_Fed_SL/Client_1", percent=args.percent, train=True,  transform=None)
    C2_trainset     = data_utils.DigitsDataset(data_path='/home/axmao2/data/Datasets_Fed_SL/Client_2', percent=args.percent,  train=True,  transform=None)
    C3_trainset     = data_utils.DigitsDataset(data_path='/home/axmao2/data/Datasets_Fed_SL/Client_3', percent=args.percent,  train=True,  transform=None)
    C4_trainset     = data_utils.DigitsDataset(data_path='/home/axmao2/data/Datasets_Fed_SL/Client_4', percent=args.percent,  train=True,  transform=None)
    C5_trainset     = data_utils.DigitsDataset(data_path='/home/axmao2/data/Datasets_Fed_SL/Client_5', percent=args.percent,  train=True,  transform=None)
    testset         = data_utils.DigitsDataset(data_path='/home/axmao2/data/Datasets_Fed_SL/Test', percent=args.percent,  train=False, transform=None)

    if args.data.lower() == 'c1':
        train_loader = torch.utils.data.DataLoader(C1_trainset, batch_size=args.batch, shuffle=True)
    elif args.data.lower() == 'c2':
        train_loader = torch.utils.data.DataLoader(C2_trainset, batch_size=args.batch, shuffle=True)
    elif args.data.lower() == 'c3':
        train_loader = torch.utils.data.DataLoader(C3_trainset, batch_size=args.batch, shuffle=True)
    elif args.data.lower() == 'c4':
        train_loader = torch.utils.data.DataLoader(C4_trainset, batch_size=args.batch, shuffle=True)
    else:
        train_loader = torch.utils.data.DataLoader(C5_trainset, batch_size=args.batch, shuffle=True)
    
    test_loader     = torch.utils.data.DataLoader(testset, batch_size=args.batch, shuffle=False)

    return train_loader, test_loader

def train(data_loader, optimizer, loss_fun, device):
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
        optimizer.zero_grad()

        data = data.to(device).float()
        target = target.to(device).long()
        output = model(data)
        loss = loss_fun(output, target)
        if args.weight_d > 0:
            loss = loss + reg_loss(model)
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

        loss.backward()
        optimizer.step()

    return loss_all / len(data_loader), correct/total

def test(data_loader,site, loss_fun, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    targets = []
    preds = []
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device).float()
            target = target.to(device).long()
            targets.extend(target.detach().cpu().numpy().tolist())
            output = model(data)
            test_loss += loss_fun(output, target).item()
            pred = output.data.max(1)[1]
            preds.extend(pred.detach().cpu().numpy().tolist())
            correct += pred.eq(target.view(-1)).sum().item()
            total += target.size(0)

    test_loss /= len(data_loader)
    correct /= total
    print(' {} | Test loss: {:.4f} | Test acc: {:.4f}'.format(site, test_loss, correct))

    if log:
        logfile.write(' {} | Test loss: {:.4f} | Test acc: {:.4f}\n'.format(site, test_loss, correct))
    
    return test_loss, correct, preds, targets

def plot_figure(title, List1, List2):
    font = {'weight' : 'normal', 'size'   : 20}
    plt.figure(figsize=(12,9))
    plt.title(title,font)
    index = list(range(1,len(List1)+1))
    plt.plot(index,List1,color='skyblue',label='Train')
    plt.plot(index,List2,color='red',label='Test')
    plt.legend(fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    plt.xlim(0,100)
    plt.xlabel('n_iter',font)
    plt.ylabel(title,font)
    
    savedpath = os.path.join(temp_PATH_2,'{}_accuracy_curve.png'.format(title))
    plt.savefig(savedpath)
    
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--percent', type = float, default= 0.1, help ='percentage of dataset to train')
    parser.add_argument('--log', action='store_true', help='whether to log')
    parser.add_argument('--gpu', type = int, default=1, help='use gpu or not')
    parser.add_argument('--epochs', type=int, default=100)                                                   #fixed
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')                              #fixed
    parser.add_argument('--batch', type = int, default= 256, help ='batch size')                             #fixed
    parser.add_argument('--weight_d', type = float, default= 0.1, help ='weight decay for regularization')   #fixed
    parser.add_argument('--seed',type=int, default=1, help='seed')
    parser.add_argument('--data', type = str, default= 'C1', help='[C1 | C2 | C3 | C4 | C5]')
    parser.add_argument('--save_path', type = str, default='./checkpoint', help='path to save the checkpoint')
    args = parser.parse_args()
    
    setup_seed(args.seed)

    exp_folder = 'singleset_equines/'

    args.save_path = os.path.join(args.save_path, exp_folder)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
    temp_PATH_1 = os.path.join(args.save_path,'SingleSet_{}'.format(args.data))
    if not os.path.exists(temp_PATH_1):
        os.makedirs(temp_PATH_1)
    temp_PATH_2 = os.path.join(temp_PATH_1, datetime.now().strftime(DATE_FORMAT))
    if not os.path.exists(temp_PATH_2):
        os.makedirs(temp_PATH_2)  ###checkpoint->singleset_equines->SingleSet_C1/2->time->saved results.
    
    
    log = args.log
    if log:
        # log_path = os.path.join('/home/axmao2/workplace/FedL/FedBN-master/logs/', exp_folder)
        # if not os.path.exists(log_path):
        #     os.makedirs(log_path)
        # t_path = os.path.join(log_path,'SingleSet_{}'.format(args.data))
        # if not os.path.exists(t_path):
        #     os.makedirs(t_path)
        logfile = open(os.path.join(temp_PATH_2, 'log.txt'), 'w+')
        logfile.write('==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        logfile.write('===Setting===\n')
        logfile.write('    seed: {}\n'.format(args.seed))
        logfile.write('    lr: {}\n'.format(args.lr))
        logfile.write('    batch: {}\n'.format(args.batch))
        logfile.write('    dataset: {}\n'.format(args.data))
        logfile.write('    epochs: {}\n'.format(args.epochs))
        logfile.write('    weight_decay: {}\n'.format(args.weight_d))
        
    #Some important results
    f = open(os.path.join(temp_PATH_2,'output.txt'), 'a')

    model = CaNet().to(device)
    
    train_loader, test_loader = prepare_data()
    
    if args.weight_d > 0:
        reg_loss=Regularization(model, args.weight_d, p=2)
    else:
        print("no regularization")
        
    loss_fun = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    best_epoch = 0
    best_acc = 0.0
    Train_accuracy = []
    Train_loss = []
    Test_accuracy = []
    Test_loss = []
    
    for epoch in range(args.epochs):
        train_scheduler.step(epoch+1)
        
        print(f"Epoch: {epoch}" , flush=True)
        loss,acc = train(train_loader, optimizer, loss_fun, device)
        Train_accuracy.append(acc)
        Train_loss.append(loss)
        
        print(' {} | Train loss: {:.4f} | Train acc : {:.4f}'.format(args.data, loss,acc), flush=True)

        if log:
            logfile.write('Epoch Number {}\n'.format(epoch))
            logfile.write(' {} | Train loss: {:.4f} | Train acc : {:.4f}\n'.format(args.data, loss, acc))
            logfile.flush()

        loss_test, acc_test, prediction, ground_truth = test(test_loader, args.data, loss_fun, device)
        Test_accuracy.append(acc_test)
        Test_loss.append(loss_test)
        
        #start to save best performance model (according to the accuracy on validation dataset) after learning rate decay to 0.01
        if epoch > 95 and best_acc < acc_test:
            best_epoch = epoch
            best_acc = acc_test
    SAVE_PATH = os.path.join(temp_PATH_2, 'best_net_epoch_{}_{:.4f}'.format(best_epoch,best_acc))
    
    ###Displaying the accuracy and loss of train and test dataset
    plot_figure(args.data, Train_accuracy, Test_accuracy)
    # plot_figure('Loss', Train_loss, Test_loss)
    
    print(' Saving the best checkpoint to {}...'.format(SAVE_PATH), flush=True)
    torch.save({
        'model': model.state_dict(),
        'epoch': epoch
    }, SAVE_PATH)

    if log:
        logfile.flush()
        logfile.close()
