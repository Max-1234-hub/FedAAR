# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 16:30:10 2022

@author: axmao2-c
"""

"""
federated learning with different aggregation strategy on benchmark exp.
"""
import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import torch
import math
import random
from torch import nn, optim
import time
import copy
from nets.CMI_Net import CaNet
import argparse
import numpy as np
from utils import data_utils
from datetime import datetime
from random import shuffle
import torch.nn.functional as F


from Regularization import Regularization

from sklearn.metrics import classification_report

def prepare_data(args):
    # Prepare data

    # Client1
    C1_trainset     = data_utils.DigitsDataset(data_path='/home/meiluzhu2/mm/data/'+args.data_path+'/Client_1', percent=args.percent, train=True,  transform=None)
    C2_trainset     = data_utils.DigitsDataset(data_path='/home/meiluzhu2/mm/data/'+args.data_path+'/Client_2', percent=args.percent,  train=True,  transform=None)
    C3_trainset     = data_utils.DigitsDataset(data_path='/home/meiluzhu2/mm/data/'+args.data_path+'/Client_3', percent=args.percent,  train=True,  transform=None)
    C4_trainset     = data_utils.DigitsDataset(data_path='/home/meiluzhu2/mm/data/'+args.data_path+'/Client_4', percent=args.percent,  train=True,  transform=None)
    C5_trainset     = data_utils.DigitsDataset(data_path='/home/meiluzhu2/mm/data/'+args.data_path+'/Client_5', percent=args.percent,  train=True,  transform=None)
    testset         = data_utils.DigitsDataset(data_path='/home/meiluzhu2/mm/data/'+args.data_path+'/Test', percent=args.percent,  train=False, transform=None)

    C1_train_loader = torch.utils.data.DataLoader(C1_trainset, batch_size=args.batch, shuffle=True)
    C2_train_loader = torch.utils.data.DataLoader(C2_trainset, batch_size=args.batch,  shuffle=True)
    C3_train_loader = torch.utils.data.DataLoader(C3_trainset, batch_size=args.batch,  shuffle=True)
    C4_train_loader = torch.utils.data.DataLoader(C4_trainset, batch_size=args.batch,  shuffle=True)
    C5_train_loader = torch.utils.data.DataLoader(C5_trainset, batch_size=args.batch,  shuffle=True)
    
    train_loaders   = [C1_train_loader, C2_train_loader, C3_train_loader, C4_train_loader, C5_train_loader]
    test_loader     = torch.utils.data.DataLoader(testset, batch_size=args.batch, shuffle=False)

    return train_loaders, test_loader

def train(args, model, train_loader, optimizer, loss_fun, client_num, device, G_feature_class, n_iteration):
    model.train()
    num_data = 0
    correct = 0
    loss_all = 0
    train_iter = iter(train_loader)
    
    feature_class = [[],[],[],[],[],[]]
    number_class = [[],[],[],[],[],[]]
    for step in range(len(train_iter)):
        loss_R = []
        optimizer.zero_grad()
        x, y = next(train_iter)
        num_data += y.size(0)
        x = x.to(device).float()
        y = y.to(device).long()
        output, features = model(x)
        ###generate the feature vectors according to class.
        labels = y.cpu().numpy()
        preds = torch.max(output, dim=1)[1].cpu().numpy()
        for i in range(6):
            feature_c = features[np.where((labels==i) & (preds==i))[0]]
            if feature_c.size()[0] != 0:
                feature_class[i].append(feature_c.mean(dim=0))
                number_class[i].append(feature_c.size()[0])
                assert torch.isnan(feature_c).sum() == 0
            # print('====0======', torch.isnan(feature_c).sum(), torch.isnan(feature_c.mean(dim=0)).sum())
             
            if n_iteration != 0:
                #print(feature_c.size()[0])
                #print(y.size()[0])
                if feature_c.size()[0] == 0 or len(G_feature_class[i]) == 0:
                    loss_R.append(0)
                else:
                    loss_R.append( (feature_c.size()[0]/y.size()[0])*((F.pairwise_distance(feature_c, G_feature_class[i],p=2)).mean()))
                    ##there is no sample for some class. Thus the mean() value would equal to nan.
        
        loss_ce = loss_fun(output, y)
        if args.weight_d > 0:
            loss_ce = loss_ce + reg_loss(model)
        
        if n_iteration != 0:    
            loss_R1, loss_R2, loss_R3, loss_R4, loss_R5, loss_R6 = loss_R[0], loss_R[1], loss_R[2], loss_R[3], loss_R[4], loss_R[5]
            loss = loss_ce + (loss_R1 + loss_R2 + loss_R3 + loss_R4 + loss_R5 + loss_R6)*args.beta
            # print("loss_ce:{}; loss_R:{}".format(loss_ce, (loss_R1 + loss_R2 + loss_R3 + loss_R4 + loss_R5 + loss_R6)*1e-3))
        else:
            loss = loss_ce
            
        #loss.backward(retain_graph=True)
        loss.backward()
        loss_all += loss.item()
        optimizer.step()

        pred = output.data.max(1)[1]
        correct += pred.eq(y.view(-1)).sum().item()
    
    for i in range(6):
        if len(feature_class[i]) != 0:
            feature_class[i] = torch.stack(feature_class[i]).mean(dim=0)
            number_class[i] = sum(number_class[i])
        else:
            number_class[i] = 0
         
        # print('====1======', torch.isnan(feature_class[i]).sum())
        
    return loss_all/len(train_iter), correct/num_data, feature_class, number_class

def train_fedprox(args, model, train_loader, optimizer, loss_fun, client_num, device):
    model.train()
    num_data = 0
    correct = 0
    loss_all = 0
    train_iter = iter(train_loader)
    for step in range(len(train_iter)):
        optimizer.zero_grad()
        x, y = next(train_iter)
        num_data += y.size(0)
        x = x.to(device).float()
        y = y.to(device).long()
        output = model(x)

        loss = loss_fun(output, y)

        #########################we implement FedProx Here###########################
        if step>0:
            w_diff = torch.tensor(0., device=device)
            for w, w_t in zip(server_model.parameters(), model.parameters()):
                w_diff += torch.pow(torch.norm(w - w_t), 2)
            loss += args.mu / 2. * w_diff
        #############################################################################

        if args.weight_d > 0:
            loss = loss + reg_loss(model)
        loss.backward()
        loss_all += loss.item()
        optimizer.step()

        pred = output.data.max(1)[1]
        correct += pred.eq(y.view(-1)).sum().item()
    return loss_all/len(train_iter), correct/num_data

def test(model, test_loader, loss_fun, device):
    model.eval()
    test_loss = 0
    correct = 0
    targets = []
    preds = []

    for data, target in test_loader:
        data = data.to(device).float()
        target = target.to(device).long()
        targets.extend(target.detach().cpu().numpy().tolist())

        output, features = model(data)
        
        test_loss += loss_fun(output, target).item()
        pred = output.data.max(1)[1]  #get the maximum value and the corresponding index
        preds.extend(pred.detach().cpu().numpy().tolist())

        correct += pred.eq(target.view(-1)).sum().item()
    
    return test_loss/len(test_loader), correct /len(test_loader.dataset), preds, targets

def get_grads(model, server_model):
    grads = []
    for key in server_model.state_dict().keys():
        if 'num_batches_tracked' not in key:
            grads.append(model.state_dict()[key].data.clone().detach().flatten() - server_model.state_dict()[key].data.clone().detach().flatten())
    return torch.cat(grads)
    
def GRA(grads):
    """ Projecting conflicting gradients (GRA). """
    client_order = list(range(len(grads)))

    # Run clients in random order
    shuffle(client_order)

    # Initialize client gradients
    grad_intial = [g.clone() for g in grads]

    for i in client_order:

        # Run other clients
        other_clients = [j for j in client_order if j != i]

        for j in other_clients:
            grad_j = grads[j]

            # Compute inner product and check for conflicting gradients
            inner_prod = torch.dot(grad_intial[i], grad_j)
            cos = torch.cosine_similarity(grad_intial[i][None,:], grad_j[None,:])[0]
            if cos < 0:
                # Sustract the conflicting component
                grad_intial[i] -= inner_prod / (grad_j ** 2).sum() * grad_j
    # Sum client gradients
    new_grads = torch.stack(grad_intial).mean(0)

    return new_grads

def set_grads(model, server_model, new_grads):
    start = 0
    for key in server_model.state_dict().keys():
        if 'num_batches_tracked' not in key:
            dims = server_model.state_dict()[key].shape
            end = start + dims.numel()
            model.state_dict()[key].data.copy_(server_model.state_dict()[key].data.clone().detach() + new_grads[start:end].reshape(dims))
            start = end
    return model
            
################# Key Function ########################
def communication(args, server_model, models, client_weights, features_class, n_class, Global_features, n_iteration):
    Grads = []
    for model in models:
        Grads.append(get_grads(model, server_model))
        
    new_grads = GRA(Grads)
    #print(new_grads[:100])
    
    for k, model in enumerate(models):
        models[k] = set_grads(model, server_model, new_grads)
    
    with torch.no_grad():
        # aggregate params
        if args.mode.lower() == 'fedbn':
            for key in server_model.state_dict().keys(): ##the name of each weight
                if ('running_mean' not in key) and ('running_var' not in key) and ('num_batches_tracked' not in key):
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(client_num):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key] ###obtain the average value of all the models from the local cleint.
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        else:
            for key in server_model.state_dict().keys():
                # num_batches_tracked is a non trainable LongTensor and
                # num_batches_tracked are the same for all clients for the given datasets
                if 'num_batches_tracked' in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(len(client_weights)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        
        new_global_features = [[],[],[],[],[],[]]
        for c in range(6):
            for client_idx in range(client_num):
                if len(features_class[client_idx][c]) !=0:
                    new_global_features[c].append(((n_class[client_idx][c])/(np.sum(n_class,axis=0)[c]))*features_class[client_idx][c])
            if len(new_global_features[c]) != 0:
                new_global_features[c] = torch.stack(new_global_features[c]).sum(dim=0)
        
        if n_iteration != 0:
            class_order = list(range(6))
            for i in range(6):
                other_order = [j for j in class_order if j != i]
                if len(new_global_features[i]) != 0:
                    if len(Global_features[i]) != 0:
                        # S = [torch.matmul(Global_features[j], Global_features[i]).item() if len(Global_features[j]) != 0 else -1 for j in other_order]
                        S = [F.pairwise_distance(Global_features[j].view(1,128), Global_features[i].view(1,128),p=2).item() if len(Global_features[j]) != 0 else -1 for j in other_order]
                        related_class = other_order[S.index(min(S))]
                        S_pos = math.exp((F.pairwise_distance(Global_features[i].view(1,128), new_global_features[i].view(1,128),p=2).item())/args.temp)
                        S_neg = math.exp((F.pairwise_distance(Global_features[related_class].view(1,128), new_global_features[i].view(1,128),p=2).item())/args.temp)
                        Global_features[i] = (S_pos/(S_pos+S_neg))*Global_features[i] + (S_neg/(S_pos+S_neg))*new_global_features[i]
                    else:
                        Global_features[i] = new_global_features[i]
        else:
            Global_features = new_global_features
        
    return server_model, models, Global_features


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print('Device:', device)
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='store_true', help ='whether to make a log')
    parser.add_argument('--test', action='store_true', help ='test the pretrained model')
    parser.add_argument('--gpu', type = int, default=1, help='use gpu or not')
    parser.add_argument('--percent', type = float, default= 0.1, help ='percentage of dataset to train')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--batch', type = int, default= 256, help ='batch size')
    parser.add_argument('--iters', type = int, default=100, help = 'iterations for communication')
    parser.add_argument('--wk_iters', type = int, default=1, help = 'optimization iters in local worker between communication')
    parser.add_argument('--weight_d', type = float, default= 0.1, help ='weight decay for regularization')   #fixed
    parser.add_argument('--seed',type=int, default=1, help='seed')
    parser.add_argument('--mode', type = str, default='fedbn', help='fedavg | fedprox | fedbn')
    parser.add_argument('--mu', type=float, default=1e-2, help='The hyper parameter for fedprox')
    parser.add_argument('--data_path', type = str, default='Datasets_Fed_SL', help='path to save the checkpoint')
    parser.add_argument('--save_path', type = str, default='./checkpoint', help='path to save the checkpoint')
    parser.add_argument('--beta', type=float, default=1., help='The hyper parameter')
    parser.add_argument('--temp', type=float, default=1., help='The hyper parameter')
    args = parser.parse_args()
    
    setup_seed(args.seed)

    exp_folder = 'federated_equines/'

    args.save_path = os.path.join(args.save_path, exp_folder)
    if not os.path.exists(args.save_path):
      os.makedirs(args.save_path)
    temp_PATH_1 = os.path.join(args.save_path, 'Mode_{}'.format(args.mode))
    if not os.path.exists(temp_PATH_1):
        os.makedirs(temp_PATH_1)
    DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
    temp_PATH_2 = os.path.join(temp_PATH_1, datetime.now().strftime(DATE_FORMAT))
    if not os.path.exists(temp_PATH_2):
        os.makedirs(temp_PATH_2)  ###checkpoint->federated_equines->Mode/FedBN,Fedavg/time->saved results.
    
    log = args.log
    if log:
        logfile = open(os.path.join(temp_PATH_2,'log.txt'), 'a')   #Recordings during training
        logfile.write('==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        logfile.write('===Setting==Baseline+PCG+Prototype-new===\n')
        logfile.write('    seed: {}\n'.format(args.seed))
        logfile.write('    lr: {}\n'.format(args.lr))
        logfile.write('    batch: {}\n'.format(args.batch))
        logfile.write('    iters: {}\n'.format(args.iters))
        logfile.write('    wk_iters: {}\n'.format(args.wk_iters))
        logfile.write('    weight_decay: {}\n'.format(args.weight_d))
        logfile.write('    beta: {}\n'.format(args.beta))
        logfile.write('    temp: {}\n'.format(args.temp))
        logfile.write('    data_path: {}\n'.format(args.data_path))
        
    #Some important results
    f = open(os.path.join(temp_PATH_2,'output.txt'), 'a')
    #Pathway of trained model
    SAVE_PATH = os.path.join(temp_PATH_2, 'net')
 
    # prepare the data
    train_loaders, test_loader = prepare_data(args)
    
    server_model = CaNet().to(device)
    if args.weight_d > 0:
        reg_loss=Regularization(server_model, args.weight_d, p=2)
    else:
        print("no regularization")
    
    # name of each client dataset
    datasets = ['C1', 'C2', 'C3', 'C4', 'C5']
    
    # federated setting
    client_num = len(datasets)
    client_weights = [1/client_num for i in range(client_num)]
    #client_weights = [23625/82274, 11071/82274, 10127/82274, 24602/82274, 12849/82274]
    models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]
    
    loss_fun = nn.CrossEntropyLoss()
    optimizers = [optim.Adam(params=models[idx].parameters(), lr=args.lr) for idx in range(client_num)]
    train_schedulers = [optim.lr_scheduler.StepLR(optimizers[idx], step_size=20, gamma=0.1) for idx in range(client_num)]

    # start training
    best_acc = 0.0
    best_epoch = 0
    best_model = server_model
    Train_accuracy = []
    Train_loss = []
    Test_accuracy = []
    Test_loss = []
    Global_features = [[],[],[],[],[],[]]
    predictions = []
    ground_truths = []

    for a_iter in range(0, args.iters):
        for wi in range(args.wk_iters):
            print("============ Train epoch {} ============".format(wi + a_iter * args.wk_iters), flush=True)
            if args.log: logfile.write("============ Train epoch {} ============\n".format(wi + a_iter * args.wk_iters)) 
            
            client_feature = []
            client_class_number = []
            for client_idx in range(client_num):
                train_schedulers[client_idx].step((wi + a_iter * args.wk_iters)+1)
                
                model, train_loader, optimizer = models[client_idx], train_loaders[client_idx], optimizers[client_idx]
                if args.mode.lower() == 'fedprox':
                    if a_iter > 0:
                        train_fedprox(args, model, train_loader, optimizer, loss_fun, client_num, device)
                    else:
                        train(model, train_loader, optimizer, loss_fun, client_num, device)
                else:
                    _, _, Features_class, N_class = train(args, model, train_loader, optimizer, loss_fun, client_num, device, Global_features, a_iter)
                    client_feature.append(Features_class)
                    client_class_number.append(N_class)
         
        # aggregation
        server_model, models, Global_features = communication(args, server_model, models, client_weights, client_feature, client_class_number, Global_features, a_iter)
        
        ###both of the train and test results are caculated by the updated models(after aggregation).
        # report after aggregation
        for client_idx in range(client_num):
                model, train_loader, optimizer = models[client_idx], train_loaders[client_idx], optimizers[client_idx]
                train_loss, train_acc, train_pred, train_gt = test(model, train_loader, loss_fun, device)
                Train_accuracy.append(train_acc)
                Train_loss.append(train_loss)
                print(' {:<11s}| Train Loss: {:.4f} | Train Acc: {:.4f}'.format(datasets[client_idx] ,train_loss, train_acc), flush=True)
                if args.log:
                    logfile.write(' {:<11s}| Train Loss: {:.4f} | Train Acc: {:.4f}\n'.format(datasets[client_idx] ,train_loss, train_acc))\

        # start testing
        test_loss, test_acc, prediction, ground_truth = test(server_model, test_loader, loss_fun, device)
        Test_accuracy.append(test_acc)
        Test_loss.append(test_loss)
        predictions.append(prediction)
        ground_truths.append(ground_truth)
        print(' Test  Loss: {:.4f} | Test  Acc: {:.4f}'.format(test_loss, test_acc), flush=True)
        if args.log:
            logfile.write(' Test  Loss: {:.4f} | Test  Acc: {:.4f}\n'.format(test_loss, test_acc))
        
        if a_iter > 20 and best_acc < test_acc:
            best_acc = test_acc
            best_epoch = a_iter
            best_model = server_model
            
    ###Saving Results
    SAVE_PATH = os.path.join(temp_PATH_2, '{}_{:.4f}'.format(best_epoch,best_acc))
    
    #saving important results into the output.txt
    print('---------------Classification report on test dataset----------------------', file=f)
    print(classification_report(ground_truths[best_epoch], predictions[best_epoch], digits=4), file=f)
    
    # Save checkpoint
    print(' Saving checkpoints to {}...'.format(SAVE_PATH), flush=True)
    torch.save({'best_model': best_model.state_dict()}, SAVE_PATH)

    if log:
        logfile.flush()
        logfile.close()
