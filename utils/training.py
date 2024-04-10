# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from utils.status import progress_bar, create_stash
from utils.tb_logger import *
from utils.loggers import *
from utils.loggers import CsvLogger
from argparse import Namespace
from models.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset
from typing import Tuple
from datasets import get_dataset
import sys
import wandb
import torch.nn.functional as F

import torch.nn as nn

def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
               dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.constant_(m.bias, 0)

# Original

def evaluate0(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.to(model.device)
    model.net.eval()
    print('model.name',model.NAME)
    accs, accs_mask_classes = [], []
    for k, test_loader in enumerate(dataset.test_loaders): # for all previous test tasks
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0       
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                if 'class-il' not in model.COMPATIBILITY:
                    outputs = model(inputs, k)
                else:
                    outputs = model(inputs)
                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]

                if dataset.SETTING == 'class-il':
                    mask_classes(outputs, dataset, k)
                    _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == labels).item()
        accs.append(correct / total * 100 if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)
    model.net.train(status)
    return accs, accs_mask_classes


def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    print('model.name1',model.NAME)
    accs, accs_mask_classes = [], []
    
    for k, test_loader in enumerate(dataset.test_loaders): # for all previous test tasks     
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0  
        # print('k',k) 
        # a = [0]*10
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                if 'class-il' not in model.COMPATIBILITY:
                    outputs = model(inputs, k)
                else:
                    outputs = model(inputs)
                _, pred = torch.max(outputs.data, 1)
                # for i in pred:
                #     # if pred==i:
                #         a[i] +=1
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]
                # print('total',total)
                if dataset.SETTING == 'class-il':
                    mask_classes(outputs, dataset, k)
                    _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == labels).item()
        # print('confusion matrix', k,a)
        accs.append(correct / total * 100 if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)
    
    model.net.train(status)
    return accs, accs_mask_classes    

## Prototype balance-train (M2)
'''
def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.to(model.device)
    print('model.name',model.NAME)
    accs, accs_mask_classes = [], []

    for k, test_loader in enumerate(dataset.test_loaders): 
        for param in model.net.bn.parameters():
            param.requires_grad = False
        model.net.classifier.apply(init_weights)  
        # count = 0  
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        if k == 0:
            count = 0
            for data in test_loader:  
                if count >4:
                    break
                if count == 0:
                    inputs, labels = data
                    inputs, labels = inputs[:32].to(model.device), labels[:32].to(model.device)
                else:
                    inputs1, labels1 = data 
                    inputs1, labels1 = inputs1[:32].to(model.device), labels1[:32].to(model.device)
                    inputs = torch.cat((inputs,inputs1),dim=0)
                    labels = torch.cat((labels,labels1),dim=0)
                count += 1
        else:
            count = 0
            for data in test_loader:    
                if count >4:
                    break         
                inputs1, labels1 = data 
                inputs1, labels1 = inputs1[:32].to(model.device), labels1[:32].to(model.device)
                inputs = torch.cat((inputs,inputs1),dim=0)
                labels = torch.cat((labels,labels1),dim=0)
                count +=1
    for j in range(3):
        for i in range(2):
            inputs_test, labels_test = inputs[-64:], labels[-64:]
            model.observe_test(inputs_test[32*i:32*i+32],labels_test[32*i:32*i+32])
    for param in model.net.bn.parameters():
        param.requires_grad = False

    for i in range(6):
        model.opt1.zero_grad()
        # model.opt.zero_grad()
        outputs = model.net(inputs)
        loss = model.loss(outputs,labels)
        loss.backward()
        model.opt1.step()  

    unique_labels = labels.unique()
    proto_features= []
    for j, label in enumerate(unique_labels):
        idx = torch.where(labels == label)[0]
        # print('idx', idx)
        temp = model.net(inputs[idx],returnt = 'features')
        # print('temp',temp.shape)
        proto_feature_j = torch.mean(temp,dim=0)
        # Normalize the Proto features
        proto_feature_j = F.normalize(proto_feature_j,dim=0)
        proto_features.append(proto_feature_j)
    proto_features = torch.stack(proto_features)
    # print('fea_type', proto_features.shape)
    
    for param in model.net.bn.parameters():
        param.requires_grad = False
    pred_num = [0]*10
    for k, test_loader in enumerate(dataset.test_loaders): 
        model.net.eval()         
        with torch.no_grad():
            correct, correct_mask_classes, total = 0.0, 0.0, 0.0
            for data in test_loader:
                inputs, labels = data
                # print('\ninputs_eval',inputs.shape)
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                if 'class-il' not in model.COMPATIBILITY:
                    outputs = model.net(inputs, k, returnt='features')
                else:
                    outputs = model.net(inputs, returnt='features')
                # print('class-il' in model.COMPATIBILITY)
                pred = []
                for j in range(len(outputs)):
                    # dist  = (F.normalize(outputs[j],dim=0)-proto_features).mean(1)
                    dist = F.cosine_similarity(outputs[j].unsqueeze(0),proto_features)
                    # print('dist',dist)
                    pred.append(torch.argmax(dist))
                pred = torch.stack(pred)

                for t in range(len(pred)):
                    pred_num[int(pred[t])] += 1
                # else:
                #     for tt in range(1,k+1):
                #        pred_num[2*tt] += torch.sum(pred == 2*tt).item()
                #        pred_num[2*tt+1] += torch.sum(pred == 2*tt+1).item()                       

                # print('pred',pred[:10])
                # print('max_pred',max(pred))
                # print('labels',labels[:10])
                
                # _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()
                # print('correct',k,correct)
                total += labels.shape[0]

                if dataset.SETTING == 'class-il':
                    mask_classes(outputs, dataset, k)
                    _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == labels).item()
        if total !=0:
            accs.append(correct / total * 100 if 'class-il' in model.COMPATIBILITY else 0)
            accs_mask_classes.append(correct_mask_classes / total * 100)
            print('pred_num',pred_num)
    model.net.train(status)
    for param in model.net.bn.parameters():
        param.requires_grad = True
    return accs, accs_mask_classes
'''
## Prototype few-shot (M1)
'''
def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    # model.net.to(model.device)
    # model.net.eval()
    print('model.name1',model.NAME)
    # if model.NAME != 'la_maml':
    #     model.net.eval()
    accs, accs_mask_classes = [], []
    for k, test_loader in enumerate(dataset.test_loaders): # for all previous test tasks
        # if last and k < len(dataset.test_loaders) - 1:
        #     continue
        
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        proto_features = []

        
        # for param in model.net._features.parameters():
        #     param.requires_grad = False
        for param in model.net.bn.parameters():
            param.requires_grad = False
        # randomly initialize classifier parameters
        model.net.classifier.apply(init_weights)
            # for param in model.net.classifier.parameters():
            #     param.requires_grad = True

        count = 0
        for data in test_loader:  
            if count >2:
                break
            if count == 0:
                inputs, labels = data
                inputs, labels = inputs[:32].to(model.device), labels[:32].to(model.device)
            else:
                inputs1, labels1 = data 
                inputs1, labels1 = inputs1[:32].to(model.device), labels1[:32].to(model.device)
                inputs = torch.cat((inputs,inputs1),dim=0)
                labels = torch.cat((labels,labels1),dim=0)
            count += 1

        unique_labels = labels.unique()
        # print('unique_labels',unique_labels)
        
        for j, label in enumerate(unique_labels):
            idx = torch.where(labels == label)[0]
            temp = model.net(inputs[idx],returnt = 'features')
            proto_feature_j = torch.mean(temp,dim=0)
            # Normalize the Proto features
            # proto_feature_j = F.normalize(proto_feature_j,dim=0)
            proto_features.append(proto_feature_j)
        proto_features = torch.stack(proto_features)

        model.net.eval()     
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                # print('\ninputs_eval',inputs.shape)
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                if 'class-il' not in model.COMPATIBILITY:
                    outputs = model.net(inputs, k, returnt='features')
                else:
                    outputs = model.net(inputs, returnt='features')
                # print('class-il' in model.COMPATIBILITY)
                pred = []
                for j in range(len(outputs)):
                    # dist  = (outputs[j]-proto_features).mean(1)
                    dist = F.cosine_similarity(outputs[j].unsqueeze(0),proto_features)
                    # dist  = (F.normalize(outputs[j],dim=0)-proto_features).mean(1)
                    # print('dist',dist)
                    pred.append(torch.argmax(dist)+2*k)
                pred = torch.stack(pred)
                # print('k', k,proto_features.shape)
                # print('pred',pred[:10])
                # print('labels',labels[:10])
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]
                # print('total',total)

                if dataset.SETTING == 'class-il':
                    mask_classes(outputs, dataset, k)
                    # _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == labels).item()
        accs.append(correct / total * 100 if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)
    model.net.train(status)
    for param in model.net.bn.parameters():
        param.requires_grad = True
    return accs, accs_mask_classes
'''
## Both Prototype M1 & M2
'''
def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    # model.net.to(model.device)
    model.net.eval()
    print('model.name',model.NAME)
    task_total = 0


    # accs1, accs_mask_classes1 = [], []
    # for k, test_loader in enumerate(dataset.test_loaders): # for all previous test tasks     
    #     correct, correct_mask_classes, total = 0.0, 0.0, 0.0   
    #     with torch.no_grad():
    #         for data in test_loader:
    #             inputs, labels = data
    #             inputs, labels = inputs.to(model.device), labels.to(model.device)
    #             if 'class-il' not in model.COMPATIBILITY:
    #                 outputs = model(inputs, k)
    #             else:
    #                 outputs = model(inputs)
    #             _, pred = torch.max(outputs.data, 1)
    #             correct += torch.sum(pred == labels).item()
    #             total += labels.shape[0]
    #             # print('total',total)
    #             if dataset.SETTING == 'class-il':
    #                 mask_classes(outputs, dataset, k)
    #                 _, pred = torch.max(outputs.data, 1)
    #                 correct_mask_classes += torch.sum(pred == labels).item()
    #     accs1.append(correct / total * 100 if 'class-il' in model.COMPATIBILITY else 0)
    #     accs_mask_classes1.append(correct_mask_classes / total * 100)
    # print('Current_class_IL', accs1)
    # print('Mean Acc (Class-IL)', np.mean(accs1))
    # print('Current_task_IL', accs_mask_classes1)
    # print('Mean Acc (Task-IL)', np.mean(accs_mask_classes1))


    

    accs1, accs_mask_classes1 = [], []      
    for k, test_loader in enumerate(dataset.test_loaders): # for all previous test tasks     
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0   
        # print('k',k)
        # print('dataset_id', dataset.task_id)
        # print('task_id', dataset.i)

        # for domain-IL
        # print('k,i',k,dataset.i)
        # if k == dataset.i/10:
        #     break

        # for Class-IL and Task-IL
        # if k == dataset.task_id:
        #     break

        # print('CL performance',k)
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                # print('labels',labels)
                # if 11 in torch.unique(labels):
                #     print('saving!!!!!')
                #     torch.save(inputs, 'inputs0.pt')
                # print('\ninputs_eval',inputs.shape)
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                if 'class-il' not in model.COMPATIBILITY:
                    outputs = model(inputs, k)
                else:
                    outputs = model(inputs)
                    # for ER-OBC
                    # outputs = model(inputs, mode = 'balance')
                # print('class-il' in model.COMPATIBILITY)
                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()
                # print('correct',correct)
                total += labels.shape[0]
                # print('total',total)

                if dataset.SETTING == 'class-il':
                    mask_classes(outputs, dataset, k)  # task-IL
                    _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == labels).item()
        accs1.append(correct / total * 100 if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes1.append(correct_mask_classes / total * 100)
    print('Current_class_IL', accs1)
    print('Mean Acc (Class-IL)', np.mean(accs1))
    print('Current_task_IL', accs_mask_classes1)
    print('Mean Acc (Task-IL)', np.mean(accs_mask_classes1))


    model.net.train()
    # model.net.train(status)
    for k, test_loader in enumerate(dataset.test_loaders): 
        # print('k',k)
        task_total += 1
        # for param in model.net.bn.parameters():
        #     param.requires_grad = False
        # model.net.classifier.apply(init_weights)  #!!!!!!!!!!!!!!!!!!!!!!!!!!! not # previous
        # count = 0  
        M = 32
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        if k == 0:
            count = 0
            for data in test_loader:  
                if count > 4:
                    break
                if count == 0:
                    inputs, labels = data
                    # print('inputs0',inputs.shape)
                    inputs, labels = inputs[:M].to(model.device), labels[:M].to(model.device)
                else:
                    inputs1, labels1 = data 
                    inputs1, labels1 = inputs1[:M].to(model.device), labels1[:M].to(model.device)
                    inputs = torch.cat((inputs,inputs1),dim=0)
                    labels = torch.cat((labels,labels1),dim=0)
                count += 1
        else:
            count = 0
            for data in test_loader:    
                if count >4:
                    break         
                inputs1, labels1 = data 
                # print('inputs1',inputs1.shape)
                # print('labels', labels1.shape)
                # labels1 += torch.tensor(k*10)
                # print('labels',labels1)
                inputs1, labels1 = inputs1[:M].to(model.device), labels1[:M].to(model.device)
                inputs = torch.cat((inputs,inputs1),dim=0)
                labels = torch.cat((labels,labels1),dim=0)
                count +=1
    # print('labels[-5*M:]',labels[-5*M:].unique())
    for j in range(25):
        for i in range(5):
            inputs_test, labels_test = inputs[-5*M:], labels[-5*M:] # the samples of the new task
            # print('labels_test', labels_test.unique())
            model.observe_test(inputs_test[M*i:M*i+M],labels_test[M*i:M*i+M],k)  # model part
            # model.observe_test(inputs_test[M*i:M*i+M],labels_test[M*i:M*i+M]-torch.tensor(k*10))  # model part

    # for param in model.net.bn.parameters():
    #     param.requires_grad = False
    # for i in range(10):
    #     model.opt1.zero_grad()
    #     # model.opt.zero_grad()
    #     outputs = model.net(inputs)
    #     loss = model.loss(outputs,labels)
    #     loss.backward()
    #     model.opt1.step()  
    # print('labels',labels)

    # Domain-IL.
    # unique_labels = labels.unique()
    # print('unique_labels',unique_labels)
    # proto_features= []
    # for k in range(task_total):
    #     for _, label in enumerate(unique_labels):
    #         idx = torch.where(labels[k*128:(k+1)*128] == label)[0]
    #         # print('idx', k, label)
    #         temp = model.net(inputs[k*128:(k+1)*128][idx],returnt = 'features')
    #         proto_feature_j = torch.mean(temp,dim=0)
    #         # Normalize the Proto features
    #         proto_feature_j = F.normalize(proto_feature_j,dim=0)
    #         proto_features.append(proto_feature_j)  

    # print('labels',labels)
    unique_labels = labels.unique()
    print('unique_labels',unique_labels)
    proto_features= []
    # features = torch.nn.Sequential(*list(model.net.children())[:-1])
    for j, label in enumerate(unique_labels):
        idx = torch.where(labels == label)[0]
        # temp = model.net(inputs[idx],returnt = 'features') #512
        temp = model.net.features(inputs[idx])
        # print('inputs',inputs[idx].shape)
        # print('temp',temp.shape)
        # temp = torch.nn.AdaptiveAvgPool2d((1, 1))(temp)
        # print('temp1',temp.shape)
        # temp = temp.view(temp.size(0), -1)
        # print('temp',temp.shape)
        proto_feature_j = torch.mean(temp,dim=0)
        # print('proto_feature_j', proto_feature_j.shape)
        # Normalize the Proto features
        proto_feature_j = F.normalize(proto_feature_j,dim=0)
        proto_features.append(proto_feature_j)
    proto_features = torch.stack(proto_features)
    print('proto_features', proto_features.shape)
    n_task_num = proto_features.shape[0]//2-1
    # n_task_num = proto_features.shape[0]
    # print('task-num', n_task_num)
    # for param in model.net.bn.parameters():
    #     param.requires_grad = False

    accs, accs_mask_classes = [], []
    model.net.eval() 
    for k, test_loader in enumerate(dataset.test_loaders): 
        with torch.no_grad():
            correct0, correct_mask_classes, total = 0.0, 0.0, 0.0
            if k == n_task_num:
                for data in test_loader:
                    inputs, labels = data
                    inputs, labels = inputs.to(model.device), labels.to(model.device)
                    if 'class-il' not in model.COMPATIBILITY:
                        outputs = model.net(inputs, k, returnt='features')
                    else:
                        outputs = model.net.features(inputs)
                        # for ER-OBC
                        # outputs = model.net(inputs, returnt='features',mode='surrogate') # the features  are the same with balance
                        # outputs = features(inputs)
                        # outputs = model.net(inputs, returnt='features')
                    pred = []
                    for j in range(len(outputs)):
                        dist = F.cosine_similarity(outputs[j].unsqueeze(0),proto_features[-2:])
                        pred.append(torch.argmax(dist)+k*2) # cifar100
                        # pred.append(torch.argmax(dist)) # domain-IL
                        # pred.append(torch.argmax(dist))
                    pred = torch.stack(pred)
                    correct0 += torch.sum(pred == labels).item()
                    # print('correct',correct, total)
                    total += labels.shape[0]
                    # if dataset.SETTING == 'class-il':
                    #     mask_classes(outputs, dataset, k)
                    #     _, pred = torch.max(outputs.data, 1)
                    #     correct_mask_classes += torch.sum(pred == labels).item()
                print('Metric 1:', correct0 / total * 100)
            
            correct, correct_mask_classes, total = 0.0, 0.0, 0.0
            for data in test_loader:
                inputs, labels = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                if 'class-il' not in model.COMPATIBILITY:
                    outputs = model.net(inputs, k, returnt='features')
                else:
                    outputs = model.net.features(inputs)
                    # for ER-OBC
                    # outputs = model.net(inputs, returnt='features',mode='surrogate') # the features  are the same with balance
                    # outputs = features(inputs)
                    # outputs = model.net(inputs, returnt='features')
                pred, dist_list = [], []
                for j in range(len(outputs)):
                    dist = F.cosine_similarity(outputs[j].unsqueeze(0),proto_features)
                    dist_list.append(dist)
                    pred.append(torch.argmax(dist))
                    # pred.append(torch.fmod(torch.argmax(dist),10)) # domain-IL
                pred = torch.stack(pred)
                dist_list = torch.stack(dist_list)
                # print('dist_list0', dist_list.shape)
                
                correct += torch.sum(pred == labels).item()
                # correct += torch.sum(pred == labels+k*10).item()
                total += labels.shape[0]

                if dataset.SETTING == 'class-il':
                    dist_list = dist_list[:,k*2:(k+1)*2]
                    # print('dist', dist_list.shape)
                    # dist_list = dist_list[-10] # domain-IL
                    # print('dist', dist_list.shape)
                    # print('dist_list', dist_list.shape)
                    pred = torch.argmax(dist_list,dim=1) + k*2
                    # pred = torch.argmax(dist_list,dim=1)
                    # pred = torch.argmax(dist_list)

                correct_mask_classes += torch.sum(pred == labels).item()
        if total !=0:
            accs.append(correct / total * 100 if 'class-il' in model.COMPATIBILITY else 0)
            accs_mask_classes.append(correct_mask_classes / total * 100)

    # accs1, accs_mask_classes1 = [], []        
    # for k, test_loader in enumerate(dataset.test_loaders): # for all previous test tasks     
    #     correct, correct_mask_classes, total = 0.0, 0.0, 0.0   
    #     if k == dataset.task_id:
    #         break
    #     # print('CL performance',k)
    #     with torch.no_grad():
    #         for data in test_loader:
    #             inputs, labels = data
    #             # print('\ninputs_eval',inputs.shape)
    #             inputs, labels = inputs.to(model.device), labels.to(model.device)
    #             if 'class-il' not in model.COMPATIBILITY:
    #                 outputs = model(inputs, k)
    #             else:
    #                 outputs = model(inputs)
    #             # print('class-il' in model.COMPATIBILITY)
    #             _, pred = torch.max(outputs.data, 1)
    #             correct += torch.sum(pred == labels).item()
    #             total += labels.shape[0]
    #             # print('total',total)

    #             if dataset.SETTING == 'class-il':
    #                 mask_classes(outputs, dataset, k)  # task-IL
    #                 _, pred = torch.max(outputs.data, 1)
    #                 correct_mask_classes += torch.sum(pred == labels).item()
    #     accs1.append(correct / total * 100 if 'class-il' in model.COMPATIBILITY else 0)
    #     accs_mask_classes1.append(correct_mask_classes / total * 100)
    # print('Current_class_IL', accs1)
    # print('Mean Acc (Class-IL)', np.mean(accs1))
    # print('Current_task_IL', accs_mask_classes1)
    # print('Mean Acc (Task-IL)', np.mean(accs_mask_classes1))

    model.net.train(status)
    # for param in model.net.bn.parameters():
    #     param.requires_grad = True
    return accs, accs_mask_classes
'''
## Both Prototype M1 & M2  with task id  imcompleted
'''
def evaluate(model: ContinualModel, dataset: ContinualDataset, current_trained_id, test_task_id, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.to(model.device)
    print('model.name',model.NAME)
    task_total = 0
    for k, test_loader in enumerate(dataset.test_loaders): 
        task_total += 1
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        if k == 0:
            count = 0
            for data in test_loader:  
                if count > 4:
                    break
                if count == 0:
                    inputs, labels = data
                    # print('inputs0',inputs.shape)
                    inputs, labels = inputs[:32].to(model.device), labels[:32].to(model.device)
                else:
                    inputs1, labels1 = data 
                    inputs1, labels1 = inputs1[:32].to(model.device), labels1[:32].to(model.device)
                    inputs = torch.cat((inputs,inputs1),dim=0)
                    labels = torch.cat((labels,labels1),dim=0)
                count += 1
        else:
            count = 0
            for data in test_loader:    
                if count >4:
                    break         
                inputs1, labels1 = data 
                # print('inputs1',inputs1.shape)
                inputs1, labels1 = inputs1[:32].to(model.device), labels1[:32].to(model.device)
                inputs = torch.cat((inputs,inputs1),dim=0)
                labels = torch.cat((labels,labels1),dim=0)
                count +=1
    for j in range(10):
        for i in range(5):
            inputs_test, labels_test = inputs[5*test_task_id-5, 5*test_task_id], labels[5*test_task_id-5, 5*test_task_id] # the samples of the new task
            model.observe_test(inputs_test[32*i:32*i+32],labels_test[32*i:32*i+32])  # model part

    # for param in model.net.bn.parameters():
    #     param.requires_grad = False
    # for i in range(10):
    #     model.opt1.zero_grad()
    #     # model.opt.zero_grad()
    #     outputs = model.net(inputs)
    #     loss = model.loss(outputs,labels)
    #     loss.backward()
    #     model.opt1.step()  
    # print('labels',labels)

    # Domain-IL.
    # unique_labels = labels.unique()
    # print('unique_labels',unique_labels)
    # proto_features= []
    # for k in range(task_total):
    #     for _, label in enumerate(unique_labels):
    #         idx = torch.where(labels[k*128:(k+1)*128] == label)[0]
    #         # print('idx', k, label)
    #         temp = model.net(inputs[k*128:(k+1)*128][idx],returnt = 'features')
    #         proto_feature_j = torch.mean(temp,dim=0)
    #         # Normalize the Proto features
    #         proto_feature_j = F.normalize(proto_feature_j,dim=0)
    #         proto_features.append(proto_feature_j)        

    unique_labels = labels.unique()
    print('unique_labels',unique_labels)
    proto_features= []
    # features = torch.nn.Sequential(*list(model.net.children())[:-1])
    for j, label in enumerate(unique_labels):
        idx = torch.where(labels == label)[0]
        temp = model.net(inputs[idx],returnt = 'features') #512
        # temp = features(inputs[idx])
        # print('inputs',inputs[idx].shape)
        # print('temp',temp.shape)
        # temp = torch.nn.AdaptiveAvgPool2d((1, 1))(temp)
        # print('temp1',temp.shape)
        # temp = temp.view(temp.size(0), -1)
        # print('temp',temp.shape)
        proto_feature_j = torch.mean(temp,dim=0)
        # print('proto_feature_j', proto_feature_j.shape)
        # Normalize the Proto features
        proto_feature_j = F.normalize(proto_feature_j,dim=0)
        proto_features.append(proto_feature_j)
    proto_features = torch.stack(proto_features)
    print('proto_features', proto_features.shape)
    n_task_num = proto_features.shape[0]//2-1
    # print('task-num', n_task_num)
    # for param in model.net.bn.parameters():
    #     param.requires_grad = False

    pred_num = [0]*20
    accs, accs_mask_classes = [], []
    for k, test_loader in enumerate(dataset.test_loaders): 
        model.net.eval()         
        # print('k',k)
        # print('n_task_num',n_task_num)
        with torch.no_grad():
            correct, correct_mask_classes, total = 0.0, 0.0, 0.0
            if k == n_task_num:
                for data in test_loader:
                    inputs, labels = data
                    inputs, labels = inputs.to(model.device), labels.to(model.device)
                    if 'class-il' not in model.COMPATIBILITY:
                        outputs = model.net(inputs, k, returnt='features')
                    else:
                        # outputs = features(inputs)
                        outputs = model.net(inputs, returnt='features')
                    pred = []
                    for j in range(len(outputs)):
                        dist = F.cosine_similarity(outputs[j].unsqueeze(0),proto_features[-2:])
                        # pred.append(torch.argmax(dist)+k*2) # cifar100
                        pred.append(torch.argmax(dist))
                        # pred.append(torch.argmax(dist))
                    pred = torch.stack(pred)
                    correct += torch.sum(pred == labels).item()
                    total += labels.shape[0]
                    # if dataset.SETTING == 'class-il':
                    #     mask_classes(outputs, dataset, k)
                    #     _, pred = torch.max(outputs.data, 1)
                    #     correct_mask_classes += torch.sum(pred == labels).item()
                print('Metric 1:', correct / total * 100)
            
            correct, correct_mask_classes, total = 0.0, 0.0, 0.0
            for data in test_loader:
                inputs, labels = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                if 'class-il' not in model.COMPATIBILITY:
                    outputs = model.net(inputs, k, returnt='features')
                else:
                    # outputs = features(inputs)
                    outputs = model.net(inputs, returnt='features')
                pred, dist_list = [], []
                for j in range(len(outputs)):
                    dist = F.cosine_similarity(outputs[j].unsqueeze(0),proto_features)
                    dist_list.append(dist)
                    pred.append(torch.argmax(dist))
                    # pred.append(torch.fmod(torch.argmax(dist),10))
                pred = torch.stack(pred)
                dist_list = torch.stack(dist_list)
                # print('dist_list0',dist_list.shape)
                
                correct += torch.sum(pred == labels).item()

                # correct += torch.sum(pred == labels+k*10).item()
                total += labels.shape[0]

                if dataset.SETTING == 'class-il':
                    dist_list = dist_list[:,k*2:(k+1)*2]
                    # print('dist_list', dist_list.shape)
                    pred = torch.argmax(dist_list,dim=1) #+k*2

                    correct_mask_classes += torch.sum(pred == labels).item()
        if total !=0:
            accs.append(correct / total * 100 if 'class-il' in model.COMPATIBILITY else 0)
            accs_mask_classes.append(correct_mask_classes / total * 100)
    accs1, accs_mask_classes1 = [], []        
    for k, test_loader in enumerate(dataset.test_loaders): # for all previous test tasks     
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0   
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                # print('\ninputs_eval',inputs.shape)
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                if 'class-il' not in model.COMPATIBILITY:
                    outputs = model(inputs, k)
                else:
                    outputs = model(inputs)
                # print('class-il' in model.COMPATIBILITY)
                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]
                # print('total',total)

                if dataset.SETTING == 'class-il':
                    mask_classes(outputs, dataset, k)  # task-IL
                    _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == labels).item()
        accs1.append(correct / total * 100 if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes1.append(correct_mask_classes / total * 100)
    print('Current_class_IL', accs1)
    print('Mean Acc (Class-IL)', np.mean(accs1))
    print('Current_task_IL', accs_mask_classes1)
    print('Mean Acc (Task-IL)', np.mean(accs_mask_classes1))


    model.net.train(status)
    # for param in model.net.bn.parameters():
    #     param.requires_grad = True
    return accs, accs_mask_classes
'''

## Both Prototype M1 & M2 (knn)
'''
from sklearn.neighbors import KNeighborsClassifier
def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.to(model.device)
    print('model.name',model.NAME)
    
    for k, test_loader in enumerate(dataset.test_loaders): 
        for param in model.net.bn.parameters():
            param.requires_grad = False
        model.net.classifier.apply(init_weights)  
        # count = 0  
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        if k == 0:
            count = 0
            for data in test_loader:  
                if count >3:
                    break
                if count == 0:
                    inputs, labels = data
                    inputs, labels = inputs[:32].to(model.device), labels[:32].to(model.device)
                else:
                    inputs1, labels1 = data 
                    inputs1, labels1 = inputs1[:32].to(model.device), labels1[:32].to(model.device)
                    inputs = torch.cat((inputs,inputs1),dim=0)
                    labels = torch.cat((labels,labels1),dim=0)
                    # print('labels',labels)
                count += 1
        else:
            count = 0
            for data in test_loader:    
                if count >3:
                    break         
                inputs1, labels1 = data 
                inputs1, labels1 = inputs1[:32].to(model.device), labels1[:32].to(model.device)
                inputs = torch.cat((inputs,inputs1),dim=0)
                labels = torch.cat((labels,labels1),dim=0)
                count +=1
    for j in range(10):
        for i in range(4):
            inputs_test, labels_test = inputs[-4*32:], labels[-4*32:]
            model.observe_test(inputs_test[32*i:32*i+32],labels_test[32*i:32*i+32])

    unique_labels = labels.unique()
    # print('unique_labels',unique_labels)
    proto_features= []
    x_train = []
    y_train = []
    knn = KNeighborsClassifier(n_neighbors = 5)
    for j, label in enumerate(unique_labels):
        idx = torch.where(labels == label)[0]
        temp = model.net(inputs[idx],returnt = 'features')
        proto_feature_j = torch.mean(temp,dim=0)
        temp1 = temp.detach().cpu().numpy()
        x_train.append(temp1)
        y_train.append(label.detach().cpu().numpy()*([1]*len(temp1)))
        # print('temp1',temp1.shape)
        # print('label', label.detach().cpu().numpy()*len(temp1))
        # print('temp',temp.detach().cpu().numpy())
        # knn.fit(temp1, label.detach().cpu().numpy()*len(temp1) )
        # Normalize the Proto features
        proto_feature_j = F.normalize(proto_feature_j,dim=0)
        proto_features.append(proto_feature_j)
    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    # print('x_train', np.concatenate(x_train, axis=0).shape)
    # print('y_train', y_train)
    knn.fit(x_train,y_train)
    proto_features = torch.stack(proto_features)
    print('proto_features', proto_features.shape)
    n_task_num = proto_features.shape[0]//10-1
    # print('task-num', n_task_num)
    
    for param in model.net.bn.parameters():
        param.requires_grad = False
    pred_num = [0]*20
    accs, accs_mask_classes = [], []
    for k, test_loader in enumerate(dataset.test_loaders): 
        model.net.eval()         
        # print('k',k)
        # print('n_task_num',n_task_num)
        with torch.no_grad():
            correct, correct_mask_classes, total = 0.0, 0.0, 0.0
            # the lastest task
            if k == n_task_num:
                for data in test_loader:
                    inputs, labels = data
                    inputs, labels = inputs.to(model.device), labels.to(model.device)
                    if 'class-il' not in model.COMPATIBILITY:
                        outputs = model.net(inputs, k, returnt='features')
                    else:
                        outputs = model.net(inputs, returnt='features')
                    pred = []
                    for j in range(len(outputs)):
                        dist = F.cosine_similarity(outputs[j].unsqueeze(0),proto_features[-10:])
                        pred.append(torch.argmax(dist)+k*10)
                    pred = torch.stack(pred)

                    correct += torch.sum(pred == labels).item()
                    total += labels.shape[0]

                    if dataset.SETTING == 'class-il':
                        mask_classes(outputs, dataset, k)
                        _, pred = torch.max(outputs.data, 1)
                        correct_mask_classes += torch.sum(pred == labels).item()
                print('Metric 1:', correct / total * 100)
                correct, correct_mask_classes, total = 0.0, 0.0, 0.0



            for data in test_loader:
                inputs, labels = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                if 'class-il' not in model.COMPATIBILITY:
                    outputs = model.net(inputs, k, returnt='features')
                else:
                    outputs = model.net(inputs, returnt='features')
                pred = []
                # for j in range(len(outputs)):
                #     dist = F.cosine_similarity(outputs[j].unsqueeze(0),proto_features)
                #     pred.append(torch.argmax(dist))
                # pred = torch.stack(pred)
                # print('pre-pred', outputs.detach().cpu().numpy().shape)
                # print('pred',knn.predict(outputs.detach().cpu().numpy()))
                pred = torch.tensor(knn.predict(outputs.detach().cpu().numpy())).cuda()
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]

                if dataset.SETTING == 'class-il':
                    mask_classes(outputs, dataset, k)
                    _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == labels).item()
        if total !=0:
            accs.append(correct / total * 100 if 'class-il' in model.COMPATIBILITY else 0)
            accs_mask_classes.append(correct_mask_classes / total * 100)
    accs1, accs_mask_classes1 = [], []        
    for k, test_loader in enumerate(dataset.test_loaders): # for all previous test tasks     
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0   
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                # print('\ninputs_eval',inputs.shape)
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                if 'class-il' not in model.COMPATIBILITY:
                    outputs = model(inputs, k)
                else:
                    outputs = model(inputs)
                # print('class-il' in model.COMPATIBILITY)
                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]
                # print('total',total)

                if dataset.SETTING == 'class-il':
                    mask_classes(outputs, dataset, k)
                    _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == labels).item()
        accs1.append(correct / total * 100 if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes1.append(correct_mask_classes / total * 100)
    print('Current_class_IL', accs1)
    print('Mean Acc (Class-IL)', np.mean(accs1))
    print('Current_task_IL', accs_mask_classes1)
    print('Mean Acc (Task-IL)', np.mean(accs_mask_classes1))

    model.net.train(status)
    for param in model.net.bn.parameters():
        param.requires_grad = True
    return accs, accs_mask_classes
'''
'''
## For balance-train the classifier and feature extractor with fixed BN
def evaluate0(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.to(model.device)
    print('model.name',model.NAME)
    accs, accs_mask_classes = [], []

    for k, test_loader in enumerate(dataset.test_loaders): 
        for param in model.net.bn.parameters():
            param.requires_grad = False
        model.net.classifier.apply(init_weights)  
        # count = 0  
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        if k == 0:
            count = 0
            for data in test_loader:  
                if count >1:
                    break
                if count == 0:
                    inputs, labels = data
                    inputs, labels = inputs[:32].to(model.device), labels[:32].to(model.device)
                else:
                    inputs1, labels1 = data 
                    inputs1, labels1 = inputs1[:32].to(model.device), labels1[:32].to(model.device)
                    inputs = torch.cat((inputs,inputs1),dim=0)
                    labels = torch.cat((labels,labels1),dim=0)
                count += 1
        else:
            count = 0
            for data in test_loader:    
                if count >1:
                    break         
                inputs1, labels1 = data 
                inputs1, labels1 = inputs1[:32].to(model.device), labels1[:32].to(model.device)
                inputs = torch.cat((inputs,inputs1),dim=0)
                labels = torch.cat((labels,labels1),dim=0)
                count +=1
    # torch.save(inputs, 'inputs.pt')
    # torch.save(labels, 'labels.pt')
    for j in range(1): #3
        for i in range(2):
            inputs_test, labels_test = inputs[-64:], labels[-64:]
            model.observe_test(inputs_test[32*i:32*i+32],labels_test[32*i:32*i+32])
    # for param in model.net.bn.parameters():
    #     param.requires_grad = False
    for i in range(1): #6
        model.opt1.zero_grad()
        # model.opt.zero_grad()
        outputs = model.net(inputs)
        loss = model.loss(outputs,labels)
        loss.backward()
        model.opt1.step()  
        # model.opt.step()      
    # torch.save(model.net.state_dict(), 'models_epoch_{}.pth'.format(t))

    for param in model.net.bn.parameters():
        param.requires_grad = False
    for k, test_loader in enumerate(dataset.test_loaders): 
        model.net.eval()         
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                # print('\ninputs_eval',inputs.shape)
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                if 'class-il' not in model.COMPATIBILITY:
                    outputs = model(inputs, k)
                else:
                    outputs = model(inputs)
                # print('class-il' in model.COMPATIBILITY)
                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]
                # print('total',total)

                if dataset.SETTING == 'class-il':
                    mask_classes(outputs, dataset, k)
                    _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == labels).item()
        accs.append(correct / total * 100 if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)
    model.net.train(status)
    for param in model.net.bn.parameters():
        param.requires_grad = True
    return accs, accs_mask_classes
def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.to(model.device)
    print('model.name',model.NAME)
    accs, accs_mask_classes = [], []

    for k, test_loader in enumerate(dataset.test_loaders): 
        for param in model.net.bn.parameters():
            param.requires_grad = False
        model.net.classifier.apply(init_weights)  
        # count = 0  
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        if k == 0:
            count = 0
            for data in test_loader:  
                if count >1:
                    break
                if count == 0:
                    inputs, labels = data
                    inputs, labels = inputs[:32].to(model.device), labels[:32].to(model.device)
                else:
                    inputs1, labels1 = data 
                    inputs1, labels1 = inputs1[:32].to(model.device), labels1[:32].to(model.device)
                    inputs = torch.cat((inputs,inputs1),dim=0)
                    labels = torch.cat((labels,labels1),dim=0)
                count += 1

        else:
            count = 0
            for data in test_loader:    
                if count >1:
                    break         
                inputs1, labels1 = data 
                inputs1, labels1 = inputs1[:32].to(model.device), labels1[:32].to(model.device)
                inputs = torch.cat((inputs,inputs1),dim=0)
                labels = torch.cat((labels,labels1),dim=0)
                count +=1
    for j in range(3):
        for i in range(2):
            inputs_test, labels_test = inputs[-64:], labels[-64:]
            model.observe_test(inputs_test[32*i:32*i+32],labels_test[32*i:32*i+32])
    # for param in model.net.bn.parameters():
    #     param.requires_grad = False
    for i in range(3):
        model.opt1.zero_grad()
        # model.opt.zero_grad()
        outputs = model.net(inputs)
        loss = model.loss(outputs,labels)
        loss.backward()
        model.opt1.step()  
        # model.opt.step() 
    torch.save(model.net.state_dict(), 'models2_epoch_{}.pth'.format(k))
    for param in model.net.bn.parameters():
        param.requires_grad = False
    for k, test_loader in enumerate(dataset.test_loaders): 
        model.net.eval()         
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                # print('\ninputs_eval',inputs.shape)
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                if 'class-il' not in model.COMPATIBILITY:
                    outputs = model(inputs, k)
                else:
                    outputs = model(inputs)
                # print('class-il' in model.COMPATIBILITY)
                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]
                # print('total',total)

                if dataset.SETTING == 'class-il':
                    mask_classes(outputs, dataset, k)
                    _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == labels).item()
        accs.append(correct / total * 100 if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)
    model.net.train(status)
    for param in model.net.bn.parameters():
        param.requires_grad = True
    return accs, accs_mask_classes
'''

'''
## For balance-train the classifier
def evaluate0(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.to(model.device)
    print('model.name',model.NAME)
    accs, accs_mask_classes = [], []

    for k, test_loader in enumerate(dataset.test_loaders): 
        for param in model.net.bn.parameters():
            param.requires_grad = False
        model.net.classifier.apply(init_weights)  
        # count = 0  
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        if k == 0:
            count = 0
            for data in test_loader:  
                if count >1:
                    break
                if count == 0:
                    inputs, labels = data
                    inputs, labels = inputs[:32].to(model.device), labels[:32].to(model.device)
                else:
                    inputs1, labels1 = data 
                    inputs1, labels1 = inputs1[:32].to(model.device), labels1[:32].to(model.device)
                    inputs = torch.cat((inputs,inputs1),dim=0)
                    labels = torch.cat((labels,labels1),dim=0)
                count += 1

        else:
            count = 0
            for data in test_loader:    
                if count >1:
                    break         
                inputs1, labels1 = data 
                inputs1, labels1 = inputs1[:32].to(model.device), labels1[:32].to(model.device)
                inputs = torch.cat((inputs,inputs1),dim=0)
                labels = torch.cat((labels,labels1),dim=0)
                count +=1
    torch.save(inputs, 'inputs.pt')
    torch.save(labels, 'labels.pt')
    for i in range(6):
        model.opt1.zero_grad()
        # model.opt.zero_grad()
        outputs = model.net(inputs)
        loss = model.loss(outputs,labels)
        loss.backward()
        model.opt1.step()  
        # model.opt.step()  

    for k, test_loader in enumerate(dataset.test_loaders): 
        model.net.eval()         
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                # print('\ninputs_eval',inputs.shape)
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                if 'class-il' not in model.COMPATIBILITY:
                    outputs = model(inputs, k)
                else:
                    outputs = model(inputs)
                # print('class-il' in model.COMPATIBILITY)
                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]
                # print('total',total)

                if dataset.SETTING == 'class-il':
                    mask_classes(outputs, dataset, k)
                    _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == labels).item()
        accs.append(correct / total * 100 if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)
    model.net.train(status)
    for param in model.net.bn.parameters():
        param.requires_grad = True
    return accs, accs_mask_classes
def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.to(model.device)
    print('model.name',model.NAME)
    accs, accs_mask_classes = [], []

    for k, test_loader in enumerate(dataset.test_loaders): 
        for param in model.net.bn.parameters():
            param.requires_grad = False
        model.net.classifier.apply(init_weights)  
        # count = 0  
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        if k == 0:
            count = 0
            for data in test_loader:  
                if count >1:
                    break
                if count == 0:
                    inputs, labels = data
                    inputs, labels = inputs[:32].to(model.device), labels[:32].to(model.device)
                else:
                    inputs1, labels1 = data 
                    inputs1, labels1 = inputs1[:32].to(model.device), labels1[:32].to(model.device)
                    inputs = torch.cat((inputs,inputs1),dim=0)
                    labels = torch.cat((labels,labels1),dim=0)
                count += 1

        else:
            count = 0
            for data in test_loader:    
                if count >1:
                    break         
                inputs1, labels1 = data 
                inputs1, labels1 = inputs1[:32].to(model.device), labels1[:32].to(model.device)
                inputs = torch.cat((inputs,inputs1),dim=0)
                labels = torch.cat((labels,labels1),dim=0)
                count +=1
    # model.opt1.zero_grad()
    for i in range(6):
        model.opt1.zero_grad()
        # model.opt.zero_grad()
        outputs = model.net(inputs)
        loss = model.loss(outputs,labels)
        loss.backward()
        # model.opt.step()  
        model.opt1.step()  
    # model.opt1.step()  
    for k, test_loader in enumerate(dataset.test_loaders): 
        model.net.eval()         
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                # print('\ninputs_eval',inputs.shape)
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                if 'class-il' not in model.COMPATIBILITY:
                    outputs = model(inputs, k)
                else:
                    outputs = model(inputs)
                # print('class-il' in model.COMPATIBILITY)
                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]
                # print('total',total)

                if dataset.SETTING == 'class-il':
                    mask_classes(outputs, dataset, k)
                    _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == labels).item()
        accs.append(correct / total * 100 if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)
    model.net.train(status)
    for param in model.net.bn.parameters():
        param.requires_grad = True
    return accs, accs_mask_classes
'''

## For Few-shot Adaptation
'''
def evaluate0(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.to(model.device)
    # model.net.eval()
    print('model.name',model.NAME)
    # if model.NAME != 'la_maml':
    #     model.net.eval()
    accs, accs_mask_classes = [], []
    for k, test_loader in enumerate(dataset.test_loaders): # for all previous test tasks
        
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        # for param in model.net._features.parameters():
        #     param.requires_grad = False
        for param in model.net.bn.parameters():
            param.requires_grad = False

        # # randomly initialize classifier parameters
        model.net.classifier.apply(init_weights)        
        count = 0
        for data in test_loader:
            if  count >2:
                break
            inputs, labels= data
            # print('inputs',inputs.shape)
            inputs, labels = inputs[:16].to(model.device), labels[:16].to(model.device)
            model.opt1.zero_grad()
            outputs = model.net(inputs)
            loss = model.loss(outputs,labels)
            loss.backward()
            model.opt1.step()   
            count +=1    
        # print('count',count)
        model.net.eval()         
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                # print('\ninputs_eval',inputs.shape)
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                if 'class-il' not in model.COMPATIBILITY:
                    outputs = model(inputs, k)
                else:
                    outputs = model(inputs)
                # print('class-il' in model.COMPATIBILITY)
                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]
                # print('total',total)

                if dataset.SETTING == 'class-il':
                    mask_classes(outputs, dataset, k)
                    _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == labels).item()
        accs.append(correct / total * 100 if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)
    model.net.train(status)
    for param in model.net.bn.parameters():
        param.requires_grad = True
    return accs, accs_mask_classes
def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    # model.net.to(model.device)
    # model.net.eval()
    print('model.name1',model.NAME)
    # if model.NAME != 'la_maml':
    #     model.net.eval()
    accs, accs_mask_classes = [], []
    for k, test_loader in enumerate(dataset.test_loaders): # for all previous test tasks
        # if last and k < len(dataset.test_loaders) - 1:
        #     continue
        
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        
        # for param in model.net._features.parameters():
        #     param.requires_grad = False
        for param in model.net.bn.parameters():
            param.requires_grad = False
        # randomly initialize classifier parameters
        model.net.classifier.apply(init_weights)
            # for param in model.net.classifier.parameters():
            #     param.requires_grad = True
        count = 0
        for data in test_loader:
            if  count > 2:
                break
            inputs, labels= data
            # print('inputs',inputs.shape)
            inputs, labels = inputs[:16].to(model.device), labels[:16].to(model.device)
            model.opt1.zero_grad()
            outputs = model.net(inputs)
            loss = model.loss(outputs,labels)
            loss.backward()
            model.opt1.step()   
            count +=1   
        # count = 0
        # for data in test_loader:
        #     if  count >100:
        #         break
        #     inputs, labels= data
        #     # print('inputs',inputs.shape)
        #     inputs, labels = inputs.to(model.device), labels.to(
        #         model.device)
        #     not_aug_inputs = inputs
        #     loss = model.observe_test(inputs, labels, k)     
        #     count +=1  

        model.net.eval()     
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                # print('\ninputs_eval',inputs.shape)
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                if 'class-il' not in model.COMPATIBILITY:
                    outputs = model(inputs, k)
                else:
                    outputs = model(inputs)
                # print('class-il' in model.COMPATIBILITY)
                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]
                # print('total',total)

                if dataset.SETTING == 'class-il':
                    mask_classes(outputs, dataset, k)
                    _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == labels).item()
        accs.append(correct / total * 100 if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)
    model.net.train(status)
    for param in model.net.bn.parameters():
        param.requires_grad = True
    return accs, accs_mask_classes
'''

def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    model.net.to(model.device)
    results, results_mask_classes = [], []
    accs_iter = []
    M2_1, M2_2 =[],[]

    if args.csv_log: #(default: True)
        csv_logger = CsvLogger(dataset.SETTING, dataset.NAME, model.NAME)
    if args.tensorboard: #(default: True)
        tb_logger = TensorboardLogger(args, dataset.SETTING)

    print(file=sys.stderr)
    for t in range(dataset.N_TASKS):
        model.net.train()
        train_loader, test_loader = dataset.get_data_loaders()  # 这个返回的dataloader 为什么可以代表不同的task? perm_mnist 每个task就是包含10个类, 其他的数据集有其各自的写法
        if hasattr(model, 'begin_task'): # hasattr 判断是否有该属性
            model.begin_task(dataset)

        # if t: # (test code) t=0 第一个任务不用测试
        #     accs = evaluate(model, dataset, last=True)
        #     results[t-1] = results[t-1] + accs[0]
        #     if dataset.SETTING == 'class-il':
        #         results_mask_classes[t-1] = results_mask_classes[t-1] + accs[1]
        scheduler = dataset.get_scheduler(model, args)
        for epoch in range(model.args.n_epochs):
            for i, data in enumerate(train_loader):
                if hasattr(dataset.train_loader.dataset, 'logits'):
                    inputs, labels, not_aug_inputs, logits = data
                    #print('inputs',inputs.shape)
                    inputs = inputs.to(model.device)
                    labels = labels.to(model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    logits = logits.to(model.device)
                    # loss = model.observe(inputs, labels, not_aug_inputs, logits, t)
                    loss = model.observe(inputs, labels, not_aug_inputs, t, logits)
                else:
                    inputs, labels, not_aug_inputs = data
                    # print('inputs',inputs.shape)
                    inputs, labels = inputs.to(model.device), labels.to(
                        model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)  # 这个部分被存入buffer
                    loss = model.observe(inputs, labels, not_aug_inputs, t)
                progress_bar(i, len(train_loader), epoch, t, loss) # 这个部分为啥运行的时候bar不合适(解决)

                if args.tensorboard:
                    tb_logger.log_loss(loss, args, epoch, t, i)

                # accs = evaluate(model, dataset) 
                # accs_iter.append(torch.tensor(accs[0][0]))    
                # # print('accs_iter', accs_iter)
                # torch.save(torch.stack(accs_iter), 'MCL_accs_iter.pt')
            
            if scheduler is not None:
                scheduler.step()

        if hasattr(model, 'end_task'): # 一些模型特殊操作的部分,比如EWC的fisher信息的计算部分
            model.end_task(dataset)

        # accs = evaluate(model, dataset)
        # results.append(accs[0])
        # results_mask_classes.append(accs[1])

        ## Save the model
        # torch.save(model.net.state_dict(), 'model_epoch_{}.pth'.format(t))


        ## Evaluate
        accs = evaluate(model, dataset) 
        acc_by_tasks = np.nan_to_num(accs[1], nan=0.0, posinf=0.0, neginf=0.0)
        acc_by_tasks = dict([(f'{i}',v) for i, v in enumerate(acc_by_tasks)])
        acc_by_classes = np.nan_to_num(accs[0], nan=0.0, posinf=0.0, neginf=0.0)
        acc_by_classes = dict([(f'{i}', v) for i, v in enumerate(acc_by_classes)])

        results.append(accs[0])
        results_mask_classes.append(accs[1])
        mean_acc = np.mean(accs, axis=1)
        print('\nAcc list:', accs[0])
        print('\nTask list:', accs[1])
        print('\nMean Class Acc:', np.mean(accs[0]))
        M2_1.append(np.mean(accs[0]))
        print('Average M2-1', np.mean(M2_1))

        print('\nMean Task ACC', np.mean(accs[1]))
        M2_2.append(np.mean(accs[1]))
        print('Average M2-2', np.mean(M2_2))        
        print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)  #utils.loggers # output the final results instead of the process info.

        if args.csv_log:
            csv_logger.log(mean_acc)
        if args.tensorboard:
            tb_logger.log_accuracy(np.array(accs), mean_acc, args, t)

    if args.csv_log:
        csv_logger.add_bwt(results, results_mask_classes)
        csv_logger.add_forgetting(results, results_mask_classes)
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            csv_logger.add_fwt(results, random_results_class,
                               results_mask_classes, random_results_task)

    if args.tensorboard:
        tb_logger.close()
    if args.csv_log:
        csv_logger.write(vars(args))

