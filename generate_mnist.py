import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from aijack.attack import GradientInversion_Attack

from dataset.utils.dataset_utils import check, separate_data, split_data, save_file
import torch.nn as nn

from system.flcore.trainmodel.models import FedAvgCNN

import math
from PIL import Image
from sklearn.preprocessing import minmax_scale


# Allocate data to users
def generate_mnist(dir_path, num_clients, num_classes, niid, balance, partition, percent_of_mal_nodes, percent_of_mal_data):

    mal_nodes = math.floor(percent_of_mal_nodes * num_clients /100.0)

    
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, num_classes, percent_of_mal_nodes, percent_of_mal_data, niid, balance, partition):
        return

    # FIX HTTP Error 403: Forbidden
    from six.moves import urllib
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    # Get MNIST data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    trainset = torchvision.datasets.MNIST(
        root=dir_path+"rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(
        root=dir_path+"rawdata", train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    # dataset = []
    # for i in range(num_classes):
    #     idx = dataset_label == i
    #     dataset.append(dataset_image[idx])

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, 
                                    niid, balance, partition)
    train_data, test_data = split_data(X, y)


    modify_data(train_data, test_data, num_clients, mal_nodes, percet_of_malicious_data)

    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, percent_of_mal_nodes, percent_of_mal_data,
        num_classes, statistic, niid, balance, partition)


def modify_data(train_data, test_data, num_clients, mal_nodes, percet_of_malicious_data):
    if mal_nodes == 0 or percet_of_malicious_data == 0:
        return
    arr = []

    malicious_nodes = random.sample(range(0, num_clients), mal_nodes)
    net = FedAvgCNN(in_features=1, num_classes=10, dim=1024).to('cpu')

    criterion = nn.CrossEntropyLoss()


    dlg_attacker = GradientInversion_Attack(
        net, (1, 28, 28), lr=1.0, log_interval=0, num_iteration=100, distancename="l2", early_stopping=25
        )

    for i, data_dict in enumerate(train_data):
        x = data_dict['x']
        y = data_dict['y']
        if i not in malicious_nodes:
            arr.append(data_dict)
            continue

        num_mal_samples = math.floor(len(x)* percet_of_malicious_data/100.0)
        malicious_indices = random.sample(range(0, num_mal_samples), num_mal_samples)
        
        # print("++++++++++++++++++++++++++", x.shape, x[0])


        # print("++++++++++++++++++++++++++", x_mal.shape, x_mal[0])

        for i in malicious_indices:
            # print("----------------", len(malicious_indices))

            x_mal = torch.Tensor(x[i:i+1])
            y_mal = torch.Tensor(y[i:i+1]).type(torch.long)
            

            pred = net(x_mal)
            loss = criterion(pred, y_mal)
            received_gradients = torch.autograd.grad(loss, net.parameters())
            received_gradients = [cg.detach() for cg in received_gradients]

            # print(x[i])
            x[i] = minmax_scale(dlg_attacker.attack(received_gradients)[0].detach().numpy().squeeze(), feature_range=(-1,1))
            # print(x[i])
            # print(x[i].shape)














if __name__ == "__main__":


    random.seed(1)
    np.random.seed(1)



    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None
    num_clients = int(sys.argv[4]) if sys.argv[4] is not None else 20
    percet_of_malicious_nodes = int(sys.argv[5]) if sys.argv[5] is not None else 0
    percet_of_malicious_data = float(sys.argv[6]) if sys.argv[6] is not None else 0
    num_classes = 10


    dir_path = "dataset/mnist/"
    generate_mnist(dir_path, num_clients, num_classes, niid, balance, partition, percet_of_malicious_nodes, percet_of_malicious_data)