#!/usr/bin/env python3
# ALL import statements
import os
import shutil
import copy
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
from torch.autograd import Variable


# GLOBAL VARIABLES
# classes are folders in each directory with these names
CLASSES = ['afraid','angry','disgusted','happy','neutral','sad','surprised']

# emotion label for KDEF photos
EMOTION_CODE = {"AF":"afraid", "AN":"angry", "DI":"disgusted", "HA":"happy", 
                "NE":"neutral", "SA":"sad", "SU":"surprised"}

# main directory filepaths
# normpath will replace fwd slash w back
ROOT_DIR = os.path.normpath('C:/Users/Lucy/Downloads/Test_Env/')
DATA_DIR = os.path.normpath('C:/Users/Lucy/Downloads/Test_Env/Faces')
DATA_DIR_FWS = os.path.normpath('C:/Users/Lucy/Downloads/Test_Env/Faces/')
KDEF_DIR = os.path.normpath('C:/Users/Lucy/Downloads/Test_Env/KDEF/')
ALEXNET_DIR = os.path.normpath('C:/Users/Lucy/Downloads/Test_Env/Faces/alexnet')


# Artifical Neural Network Architecture
class ANNClassifier_Alexnet(nn.Module):
    def __init__(self):
        super(ANNClassifier_Alexnet, self).__init__()
        self.name = "alexnet_ann"
        self.fc1 = nn.Linear(256 * 6 * 6, 300)
        self.fc2 = nn.Linear(300, 7)

    def forward(self, x):
        x = x.view(-1, 256 * 6 * 6) #flatten feature data
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


""" TRAINING FUNCTIONS """
def load_feature(loc): 
    return torch.load(loc)

def get_features_data_loader(batch_size): # Data Loading
    # define training and test data directories
    test_dir = os.path.normpath('C:/Users/Lucy/Downloads/Test_Env/test')
    test_data = datasets.DatasetFolder(test_dir, loader=load_feature, extensions = '.tensor')
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
    
    return test_loader

def get_model_name(name, batch_size, learning_rate, epoch):
    """ Generate a name for the model consisting of all the hyperparameter values
    Args:
        config: Configuration object containing the hyperparameters
    Returns:
        path: A string with the hyperparameter name and value concatenated
    """
    path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(name,
                batch_size, learning_rate, epoch)
    return path
  

def evaluate(net, loader, criterion):
    """ Evaluate the network on the validation set.
     Args:
         net: PyTorch neural network object
         loader: PyTorch data loader for the validation set
         criterion: The loss function
     Returns:
         acc: A scalar for the avg classification acc over the validation set
         loss: A scalar for the average loss function over the validation set
     """
    total_loss = 0.0
    total_epoch = 0
    correct = 0
    total = 0

    for i, data in enumerate(loader, 0):
        imgs, labels = data

        imgs = torch.from_numpy(imgs.detach().numpy())              
        out = model(imgs) # forward pass
        prob = F.softmax(out)
        loss = criterion(prob, labels)
        
        #select index with maximum prediction score
        pred = prob.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.shape[0]
        
        total_loss += loss
        total_epoch += len(labels)
    
    acc = correct / total
    loss = float(total_loss) / (i + 1)
    
    return acc, loss

def test(model, filename):
    img = Image.open(filename).convert('L')
    new_img = img.resize((256, 256))
    new_img = new_img.crop((16, 16, 240, 240))
    trans = transforms.ToTensor()
    imgs = trans(new_img)    
    out = model(imgs)
    print(out)


""" MODEL TRAINING + TESTING """
if __name__ == "__main__":
    model = ANNClassifier_Alexnet()
    model_path = get_model_name(model.name, batch_size=128, learning_rate=0.001, epoch=149)
    
    state = torch.load(os.path.normpath('C:/Users/Lucy/Downloads/Test_Env/model_alexnet_ann_bs128_lr0_001_epoch149'))
    model.load_state_dict(state)

    test(model, os.path.normpath('C:/Users/Lucy/Downloads/Test_Env/KDEF/AF01/AF01AFS.JPG'))
    # test_loader = get_features_data_loader(batch_size=128)

    # criterion = nn.CrossEntropyLoss()
    # test_acc, test_loss = evaluate(model, test_loader, criterion)
    # print("Test classification accuracy:", test_acc)