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
ROOT_DIR = '/c/Users/Lucy/Downloads/Test_Env'
DATA_DIR = '/c/Users/Lucy/Downloads/Test_Env/Faces'
DATA_DIR_FWS = '/c/Users/Lucy/Downloads/Test_Env/Faces/'
KDEF_DIR = '/c/Users/Lucy/Downloads/Test_Env/KDEF/'
ALEXNET_DIR = '/c/Users/Lucy/Downloads/Test_Env/Faces/alexnet'

def generate_main_info():
    # crop all images to 224 x 224 for all datasets
    # generate image folders + data loaders for train, val, test
    data_transform = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor()])

    dir_paths = {
        'train': os.path.join(DATA_DIR_FWS, 'train/'),
        'val': os.path.join(DATA_DIR_FWS, 'val/'),
        'test': os.path.join(DATA_DIR_FWS, 'test/')
    }

    image_datasets = {
        'train': datasets.ImageFolder(
            dir_paths['train'], 
            transform=data_transform
        ),
        'val': datasets.ImageFolder(
            dir_paths['val'], 
            transform=data_transform
        ),
        'test': datasets.ImageFolder(
            dir_paths['test'], 
            transform=data_transform
        )
    }

    data_loaders = {
        'train': torch.utils.data.DataLoader(
            image_datasets['train'], batch_size=1
        ),
        'val': torch.utils.data.DataLoader(
            image_datasets['val'], batch_size=1
        ),
        'test': torch.utils.data.DataLoader(
            image_datasets['test'], batch_size=1
        )
    }

    # get size of each dataset
    dataset_sizes = {
        'train': len(image_datasets['train']),
        'val': len(image_datasets['val']),
        'test': len(image_datasets['test']) 
    }

    return data_loaders


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


""" DATA PROCESSING FUNCTIONS """
def create_useful_dataset():
    # Call this function once only!
    """
    Logic for sorting thru dataseta for desired images:
    KDEF:
    - Example file name: AF01ANS.JPG
    - Check:
        - length of name = 7, for straight profile only, ends with "S.jpg"
        - str[4:5] = {"AF":"afraid", "AN":"angry", "DI":"disgusted", "HA":"happy",
        "NE":"neutral", "SA":sad", "SU":"surprised"}
    """
    # go thru KDEF data + sort out desired photos
    for subdir, dirs, files in os.walk(KDEF_DIR):
        for file in files:
            filename = subdir + os.sep + file
            if (file.endswith("S.jpg") or file.endswith("S.JPG")): 
                """
                For each straight profile photo:
                    - convert RGB --> Grayscale
                    - make 4 copies of photo: original orientation, rotate 5 degrees
                        clockwise (cw), rotate counter-clockwise (ccw), flip horizontally
                    - resize all to 256 x 256 pixels, b/c will center crop to 224 x 224 later
                    - then save in the corresponding emotion class folder
                """
                img = Image.open(filename).convert('L')
                img_cw = img.rotate(350)
                img_ccw = img.rotate(10)
                img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)

                new_img = img.resize((256, 256))
                new_img_cw = img_cw.resize((256, 256))
                new_img_ccw = img_ccw.resize((256, 256))
                new_img_flip = img_flip.resize((256, 256))

                label = file[4:6]
                new_img.save(DATA_DIR+'/'+EMOTION_CODE[label]+'/'+file)
                new_img_cw.save(DATA_DIR+'/'+EMOTION_CODE[label]+'/'+'1'+file)
                new_img_ccw.save(DATA_DIR+'/'+EMOTION_CODE[label]+'/'+'2'+file)
                new_img_flip.save(DATA_DIR+'/'+EMOTION_CODE[label]+'/'+'3'+file)
    
    print("Finished creating useful dataset!")

def split_data_to_subsets():   
    """
    Split data into train, val, test datasets (60:20:20)
    each class = ~568 images --> ~340 train, ~114 val, ~114 test
    """
    # divide data into train, val, + test
    # for each emotion class, get filenames, shuffle, 
    # divide, move to corresponding folders
    for cla in CLASSES:
        filepath = DATA_DIR+'/'+cla
        names = []

        for file in os.listdir(filepath):
            names.append(file)

        random.shuffle(names)
        num_files = len(names)

        for ind, name in enumerate(names):
            if(ind <= math.ceil(0.6 * num_files)):
                # Move to train
                shutil.move(filepath+'/'+name, DATA_DIR+'/train/'+cla+'/'+name)
            elif(ind <= math.ceil(0.8 * num_files)):
                # Move to val
                shutil.move(filepath+'/'+name, DATA_DIR+'/val/'+cla+'/'+name)
            else:
                # Move to test
                shutil.move(filepath+'/'+name, DATA_DIR+'/test/'+cla+'/'+name)
    
    print("Finished splitting data to training, val, and test subsets")

def save_tensor_helper(dir_name, features, label, img_num):
    # save tensor to appropriate emotion folder
    path = DATA_DIR + '/'+dir_name

    if (label.item() == 0):
        torch.save(features, path + '/afraid/features_' + str(img_num) + '.tensor')
    if (label.item() == 1):
        torch.save(features, path + '/angry/features_' + str(img_num) + '.tensor')
    if (label.item() == 2):
        torch.save(features, path + '/disgusted/features_' + str(img_num) + '.tensor')
    if (label.item() == 3):
        torch.save(features, path + '/happy/features_' + str(img_num) + '.tensor')
    if (label.item() == 4):
        torch.save(features, path + '/neutral/features_' + str(img_num) + '.tensor')
    if (label.item() == 5):
        torch.save(features, path + '/sad/features_' + str(img_num) + '.tensor')
    if (label.item() == 6):
        torch.save(features, path + '/surprised/features_' + str(img_num) + '.tensor')

def save_tensors(model, data_loaders, model_name='alexnet'):
    # save tensors to train, val, test folders
    i = 0
    for img, label in data_loaders['train']:
        features = model.features(img)
        save_tensor_helper(model_name+'/train', features, label, i)
        i+=1

    i = 0
    for img, label in data_loaders['val']:
        features = model.features(img)
        save_tensor_helper(model_name+'/val', features, label, i)
        i+=1

    i = 0
    for img, label in data_loaders['test']:
        features = model.features(img)
        save_tensor_helper(model_name+'/test', features, label, i)
        i+=1


""" TRAINING FUNCTIONS """
def load_feature(loc): 
    return torch.load(loc)

def get_features_data_loader(data_dir, batch_size): # Data Loading
    # define training and test data directories
    train_dir = os.path.join(data_dir, '/train/')
    val_dir = os.path.join(data_dir, '/val/')
    test_dir = os.path.join(data_dir, '/test/')

    train_data = datasets.DatasetFolder(train_dir, loader=load_feature, extensions = '.tensor')
    val_data = datasets.DatasetFolder(val_dir, loader=load_feature, extensions = '.tensor')
    test_data = datasets.DatasetFolder(test_dir, loader=load_feature, extensions = '.tensor')

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

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
  
def get_accuracy(model, loader):
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs = torch.from_numpy(imgs.detach().numpy())
        output = model(imgs)
        prob = F.softmax(output)
        
        #select index with maximum prediction score
        pred = prob.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.shape[0]
        
    return correct / total

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

def plot_training_curve(path):
    """ Plots the training curve for a model run, given the csv files
    containing the train/validation accuracy/loss.
    Args:
        path: The base path of the csv files produced during training
    """
    train_acc = np.loadtxt("{}_train_acc.csv".format(path))
    val_acc = np.loadtxt("{}_val_acc.csv".format(path))
    train_loss = np.loadtxt("{}_train_loss.csv".format(path))
    val_loss = np.loadtxt("{}_val_loss.csv".format(path))
    
    n = len(train_acc) # number of epochs
    
    plt.title("Train vs Validation Loss")
    plt.plot(range(1,n+1), train_loss, label="Train")
    plt.plot(range(1,n+1), val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()
    
    plt.title("Train vs Validation Accuracy")
    plt.plot(range(1,n+1), train_acc, label="Train")
    plt.plot(range(1,n+1), val_acc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()
    
    print("Final Training Accuracy: {}".format(train_acc[-1]))
    print("Final Validation Accuracy: {}".format(val_acc[-1]))

    
def train_net(net, batch_size=64, learning_rate=0.01, num_epochs=30,
    data_dir=DATA_DIR):

    # Fixed PyTorch random seed for reproducible result
    torch.manual_seed(1000)
    
    # Obtain the PyTorch data loader objects to load batches of the datasets
    train_loader, val_loader, test_loader = get_features_data_loader(data_dir, batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    # Set up some numpy arrays to store the training/test loss/erruracy
    train_acc = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    val_acc = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)
    
    # Train the network
    # Loop over the data iterator and sample a new batch of training data
    # Get the output from the network, and optimize our loss function.
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        total_train_loss = 0.0
        total_train_acc = 0.0
        total_epoch = 0

        for i, data in enumerate(train_loader, 0):
            # Get the inputs
            imgs, labels = data
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass, backward pass, and optimize
            imgs = torch.from_numpy(imgs.detach().numpy())
              
            out = model(imgs) # forward pass
            prob = F.softmax(out)
            loss = criterion(prob, labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss
            total_epoch += len(labels)

        train_acc[epoch] = get_accuracy(net, train_loader)
        train_loss[epoch] = float(total_train_loss) / (i+1)
        val_acc[epoch], val_loss[epoch] = evaluate(net, val_loader, criterion)
        
        print(("Epoch {}: Train acc: {}, Train loss: {} |"+
               "Validation acc: {}, Validation loss: {}").format(
                   epoch + 1,
                   train_acc[epoch],
                   train_loss[epoch],
                   val_acc[epoch],
                   val_loss[epoch]))
        
        # Save the current model (checkpoint) to a file
        model_path = get_model_name(net.name, batch_size, learning_rate, epoch)
        torch.save(net.state_dict(), model_path)
    
    print('Finished Training')
    
    # Write the train/test loss/accuracy into CSV file for plotting later
    epochs = np.arange(1, num_epochs + 1)
    np.savetxt("{}_train_acc.csv".format(model_path), train_acc)
    np.savetxt("{}_train_loss.csv".format(model_path), train_loss)
    np.savetxt("{}_val_acc.csv".format(model_path), val_acc)
    np.savetxt("{}_val_loss.csv".format(model_path), val_loss)


""" MODEL TRAINING + TESTING """
if __name__ == "__main__":
    # set up AlexNet model
    alexnet = torchvision.models.alexnet(pretrained=True)
    torch.manual_seed(1)

    # data processing
    create_useful_dataset()
    split_data_to_subsets()
    data_loaders = generate_main_info()
    save_tensors(model, data_loaders)

    # transfer learning + model training
    model = ANNClassifier_Alexnet()
    train_net(model, batch_size=128, learning_rate=0.001, num_epochs=150,
        data_dir=ALEXNET_DIR)
    model_path = get_model_name(model.name, batch_size=128, learning_rate=0.001, epoch=149)
    plot_training_curve(ROOT_DIR + model_path)

    model = ANNClassifier_Alexnet()
    state = torch.load(ROOT_DIR + model_path)
    model.load_state_dict(state)

    train_loader, val_loader, test_loader = get_features_data_loader(data_dir=ALEXNET_DIR, batch_size=128)

    criterion = nn.CrossEntropyLoss()
    test_acc, test_loss = evaluate(model, test_loader, criterion)
    print("Test classification accuracy:", test_acc)