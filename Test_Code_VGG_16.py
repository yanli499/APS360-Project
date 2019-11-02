import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.models
import matplotlib.pyplot as plt

# import os
# import time
# import torch.nn.functional as F
# import torch.optim as optim
# import torchvision.transforms as transforms
# from sklearn.model_selection import train_test_split
# from torchvision import datasets, transforms
# from torch.utils.data.sampler import SubsetRandomSampler

vgg16 = torchvision.models.vgg16(pretrained=True)
vgg16.features
vgg16.classifier