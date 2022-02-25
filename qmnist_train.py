import shutil
import torch
import torch.utils.data as data_utils
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import random

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from torch.nn import BCELoss

from miltrick.miltrick import MILModel
from qmnist_dataloaders import QmnistBags
from qmnist_utils import set_seed, setup_qmnist
from qmnist_models import QMNISTFeatureModel, QMNISTAttentionModel


# Create folder to save models while training. 
if os.path.exists('Temp_models'):
    shutil.rmtree('Temp_models')
    os.mkdir('Temp_models')
else:
    os.mkdir('Temp_models')

# Set the seed
set_seed()

# Hyperparameters
nepochs = 300                                   # Number of epochs
max_instances_forward_pass = 25                 # Max number of instances to process on GPU
number_of_bags_to_average_gradient_over = 25    # How many bags should we average gradient over   
mean_data = 0.135                               # Mean of QMNIST
std_data = 0.305                                # STD of QMNIST
key_instance_digit = '9'                        # Key instance digit in QMNIST
img_size_value = 84                             # Image width
window_length = 15                              # Window which we average training stats. over
gpu = 'cuda'                                    # GPU
lr = 5e-5                                       # Learning rate. Preferably small
num_train_bags = 300                            # Number of training bags
num_instances_bag = 50                          # Number of isntances in each bag
num_bags_valid = 30                             # Number of validation bags
num_bags_test = 60                              # Number of bags for testing
percent_key_instances = 5                       # Percent of key instances
sampling_percent = 100
sampling_size_in_instances = np.ceil((sampling_percent*num_instances_bag)/100)
image_sizeQMNIST = (1,img_size_value,img_size_value)

# Where to save trained models
base_path = f'./benchmark/QMNIST/data/QMNIST_{num_train_bag:04d}_{num_instances:04d}_{percent_key_instances:04d}/{fold}'

# Create QMNIST bags
setup_qmnist(
    num_bags=[num_train_bags],
    num_instances=[num_instances_bag],
    num_bags_valid=num_bags_valid,
    num_bags_test=num_bags_test,
    percent_key_instances=percent_key_instances
)

# Create model
feature_model = QMNISTFeatureModel()
attention_model = QMNISTAttentionModel()
feature_model.load_state_dict(torch.load((f'benchmark/QMNIST/models/qmnist_feature_model_1.pt')), strict=True)
attention_model.load_state_dict(torch.load((f'benchmark/QMNIST/models/qmnist_attention_model_1.pt')), strict=True)
model = MILModel(feature_model, attention_model, BCELoss())
model = model.to(gpu)



# Create dataloaders
train_loader = data_utils.DataLoader(QmnistBags(train=True,
                                                valid=False,
                                                test=False,
                                                image_size = image_sizeQMNIST,
                                                key_ins_digit = key_instance_digit,
                                                transform=transforms.Compose([transforms.Resize(img_size_value),
                                                                transforms.ToTensor(),
                                                                transforms.Normalize((mean_data,), (std_data,))
                                    ]), sampling_size=sampling_size_in_instances, data_path = base_path,),
                                    batch_size=1,
                                    shuffle=True)

valid_loader = data_utils.DataLoader(QmnistBags(train=False,
                                                valid=True,
                                                test=False,
                                                image_size = image_sizeQMNIST,
                                                key_ins_digit = key_instance_digit,
                                    transform=transforms.Compose([transforms.Resize(img_size_value),
                                                                transforms.ToTensor(),
                                                                transforms.Normalize((mean_data,), (std_data,))
                                    ]), sampling_size=sampling_size_in_instances, data_path = base_path), 
                                    batch_size=1,
                                    shuffle=False)



test_loader = data_utils.DataLoader(QmnistBags(train=False,
                                            valid=False,
                                            test=True,
                                            image_size = image_sizeQMNIST,
                                            key_ins_digit = key_instance_digit,
                                    transform=transforms.Compose([transforms.Resize(img_size_value),
                                                                transforms.ToTensor(),
                                                                transforms.Normalize((mean_data,), (std_data,))
                                    ]), sampling_size=sampling_size_in_instances, data_path = base_path),
                                    batch_size=1,
                                    shuffle=False)



# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)


train_loss = 0
train_error = 0
valid_loss = 0
valid_error = 0
best_avg_val_error = np.inf

for epoch in range(nepochs):
    train_loss = 0
    train_error = 0
    valid_loss = 0
    valid_error = 0

    model.train()
    for batch_idx, (bag, target, sample_indices, index, bag_names) in enumerate(train_loader):
        target = target[0]; bag = bag[0]
        if cuda:
            target, bag = target.to(gpu), bag.to(gpu)
        bag, target = Variable(bag), Variable(target)
        
        pred, attention, loss = model.forward_train(
            bag, 
            target, 
            max_instances_per_forward_pass=max_instances_per_forward_pass, 
            scale_loss=1.0/number_of_bags_to_average_gradient_over
        )
       
        train_error += 1. - pred.ge(0.5).eq(target).cpu().float().item()        
        train_loss += loss

        if batch_idx % number_of_bags_to_average_gradient_over == 0:
            optimizer.step()
            optimizer.zero_grad()


    train_loss /= len(train_loader) / number_of_bags_to_average_gradient_over
    train_error /= len(train_loader)

    # Validation
    model.eval()
    with torch.no_grad():
        for batch_idx, (bag, target, sample_indices, index, bag_names) in enumerate(valid_loader):
            target = target[0]; bag = bag[0]
            if cuda:
                target, bag = target.to(gpu), bag.to(gpu)
            bag, target = Variable(bag), Variable(target)
            pred, attention, loss = model.forward_test(bag, target, max_instances_per_forward_pass=max_instances_per_forward_pass)
            valid_loss += loss
            valid_error += 1. - pred.ge(0.5).eq(target).cpu().float().item()        
        valid_loss /= len(valid_loader)
        valid_error /= len(valid_loader)
        print(f'Epoch: {epoch}, Train loss: {train_loss:02f}, Train error: {train_error:02f},  Valid loss: {valid_loss:02f}, Valid error {valid_error:02f}')





