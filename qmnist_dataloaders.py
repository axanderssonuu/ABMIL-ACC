"""Pytorch dataset objects that loads QMNIST-bags and Imagenette-bags datasets for experiments with/without within-bag sampling"""

import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from pathlib import Path
import os
from PIL import Image

def list_files_in_folder(image_folder):
    """Lists file names in a given directory"""
    list_of_files = []
    for file in os.listdir(image_folder):
        if os.path.isfile(os.path.join(image_folder, file)):
            list_of_files.append(file)
    return list_of_files

class QmnistBags(data_utils.Dataset):
    def __init__(self, transform, sampling_size, train, valid, test, data_path, image_size, key_ins_digit):
        self.train = train; self.valid = valid; self.test = test
        self.transform = transform
        self.sampling_size = sampling_size
        self.image_size = image_size
        self.key_ins_digit = key_ins_digit
        if self.train==True:
            self.datapath = os.path.join(data_path, 'train')
            self.list_paths = self._list_bags_paths()
            self.train_bags_list, self.train_labels_list, self.train_imgs_lists, self.bags_names_train = self._create_bags()
            np.save(os.path.join(data_path, 'train_imgs_lists.npy'), np.asarray(self.train_imgs_lists))
        elif self.valid==True:
            self.datapath = os.path.join(data_path, 'valid')
            self.list_paths = self._list_bags_paths()
            self.valid_bags_list, self.valid_labels_list, self.valid_imgs_lists, self.bags_names_valid = self._create_bags()
            np.save(os.path.join(data_path, 'valid_imgs_lists.npy'), np.asarray(self.valid_imgs_lists))            
        elif self.test==True:
            self.datapath = os.path.join(data_path, 'test')
            self.list_paths = self._list_bags_paths()
            self.test_bags_list, self.test_labels_list, self.test_imgs_lists, self.bags_names_test = self._create_bags()
            np.save(os.path.join(data_path,'test_imgs_lists.npy'), np.asarray(self.test_imgs_lists))

    def _list_bags_paths(self):
        list_paths = []
        basePath = Path(self.datapath)
        for child in basePath.iterdir():
            if child.is_dir():
                for grandchild in child.iterdir():
                    if grandchild.is_dir():
                        list_paths.append(grandchild)
        return list_paths
    
    def _create_bags(self):
        all_bags = [] 
        all_bags_labels = [] 
        lists_imgs_all = []
        bags_names_all=[]
        for p in range(len(self.list_paths)):
            print(str(self.list_paths[p]))
            list_img_names = list_files_in_folder(str(self.list_paths[p]))
            basePath = Path(self.list_paths[p])
            bags_names_all.append(np.asarray(int(basePath.parts[-1])))
            list_img_names_bag = ()
            
            for i in range(len(list_img_names)):
                temp = (list_img_names[i],)
                list_img_names_bag = list_img_names_bag+temp
            lists_imgs_all.append(np.asarray(list_img_names_bag))
            images = [self.transform(Image.open(os.path.join(str(self.list_paths[p]), list_img_names_bag[i]))).float() for i in range(len(list_img_names_bag))]
            images = torch.stack(images)
            X = images                
#             X = torch.empty(1, self.image_size[0], self.image_size[1], self.image_size[2]) 
            y = torch.empty(1)
            for i in range(len(list_img_names)):
                if list_img_names[i][0]==self.key_ins_digit:
                    y = torch.cat((y,torch.ones(1)),0)
                else:
                    y = torch.cat((y,torch.zeros(1)),0)

            all_bags.append(X) #
#             all_bags.append(X[1:,:,:,:])
            all_bags_labels.append(y[1:])
        return all_bags, all_bags_labels, lists_imgs_all, bags_names_all 
    
    def __len__(self):
        return len(self.list_paths)

    def __getitem__(self, index):
        # Introduce sampling, with replacement for train, test and valid
        max_number_of_imgs_in_bag = int(self.sampling_size)
        if self.train:
            bag = self.train_bags_list[index]
            label = [max(self.train_labels_list[index]), self.train_labels_list[index]]
            lists_of_names = self.train_imgs_lists[index]
            bages_names = self.bags_names_train[index]
        elif self.valid:
            bag = self.valid_bags_list[index]
            label = [max(self.valid_labels_list[index]), self.valid_labels_list[index]]
            lists_of_names = self.valid_imgs_lists[index]
            bages_names = self.bags_names_valid[index]            
        else:
            bag = self.test_bags_list[index]
            label = [max(self.test_labels_list[index]), self.test_labels_list[index]]
            lists_of_names = self.test_imgs_lists[index]
            bages_names = self.bags_names_test[index]

        if bag.shape[0] > max_number_of_imgs_in_bag:  
            sample_indices = torch.randint(bag.shape[0], (max_number_of_imgs_in_bag,))
            bag_sampled = bag[sample_indices,:,:,:]
            label_sampled = [label[0],label[1][sample_indices]]
            lists_of_names_sampled = [lists_of_names[x] for x in sample_indices]
            bages_names_sampled = bages_names      
        else:
            sample_indices = np.arange(bag.shape[0])
            bag_sampled = bag
            label_sampled = label
            lists_of_names_sampled = lists_of_names
            bages_names_sampled = bages_names     
            
        return bag_sampled, label_sampled, torch.as_tensor(sample_indices), torch.as_tensor(index), torch.as_tensor(bages_names_sampled)

