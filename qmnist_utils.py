import torch
import torchvision
import shutil
import numpy as np
import random
from qmnist_models import QMNISTFeatureModel, QMNISTAttentionModel
import os


def set_seed():
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    #torch.use_deterministic_algorithms(True)


def setup_qmnist(
    num_bags: list = [300], 
    num_instances_bag: list = [50],
    num_bags_valid: int = 30,
    num_bags_test: int = 60,
    percent_key_instances: int = 5, 
    nfolds: int = 1):

    set_seed()
    if not os.path.exists('benchmark'):
        os.makedirs('benchmark')
    if not os.path.exists('benchmark/QMNIST'):
        os.makedirs('benchmark/QMNIST')
    model_path = 'benchmark/QMNIST/models'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    for i in range(nfolds):
        if not os.path.exists(f'{model_path}/qmnist_feature_model_fold{i+1}.pt'):
            torch.save(QMNISTFeatureModel(without_batchnorm=False).state_dict(), f'benchmark/QMNIST/models/qmnist_feature_model_fold{i+1}.pt')
        if not os.path.exists(f'{model_path}/qmnist_attention_model_fold{i+1}.pt'):
            torch.save(QMNISTAttentionModel().state_dict(), f'benchmark/QMNIST/models/qmnist_attention_model_fold{i+1}.pt')
    data_path = 'benchmark/QMNIST/data'
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        __create_qmnist_data(data_path, nfolds, num_bags, num_instances_bag, percent_key_instances, num_bags_valid, num_bags_test)

    if not os.path.exists('benchmark/QMNIST/results'):
        os.makedirs('benchmark/QMNIST/results')



def __create_qmnist_data(data_path, number_of_folds, num_bags_trains, num_instances_in_bags, percent_key_instances, num_bags_valid, num_bags_test):


    set_seed()

    def list_files_in_folder(image_folder):
        """Lists file names in a given directory"""
        list_of_files = []
        for file in os.listdir(image_folder):
            if os.path.isfile(os.path.join(image_folder, file)):
                list_of_files.append(file)
        return list_of_files

    def create_save_dir(direct, name_subdirectory):
        if not os.path.exists(os.path.join(direct, name_subdirectory)):
            os.mkdir(os.path.join(direct, name_subdirectory))
        return os.path.join(direct, name_subdirectory)

    # Load and save data to train, validation and test sets
    root = './'
    dataloader_train = torchvision.datasets.QMNIST(root, train=True, transform=None, target_transform=None, download=True)
    dataloader_test = torchvision.datasets.QMNIST(root, train=False, transform=None, target_transform=None, download=True)

    train_dir = create_save_dir('./', 'QMNIST_train')
    if not os.listdir(train_dir):
        for i, (image,target) in enumerate(dataloader_train): 
            image.save(os.path.join(train_dir,str(target)+'_'+str(i)+'_train'+'.jpg'), "JPEG")
        
    test_dir = create_save_dir('./', 'QMNIST_test')
    if not os.listdir(test_dir):
        for i, (image,target) in enumerate(dataloader_test): 
            image.save(os.path.join(test_dir,str(target)+'_'+str(i)+'_test'+'.jpg'), "JPEG")

    # Take 1/3 from original QMNIST test set for validation set
    folder = './QMNIST_test'
    valid_dir = create_save_dir('./', 'QMNIST_valid')
    if not os.listdir(valid_dir):
        name_list = list_files_in_folder(folder)
        for digit in range(10):
            count = len([elem for elem in name_list if elem[0]==str(digit)])
            to_move = np.around(count/3)
            count_moved=0
            for i in range(len(name_list)):
                if name_list[i][0]==str(digit):
                    if count_moved>=to_move:
                        break
                    src = os.path.join(folder, name_list[i])
                    dst = os.path.join(valid_dir, name_list[i])
                    shutil.move(src,dst)
                    count_moved+=1

    # Choose key instance digit
    key_ins_digit = '9'
        
    # Create separate folder for key instances
    def key_instances_to_folder(direct):
        key_ins_dir = create_save_dir(direct, key_ins_digit)
        list_img_names = list_files_in_folder(direct)
        for i in range(len(list_img_names)):
            if list_img_names[i][0]==key_ins_digit:
                src = os.path.join(direct, list_img_names[i])
                dst = os.path.join(key_ins_dir, list_img_names[i])
                shutil.move(src,dst)

    key_instances_to_folder(train_dir)
    key_instances_to_folder(valid_dir)
    key_instances_to_folder(test_dir)

    def auxiliary_function(count_bags, i_range_min, i_range_max, max_num_bags_of_type, dir_qmnist, 
                        max_num_inst_of_type, img_names_list, save_f):
        count_instances = 0
        list_sampled=[]
        for i in range(i_range_min, i_range_max): 
            if count_instances==max_num_inst_of_type: 
                count_bags+=1
                list_sampled=[]
                if count_bags>max_num_bags_of_type: 
                    break
                count_instances = 0

            random_name = random.choice(img_names_list)
            while random_name in list_sampled: # not to repeat instances in one bag
                random_name = random.choice(img_names_list)
                if random_name not in list_sampled:
                    break
                    
            src = os.path.join(dir_qmnist, random_name) 
            bag_folder = create_save_dir(save_f, str(count_bags).zfill(4))

            dst = os.path.join(bag_folder, random_name) 
            shutil.copy(src, dst)     
            list_sampled.append(random_name) 

            count_instances+=1
            last_used_index_in_list = i
        return last_used_index_in_list, count_bags
            

    def compose_dataset(dir_qmnist, save_subfolder, num_bags, num_instances_in_bag, percent_key_instances):
        save_f = create_save_dir(save_subfolder, 'positive')
        num_key_ins_per_bag = np.ceil((num_instances_in_bag*percent_key_instances)/100)
        if num_key_ins_per_bag<1:
            print('WARNING!')

        img_names_list = list_files_in_folder(dir_qmnist)
        random.shuffle(img_names_list)

        ''' Positive bags '''
        # Negative instances in positive bags
        count_bags = 0; i_range_min=0; i_range_max=int(1e20)
        max_num_inst_of_type = num_instances_in_bag-num_key_ins_per_bag
        max_num_bags_of_type = np.ceil(num_bags/2)-1
        last_used_index_in_list, _ = auxiliary_function(count_bags, i_range_min, i_range_max, max_num_bags_of_type, 
                                                    dir_qmnist, max_num_inst_of_type, img_names_list, save_f)

        # Key instances in positive bags       
        key_instance_list = list_files_in_folder(os.path.join(dir_qmnist, key_ins_digit))       
        count_bags = 0; i_range_min=0; i_range_max=int(1e20)
        max_num_inst_of_type = num_key_ins_per_bag
        max_num_bags_of_type = np.ceil(num_bags/2)-1
        _, count_bags_pos = auxiliary_function(count_bags, i_range_min, i_range_max, max_num_bags_of_type, 
                            os.path.join(dir_qmnist, key_ins_digit), max_num_inst_of_type, key_instance_list, save_f)

        ''' Negative bags '''
        save_f = create_save_dir(save_subfolder, 'negative')
        i_range_max = int(1e20)
        count_bags = count_bags_pos
        max_num_bags_of_type = count_bags_pos+np.ceil(num_bags/2)-1
        
        _,_ = auxiliary_function(count_bags, last_used_index_in_list, i_range_max, max_num_bags_of_type, 
                                                    dir_qmnist, num_instances_in_bag, img_names_list, save_f)



    for num_bags_train in num_bags_trains:
        for num_instances_in_bag in num_instances_in_bags:
            for fold in range(1,number_of_folds+1):
                name = 'QMNIST'+'_'+str(num_bags_train).zfill(4)+'_'+str(num_instances_in_bag).zfill(4)+'_'+str(percent_key_instances).zfill(4)
                fold_dir = create_save_dir(data_path, name) 
                save_folder = create_save_dir(fold_dir,f'fold{fold}')
                if not os.listdir(save_folder):    
                    dir_qmnist = './QMNIST_train/'
                    save_subfolder = create_save_dir(save_folder, 'train')
                    compose_dataset(dir_qmnist, save_subfolder, num_bags_train, num_instances_in_bag, percent_key_instances)

                    # Validation
                    dir_qmnist = './QMNIST_valid/'
                    save_subfolder = create_save_dir(save_folder, 'valid')
                    compose_dataset(dir_qmnist, save_subfolder, num_bags_valid, num_instances_in_bag, percent_key_instances)

                    # Test set
                    dir_qmnist = './QMNIST_test/'
                    save_subfolder = create_save_dir(save_folder, 'test')
                    compose_dataset(dir_qmnist, save_subfolder, num_bags_test, num_instances_in_bag, percent_key_instances)

    if os.path.exists('QMNIST'):
        shutil.rmtree('QMNIST')
    if os.path.exists('QMNIST_train'):
        shutil.rmtree('QMNIST_train')
    if os.path.exists('QMNIST_valid'):
        shutil.rmtree('QMNIST_valid')
    if os.path.exists('QMNIST_test'):
        shutil.rmtree('QMNIST_test')
        

setup_qmnist()    