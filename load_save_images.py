import os
import pickle

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

from datasets.datasets import register
from datasets.transforms import get_transform


def main():
    split = "meta-test"
    root_path = "./materials/mini-imagenet/"
    split_dict = {'train': 'train_phase_train',        # standard train
                    'val': 'train_phase_val',            # standard val
                    'trainval': 'train_phase_trainval',  # standard train and val
                    'test': 'train_phase_test',          # standard test
                    'meta-train': 'train_phase_train',   # meta-train
                    'meta-val': 'val',                   # meta-val
                    'meta-test': 'test',                 # meta-test
                    }
    split_tag = split_dict[split]

    split_file = os.path.join(root_path, 'miniImageNet_category_split_' + split_tag + '.pickle')
    assert os.path.isfile(split_file)
    with open(split_file, 'rb') as f:
        pack = pickle.load(f, encoding='latin1')
    data, label = pack['data'], pack['labels']

    data = [Image.fromarray(x) for x in data]
    label = np.array(label)
    label_key = sorted(np.unique(label))
    label_map = dict(zip(label_key, range(len(label_key))))
    new_label = np.array([label_map[x] for x in label])
    label = new_label
    
    print("label_key", label_key)
    for image, image_label in zip(data, label):
        dir_path = "./materials/mini_imagenet_raw/test/" + str(image_label) + "/" 
        if os.path.exists(dir_path) is False:
            os.mkdir(dir_path)
        file_path = dir_path + str(data.index(image))  + ".png"
        image.save(file_path)
        print("Image {} saved.".format(file_path))
    



if __name__ == '__main__':
    main()







