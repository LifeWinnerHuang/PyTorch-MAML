import os
import pickle

import argparse
import random

import yaml
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image

import datasets
import models
import utils

from datasets.datasets import register
from datasets.transforms import get_transform


def main(config):
  random.seed(0)
  np.random.seed(0)
  torch.manual_seed(0)
  torch.cuda.manual_seed(0)
  # torch.backends.cudnn.deterministic = True
  # torch.backends.cudnn.benchmark = False

  ##### Dataset #####

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
  n_classes = len(label_key)

  # n_way k_shot and query numbers
  n_way = config['test']['n_way']
  n_shot = config['test']['n_shot']
  n_query = config['test']['n_query']

  ##### Model #####

  ckpt = torch.load(config['load'])
  inner_args = utils.config_inner_args(config.get('inner_args'))
  model = models.load(ckpt, load_clf=(not inner_args['reset_classifier']))

  if args.efficient:
    model.go_efficient()

  if config.get('_parallel'):
    model = nn.DataParallel(model)

  utils.log('num params: {}'.format(utils.compute_n_params(model)))

  ##### Evaluation #####
  model.eval()

  normalization = config['test']['normalization']
  image_size = 84
  transform = None
  if normalization:
      norm_params = {'mean': [0.471, 0.450, 0.403],
                          'std':  [0.278, 0.268, 0.284]}
  else:
    norm_params = {'mean': [0., 0., 0.],
                        'std':  [1., 1., 1.]}
  transform = get_transform(transform, image_size, norm_params)

  catlocs = tuple()
  for cat in range(n_classes):
    catlocs += (np.argwhere(label == cat).reshape(-1),)



  acc_per_class = {} # accuracy per calss 
  # Random time:
  total_tasks = config['test']['n_batch']
  # total_tasks = 200
  for i_sample in range(0, total_tasks):
    print("Running sample {} .".format(i_sample))
    
    shot_ids = {}
    query_ids = {}
    shot, query = [], []
    # Random choose n way classes of all test classes
    cats = np.random.choice(n_classes, n_way, replace=False)
    # Compute specific categories accuracies
    # cats = [5, 10, 12, 18, 19] # All best
    # cats = [3, 6, 11, 14, 15] # All worst
    # cats = [5, 10, 12, 14, 15] # mixture
    # construct support set and query set for each category. 
    for c in cats:
      c_shot, c_query = [], []
      idx_list = np.random.choice(
        catlocs[c], n_shot + n_query, replace=False)
      shot_idx, query_idx = idx_list[:n_shot], idx_list[-n_query:]
      shot_ids[c] = shot_idx
      query_ids[c] = query_idx
      for idx in shot_idx:
        c_shot.append(transform(data[idx]))
      for idx in query_idx:
        c_query.append(transform(data[idx]))
      shot.append(torch.stack(c_shot))
      query.append(torch.stack(c_query))
    
    shot = torch.cat(shot, dim=0)             # [n_way * n_shot, C, H, W]
    query = torch.cat(query, dim=0)           # [n_way * n_query, C, H, W]
    cls = torch.arange(n_way)[:, None]
    shot_labels = cls.repeat(1, n_shot).flatten()    # [n_way * n_shot]
    query_labels = cls.repeat(1, n_query).flatten()  # [n_way * n_query]

    x_shot, x_query, y_shot, y_query = shot[None, :, :, :, :], query[None, :, :, :, :], shot_labels[None, :], query_labels[None, :]
    x_shot, y_shot = x_shot.cuda(), y_shot.cuda()
    x_query, y_query = x_query.cuda(), y_query.cuda()

    logits = model(x_shot, x_query, y_shot, inner_args, meta_train=False)
    logits = logits.view(-1, config['test']['n_way'])
    labels = y_query.view(-1)           # [n_way * n_query]
    pred = torch.argmax(logits, dim=1)  # [n_way * n_query]
    # print("predicts", pred)
    # print("labels", labels)

    # Calculate per class accuracy
    for i, c in enumerate(cats):
      result_c = (labels[i * n_query : (i + 1) * n_query] == pred[i * n_query : (i + 1) * n_query]).float().mean().item()
      if c in acc_per_class:
        acc_per_class[c][1] = acc_per_class[c][1] + result_c
        acc_per_class[c][2] += 1
        acc_per_class[c][0] = acc_per_class[c][1] / acc_per_class[c][2]
      else:
        acc_per_class[c] = [result_c, result_c, 1] # acc, total, count
    
  
  for key, value in acc_per_class.items():
    print("Class {}: average accuracy {}".format(key, value[0]))

  print(acc_per_class)


  # print("Accuracy value list", va_lst)
  # print('Meta test: acc={:.2f} +- {:.2f} (%)'.format(aves_va.item() * 100, 
  #   utils.mean_confidence_interval(va_lst) * 100))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', 
                      help='configuration file')
  parser.add_argument('--gpu', 
                      help='gpu device number', 
                      type=str, default='0')
  parser.add_argument('--efficient', 
                      help='if True, enables gradient checkpointing',
                      action='store_true')
  args = parser.parse_args()
  config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
  
  if len(args.gpu.split(',')) > 1:
    config['_parallel'] = True
    config['_gpu'] = args.gpu

  utils.set_gpu(args.gpu)
  main(config)


