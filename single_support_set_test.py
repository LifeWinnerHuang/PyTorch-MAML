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

  # n_way k_shot indexs

  # very good and bad support set examples
  # Good Example 1: very similar image, result all correct.
  support_set_idlist = [[70], [650], [1200], [1809], [2400]]
  support_set_labellist = [0, 1, 2, 3, 4]
  query_set_idlist = [[71], [652], [1203], [1812], [2401]]
  query_set_labellist = [0, 1, 2, 3, 4]
  # Bad Exmple 1: accuracy can be zero, zero example:
  # support_set_idlist = [[0], [600], [1200], [1800], [2400]]
  # support_set_labellist = [0, 1, 2, 3, 4]
  # query_set_idlist = [[409], [601], [1201], [1813], [2674]]
  # query_set_labellist = [0, 1, 2, 3, 4]

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
  aves_va = utils.AverageMeter()
  va_lst = []

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
    
  shot, query = [], []
  # construct support set and query set for each category. 
  for i, c in enumerate(support_set_idlist):
    print(i, c)
    c_shot, c_query = [], []
    shot_idx, query_idx = c, query_set_idlist[i]
    for idx in shot_idx:
      c_shot.append(transform(data[idx]))
    for idx in query_idx:
      c_query.append(transform(data[idx]))
    shot.append(torch.stack(c_shot))
    query.append(torch.stack(c_query))
  # print("shot", shot)
  # print("query", query)
  
  shot = torch.cat(shot, dim=0)             # [n_way * n_shot, C, H, W]
  query = torch.cat(query, dim=0)           # [n_way * n_query, C, H, W]
  cls = torch.arange(n_way)[:, None]
  shot_labels = cls.repeat(1, n_shot).flatten()    # [n_way * n_shot]
  query_labels = cls.repeat(1, n_query).flatten()  # [n_way * n_query]

  x_shot, x_query, y_shot, y_query = shot[None, :, :, :, :], query[None, :, :, :, :], shot_labels[None, :], query_labels[None, :]
  x_shot, y_shot = x_shot.cuda(), y_shot.cuda()
  x_query, y_query = x_query.cuda(), y_query.cuda()
  # print("data", data)
  # print(x_shot)
  # print('meta-test x_shot: {} (x{})'.format(x_shot.shape, len(x_shot) ) )
  # print('meta-test x_query: {} (x{})'.format(x_query.shape, len(x_query) ) )
  # print('meta-test y_shot: {} (x{})'.format(y_shot.shape, len(y_shot) ) )
  # print('meta-test y_query: {} (x{})'.format(y_query.shape, len(y_query) ) )

  # if inner_args['reset_classifier']:
  #     if config.get('_parallel'):
  #     model.module.reset_classifier()
  #     else:
  #     model.reset_classifier()

  logits = model(x_shot, x_query, y_shot, inner_args, meta_train=False)
  logits = logits.view(-1, config['test']['n_way'])
  labels = y_query.view(-1)

  pred = torch.argmax(logits, dim=1)
  print("predicts", pred)
  print("labels", labels)
  

  acc = utils.compute_acc(pred, labels)
  print("Single support set accuracy", acc)
  aves_va.update(acc, 1)
  # va_lst.append(acc)
  print("#####\n")

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