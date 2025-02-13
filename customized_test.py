import argparse
import random

import yaml
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

import datasets
import models
import utils


def main(config):
  random.seed(0)
  np.random.seed(0)
  torch.manual_seed(0)
  torch.cuda.manual_seed(0)
  # torch.backends.cudnn.deterministic = True
  # torch.backends.cudnn.benchmark = False

  ##### Dataset #####

  dataset = datasets.make(config['dataset'], **config['test'])
  utils.log('meta-test set: {} (x{}), {}'.format(
    dataset[0][0].shape, len(dataset), dataset.n_classes))
  print("dataset object:", dataset)
  print(dataset.__dict__.keys())
  # keys: ['root_path', 'split_tag', 'image_size', 'data', 'label', 
  # 'n_classes', 'norm_params', 'transform', 'convert_raw', 'n_batch', 
  # 'n_episode', 'n_way', 'n_shot', 'n_query', 'catlocs', 'val_transform'])

  print('root_path', dataset.root_path)
  print('split_tag', dataset.split_tag)
  print('label', dataset.label)

  loader = DataLoader(dataset, config['test']['n_episode'],
    collate_fn=datasets.collate_fn, num_workers=1, pin_memory=True)

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

  print(loader)
  
  # for data in tqdm(loader, leave=False):
  for data in loader:
    x_shot, x_query, y_shot, y_query = data
    x_shot, y_shot = x_shot.cuda(), y_shot.cuda()
    x_query, y_query = x_query.cuda(), y_query.cuda()
    # print("data", data)
    # print('meta-test x_shot: {} (x{})'.format(x_shot.shape, len(x_shot) ) )
    # print('meta-test x_query: {} (x{})'.format(x_query.shape, len(x_query) ) )
    # print('meta-test y_shot: {} (x{})'.format(y_shot.shape, len(y_shot) ) )
    # print('meta-test y_query: {} (x{})'.format(y_query.shape, len(y_query) ) )
    

    if inner_args['reset_classifier']:
      if config.get('_parallel'):
        model.module.reset_classifier()
      else:
        model.reset_classifier()

    logits = model(x_shot, x_query, y_shot, inner_args, meta_train=False)
    logits = logits.view(-1, config['test']['n_way'])
    labels = y_query.view(-1)
    
    pred = torch.argmax(logits, dim=1)
    acc = utils.compute_acc(pred, labels)
    # print("accuracy", acc)
    aves_va.update(acc, 1)
    va_lst.append(acc)
    # print("#####\n")

  print("Accuracy value list", va_lst)
  print('Meta test: acc={:.2f} +- {:.2f} (%)'.format(aves_va.item() * 100, 
    utils.mean_confidence_interval(va_lst) * 100))


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