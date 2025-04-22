from __future__ import print_function

import argparse
import os

# internal imports
from utils.file_utils import save_pkl
from utils.utils import *
from utils.core_utils import train
from datasets.dataset_generic import Generic_MIL_Dataset

# pytorch imports
import torch
import pandas as pd
import numpy as np


def main(args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    all_test_loss = []
    all_val_loss = []
    folds = np.arange(start, end)
    print('folds:',folds)
    for i in folds:
        seed_torch(args.seed+i)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False,
                csv_path='{}/splits_{}.csv'.format(args.split_dir, i))

        datasets = (train_dataset, val_dataset, test_dataset)
        results, test_auc, val_auc, test_acc, val_acc, test_loss, val_loss  = train(datasets, i, args)
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        all_test_loss.append(test_loss)
        all_val_loss.append(val_loss)
        
        #write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

    "Compute the average and std of the metrics"
    test_auc_ave= np.mean(all_test_auc)
    test_acc_ave= np.mean(all_test_acc)
    test_loss_ave= np.mean(all_test_loss)

    test_auc_std= np.std(all_test_auc)
    test_acc_std= np.std(all_test_acc)
    test_loss_std= np.std(all_test_loss)
    
    val_auc_ave= np.mean(all_val_auc)
    val_acc_ave= np.mean(all_val_acc)
    val_loss_ave= np.mean(all_val_loss)

    val_auc_std= np.std(all_val_auc)
    val_acc_std= np.std(all_val_acc)
    val_loss_std= np.std(all_val_loss)
    
    
    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc,
        'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc, 'test_loss' : all_test_loss, 'val_loss' : all_val_loss})

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))

    print('\n Val:\n loss ± std: {0:.3f} ± {1:.3f}'.format(val_loss_ave, val_loss_std))

    print('\n\n Test:\n auc ± std: {0:.3f} ± {1:.3f}, acc ± std: {2:.3f} ± {3:.3f}'.format(
        test_auc_ave, test_auc_std, test_acc_ave, test_acc_std))
    
    print('\n\n Misc:\n auc ± std (val): {0:.3f} ± {1:.3f}, acc ± std (val): {2:.3f} ± {3:.3f},'
          'loss ± std (test): {4:.3f} ± {5:.3f}'.format(val_auc_ave, val_auc_std, val_acc_ave, val_acc_std, test_loss_ave, test_loss_std))
        


# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None,help='data directory')
parser.add_argument('--max_epochs', type=int, default=50,help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4,help='learning rate (default: 0.0001)')
parser.add_argument('--reg', type=float, default=1e-2,help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1,help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=5, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=5, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None,help='manually specify the set of splits to use')
parser.add_argument('--log_data', action='store_true', default=True, help='log data using tensorboard')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--task', type=str, choices=['task_2_tumor_subtyping_brca', 'task_2_tumor_subtyping_lung'],help='task to train')
parser.add_argument('--model_type', type=str, choices=['gabmil', 'gabmil_in', 'transmil'], default='gabmil', help='type of model')
parser.add_argument('--use_local', action='store_true', default=False, help='no global information')
parser.add_argument('--use_grid', action='store_true', default=False, help='enable grid information')
parser.add_argument('--use_block', action='store_true', default=False, help='enable block information')
parser.add_argument('--win_size_b', type=int, default=1, help='block window size')
parser.add_argument('--win_size_g', type=int, default=1, help='grid window size')

args = parser.parse_args()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.early_stopping =True
args.weighted_sample =True
args.task = "task_2_tumor_subtyping_lung"
args.split_dir = "tcga_lung"
args.csv_path = './dataset_csv/tcga_lung_subset.csv'
args.data_root_dir = "/home/bme001/20215294/Data/LUNG/LUNG_ostu_10x/"
sub_feat_dir = 'feat_ssl'
args.seed = 2021
args.k = 10
args.k_end = 10

def seed_torch(seed=7):
    import random
    random.seed(int(seed))  # Ensure the seed is a Python int
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


settings = {'num_splits': args.k,
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs,
            'results_dir': args.results_dir,
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'seed': args.seed,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt,
            'model_type': args.model_type,
            'local': args.use_local,
            'grid': args.use_grid,
            'block': args.use_block,
            'win_size_b': args.win_size_b,
            'win_size_g': args.win_size_g,
            'sub_feat_dir': sub_feat_dir,}


print('\nLoad Dataset')

if args.task == 'task_2_tumor_subtyping_brca':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = args.csv_path,
                            data_dir= os.path.join(args.data_root_dir, sub_feat_dir),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_col='oncotree_code',
                            label_dict = {'IDC':0, 'ILC':1},
                            patient_strat=False,
                            ignore=['MDLC', 'PD', 'ACBC', 'IMMC', 'BRCNOS', 'BRCA', 'SPC', 'MBC', 'MPT'])
elif args.task == 'task_2_tumor_subtyping_lung':    
    args.n_classes = 2
    dataset = Generic_MIL_Dataset(csv_path = args.csv_path,
                            data_dir= os.path.join(args.data_root_dir, sub_feat_dir),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_col='oncotree_code',
                            label_dict = {'LUAD':0, 'LUSC':1},
                            patient_strat=False,
                            ignore=[])    
else:
    raise NotImplementedError

if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

args.split_dir = os.path.join('splits', args.split_dir)

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})


with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))

if __name__ == "__main__":
    results = main(args)
    print("finished!")
    print("end script")