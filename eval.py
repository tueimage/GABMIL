from __future__ import print_function


import argparse
import torch
import os
import pandas as pd
from utils.utils import *
from datasets.dataset_generic import Generic_MIL_Dataset
from utils.eval_utils import *


# Generic settings
parser = argparse.ArgumentParser(description='Configurations for WSI Evaluation')
parser.add_argument('--data_root_dir', type=str, default=None, help='data directory')
parser.add_argument('--results_dir', type=str, default='./results', help='(default: ./results)')
parser.add_argument('--save_exp_code', type=str, default=None, help='experiment code to save eval results')
parser.add_argument('--models_exp_code', type=str, default=None, help='experiment code to load trained models')
parser.add_argument('--k', type=int, default=5, help='number of folds')
parser.add_argument('--k_start', type=int, default=-1, help='start fold')
parser.add_argument('--k_end', type=int, default=5, help='end fold')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
parser.add_argument('--splits_dir', type=str, default=None,help='manually specify the set of splits to use')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
parser.add_argument('--task', type=str, choices=['task_2_tumor_subtyping_brca', 'task_2_tumor_subtyping_lung'],help='task to train')
parser.add_argument('--model_type', type=str, choices=['gabmil', 'transmil'], default='gabmil', help='type of model (default: gabmil)')
parser.add_argument('--use_local', action='store_true', default=False, help='no global information')
parser.add_argument('--use_grid', action='store_true', default=False, help='enable grid information')
parser.add_argument('--use_block', action='store_true', default=False, help='enable block information')
parser.add_argument('--win_size_b', type=int, default=1, help='block window size')
parser.add_argument('--win_size_g', type=int, default=1, help='grid window size')


args = parser.parse_args()

args.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.task = "task_2_tumor_subtyping_brca"
args.splits_dir = "splits/tcga_brca"
args.csv_path = './dataset_csv/tcga_brca_subset.csv'
args.data_root_dir = "/home/bme001/20215294/Data/BRCA/BRCA_ostu_10x/"
sub_feat_dir = 'feat'
args.seed = 2021
args.k = 10
args.k_end = 10
args.save_dir = os.path.join('./eval_results', 'EVAL_' + str(args.save_exp_code))
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))

os.makedirs(args.save_dir, exist_ok=True)

if args.splits_dir is None:
    args.splits_dir = args.models_dir

assert os.path.isdir(args.models_dir)
assert os.path.isdir(args.splits_dir)


settings = {
            'local': args.use_local,
            'grid': args.use_grid,
            'block': args.use_block,
            'win_size_b': args.win_size_b,
            'win_size_g': args.win_size_g,
            'model_type': args.model_type,
            'sub_feat_dir': sub_feat_dir,}

with open(args.save_dir + '/eval_experiment_{}.txt'.format(args.save_exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print(settings)

if args.task == 'task_2_tumor_subtyping_brca':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = args.csv_path,
                            data_dir= os.path.join(args.data_root_dir, sub_feat_dir),
                            shuffle = False, 
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

if args.k_start == -1:
    start = 0
else:
    start = args.k_start
if args.k_end == -1:
    end = args.k
else:
    end = args.k_end

if args.fold == -1:
    folds = range(start, end)
else:
    folds = range(args.fold, args.fold+1)
ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds]
datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}

if __name__ == "__main__":
    all_results = []
    all_auc = []
    all_acc = []
    all_f1 = []
    all_prc = []
    all_recall = []
    all_pr_area = []
    all_loss = []
    all_kappa = []
    for ckpt_idx in range(len(ckpt_paths)):
        if datasets_id[args.split] < 0:
            split_dataset = dataset
        else:
            csv_path = '{}/splits_{}.csv'.format(args.splits_dir, folds[ckpt_idx])
            datasets = dataset.return_splits(from_id=False, csv_path=csv_path)
            split_dataset = datasets[datasets_id[args.split]]

        model, patient_results, auc, acc, f1, prc, recall, pr_area, loss, kappa, df  = eval(split_dataset, args, ckpt_paths[ckpt_idx])

        all_results.append(patient_results)
        all_auc.append(auc)
        all_acc.append(acc)
        all_f1.append(f1)
        all_prc.append(prc)
        all_recall.append(recall)
        all_pr_area.append(pr_area)
        all_loss.append(loss)
        all_kappa.append(kappa)

        df.to_csv(os.path.join(args.save_dir, 'fold_{}.csv'.format(folds[ckpt_idx])), index=False)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_auc, 'test_acc': all_acc, 'test_f1': all_f1, 'test_prc': all_prc, 'test_recall': all_recall, 'test_pr_area': all_pr_area, 'test_loss': all_loss, 'test_kappa': all_kappa})
    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
    else:
        save_name = 'summary.csv'
        
    final_df.to_csv(os.path.join(args.save_dir, save_name))
    
    "Compute the average and std of the metrics"
    test_auc_ave= np.mean(all_auc)
    test_acc_ave= np.mean(all_acc)
    test_f1_ave= np.mean(all_f1)
    test_prc_ave= np.mean(all_prc)
    test_recall_ave= np.mean(all_recall)
    test_pr_area_ave= np.mean(all_pr_area)
    test_loss_ave= np.mean(all_loss)
    test_kappa_ave= np.mean(all_kappa)

    test_auc_std= np.std(all_auc)
    test_acc_std= np.std(all_acc)
    test_f1_std= np.std(all_f1)
    test_prc_std= np.std(all_prc)
    test_recall_std= np.std(all_recall)
    test_pr_area_std= np.std(all_pr_area)
    test_loss_std= np.std(all_loss)
    test_kappa_std= np.std(all_kappa)

    print('\n\nTest:\nauc ± std: {0:.3f} ± {1:.3f}, acc ± std: {2:.3f} ± {3:.3f}, f1 ± std: {4:.3f} ± {5:.3f}, prc ± std: {6:.3f} ± {7:.3f}, recall ± std: {8:.3f} ± {9:.3f}, pr_area ± std: {10:.3f} ± {11:.3f}, kappa ± std: {12:.3f} ± {13:.3f}, loss ± std: {14:.3f} ± {15:.3f} \n\n'.
          format(test_auc_ave, test_auc_std, test_acc_ave, test_acc_std, test_f1_ave, test_f1_std, test_prc_ave, test_prc_std, test_recall_ave, test_recall_std, test_pr_area_ave, test_pr_area_std, test_kappa_ave, test_kappa_std, test_loss_ave, test_loss_std))
    
        