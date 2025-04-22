import numpy as np

import torch
import pandas as pd
from models.model_gabmil import GABMIL
from models.model_transmil import TransMIL
from utils.utils import *
from sklearn.metrics import roc_auc_score, precision_score, recall_score,  auc, precision_recall_curve, f1_score, cohen_kappa_score

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        self.data_all = {'y_true':[],'y_pred':[]}

    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)

        self.data_all['y_true'].append(Y)
        self.data_all['y_pred'].append(Y_hat)

    def get_accuracy(self):
        correct = sum([x["correct"] for x in self.data])
        count = sum([x["count"] for x in self.data])
        accuracy = correct / count
        return accuracy

    def get_f1(self):
        y_true = np.asarray(self.data_all['y_true']).reshape(-1,)
        y_pred = np.asarray(self.data_all['y_pred']).reshape(-1,)
        f1 = f1_score(y_true,y_pred,average='macro')
        return f1
    
    def get_prc(self):
        y_true = np.asarray(self.data_all['y_true']).reshape(-1,)
        y_pred = np.asarray(self.data_all['y_pred']).reshape(-1,)
        prc = precision_score(y_true, y_pred, average='macro')
        return prc

    def get_recall(self):
        y_true = np.asarray(self.data_all['y_true']).reshape(-1,)
        y_pred = np.asarray(self.data_all['y_pred']).reshape(-1,)
        recall = recall_score(y_true, y_pred, average='macro')
        return recall

    def cohen_kappa(self):
        y_true = np.asarray(self.data_all['y_true']).reshape(-1,)
        y_pred = np.asarray(self.data_all['y_pred']).reshape(-1,)
        kappa = cohen_kappa_score(y_true, y_pred)
        return kappa

        
def initiate_model(args, ckpt_path):
    print('\nInit Model...', end=' ')
    model_dict = {"n_classes": args.n_classes,"use_local": args.use_local, "use_block": args.use_block, "use_grid": args.use_grid, "win_size_b": args.win_size_b, "win_size_g": args.win_size_g}
    if args.model_type == 'transmil':
        model = TransMIL()
    else:    
        model = GABMIL(**model_dict)   
          
    print_network(model)

    ckpt = torch.load(ckpt_path)
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
    model.load_state_dict(ckpt_clean, strict=True)

    model.relocate()
    model.eval()
    return model

def eval(dataset, args, ckpt_path):
    model = initiate_model(args, ckpt_path)
    
    print('Init Loaders')
    loader = get_simple_loader(dataset)

    print('\nInit loss function...', end=' ')
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(args.device)

    patient_results, auc, pr_area, loss, df, test_logger = summary(model, loader, args, loss_fn)

    acc = test_logger.get_accuracy()
    f1 = test_logger.get_f1()
    prc = test_logger.get_prc()
    recall = test_logger.get_recall()
    kappa = test_logger.cohen_kappa()

    return model, patient_results, auc, acc, f1, prc, recall, pr_area, loss, kappa, df

def summary(model, loader, args, loss_fn):
    device = args.device
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    model.eval()
    test_error = 0.
    test_loss = 0.

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    for batch_idx, (data, label, coord) in enumerate(loader):
        data, label, coord = data.to(device, non_blocking=True), label.to(device, non_blocking=True), coord.to(device, non_blocking=True)    
        slide_id = slide_ids.iloc[batch_idx]

        with torch.no_grad():
            logits, Y_prob, Y_hat, _ = model(data, coords= coord)
        
        loss = loss_fn(logits, label)
        test_loss += loss.item()

        acc_logger.log(Y_hat, label)
        
        probs = Y_prob.cpu().numpy()

        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        
        error = calculate_error(Y_hat, label)
        test_error += error

    del data
    test_loss /= len(loader)
    test_error /= len(loader)

    if args.n_classes == 2:
        auc_score = roc_auc_score(all_labels, all_probs[:, 1])

        precision, recall, _ = precision_recall_curve(all_labels, all_probs[:, 1])
        pr_area = auc(recall, precision)
    else:
        # Compute ROC AUC for multi-class
        auc_score = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')

        # Compute Precision-Recall AUC for each class and average
        pr_area = 0
        for i in range(args.n_classes):
            precision, recall, _ = precision_recall_curve((all_labels == i).astype(int), all_probs[:, i])
            pr_area += auc(recall, precision)
        pr_area /= args.n_classes

    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})

    df = pd.DataFrame(results_dict)
    return patient_results, auc_score, pr_area, test_loss, df, acc_logger
