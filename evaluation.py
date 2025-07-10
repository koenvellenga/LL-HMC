import torch
import sklearn
import sklearn.metrics
import uncertainty_metrics.numpy as um
import numpy as np
import pandas as pd
from bayesian_torch.utils.util import predictive_entropy
from torch_uncertainty.metrics.classification import FPR95

def AULC(accs, uncertainties):
    idxs = np.argsort(uncertainties)
    uncs_s = uncertainties[idxs]
    error_s = accs[idxs]

    mean_error = error_s.mean()
    error_csum = np.cumsum(error_s)

    Fs = error_csum / np.arange(1, len(error_s) + 1)
    s = 1 / len(Fs)
    return -1 + s * Fs.sum() / mean_error, Fs

def rAULC(uncertainties, accs):
    perf_aulc, Fsp = AULC(accs, -accs.astype("float"))
    curr_aulc, Fsc = AULC(accs, uncertainties)
    return curr_aulc / perf_aulc, Fsp, Fsc

def compute_AUCs(uc_id, uc_labels,uc_ood):
    id_labels = np.zeros_like(uc_id)
    ood_labels = np.ones_like(uc_ood)
    
    uc_labels_ood = np.concatenate([id_labels, ood_labels])
    uc_values = np.concatenate([uc_id, uc_ood])
    
    raulc, r1, r2 = rAULC(np.array(uc_id), np.array(uc_labels))
    roc_auc = sklearn.metrics.roc_auc_score(np.array(uc_labels_ood), np.array(uc_values))
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(np.array(uc_labels_ood), np.array(uc_values))
    pr_auc = sklearn.metrics.auc(recall, precision)
    return raulc, roc_auc, pr_auc

def compute_FPR95(uc_id, uc_ood):
    fpr95 = FPR95(pos_label=1)
    fpr95.update(uc_id, torch.zeros_like(uc_id))
    fpr95.update(uc_ood, torch.ones_like(uc_ood))
    result =  fpr95.compute().item()
    return result


def evaluate(model_type, model, X_test, y_test, ood=None):
    num_classes = len(torch.unique(y_test))
    with torch.no_grad():
        if model_type == "llhmc":
            logits = model(X_test)
            avg_logits = logits.mean(0)
            if isinstance(ood, torch.Tensor):
                logits_ood = model(ood)

    pred_one_hot = F.one_hot(argmax_outputs, num_classes=num_classes)
    gt_onehot  = F.one_hot(y_test, num_classes=num_classes)
    acc = sklearn.metrics.accuracy_score(y_true=gt_onehot, y_pred=pred_one_hot)
    f1 = sklearn.metrics.f1_score(y_true=gt_onehot, 
                                  y_pred=pred_one_hot, average="macro")
    ace = um.gce(labels=y_test,
                   probs=np.array(softmax_pred.detach()),
                   num_bins=10,
                   binning_scheme='adaptive',
                   class_conditional=True,
                   max_prob=False,
                   norm='l1')

    softmax_pred = F.softmax(avg_logits, dim=-1)
    argmax_outputs = softmax_pred.argmax(-1)

    if isinstance(ood, torch.Tensor):
        softmax_ood= F.softmax(logits_ood, dim=-1)
        fpr_result = compute_FPR95(torch.from_numpy(uc_id), torch.from_numpy(uc_ood))
        _, roc_auc, pr_auc = compute_AUCs(np.array(uc_id), np.array(uc_labels), torch.from_numpy(uc_ood))
    else:
        fpr_result = None
        roc_auc = None
        pr_auc = None

    return {
            "acc": acc,
            "f1": f1,
            "ace": ace,
            "raulc": raulc,
            "fpr95": fpr_result,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
        }
