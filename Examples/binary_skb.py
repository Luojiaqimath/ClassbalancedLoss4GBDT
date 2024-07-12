import numpy as np 
from py_boost import GradientBoosting
from py_boost.multioutput.sketching import RandomProjectionSketch
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import  StratifiedKFold
import optuna
import os
from imblearn.datasets import fetch_datasets
from sklearn.model_selection import train_test_split
from gbdtCBL.binarycupy import CupyFLLoss

    

class PyBObjective(object):
    def __init__(self, X, y):

        self.X = X
        self.y = y

    def __call__(self, trial):
        params = {
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, 2.0),
        "lr": trial.suggest_float("lr", 0.01, 1.0),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "max_bin": trial.suggest_int("max_bin", 64, 256),
        "r": trial.suggest_categorical("r", [0.5, 1.0, 2.0]),
        }
    
        folds = StratifiedKFold(5, random_state=42, shuffle=True)
        scores = []
        for _, (train_idx, val_idx) in enumerate(folds.split(self.X, self.y)):
            X_tr, y_tr = self.X[train_idx], self.y[train_idx]
            X_val, y_val = self.X[val_idx], self.y[val_idx]
            
            clf = GradientBoosting(
            CupyFLLoss(params["r"]),
            ntrees=1000,
            es=50,
            verbose=-1, 
            multioutput_sketch=RandomProjectionSketch(5),
            lr=params["lr"],
            lambda_l2=params["lambda_l2"], 
            subsample=params["subsample"], 
            max_bin=params["max_bin"],
            max_depth=params["max_depth"], 
            )
            
            clf.fit(
                X_tr, y_tr, eval_sets=[{'X': X_val, 'y': y_val}]
            )
            y_val_pred_prob = clf.predict(X_val)
            y_val_pred = (y_val_pred_prob >=0.5)+0
            scores.append(roc_auc_score(y_val, y_val_pred_prob))
        return np.mean(scores)
    
    
def test(best_params, X, y, X_test, y_test):
    folds = StratifiedKFold(5, random_state=42, shuffle=True)
    scores1 = []
    scores2 = []
    scores3 = []
    scores4 = []
    scores5 = []
    scores6 = []
    for _, (train_idx, val_idx) in enumerate(folds.split(X, y)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        clf = GradientBoosting(
        CupyFLLoss(best_params["r"]),
        ntrees=1000,
        es=50,
        verbose=-1, 
        multioutput_sketch=RandomProjectionSketch(5),
        lr=best_params["lr"],
        lambda_l2=best_params["lambda_l2"], 
        subsample=best_params["subsample"], 
        max_bin=best_params["max_bin"],
        max_depth=best_params["max_depth"], 
        )
        
        clf.fit(X_tr, y_tr, eval_sets=[{'X': X_val, 'y': y_val}])
        
        y_test_pred_prob = clf.predict(X_test)
        y_test_pred = (y_test_pred_prob >=0.5)+0
        scores1.append(roc_auc_score(y_test, y_test_pred_prob))
        scores2.append(f1_score(y_test, y_test_pred))
    return [scores1, scores2]



data = fetch_datasets()['ecoli']
X, y = data.data, data.target
y = 0.5*y+0.5
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

optuna.logging.set_verbosity(optuna.logging.WARNING)  
    
sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction="maximize",
                            sampler=sampler,
                            study_name='model_eval',
                            )

study.optimize(PyBObjective(X_train, y_train), n_trials=100)
best_params = study.best_trial.params
scores = test(best_params, X_train, y_train, X_test, y_test)



print(f"AUC Mean: {np.mean(scores[0])}\n")
print(f"AUC Std: {np.std(scores[0])}\n")

print(f"F1 Mean: {np.mean(scores[1])}\n")
print(f"F1 Std: {np.std(scores[1])}\n")

