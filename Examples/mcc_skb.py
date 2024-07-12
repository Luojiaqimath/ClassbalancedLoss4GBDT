import numpy as np 
from py_boost import GradientBoosting
from py_boost.multioutput.sketching import RandomProjectionSketch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
import optuna
import os
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from gbdtCBL.multicupy import CupyASLMulti
    


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
        "k": trial.suggest_int("k", 1, 10),
        "r1": trial.suggest_categorical("r1", [0.0, 0.1]),
        "r2": trial.suggest_categorical("r2", [0.5, 1.0, 2.0]),
        "m": trial.suggest_float("m", 0.05, 0.2),
        }
    
        folds = StratifiedKFold(5, random_state=42, shuffle=True)
        scores = []
        for _, (train_idx, val_idx) in enumerate(folds.split(self.X, self.y)):
            X_tr, y_tr = self.X[train_idx], self.y[train_idx]
            X_val, y_val = self.X[val_idx], self.y[val_idx]
            
            clf = GradientBoosting(
            CupyASLMulti(r1=params['r1'], r2=params['r2'], m=params['m']),
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
            y_val_pred = np.argmax(y_val_pred_prob, axis=1)
            scores.append(accuracy_score(y_val, y_val_pred, normalize=True, sample_weight=None))
        return np.mean(scores)
    
    
def test(best_params, X, y, X_test, y_test):
    folds = StratifiedKFold(5, random_state=42, shuffle=True)
    scores1 = []
    scores2 = []
    for _, (train_idx, val_idx) in enumerate(folds.split(X, y)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        clf = GradientBoosting(
        CupyASLMulti(r1=best_params['r1'], r2=best_params['r2'], m=best_params['m']),
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
        y_test_pred = np.argmax(y_test_pred_prob, axis=1)
        scores1.append(accuracy_score(y_test, y_test_pred))
        scores2.append(f1_score(y_test, y_test_pred,  average="weighted"))
    return [scores1, scores2]


def load_data(path):
    with open(path) as f:
        metadata_lines = 0
        for line in f:
            if line.startswith('@'):
                metadata_lines += 1

                if line.startswith('@input'):
                    inputs = [l.strip() for l in line[8:].split(',')]
                elif line.startswith('@output'):
                    outputs = [l.strip() for l in line[8:].split(',')]
            else:
                break
        
    df = pd.read_csv(path, skiprows=metadata_lines, header=None)
    df.columns = inputs + outputs
    df = pd.concat([pd.get_dummies(df[inputs]), df[outputs]], axis=1)

    matrix = df.values
    X, y = matrix[:, :-1], matrix[:, -1]
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    return X, y


X, y = load_data('./data/classification/'+'automobile-full.data')
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


print(f"ACC Mean: {np.mean(scores[0])}\n")
print(f"ACC Std: {np.std(scores[0])}\n")

print(f"F1 Mean: {np.mean(scores[1])}\n")
print(f"F1 Std: {np.std(scores[1])}\n")



