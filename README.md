This is a **Python package** containing class-balanced loss functions for gradient boosting decision tree. 
Please refer to the paper "Improving GBDT Performance on Imbalanced Datasets: An Empirical Study of Class-Balanced Loss Functions" for more details.
The python package link: \url{https://pypi.org/project/gbdtCBL/#description}

You can install the package by:
```python
pip install gbdtCBL
```

To use XGBoost, you need to first install the following packages:
```python
pip install xgboost
```

To use LightGBM, you need to first install the following packages:
```python
pip install lightgbm --config-settings=cmake.define.USE_GPU=ON
```

To use SketchBoost, you need to first install the following packages:
```python
pip install -U cupy-cuda11x py-boost  # for cuda 11.x

pip install -U cupy-cuda12x py-boost  # for cuda 12.x
```


Some examples are provided in the **Examples** file. Datasets used in the paper are also provided.



**Feel free to reach out if you have any ideas or questions!**




## Demo for binary classification using LightGBM
```python
import numpy as np 
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold
import optuna
from imblearn.datasets import fetch_datasets
from sklearn.model_selection import train_test_split
from gbdtCBL.binary import ACELoss



# define some functions
def sigmoid(x):
    kEps = 1e-16 #  avoid 0 div
    x = np.minimum(-x, 88.7)  # avoid exp overflow
    return 1 / (1 + np.exp(x)+kEps)


def predict_proba(model, X):
    # Lightgbm: Cannot compute class probabilities or labels due to the usage of customized objective function.
    prediction = model.predict(X)
    
    prediction_probabilities = sigmoid(prediction).reshape(-1, 1)
    prediction_probabilities = np.concatenate((1 - prediction_probabilities,
                                                    prediction_probabilities), 1)
    return prediction_probabilities

def eval_auc(labels, preds):  # auc
    p = sigmoid(preds)
    return 'auc', roc_auc_score(labels, p), True


class LGBBinary(object):
    def __init__(self, X, y):

        self.X = X
        self.y = y

    def __call__(self, trial):
        params = {
        "num_leaves": trial.suggest_int("num_leaves", 8, 32),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 2.0),
        "m": trial.suggest_float("m", 0.05, 0.2),
        }
        
        folds = StratifiedKFold(5, random_state=42, shuffle=True)
        scores = []
        for _, (train_idx, val_idx) in enumerate(folds.split(self.X, self.y)):
            X_tr, y_tr = self.X[train_idx], self.y[train_idx]
            X_val, y_val = self.X[val_idx], self.y[val_idx]
            
            clf = lgb.LGBMClassifier(
                n_estimators=1000,
                learning_rate=params["learning_rate"],
                reg_alpha=params["reg_alpha"], 
                reg_lambda=params["reg_lambda"], 
                num_leaves=params["num_leaves"],
                device= "gpu",
                objective=ACELoss(m=params['m']),
                )
            
            clf.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric=eval_auc,
                callbacks=[lgb.early_stopping(stopping_rounds=50)],
                )
            
            y_val_pred_prob = predict_proba(clf, X_val)[:, 1]
            scores.append(roc_auc_score(y_val, y_val_pred_prob))
        return np.mean(scores)
    
    
def test(best_params, X, y, X_test, y_test):
    folds = StratifiedKFold(5, random_state=42, shuffle=True)
    scores1 = []
    scores2 = []
    for _, (train_idx, val_idx) in enumerate(folds.split(X, y)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        clf = lgb.LGBMClassifier(
                n_estimators=1000,
                learning_rate=best_params["learning_rate"],
                reg_alpha=best_params["reg_alpha"], 
                reg_lambda=best_params["reg_lambda"], 
                num_leaves=best_params["num_leaves"],
                device= "gpu",
                objective=ACELoss(m=best_params['m']),
                )
        
        clf.fit(X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric=eval_auc,
                callbacks=[lgb.early_stopping(stopping_rounds=50)],
                )
        
        y_test_pred_prob = predict_proba(clf, X_test)[:, 1]
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

study.optimize(LGBBinary(X_train, y_train), n_trials=100)
best_params = study.best_trial.params
scores = test(best_params, X_train, y_train, X_test, y_test)

    
print(f"AUC Mean: {np.mean(scores[0])}\n")
print(f"AUC Std: {np.std(scores[0])}\n")

print(f"F1 Mean: {np.mean(scores[1])}\n")
print(f"F1 Std: {np.std(scores[1])}\n")
```





## Demo for multi-class classification using XGBoost


```python
import numpy as np 
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
import optuna
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from gbdtCBL.xgbmulti import XGBFLMulti



class XGBMulti(object):
    def __init__(self, X, y):

        self.X = X
        self.y = y

    def __call__(self, trial):
        params = {
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 5.0),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1.0),
        "r": trial.suggest_categorical("r", [0.5, 1.0, 2.0]),
        }
        
        folds = StratifiedKFold(5, random_state=42, shuffle=True)
        scores = []
        for _, (train_idx, val_idx) in enumerate(folds.split(self.X, self.y)):
            X_tr, y_tr = self.X[train_idx], self.y[train_idx]
            X_val, y_val = self.X[val_idx], self.y[val_idx]
            
            clf = xgb.XGBClassifier(
                n_estimators=1000,
                learning_rate=params["learning_rate"],
                reg_alpha=params["reg_alpha"], 
                reg_lambda=params["reg_lambda"], 
                max_depth=params["max_depth"],
                device= "cuda",
                tree_method= "hist",
                early_stopping_rounds=50,
                objective=XGBFLMulti(r=params['r'])
                )
            
            clf.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                )
            
            y_val_pred_prob = clf.predict_proba(X_val)
            y_val_pred = clf.predict(X_val)
            scores.append(accuracy_score(y_val, y_val_pred))
        return np.mean(scores)
    
    
def test(best_params, X, y, X_test, y_test):
    folds = StratifiedKFold(5, random_state=42, shuffle=True)
    scores1 = []
    scores2 = []
    for _, (train_idx, val_idx) in enumerate(folds.split(X, y)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        clf = xgb.XGBClassifier(
                n_estimators=1000,
                learning_rate=best_params["learning_rate"],
                reg_alpha=best_params["reg_alpha"], 
                reg_lambda=best_params["reg_lambda"], 
                max_depth=best_params["max_depth"],
                device= "cuda",
                tree_method= "hist",
                early_stopping_rounds=50,
                objective=XGBFLMulti(r=best_params['r'])
                )
        
        clf.fit(X_tr, y_tr,
                eval_set=[(X_val, y_val)],
        
                )
        
        y_test_pred_prob = clf.predict_proba(X_test)
        y_test_pred = clf.predict(X_test)
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

study.optimize(XGBMulti(X_train, y_train), n_trials=100)
best_params = study.best_trial.params
scores = test(best_params, X_train, y_train, X_test, y_test)


print(f"ACC Mean: {np.mean(scores[0])}\n")
print(f"ACC Std: {np.std(scores[0])}\n")

print(f"F1 Mean: {np.mean(scores[1])}\n")
print(f"F1 Std: {np.std(scores[1])}\n")
```



## Demo for multi-label classification using SketchBoost

```python
import numpy as np 
from py_boost import GradientBoosting
from py_boost.multioutput.sketching import RandomProjectionSketch
from sklearn.metrics import  accuracy_score, f1_score
from sklearn.model_selection import KFold
import optuna
from skmultilearn.dataset import load_dataset
from gbdtCBL.binarycupy import CupyAWELoss

    

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
        "m": trial.suggest_float("m", 0.05, 0.2),
        "w": trial.suggest_categorical("w", [2.0, 3.0, 5.0]),
        }
    
        folds = KFold(5, random_state=42, shuffle=True)
        scores = []
        for _, (train_idx, val_idx) in enumerate(folds.split(self.X, self.y)):
            X_tr, y_tr = self.X[train_idx], self.y[train_idx]
            X_val, y_val = self.X[val_idx], self.y[val_idx]
            
            clf = GradientBoosting(
            CupyAWELoss(w=params['w'], m=params['m']),
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
            scores.append(accuracy_score(y_val, y_val_pred, normalize=True, sample_weight=None))
        return np.mean(scores)
    
    
def test(best_params, X, y, X_test, y_test):
    folds = KFold(5, random_state=42, shuffle=True)
    scores1 = []
    scores2 = []
    for _, (train_idx, val_idx) in enumerate(folds.split(X, y)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        clf = GradientBoosting(
        CupyAWELoss(w=best_params['w'], m=best_params['m']),
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
        scores1.append(accuracy_score(y_test, y_test_pred, normalize=True, sample_weight=None))
        scores2.append(f1_score(y_test, y_test_pred, average='samples', zero_division=0))
    return [scores1, scores2]


X_train, y_train, feature_names, label_names = load_dataset('Corel5k', 'train')
X_test, y_test, _, _ = load_dataset('Corel5k', 'test')

optuna.logging.set_verbosity(optuna.logging.WARNING)  
    
sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction="maximize",
                            sampler=sampler,
                            study_name='model_eval',
                            )

study.optimize(PyBObjective(X_train.toarray(), y_train.toarray()), n_trials=100)
best_params = study.best_trial.params
scores = test(best_params, X_train.toarray(), y_train.toarray(), X_test.toarray(), y_test.toarray())


            
print(f"ACC Mean: {np.mean(scores[0])}\n")
print(f"ACC Std: {np.std(scores[0])}\n")

print(f"F1 Mean: {np.mean(scores[1])}\n")
print(f"F1 Std: {np.std(scores[1])}\n")
```
