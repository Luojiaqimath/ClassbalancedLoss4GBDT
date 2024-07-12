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


