import numpy as np 
import xgboost as xgb
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold
import optuna
from imblearn.datasets import fetch_datasets
from sklearn.model_selection import train_test_split
from gbdtCBL.binary import ASLLoss



class XGBBinary(object):
    def __init__(self, X, y):

        self.X = X
        self.y = y

    def __call__(self, trial):
        params = {
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 5.0),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1.0),
        "r1": trial.suggest_categorical("r1", [0.0, 0.1]),
        "r2": trial.suggest_categorical("r2", [0.5, 1.0, 2.0]),
        "m": trial.suggest_float("m", 0.05, 0.2),
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
                objective=ASLLoss(r1=params['r1'],
                                  r2=params['r2'],
                                  m=params['m']),
                )
            
            clf.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                )
            
            y_val_pred_prob = clf.predict_proba(X_val)[:, 1]
            y_val_pred = clf.predict(X_val)
            scores.append(roc_auc_score(y_val, y_val_pred_prob))
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
                objective=ASLLoss(r1=best_params['r1'],
                                  r2=best_params['r2'],
                                  m=best_params['m']),
                )
        
        clf.fit(X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                )
        
        y_test_pred_prob = clf.predict_proba(X_test)[:, 1]
        y_test_pred = clf.predict(X_test)
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

study.optimize(XGBBinary(X_train, y_train), n_trials=100)
best_params = study.best_trial.params
scores = test(best_params, X_train, y_train, X_test, y_test)


print(f"AUC Mean: {np.mean(scores[0])}\n")
print(f"AUC Std: {np.std(scores[0])}\n")

print(f"F1 Mean: {np.mean(scores[1])}\n")
print(f"F1 Std: {np.std(scores[1])}\n")




# data_list = ['ecoli',
#             'satimage',
#             'sick_euthyroid',
#             'spectrometer',
#             'car_eval_34',
#             'isolet',
#             'us_crime',
#             'libras_move',
#             'thyroid_sick',
#             'arrhythmia',
#             'oil',
#             'yeast_me2',
#             'webpage',
#             'mammography',
#             'protein_homo',
#         ]
