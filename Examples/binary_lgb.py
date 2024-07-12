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



# data_list = ['ecoli',
#             'optical_digits',
#             'satimage',
#             'pen_digits',
#             'sick_euthyroid',
#             'spectrometer',
#             'car_eval_34',
#             'isolet',
#             'us_crime',
#             'libras_move',
#             'thyroid_sick',
#             'arrhythmia',
#             'oil',
#             'car_eval_4',
#             'letter_img',
#             'webpage',
#             'mammography',
#             'protein_homo',
#         ]
