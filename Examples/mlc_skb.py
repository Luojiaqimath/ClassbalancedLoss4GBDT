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



# datalist= ['Corel5k',
#             'bibtex',
#             'birds',
#             'delicious',
#             'emotions',
#             'enron',
#             'genbase',
#             'mediamill',
#             'medical',
#             'scene',
#             'tmc2007_500',
#             'yeast']
