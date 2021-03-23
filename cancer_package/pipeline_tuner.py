import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from time import time
from functools import partial
from collections.abc import Callable
from datetime import datetime, timedelta
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from typing import Optional, List, Dict, Union, Any
from sklearn.base import clone
from sklearn.utils import shuffle
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV, cross_val_score


class PipelineTuner():
    """
    Tunes the hyper params and performs feature selection
    """
    def __init__(
            self,
            create_pipe: Callable,
            hyper_params: Dict[str, Union[list, np.array]],
            X_train: pd.DataFrame,
            y_train: pd.Series,
            scorer: str,
            cv: Any,
            weight_array: np.array,
            patience: Optional[int] = None,
        ) -> None:
        self.ts = datetime.now().strftime("%Y%m%d_%H%M")
        self.create_pipe = create_pipe
        self.hyper_params = hyper_params
        self.original_X_train = X_train
        self.y_train = y_train
        self.scorer = scorer
        self.cv = cv
        self.weight_array = weight_array
        self.patience = patience if patience else np.inf
        #
        self.feature_names = self.original_X_train.columns
        self.n_features = len(self.feature_names)
        self.fs_pipes = pd.Series([], dtype=object)
        self.fs_scores = pd.Series([], dtype=float)
        self.fs_features = pd.Series([], dtype=object)
        self.offset = 0
        self.max_score = -np.inf
        #
        self.bayes_n_iter = None
        self.bayes_init_points = None
        self.bayes_discrete_vars = None
        self.save_dir = None
        self.best_pipe = None
        self.best_features = None
    

    def continue_tuning(self, path):
        previous_tuning = joblib.load(path)
        self.fs_pipes = previous_tuning["fs_pipes"]
        self.fs_scores = previous_tuning["fs_scores"]
        self.fs_features = previous_tuning["fs_features"]
        self.feature_names = previous_tuning["fs_features"].iloc[-1]
        self.n_features = len(self.feature_names)
        self.offset = len(self.fs_pipes)
        self.max_score = self.fs_scores.max()
        
    
    def bayes_config(
            self,
            n_iter: int,
            init_points: int,
            discrete_vars: Optional[List[str]] = None
        ) -> None:
        self.bayes_n_iter = n_iter
        self.bayes_init_points = init_points
        self.bayes_discrete_vars = discrete_vars
    
    
    def save(self, path, base_name):
        sub_dir = f'{base_name}_{self.ts}'
        print(f"model save in: {sub_dir}")
        self.save_dir = os.path.join(path, sub_dir)
        os.mkdir(self.save_dir)
    
    
    def _feature_selection(self):
        """
        to be removed
        """
        if self.fs_pipes.empty:
            n_features = self.n_features
            selected_cols = self.feature_names
        else:
            n_features = self.n_features - 1
            selector = SelectFromModel(
                estimator=self.fs_pipes.iloc[-1]["model"],
                threshold=-np.inf,
                max_features=n_features,
                prefit=True
            )
            selected_cols = self.fs_features.iloc[-1][selector.get_support()]
        
        self.n_features = n_features
        self.fs_features = self.fs_features.append(
            pd.Series([np.array(selected_cols)], index=[n_features])
        )
    
    
    def _new_feature_selection(self):
        if self.fs_pipes.empty:
            n_features = self.n_features
            selected_cols = self.feature_names
        else:
            for n_features in reversed(range(self.n_features)):
                selector = RFECV(
                    self.fs_pipes.iloc[-1]["model"],
                    step=1,
                    cv=self.cv,
                    min_features_to_select=n_features)
                selector = selector.fit(X, y)
                print(selector.support_)
                if selector.support_.sum() == i:
                    print("broke")
                    break
                else:
                    print("continue")
    
    
    def _discretize_params(self, params):
        params = {
                key: int(round(val)) if key in self.bayes_discrete_vars else val
                for key, val in params.items()
            }
        return params
    
    
    def _bayes_opt_func(self, model_pipe, X_train, **params):
        if self.bayes_discrete_vars is not None:
            params = self._discretize_params(params)
        print(f"-- bayes func: {params}")
        cv_score = cross_val_score(
            model_pipe.set_params(**params),
            X_train,
            self.y_train,
            scoring=self.scorer,
            cv=self.cv,
            fit_params={'model__sample_weight': self.weight_array},
            n_jobs=-1
        )
        return np.mean(cv_score)
    
    
    def _param_optimization(self):
        pipe = self.create_pipe()
        selected_cols = self.fs_features.iloc[-1]
        X_train = self.original_X_train[selected_cols]
        if self.bayes_n_iter is None:
            gscv = GridSearchCV(
              pipe,
              self.hyper_params,
              scoring=self.scorer,
              n_jobs=-1,
              cv=self.cv,
              verbose=1
            )
            gscv.fit(X_train, self.y_train, model__sample_weight=self.weight_array)
            best_score_iter = gscv.best_score_
            best_pipe_iter = gscv.best_estimator_
        else:
            optimizer = BayesianOptimization(
                f=partial(self._bayes_opt_func, model_pipe=pipe, X_train=X_train),
                pbounds=self.hyper_params,
                random_state=9,
                verbose=1
            )
            if self.save_dir:
                log_fn = os.path.join(self.save_dir, f"{self.n_features}_features_log.json")
                logger = JSONLogger(path=log_fn)
                optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
            
            # probing all hyper-parameters that were optimal in previous steps
            if not self.fs_pipes.empty:
                for previous_pipe in self.fs_pipes:
                    previous_params = previous_pipe["model"].get_params()
                    previous_params = {
                        f"model__{key}": val for key, val in previous_params.items()
                        if f"model__{key}" in self.hyper_params
                    }
                    optimizer.probe(
                        params=previous_params
                    )
            
            optimizer.maximize(
                init_points=self.bayes_init_points,
                n_iter=self.bayes_n_iter,
            )
            best_score_iter = optimizer.max["target"]
            best_pipe_iter = clone(pipe).set_params(**self._discretize_params(optimizer.max["params"]))
            best_pipe_iter.fit(X_train, self.y_train)
            
        print(f"len(selected_cols) {len(selected_cols)}")
        print(f"X_train.shape {X_train.shape}")
        print(f"best_score_iter {best_score_iter}")
        print(f"score with {self.n_features} features: {best_score_iter:.2%}")
        self.fs_scores = self.fs_scores.append(
            pd.Series([best_score_iter], index=[self.n_features])
        )
        self.fs_pipes = self.fs_pipes.append(
            pd.Series([best_pipe_iter], index=[self.n_features])
        )
        
    
    def _update_max_score(self):
        if self.fs_scores.iloc[-1] >= self.max_score:
            self.max_score = self.fs_scores.iloc[-1]
            print(f'best score so far, model:\n {self.fs_pipes.iloc[-1]}')
    
    
    def _save_tuning_iter(self):
        if self.save_dir:
            joblib.dump(
              {"fs_pipes": self.fs_pipes, "fs_scores": self.fs_scores, "fs_features": self.fs_features}, 
              os.path.join(self.save_dir, "model.pkl"),
              compress=1
            )
    
    
    def tune(self):
        ts_total = time()
        for i in range(len(self.fs_pipes), len(self.feature_names)):
            print(f"\nstep {i} starts")
            ts_step = time()
            
            self._feature_selection()
            self._param_optimization()
            self._update_max_score()
            self._save_tuning_iter()
            
            enough_iters = len(self.fs_scores) > self.offset + self.patience
            non_productive_iters = len(self.fs_scores) - pd.Series(self.fs_scores).argmax() > self.patience
            if enough_iters and non_productive_iters:
                break

            print(f"step {i}: { str(timedelta(seconds=time() - ts_step)).split('.')[0] }")
        print(f"total time { str(timedelta(seconds=time() - ts_total)).split('.')[0] }")
        
        self.best_pipe = self.fs_pipes[ self.fs_scores.idxmax()]
        self.best_features = self.fs_features[ self.fs_scores.idxmax()]


    def plot_tuning(self, figsize: tuple = (12, 6)):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_xlim(self.fs_pipes.index.max(), self.fs_pipes.index.min())
        self.fs_scores.plot.line()
        plt.show()