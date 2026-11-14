import os
import joblib
import optuna
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import f1_score, recall_score
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ChampionModelTrainer(BaseEstimator):
    """
    Object-oriented Optuna orchestrator for the LightGBM champion model.
    Replaces standalone optimize_pipeline.py and tune_models.py scripts.
    """
    def __init__(self, random_state: int = 42, n_trials: int = 50, cv_folds: int = 5):
        self.random_state = random_state
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.best_params_ = None
        self.champion_estimator_ = None
        self.opt_f1_threshold_ = 0.50
        self.clinical_threshold_ = 0.50

    def _objective(self, trial, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Optuna objective function for tuning LightGBM hyperparameters.
        Optimized specifically for imbalanced clinical PR-AUC.
        """
        param_grid = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0),
            'max_depth': trial.suggest_int('max_depth', 4, 15)
        }
        
        model = lgb.LGBMClassifier(**param_grid, random_state=self.random_state, verbose=-1)
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        #Calculate PR-AUC across the 5 folds
        scores = cross_val_score(
            model, X, y, cv=cv, scoring='average_precision', n_jobs=-1
        )
        return scores.mean()

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Executes the Optuna TPE search and fits the final Champion LightGBM model.
        """
        logging.info("Starting Optuna TPE optimization for Champion LightGBM...")
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.random_state))
        
        #Pass objective via lambda to inject X_train and y_train
        study.optimize(lambda trial: self._objective(trial, X_train, y_train), n_trials=self.n_trials)
        
        logging.info(f"Best CV PR-AUC Score: {study.best_value:.4f}")
        logging.info(f"Best Hyperparameters: {study.best_params}")
        
        self.best_params_ = study.best_params
        
        #Retrain the champion on the full training set using best parameters
        self.champion_estimator_ = lgb.LGBMClassifier(
            **self.best_params_, random_state=self.random_state, verbose=-1
        )
        self.champion_estimator_.fit(X_train, y_train)
        
        return self

    def calibrate_thresholds(self, y_true: np.ndarray, y_prob: np.ndarray):
        """
        Sweeps decision thresholds on holdout predictions to find Max-F1 and 
        Clinical Safety (Recall >= 65%) operating points.
        """
        logging.info("Calibrating asymmetric medical risk thresholds...")
        thresholds = np.arange(0.20, 0.81, 0.01)
        
        best_f1_thresh, best_f1 = 0.50, 0.0
        clinical_thresh, best_clinical_f1 = 0.50, 0.0

        for t in thresholds:
            preds = (y_prob >= t).astype(int)
            f1 = f1_score(y_true, preds, zero_division=0)
            rec = recall_score(y_true, preds, zero_division=0)

            #Max F1
            if f1 > best_f1:
                best_f1 = f1
                best_f1_thresh = t

            #Clinical Safety (Catch >= 65% of crashes while maximizing F1)
            if rec >= 0.65 and f1 > best_clinical_f1:
                best_clinical_f1 = f1
                clinical_thresh = t

        self.opt_f1_threshold_ = round(float(best_f1_thresh), 2)
        self.clinical_threshold_ = round(float(clinical_thresh), 2)
        
        logging.info(f"Max-F1 Threshold: {self.opt_f1_threshold_} | Clinical Threshold: {self.clinical_threshold_}")
        return self.opt_f1_threshold_, self.clinical_threshold_

    def get_champion_bundle(self, X_test, y_test):
        """
        Packages the tuned estimator, holdout data, and thresholds 
        into the exact dictionary structure expected by Track B's SHAP scripts.
        """
        if self.champion_estimator_ is None:
            raise NotFittedError("Champion model is not fitted. Call 'fit' first.")
            
        y_prob_test = self.champion_estimator_.predict_proba(X_test)[:, 1]
        self.calibrate_thresholds(y_test, y_prob_test)
        
        bundle = {
            "X_test": X_test,
            "y_test": y_test,
            "models": {
                "LightGBM (Weighted)": {
                    "best_estimator": self.champion_estimator_,
                    "best_params": self.best_params_,
                    "opt_f1_threshold": self.opt_f1_threshold_,
                    "clinical_threshold": self.clinical_threshold_,
                    "y_prob_test": y_prob_test
                }
            }
        }
        return bundle