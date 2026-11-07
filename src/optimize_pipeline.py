import optuna
import joblib
import os
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb


def create_objective(X, y, model_name):
    """
    Creates an Optuna objective function dynamically based on the chosen model.
    """
    def objective(trial):
        
        if model_name == 'random_forest':
            param_grid = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10)
            }
            model = RandomForestClassifier(**param_grid, random_state=42, class_weight='balanced')
            
        elif model_name == 'xgboost':
            param_grid = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0)
            }
            model = XGBClassifier(**param_grid, random_state=42, use_label_encoder=False, eval_metric='logloss')
            
        elif model_name == 'lightgbm':
            param_grid = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0)
            }
            model = lgb.LGBMClassifier(**param_grid, random_state=42)
        else:
            raise ValueError("Unsupported model name.")

        #The Leak-Free Pipeline
        
        if model_name == 'lightgbm':
            pipeline = Pipeline([
                ('classifier', model)
            ])
        else:
            #Keep SMOTE for Random Forest
            pipeline = Pipeline([
                ('oversampler', SMOTE(random_state=42)),
                ('classifier', model)
            ])

        #Stratified 5-Fold Cross-Validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        #Calculate PR-AUC scores across the 5 folds
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring='average_precision', n_jobs=-1)
        
        return scores.mean()
    
    return objective

def run_optimization_and_serialize(X, y, model_name, n_trials=50):
    """
    Executes the Optuna study and saves the winning pipeline for the Week 5 handoff.
    """
    print(f"Starting TPE Optimization for {model_name}...")
    
    #Create the study prioritizing maximum F1 Score
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    objective_func = create_objective(X, y, model_name)
    study.optimize(objective_func, n_trials=n_trials)
    
    print(f"Best Trial PR-AUC Score: {study.best_value:.4f}")
    print(f"Best Parameters: {study.best_params}")
    
    #Automating the Handoff (Serialization)
    if model_name == 'random_forest':
        best_model = RandomForestClassifier(**study.best_params, random_state=42, class_weight='balanced')
        
        final_pipeline = Pipeline([
            ('oversampler', SMOTE(random_state=42)),
            ('classifier', best_model)
        ])
    elif model_name == 'xgboost':
        best_model = XGBClassifier(**study.best_params, random_state=42, use_label_encoder=False, eval_metric='logloss')
        final_pipeline = Pipeline([
            ('oversampler', SMOTE(random_state=42)),
            ('classifier', best_model)
        ])
    elif model_name == 'lightgbm':
        best_model = lgb.LGBMClassifier(**study.best_params, random_state=42)
        #LightGBM strictly uses scale_pos_weight. NO SMOTE.
        final_pipeline = Pipeline([
            ('classifier', best_model)
        ])
    
    
    final_pipeline.fit(X, y)
    
  
    os.makedirs('models', exist_ok=True)
    filepath = f"models/best_pipeline_{model_name}.pkl"
    joblib.dump(final_pipeline, filepath)
        
    final_pipeline = Pipeline([
        ('oversampler', SMOTE(random_state=42)),
        ('classifier', best_model)
    ])
    
    #Train the final leak-free pipeline on the entire training dataset
   
    
    print(f"Successfully serialized tuned model weights to {filepath}")
    
    return study, final_pipeline



from train_baseline import load_processed_data

if __name__ == "__main__":
    print("Loading raw baseline data...")
    
    X, y=load_processed_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print("Initiating Week 5 Optuna Pipeline...")
    
    study, final_model = run_optimization_and_serialize(
        X=X_train, 
        y=y_train, 
        model_name='lightgbm', 
        n_trials=100
    )