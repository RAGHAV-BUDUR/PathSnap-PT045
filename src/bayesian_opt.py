"""
PathSnap - Bayesian Hyperparameter Optimisation Module
Team: PT-045 | P3: Palleneni Sloka (PES1UG24AM183)
Optimises the chosen Random Forest model using Optuna (Tree-structured Parzen Estimator).
Uses 5-fold GroupKFold cross-validation on training trips to prevent data leakage.
Compares untuned vs tuned model on CV Macro F1, Student-Test Top-1 Accuracy, Macro F1, Log Loss, and Inference Time.
"""

import sys, os, time, json, joblib
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss
from src.features import FEATURE_COLS
from src.train_models import compute_observation_top1_accuracy, compute_route_continuity, load_all_arcs

def run_bayesian_optimization(n_trials=20):
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    print("Loading training and testing data for Bayesian Optimization...")
    df_train = pd.read_csv('results/train_features.csv', dtype={'trip_id': str})
    df_test = pd.read_csv('results/student_test_features.csv', dtype={'trip_id': str})

    # Balanced subset for rapid, high-signal tuning
    pos_mask = df_train['label'] == 1
    df_pos = df_train[pos_mask]
    df_neg = df_train[~pos_mask].sample(n=len(df_pos)*3, random_state=42)
    df_sub = pd.concat([df_pos, df_neg]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    X_train = df_sub[FEATURE_COLS].values
    y_train = df_sub['label'].values
    groups = df_sub['trip_id'].values

    X_test = df_test[FEATURE_COLS].values
    y_test = df_test['label'].values

    arcs_dict = load_all_arcs()
    gkf = GroupKFold(n_splits=3)

    trials_log = []

    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 60, 160, step=20)
        max_depth = trial.suggest_int('max_depth', 10, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 4, 16)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 6)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.6])

        f1s = []
        for trn_idx, val_idx in gkf.split(X_train, y_train, groups=groups):
            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                n_jobs=-1,
                random_state=42
            )
            clf.fit(X_train[trn_idx], y_train[trn_idx])
            preds = clf.predict(X_train[val_idx])
            f1s.append(f1_score(y_train[val_idx], preds, average='macro'))

        mean_f1 = np.mean(f1s)
        trials_log.append({
            'trial_number': trial.number,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': str(max_features),
            'cv_macro_f1': mean_f1
        })
        return mean_f1

    print(f"\nStarting Bayesian Optimization ({n_trials} trials with TPE sampler)...")
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    t0_opt = time.time()
    study.optimize(objective, n_trials=n_trials)
    opt_duration = time.time() - t0_opt

    print(f"Optimization completed in {opt_duration:.1f}s")
    print(f"Best Trial: #{study.best_trial.number} with CV Macro F1: {study.best_value:.4f}")
    print("Best Hyperparameters:", study.best_params)

    # Save trials log
    df_trials = pd.DataFrame(trials_log)
    df_trials.to_csv('results/bayesian_trials.csv', index=False)

    with open('models/best_hyperparameters.json', 'w') as f:
        json.dump(study.best_params, f, indent=4)

    # Plot Optimization History
    plt.figure(figsize=(8, 4))
    plt.plot(df_trials['trial_number'], df_trials['cv_macro_f1'], marker='o', color='teal', linewidth=2)
    plt.axhline(study.best_value, color='crimson', linestyle='--', label=f'Best CV Macro F1: {study.best_value:.4f}')
    plt.xlabel('Trial Number')
    plt.ylabel('Group-Aware CV Macro F1')
    plt.title('Bayesian Optimization Search History (Random Forest with TPE)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('plots/bayes_opt_history.png', dpi=200)
    plt.close()

    # Compare Untuned vs Tuned Model
    print("\n--- Evaluating Untuned vs Tuned Model on Student Test Partition ---")
    untuned_params = {'n_estimators': 100, 'max_depth': 14, 'min_samples_split': 10, 'max_features': 'sqrt', 'random_state': 42, 'n_jobs': -1}
    tuned_params = {**study.best_params, 'random_state': 42, 'n_jobs': -1}

    models_to_compare = {
        'Untuned Random Forest': RandomForestClassifier(**untuned_params),
        'Tuned Random Forest (Bayes Opt)': RandomForestClassifier(**tuned_params)
    }

    comparison_records = []
    gkf_eval = GroupKFold(n_splits=5)

    for name, clf in models_to_compare.items():
        # CV Macro F1
        cv_f1s = []
        for trn_idx, val_idx in gkf_eval.split(X_train, y_train, groups=groups):
            clf_clone = RandomForestClassifier(**clf.get_params())
            clf_clone.fit(X_train[trn_idx], y_train[trn_idx])
            preds = clf_clone.predict(X_train[val_idx])
            cv_f1s.append(f1_score(y_train[val_idx], preds, average='macro'))
        cv_score = np.mean(cv_f1s)

        # Fit on full training subset
        clf.fit(X_train, y_train)

        # Test evaluation
        t0 = time.time()
        test_probs = clf.predict_proba(X_test)[:, 1]
        inf_time = (time.time() - t0) / len(X_test) * 1000.0

        test_preds = (test_probs >= 0.5).astype(int)
        test_f1 = f1_score(y_test, test_preds, average='macro')
        test_log_loss = log_loss(y_test, np.clip(test_probs, 1e-6, 1 - 1e-6))
        top1_acc, top1_df = compute_observation_top1_accuracy(df_test, test_probs)
        continuity = compute_route_continuity(top1_df, arcs_dict)

        comparison_records.append({
            'Model Configuration': name,
            'CV Macro F1-score': round(cv_score, 4),
            'Student-testing Top-1 Accuracy': f"{top1_acc*100:.2f}%",
            'Student-testing Macro F1-score': round(test_f1, 4),
            'Log Loss': round(test_log_loss, 4),
            'Route Continuity': f"{continuity*100:.2f}%",
            'Inference Time': f"{inf_time:.3f} ms"
        })

    df_bayes_comp = pd.DataFrame(comparison_records)
    df_bayes_comp.to_csv('results/bayesian_comparison.csv', index=False)
    print("\nBayesian Optimization Comparison Table:\n", df_bayes_comp.to_string(index=False))

    # Refit and save best tuned model for production inference
    best_clf = models_to_compare['Tuned Random Forest (Bayes Opt)']
    joblib.dump(best_clf, 'models/best_tuned_model.joblib')
    print("\nSaved best tuned model to models/best_tuned_model.joblib")

    return df_bayes_comp

if __name__ == '__main__':
    run_bayesian_optimization()
