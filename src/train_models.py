"""
PathSnap - Model Training and Comparison Module
Team: PT-045 | P2: Pabbisetty Harshitha Sai (PES1UG24AM182)
Trains and evaluates 5 distinct ML models across different algorithm families:
1. Logistic Regression (Linear/Probabilistic)
2. K-Nearest Neighbors (Instance-based)
3. Decision Tree (Tree-based)
4. Random Forest (Ensemble)
5. HistGradientBoosting (Boosting)
Computes GroupKFold CV Macro F1, Observation-level Top-1 Accuracy, Route Continuity, Student-Test Macro F1,
Log Loss, Training Time, and Inference Time.
"""

import sys, os, time, joblib
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from src.features import FEATURE_COLS

def compute_observation_top1_accuracy(df, pred_probs):
    """
    Observation-level Top-1 Accuracy:
    For each GPS observation, select candidate with highest predicted score;
    check if that candidate has label == 1.
    """
    df_eval = df[['trip_id', 'obs_idx', 'arc_id', 'gt_arc_id', 'label']].copy()
    df_eval['score'] = pred_probs

    # Pick candidate with max score per observation
    top1_idx = df_eval.groupby(['trip_id', 'obs_idx'])['score'].idxmax()
    top1_df = df_eval.loc[top1_idx]
    acc = (top1_df['label'] == 1).mean()
    return acc, top1_df

def compute_route_continuity(top1_df, arcs_df_dict):
    """
    Route continuity: fraction of consecutive predicted arcs that are connected
    (i.e., v_prev == u_curr) or identical (vehicle stationary on same segment).
    """
    continuity_scores = []
    for tid, grp in top1_df.groupby('trip_id'):
        arcs_df = arcs_df_dict.get(tid)
        if arcs_df is None:
            continue
        u_map = arcs_df['u'].values
        v_map = arcs_df['v'].values
        pred_arcs = grp['arc_id'].values
        if len(pred_arcs) < 2:
            continue
        connected = 0
        total = len(pred_arcs) - 1
        for i in range(total):
            a_prev, a_curr = pred_arcs[i], pred_arcs[i+1]
            if a_prev == a_curr:
                connected += 1
            elif a_prev < len(v_map) and a_curr < len(u_map) and v_map[a_prev] == u_map[a_curr]:
                connected += 1
        continuity_scores.append(connected / total)
    return np.mean(continuity_scores) if continuity_scores else 0.0

def load_all_arcs(folders=['train', 'student_test']):
    arcs_dict = {}
    for f in folders:
        if not os.path.exists(f): continue
        for tid in os.listdir(f):
            p = os.path.join(f, tid, f"{tid}.arcs")
            if os.path.exists(p):
                arcs_dict[tid] = pd.read_csv(p, sep=r'\s+', header=None, names=['u', 'v'])
    return arcs_dict

def train_and_evaluate_all():
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    print("Loading datasets...")
    df_train = pd.read_csv('results/train_features.csv', dtype={'trip_id': str})
    df_test = pd.read_csv('results/student_test_features.csv', dtype={'trip_id': str})

    # Subsample training data to 150k rows for rapid, balanced training across algorithms
    # Ensure all positive examples are kept, and negative examples sampled
    pos_mask = df_train['label'] == 1
    df_pos = df_train[pos_mask]
    df_neg = df_train[~pos_mask].sample(n=min(len(df_train[~pos_mask]), len(df_pos)*3), random_state=42)
    df_train_sub = pd.concat([df_pos, df_neg]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"Subsampled training set: {len(df_train_sub)} samples ({df_train_sub['label'].mean()*100:.1f}% positive)")

    X_train_raw = df_train_sub[FEATURE_COLS].values
    y_train = df_train_sub['label'].values
    groups_train = df_train_sub['trip_id'].values

    X_test_raw = df_test[FEATURE_COLS].values
    y_test = df_test['label'].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    joblib.dump(scaler, 'models/scaler.joblib')

    arcs_dict = load_all_arcs()

    # Define 5 Models across different algorithm families
    models = {
        'Logistic Regression': {
            'family': 'Linear and probabilistic',
            'config': 'C=1.0, penalty=l2, max_iter=300, class_weight=balanced',
            'model': LogisticRegression(C=1.0, max_iter=300, class_weight='balanced', random_state=42),
            'scaled': True
        },
        'K-Nearest Neighbors': {
            'family': 'Instance-based',
            'config': 'n_neighbors=9, weights=distance, metric=euclidean',
            'model': KNeighborsClassifier(n_neighbors=9, weights='distance', n_jobs=-1),
            'scaled': True
        },
        'Decision Tree': {
            'family': 'Tree-based',
            'config': 'max_depth=12, min_samples_split=20, criterion=gini',
            'model': DecisionTreeClassifier(max_depth=12, min_samples_split=20, random_state=42),
            'scaled': False
        },
        'Random Forest': {
            'family': 'Ensemble',
            'config': 'n_estimators=100, max_depth=14, min_samples_split=10, max_features=sqrt',
            'model': RandomForestClassifier(n_estimators=100, max_depth=14, min_samples_split=10, max_features='sqrt', n_jobs=-1, random_state=42),
            'scaled': False
        },
        'HistGradientBoosting': {
            'family': 'Boosting',
            'config': 'learning_rate=0.08, max_iter=120, max_depth=8, l2_regularization=1.0',
            'model': HistGradientBoostingClassifier(learning_rate=0.08, max_iter=120, max_depth=8, l2_regularization=1.0, random_state=42),
            'scaled': False
        }
    }

    comparison_results = []
    gkf = GroupKFold(n_splits=5)

    confusion_matrices = {}

    for name, info in models.items():
        print(f"\n==========================================")
        print(f"Training Model: {name} ({info['family']})")
        print(f"Config: {info['config']}")
        clf = info['model']
        is_scaled = info['scaled']

        X_tr = X_train_scaled if is_scaled else X_train_raw
        X_te = X_test_scaled if is_scaled else X_test_raw

        # 1. 5-Fold Group Cross Validation on Train Trips
        cv_f1s = []
        for fold, (trn_idx, val_idx) in enumerate(gkf.split(X_tr, y_train, groups=groups_train)):
            clf_fold = info['model'].__class__(**clf.get_params())
            clf_fold.fit(X_tr[trn_idx], y_train[trn_idx])
            val_preds = clf_fold.predict(X_tr[val_idx])
            f1 = f1_score(y_train[val_idx], val_preds, average='macro')
            cv_f1s.append(f1)
        mean_cv_f1 = np.mean(cv_f1s)
        print(f"  CV Macro F1 (5-fold GroupKFold): {mean_cv_f1:.4f} (+/-{np.std(cv_f1s):.4f})")

        # 2. Fit on full training set
        t0 = time.time()
        clf.fit(X_tr, y_train)
        train_time = time.time() - t0
        joblib.dump(clf, f"models/{name.lower().replace(' ', '_').replace('-', '_')}.joblib")

        # 3. Predict on Student Test set
        t0 = time.time()
        if hasattr(clf, 'predict_proba'):
            test_probs = clf.predict_proba(X_te)[:, 1]
        else:
            test_probs = clf.predict(X_te)
        inference_time = (time.time() - t0) / len(X_te) * 1000.0  # ms per sample

        test_preds = (test_probs >= 0.5).astype(int)
        test_macro_f1 = f1_score(y_test, test_preds, average='macro')
        test_acc_binary = accuracy_score(y_test, test_preds)

        # Log Loss
        test_log_loss = log_loss(y_test, np.clip(test_probs, 1e-6, 1 - 1e-6))

        # Observation-level Top-1 Accuracy
        top1_acc, top1_df = compute_observation_top1_accuracy(df_test, test_probs)

        # Route Continuity
        continuity = compute_route_continuity(top1_df, arcs_dict)

        cm = confusion_matrix(y_test, test_preds)
        confusion_matrices[name] = cm

        print(f"  Student-Test Macro F1: {test_macro_f1:.4f}")
        print(f"  Top-1 Accuracy: {top1_acc*100:.2f}%")
        print(f"  Route Continuity: {continuity*100:.2f}%")
        print(f"  Log Loss: {test_log_loss:.4f}")
        print(f"  Training Time: {train_time:.2f}s | Inference: {inference_time:.3f}ms/obs")

        comparison_results.append({
            'Model': name,
            'Algorithm Family': info['family'],
            'Key Configuration': info['config'],
            'CV Macro F1-score': round(mean_cv_f1, 4),
            'Top-1 Accuracy': f"{top1_acc*100:.2f}%",
            'Route continuity': f"{continuity*100:.2f}%",
            'Student-Test Macro F1-score': round(test_macro_f1, 4),
            'Log Loss': round(test_log_loss, 4),
            'Inference Time': f"{inference_time:.3f} ms"
        })

    df_comp = pd.DataFrame(comparison_results)
    df_comp.to_csv('results/model_comparison.csv', index=False)
    print("\n\n================ FINAL MODEL COMPARISON TABLE ================")
    print(df_comp.to_string(index=False))

    # ================= PLOTS =================
    # Plot 1: CV Macro F1 vs Student-Test Macro F1
    plt.figure(figsize=(9, 5))
    x = np.arange(len(df_comp))
    width = 0.35
    plt.bar(x - width/2, df_comp['CV Macro F1-score'], width, label='CV Macro F1 (Train GroupKFold)', color='steelblue')
    plt.bar(x + width/2, df_comp['Student-Test Macro F1-score'], width, label='Student-Test Macro F1', color='coral')
    plt.xticks(x, df_comp['Model'], rotation=15, ha='right')
    plt.ylabel('Macro F1-Score')
    plt.ylim(0.5, 1.0)
    plt.title('Comparison of Cross-Validation vs. Student-Test Macro F1 Across Models')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('plots/model_f1_comparison.png', dpi=200)
    plt.close()

    # Plot 2: Top-1 Accuracy & Route Continuity Comparison
    plt.figure(figsize=(9, 5))
    top1_vals = [float(v.replace('%', '')) for v in df_comp['Top-1 Accuracy']]
    cont_vals = [float(v.replace('%', '')) for v in df_comp['Route continuity']]
    plt.bar(x - width/2, top1_vals, width, label='Observation Top-1 Accuracy (%)', color='forestgreen')
    plt.bar(x + width/2, cont_vals, width, label='Route Continuity (%)', color='purple')
    plt.xticks(x, df_comp['Model'], rotation=15, ha='right')
    plt.ylabel('Percentage (%)')
    plt.ylim(0, 100)
    plt.title('Observation-Level Top-1 Accuracy & Raw Route Continuity Across Models')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('plots/model_top1_continuity.png', dpi=200)
    plt.close()

    # Plot 3: Log Loss Comparison
    plt.figure(figsize=(7, 4.5))
    plt.bar(df_comp['Model'], df_comp['Log Loss'], color='indianred', edgecolor='black', width=0.45)
    plt.xticks(rotation=15, ha='right')
    plt.ylabel('Cross-Entropy Log Loss')
    plt.title('Log Loss Comparison on Student-Testing Set (Lower is Better)')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('plots/model_log_loss.png', dpi=200)
    plt.close()

    # Plot 4: Confusion Matrices
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.5))
    for idx, (m_name, cm) in enumerate(confusion_matrices.items()):
        ax = axes[idx]
        im = ax.imshow(cm, cmap='Blues')
        ax.set_title(m_name, fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Pred 0', 'Pred 1'])
        ax.set_yticklabels(['True 0', 'True 1'])
        for r in range(2):
            for c in range(2):
                ax.text(c, r, f"{cm[r, c]}", ha='center', va='center',
                        color='white' if cm[r, c] > cm.max()/2 else 'black', fontsize=9)
    plt.tight_layout()
    plt.savefig('plots/confusion_matrices.png', dpi=200)
    plt.close()

    print("\nSaved all model comparison plots in plots/")
    return df_comp

if __name__ == '__main__':
    train_and_evaluate_all()
