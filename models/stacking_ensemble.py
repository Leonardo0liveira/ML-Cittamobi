import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_auc_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, roc_curve, classification_report)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import resample
from catboost import CatBoostClassifier
import warnings
import time
from google.cloud import bigquery
import os
warnings.filterwarnings('ignore')

os.makedirs('visualizations', exist_ok=True)
os.makedirs('reports', exist_ok=True)

print(f"\n{'='*80}")
print(f"STACKING ENSEMBLE - TOP 3 MODELS")
print(f"Base Models: Random Forest + CatBoost + SGD")
print(f"Meta Model: Logistic Regression")
print(f"{'='*80}\n")

# Carregar dados
project_id = "proj-ml-469320"
client = bigquery.Client(project=project_id)
query = "SELECT * FROM `proj-ml-469320.app_cittamobi.dataset-updated` TABLESAMPLE SYSTEM (20 PERCENT) LIMIT 50000"
df = client.query(query).to_dataframe()
target = "target"

# Preparação
df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])
df = df.sort_values('event_timestamp').reset_index(drop=True)
df['hour'] = df['event_timestamp'].dt.hour
df['day_of_week'] = df['event_timestamp'].dt.dayofweek
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

print("Processando expanding windows...")
df['user_hist_rate'] = 0.0
df['stop_hist_rate'] = 0.0
for i in range(100, len(df)):
    hist = df.iloc[:i]
    curr = df.iloc[i]
    user_hist = hist[hist['user_pseudo_id'] == curr['user_pseudo_id']]
    if len(user_hist) > 0:
        df.at[i, 'user_hist_rate'] = user_hist[target].mean()
    stop_hist = hist[hist['gtfs_stop_id'] == curr['gtfs_stop_id']]
    if len(stop_hist) > 0:
        df.at[i, 'stop_hist_rate'] = stop_hist[target].mean()

# Feature engineering
if 'device_lat' in df.columns and 'stop_lat_event' in df.columns:
    df['geo_distance'] = np.sqrt((df['device_lat']-df['stop_lat_event'])**2+(df['device_lon']-df['stop_lon_event'])**2)
if 'stop_event_rate' in df.columns and 'stop_total_samples' in df.columns:
    df['stop_density'] = df['stop_event_count'] / (df['stop_total_samples'] + 1)

df = df.fillna(0)
if 'user_frequency' in df.columns:
    df = df[df['user_frequency'] >= 2]

numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_features.remove(target)
for col in ['user_pseudo_id', 'gtfs_stop_id', 'gtfs_route_id', 'session_id']:
    if col in numeric_features:
        numeric_features.remove(col)

X, y = df[numeric_features], df[target]
print(f"✓ Features: {len(numeric_features)} | Target: 0={sum(y==0):,}, 1={sum(y==1):,}")

# Divisão temporal
tscv = TimeSeriesSplit(n_splits=3)
train_idx, test_idx = list(tscv.split(X))[-1]
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# Undersampling
train_df = pd.concat([X_train, y_train], axis=1)
class_0, class_1 = train_df[train_df[target]==0], train_df[train_df[target]==1]
class_0_down = resample(class_0, n_samples=len(class_1)*2, random_state=42, replace=False)
train_balanced = pd.concat([class_0_down, class_1]).sample(frac=1, random_state=42)
X_train_bal, y_train_bal = train_balanced[numeric_features], train_balanced[target]
print(f"✓ Undersampling: {len(class_0):,} → {len(class_0_down):,} (ratio {len(class_0_down)/len(class_1):.1f}:1)\n")

# Normalização para SVM
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_bal), columns=X_train_bal.columns, index=X_train_bal.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

print(f"{'='*80}")
print(f"TREINANDO MODELOS BASE")
print(f"{'='*80}\n")

# Modelo 1: Random Forest (ROC-AUC 0.8083)
print("1. Random Forest...")
start = time.time()
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=25,
    min_samples_split=50,
    min_samples_leaf=20,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_bal, y_train_bal)
rf_time = time.time() - start
rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
rf_roc = roc_auc_score(y_test, rf_pred_proba)
print(f"   ROC-AUC: {rf_roc:.4f} | Time: {rf_time:.1f}s")

# Modelo 2: CatBoost (ROC-AUC 0.8040)
print("2. CatBoost...")
start = time.time()
cb_model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    random_state=42,
    verbose=0
)
cb_model.fit(X_train_bal, y_train_bal)
cb_time = time.time() - start
cb_pred_proba = cb_model.predict_proba(X_test)[:, 1]
cb_roc = roc_auc_score(y_test, cb_pred_proba)
print(f"   ROC-AUC: {cb_roc:.4f} | Time: {cb_time:.1f}s")

# Modelo 3: SGD Classifier (mais compatível com stacking)
print("3. SGD Classifier...")
start = time.time()
sgd_base = SGDClassifier(loss='log_loss', alpha=0.001, max_iter=1000, random_state=42)
sgd_model = sgd_base.fit(X_train_scaled, y_train_bal)
sgd_time = time.time() - start
sgd_pred_proba = sgd_model.predict_proba(X_test_scaled)[:, 1]
sgd_roc = roc_auc_score(y_test, sgd_pred_proba)
print(f"   ROC-AUC: {sgd_roc:.4f} | Time: {sgd_time:.1f}s\n")

print(f"{'='*80}")
print(f"TREINANDO STACKING ENSEMBLE")
print(f"{'='*80}\n")

print("Treinando meta-model (Logistic Regression)...")
start = time.time()

# Criar Pipeline para SVM com normalização integrada
from sklearn.pipeline import Pipeline
sgd_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('sgd', SGDClassifier(loss='log_loss', alpha=0.001, max_iter=1000, random_state=42))
])

# Stacking com modelos que não precisam de normalização + pipeline para SGD
estimators = [
    ('random_forest', rf_model),
    ('catboost', cb_model),
    ('sgd', sgd_pipeline)
]

stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=5,
    stack_method='predict_proba',
    n_jobs=-1,
    verbose=0
)

stacking_model.fit(X_train_bal, y_train_bal)
stacking_time = time.time() - start

print(f"✓ Stacking treinado em {stacking_time:.1f}s\n")

# Cross-validation do stacking
print("Executando cross-validation...")
tscv_cv = TimeSeriesSplit(n_splits=5)
cv_scores = cross_val_score(stacking_model, X_train_bal, y_train_bal, cv=tscv_cv, scoring='roc_auc', n_jobs=-1)
print(f"CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")

# Predições
y_pred_proba = stacking_model.predict_proba(X_test)[:, 1]
y_pred = stacking_model.predict(X_test)

# Métricas
roc_auc = roc_auc_score(y_test, y_pred_proba)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"{'='*80}")
print(f"RESULTADOS STACKING ENSEMBLE")
print(f"{'='*80}\n")
print(f"CV ROC-AUC:  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"Test ROC-AUC: {roc_auc:.4f}")
print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}\n")
print(classification_report(y_test, y_pred, zero_division=0))

print(f"\nCONFUSION MATRIX:")
print(f"                Predicted 0    Predicted 1")
print(f"Actual 0        {cm[0,0]:>11,}    {cm[0,1]:>11,}")
print(f"Actual 1        {cm[1,0]:>11,}    {cm[1,1]:>11,}")

# Comparação com modelos individuais
print(f"\n{'='*80}")
print(f"COMPARAÇÃO: STACKING vs MODELOS BASE")
print(f"{'='*80}\n")
print(f"{'Modelo':<20} {'ROC-AUC':>10} {'F1-Score':>10} {'Recall':>10}")
print(f"{'-'*50}")
print(f"{'Stacking Ensemble':<20} {roc_auc:>10.4f} {f1:>10.4f} {recall:>10.4f}")
print(f"{'Random Forest':<20} {rf_roc:>10.4f} {'-':>10} {'-':>10}")
print(f"{'CatBoost':<20} {cb_roc:>10.4f} {'-':>10} {'-':>10}")
print(f"{'SGD':<20} {sgd_roc:>10.4f} {'-':>10} {'-':>10}")

# Ganho percentual
rf_gain = ((roc_auc - rf_roc) / rf_roc) * 100
cb_gain = ((roc_auc - cb_roc) / cb_roc) * 100
sgd_gain = ((roc_auc - sgd_roc) / sgd_roc) * 100
avg_gain = ((roc_auc - np.mean([rf_roc, cb_roc, sgd_roc])) / np.mean([rf_roc, cb_roc, sgd_roc])) * 100

print(f"\nGANHO vs MODELOS BASE:")
print(f"  vs Random Forest: {rf_gain:+.2f}%")
print(f"  vs CatBoost:      {cb_gain:+.2f}%")
print(f"  vs SGD:           {sgd_gain:+.2f}%")
print(f"  vs Média dos 3:   {avg_gain:+.2f}%")

# Visualizações
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# ROC Curves
ax1 = fig.add_subplot(gs[0, :2])
fpr_stack, tpr_stack, _ = roc_curve(y_test, y_pred_proba)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_pred_proba)
fpr_cb, tpr_cb, _ = roc_curve(y_test, cb_pred_proba)
fpr_sgd, tpr_sgd, _ = roc_curve(y_test, sgd_pred_proba)

ax1.plot(fpr_stack, tpr_stack, linewidth=3, label=f'Stacking ({roc_auc:.4f})', color='#e74c3c')
ax1.plot(fpr_rf, tpr_rf, linewidth=2, label=f'Random Forest ({rf_roc:.4f})', color='#3498db', alpha=0.7)
ax1.plot(fpr_cb, tpr_cb, linewidth=2, label=f'CatBoost ({cb_roc:.4f})', color='#2ecc71', alpha=0.7)
ax1.plot(fpr_sgd, tpr_sgd, linewidth=2, label=f'SGD ({sgd_roc:.4f})', color='#f39c12', alpha=0.7)
ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax1.set_xlabel('False Positive Rate', fontweight='bold')
ax1.set_ylabel('True Positive Rate', fontweight='bold')
ax1.set_title('ROC Curves Comparison', fontweight='bold', fontsize=14)
ax1.legend(loc='lower right')
ax1.grid(alpha=0.3)

# Confusion Matrix
ax2 = fig.add_subplot(gs[0, 2])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, cbar=False)
ax2.set_title('Confusion Matrix', fontweight='bold')
ax2.set_ylabel('True', fontweight='bold')
ax2.set_xlabel('Predicted', fontweight='bold')

# ROC-AUC Comparison
ax3 = fig.add_subplot(gs[1, 0])
models = ['Stacking', 'RF', 'CatBoost', 'SGD']
rocs = [roc_auc, rf_roc, cb_roc, sgd_roc]
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
bars = ax3.bar(models, rocs, color=colors, alpha=0.7)
ax3.axhline(y=roc_auc, color='r', linestyle='--', alpha=0.3)
ax3.set_ylabel('ROC-AUC', fontweight='bold')
ax3.set_title('ROC-AUC Comparison', fontweight='bold')
ax3.set_ylim(0.75, 0.82)
ax3.grid(axis='y', alpha=0.3)
for bar, roc in zip(bars, rocs):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{roc:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# Training Time Comparison
ax4 = fig.add_subplot(gs[1, 1])
times = [stacking_time, rf_time, cb_time, sgd_time]
bars = ax4.bar(models, times, color=colors, alpha=0.7)
ax4.set_ylabel('Training Time (s)', fontweight='bold')
ax4.set_title('Training Time Comparison', fontweight='bold')
ax4.grid(axis='y', alpha=0.3)
for bar, t in zip(bars, times):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{t:.1f}s', ha='center', va='bottom', fontweight='bold', fontsize=9)

# Metrics Comparison
ax5 = fig.add_subplot(gs[1, 2])
metrics = ['Precision', 'Recall', 'F1-Score']
values = [precision, recall, f1]
colors_metrics = ['#3498db', '#e74c3c', '#2ecc71']
bars = ax5.bar(metrics, values, color=colors_metrics, alpha=0.7)
ax5.set_ylabel('Score', fontweight='bold')
ax5.set_title('Stacking Metrics', fontweight='bold')
ax5.set_ylim(0, 1)
ax5.grid(axis='y', alpha=0.3)
for bar, v in zip(bars, values):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height+0.02,
             f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

# CV Scores Distribution
ax6 = fig.add_subplot(gs[2, :])
folds = range(1, len(cv_scores)+1)
ax6.plot(folds, cv_scores, marker='o', linewidth=2, markersize=10, color='#e74c3c', label='CV Scores')
ax6.axhline(y=cv_scores.mean(), color='g', linestyle='--', linewidth=2, label=f'Mean: {cv_scores.mean():.4f}')
ax6.axhline(y=roc_auc, color='b', linestyle='--', linewidth=2, label=f'Test: {roc_auc:.4f}')
ax6.fill_between(folds, cv_scores.mean()-cv_scores.std(), cv_scores.mean()+cv_scores.std(), alpha=0.2, color='g')
ax6.set_xlabel('Fold', fontweight='bold')
ax6.set_ylabel('ROC-AUC', fontweight='bold')
ax6.set_title('Cross-Validation Scores', fontweight='bold', fontsize=14)
ax6.set_xticks(folds)
ax6.legend()
ax6.grid(alpha=0.3)

plt.savefig('visualizations/stacking_ensemble.png', dpi=300, bbox_inches='tight')
plt.close()

# Salvar relatório
with open('reports/stacking_ensemble_report.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("STACKING ENSEMBLE - TOP 3 MODELS\n")
    f.write("="*80 + "\n\n")
    f.write("BASE MODELS:\n")
    f.write(f"  1. Random Forest (ROC-AUC: {rf_roc:.4f})\n")
    f.write(f"  2. CatBoost      (ROC-AUC: {cb_roc:.4f})\n")
    f.write(f"  3. SGD Classifier (ROC-AUC: {sgd_roc:.4f})\n\n")
    f.write("META MODEL: Logistic Regression\n")
    f.write(f"STACKING METHOD: predict_proba with 5-fold CV\n\n")
    f.write("="*80 + "\n")
    f.write("RESULTS\n")
    f.write("="*80 + "\n\n")
    f.write(f"CV ROC-AUC:  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")
    f.write(f"Test ROC-AUC: {roc_auc:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1-Score: {f1:.4f}\n\n")
    f.write(classification_report(y_test, y_pred, zero_division=0))
    f.write("\n\nGAINS vs BASE MODELS:\n")
    f.write(f"  vs Random Forest: {rf_gain:+.2f}%\n")
    f.write(f"  vs CatBoost:      {cb_gain:+.2f}%\n")
    f.write(f"  vs SGD:           {sgd_gain:+.2f}%\n")
    f.write(f"  vs Average:       {avg_gain:+.2f}%\n")

print(f"\n✓ Salvo: reports/stacking_ensemble_report.txt")
print(f"✓ Salvo: visualizations/stacking_ensemble.png\n")
