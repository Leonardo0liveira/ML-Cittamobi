"""
Script otimizado para comparar todos os modelos com:
- Undersampling (2:1 ratio)
- Cross-validation (5-fold TimeSeriesSplit)
- Classification report
- Código limpo e eficiente
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_auc_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, classification_report, roc_curve)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import resample
import warnings
import time
from google.cloud import bigquery
import os

warnings.filterwarnings('ignore')
os.makedirs('visualizations/comparison', exist_ok=True)
os.makedirs('reports', exist_ok=True)

print(f"\n{'='*80}")
print(f"COMPARAÇÃO DE MODELOS - OPTIMIZED")
print(f"{'='*80}\n")

# ===========================================================================
# CARREGAR E PREPARAR DADOS
# ===========================================================================
project_id = "proj-ml-469320"
client = bigquery.Client(project=project_id)

query = """
    SELECT * FROM `proj-ml-469320.app_cittamobi.dataset-updated` 
    TABLESAMPLE SYSTEM (20 PERCENT)
    LIMIT 50000
"""

df = client.query(query).to_dataframe()
target = "target"

# Preparação temporal
df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])
df = df.sort_values('event_timestamp').reset_index(drop=True)

# Features temporais básicas
df['hour'] = df['event_timestamp'].dt.hour
df['day_of_week'] = df['event_timestamp'].dt.dayofweek
df['day_of_month'] = df['event_timestamp'].dt.day
df['month'] = df['event_timestamp'].dt.month
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

# Expanding windows (simplificado)
print("Processando expanding windows...")
df['user_hist_rate'] = 0.0
df['stop_hist_rate'] = 0.0
df['user_hist_count'] = 0
df['stop_hist_count'] = 0

for i in range(100, len(df)):
    hist = df.iloc[:i]
    curr = df.iloc[i]
    
    user_hist = hist[hist['user_pseudo_id'] == curr['user_pseudo_id']]
    if len(user_hist) > 0:
        df.at[i, 'user_hist_rate'] = user_hist[target].mean()
        df.at[i, 'user_hist_count'] = len(user_hist)
    
    stop_hist = hist[hist['gtfs_stop_id'] == curr['gtfs_stop_id']]
    if len(stop_hist) > 0:
        df.at[i, 'stop_hist_rate'] = stop_hist[target].mean()
        df.at[i, 'stop_hist_count'] = len(stop_hist)

# Feature engineering básico
if 'device_lat' in df.columns and 'stop_lat_event' in df.columns:
    df['geo_distance'] = np.sqrt((df['device_lat'] - df['stop_lat_event'])**2 + 
                                  (df['device_lon'] - df['stop_lon_event'])**2)

# Limpeza
df = df.fillna(0)
if 'user_frequency' in df.columns:
    df = df[df['user_frequency'] >= 2]
if 'dist_device_stop' in df.columns:
    df = df[df['dist_device_stop'] < df['dist_device_stop'].quantile(0.99)]

# Selecionar features
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_features.remove(target)
for col in ['user_pseudo_id', 'gtfs_stop_id', 'gtfs_route_id', 'session_id']:
    if col in numeric_features:
        numeric_features.remove(col)

X = df[numeric_features]
y = df[target]

print(f"✓ Total: {len(df):,} | Features: {len(numeric_features)}")
print(f"✓ Target: 0={sum(y==0):,} ({100*sum(y==0)/len(y):.1f}%), 1={sum(y==1):,} ({100*sum(y==1)/len(y):.1f}%)")

# Divisão temporal
tscv = TimeSeriesSplit(n_splits=3)
train_idx, test_idx = list(tscv.split(X))[-1]
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# Undersampling
train_df = pd.concat([X_train, y_train], axis=1)
class_0 = train_df[train_df[target] == 0]
class_1 = train_df[train_df[target] == 1]
class_0_down = resample(class_0, n_samples=len(class_1)*2, random_state=42, replace=False)
train_balanced = pd.concat([class_0_down, class_1]).sample(frac=1, random_state=42)
X_train_bal = train_balanced[numeric_features]
y_train_bal = train_balanced[target]

print(f"✓ Undersampling: {len(class_0):,} → {len(class_0_down):,} (ratio {len(class_0_down)/len(class_1):.1f}:1)\n")

# ===========================================================================
# DEFINIR MODELOS
# ===========================================================================
models = {
    'Random Forest': RandomForestClassifier(n_estimators=300, max_depth=25, min_samples_split=30, 
                                            max_features='sqrt', random_state=42, n_jobs=-1),
    
    'Decision Tree': DecisionTreeClassifier(max_depth=20, min_samples_split=100, 
                                            min_samples_leaf=50, random_state=42),
    
    'KNN': KNeighborsClassifier(n_neighbors=31, weights='distance', n_jobs=-1),
    
    'SGD': SGDClassifier(loss='log_loss', alpha=0.001, max_iter=1000, random_state=42),
    
    'Gaussian NB': GaussianNB(),
    
    'Bernoulli NB': BernoulliNB(),
}

# ===========================================================================
# TREINAR E AVALIAR
# ===========================================================================
results = []
tscv_cv = TimeSeriesSplit(n_splits=5)

print(f"{'='*80}")
print(f"TREINAMENTO E AVALIAÇÃO")
print(f"{'='*80}\n")

for name, model in models.items():
    print(f"Treinando: {name}")
    start = time.time()
    
    # Preparar dados específicos
    X_tr, y_tr = X_train_bal.copy(), y_train_bal.copy()
    X_te = X_test.copy()
    
    # Normalização para modelos que precisam
    if name in ['KNN', 'SGD']:
        scaler = StandardScaler()
        X_tr = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index)
        X_te = pd.DataFrame(scaler.transform(X_te), columns=X_te.columns, index=X_te.index)
    elif name == 'Bernoulli NB':
        # Binarizar para Bernoulli
        X_tr = (X_tr > X_tr.median()).astype(int)
        X_te = (X_te > X_test.median()).astype(int)
    
    # Treinar
    model.fit(X_tr, y_tr)
    
    # Cross-validation
    try:
        cv_scores = cross_val_score(model, X_tr, y_tr, cv=tscv_cv, scoring='roc_auc', n_jobs=-1)
        cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
    except:
        cv_mean, cv_std = 0, 0
    
    # Predições
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_te)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_pred_proba = model.decision_function(X_te)
    else:
        y_pred_proba = model.predict(X_te)
    
    y_pred = model.predict(X_te)
    
    # Métricas
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    train_time = time.time() - start
    
    print(f"  CV: {cv_mean:.4f} ± {cv_std:.4f} | Test: {roc_auc:.4f} | "
          f"P: {precision:.3f} R: {recall:.3f} F1: {f1:.3f} | Time: {train_time:.1f}s\n")
    
    results.append({
        'model': name,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'time': train_time,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'cm': cm,
        'clf_report': classification_report(y_test, y_pred, zero_division=0)
    })

# ===========================================================================
# RANKING E MELHOR MODELO
# ===========================================================================
results_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['y_pred', 'y_pred_proba', 'cm', 'clf_report']} 
                           for r in results])
results_df = results_df.sort_values('roc_auc', ascending=False)

print(f"\n{'='*80}")
print(f"RANKING POR ROC-AUC")
print(f"{'='*80}\n")
print(f"{'Rank':<5} {'Model':<20} {'CV Mean':<10} {'Test':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
print(f"{'-'*75}")
for i, (_, row) in enumerate(results_df.iterrows(), 1):
    emoji = "🏆" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
    print(f"{emoji} {i:<3} {row['model']:<20} {row['cv_mean']:<10.4f} {row['roc_auc']:<10.4f} "
          f"{row['precision']:<10.3f} {row['recall']:<10.3f} {row['f1']:<10.3f}")

# Melhor modelo
best = results[results_df.index[0]]

print(f"\n{'='*80}")
print(f"MELHOR MODELO: {best['model']}")
print(f"{'='*80}\n")
print(f"CV ROC-AUC:  {best['cv_mean']:.4f} ± {best['cv_std']:.4f}")
print(f"Test ROC-AUC: {best['roc_auc']:.4f}")
print(f"Precision:    {best['precision']:.4f}")
print(f"Recall:       {best['recall']:.4f}")
print(f"F1-Score:     {best['f1']:.4f}")

print(f"\n{'='*70}")
print(f"CLASSIFICATION REPORT")
print(f"{'='*70}\n")
print(best['clf_report'])

print(f"{'='*70}")
print(f"CONFUSION MATRIX")
print(f"{'='*70}")
cm = best['cm']
print(f"                Predicted 0    Predicted 1")
print(f"Actual 0        {cm[0,0]:>11,}    {cm[0,1]:>11,}")
print(f"Actual 1        {cm[1,0]:>11,}    {cm[1,1]:>11,}")

# ===========================================================================
# VISUALIZAÇÕES
# ===========================================================================
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. ROC Curves
ax1 = fig.add_subplot(gs[0, :2])
for r in results:
    fpr, tpr, _ = roc_curve(y_test, r['y_pred_proba'])
    ax1.plot(fpr, tpr, linewidth=2, label=f"{r['model']} ({r['roc_auc']:.3f})")
ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax1.set_xlabel('False Positive Rate', fontweight='bold')
ax1.set_ylabel('True Positive Rate', fontweight='bold')
ax1.set_title('ROC Curves Comparison', fontweight='bold', fontsize=14)
ax1.legend(loc='lower right')
ax1.grid(alpha=0.3)

# 2. ROC-AUC Comparison
ax2 = fig.add_subplot(gs[0, 2])
models_list = results_df['model'].tolist()
roc_scores = results_df['roc_auc'].tolist()
colors = ['#2ecc71' if i == 0 else '#3498db' if i < 3 else '#95a5a6' for i in range(len(models_list))]
bars = ax2.barh(range(len(models_list)), roc_scores, color=colors, alpha=0.7)
ax2.set_yticks(range(len(models_list)))
ax2.set_yticklabels(models_list, fontsize=9)
ax2.set_xlabel('ROC-AUC', fontweight='bold')
ax2.set_title('Model Ranking', fontweight='bold')
ax2.grid(axis='x', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, roc_scores)):
    ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
             va='center', fontweight='bold', fontsize=8)

# 3. Precision-Recall-F1
ax3 = fig.add_subplot(gs[1, :])
x = np.arange(len(models_list))
width = 0.25
ax3.bar(x - width, results_df['precision'], width, label='Precision', alpha=0.8)
ax3.bar(x, results_df['recall'], width, label='Recall', alpha=0.8)
ax3.bar(x + width, results_df['f1'], width, label='F1-Score', alpha=0.8)
ax3.set_xlabel('Model', fontweight='bold')
ax3.set_ylabel('Score', fontweight='bold')
ax3.set_title('Precision, Recall, F1-Score Comparison', fontweight='bold', fontsize=14)
ax3.set_xticks(x)
ax3.set_xticklabels(models_list, rotation=45, ha='right')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# 4. Training Time
ax4 = fig.add_subplot(gs[2, 0])
times = results_df['time'].tolist()
ax4.bar(range(len(models_list)), times, color='coral', alpha=0.7)
ax4.set_xticks(range(len(models_list)))
ax4.set_xticklabels(models_list, rotation=45, ha='right', fontsize=8)
ax4.set_ylabel('Time (seconds)', fontweight='bold')
ax4.set_title('Training Time', fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

# 5. Best Model Confusion Matrix
ax5 = fig.add_subplot(gs[2, 1])
sns.heatmap(best['cm'], annot=True, fmt='d', cmap='Blues', ax=ax5, cbar=False)
ax5.set_title(f"Confusion Matrix: {best['model']}", fontweight='bold')
ax5.set_ylabel('True', fontweight='bold')
ax5.set_xlabel('Predicted', fontweight='bold')

# 6. CV vs Test
ax6 = fig.add_subplot(gs[2, 2])
cv_means = results_df['cv_mean'].tolist()
test_scores = results_df['roc_auc'].tolist()
cv_stds = results_df['cv_std'].tolist()
x = np.arange(len(models_list))
ax6.scatter(x, cv_means, s=100, label='CV Mean', alpha=0.7)
ax6.scatter(x, test_scores, s=100, label='Test', alpha=0.7)
ax6.errorbar(x, cv_means, yerr=cv_stds, fmt='none', color='gray', alpha=0.5)
ax6.set_xticks(x)
ax6.set_xticklabels(models_list, rotation=45, ha='right', fontsize=8)
ax6.set_ylabel('ROC-AUC', fontweight='bold')
ax6.set_title('CV vs Test Scores', fontweight='bold')
ax6.legend()
ax6.grid(alpha=0.3)

plt.savefig('visualizations/comparison/all_models_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ===========================================================================
# SALVAR RESULTADOS
# ===========================================================================
results_df.to_csv('reports/all_models_comparison.csv', index=False)

with open('reports/all_models_comparison_report.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("COMPARAÇÃO DE MODELOS - OPTIMIZED\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"{'Rank':<5} {'Model':<20} {'CV Mean':<10} {'Test':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}\n")
    f.write("-"*75 + "\n")
    for i, (_, row) in enumerate(results_df.iterrows(), 1):
        f.write(f"{i:<5} {row['model']:<20} {row['cv_mean']:<10.4f} {row['roc_auc']:<10.4f} "
                f"{row['precision']:<10.3f} {row['recall']:<10.3f} {row['f1']:<10.3f}\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write(f"BEST MODEL: {best['model']}\n")
    f.write("="*80 + "\n\n")
    f.write(f"CV ROC-AUC:  {best['cv_mean']:.4f} ± {best['cv_std']:.4f}\n")
    f.write(f"Test ROC-AUC: {best['roc_auc']:.4f}\n")
    f.write(f"Precision:    {best['precision']:.4f}\n")
    f.write(f"Recall:       {best['recall']:.4f}\n")
    f.write(f"F1-Score:     {best['f1']:.4f}\n\n")
    
    f.write("CLASSIFICATION REPORT\n")
    f.write("-"*80 + "\n")
    f.write(best['clf_report'] + "\n")
    
    f.write("CONFUSION MATRIX\n")
    f.write("-"*80 + "\n")
    f.write(f"                Predicted 0    Predicted 1\n")
    f.write(f"Actual 0        {cm[0,0]:>11,}    {cm[0,1]:>11,}\n")
    f.write(f"Actual 1        {cm[1,0]:>11,}    {cm[1,1]:>11,}\n")

print(f"\n✓ Relatório: reports/all_models_comparison_report.txt")
print(f"✓ Visualização: visualizations/comparison/all_models_comparison.png")
print(f"\n{'='*80}")
print(f"✅ COMPARAÇÃO COMPLETA")
print(f"{'='*80}\n")
