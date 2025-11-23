import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, 
                             recall_score, f1_score, confusion_matrix, roc_curve)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
import warnings
import time
from google.cloud import bigquery
import os
warnings.filterwarnings('ignore')

os.makedirs('visualizations', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# ===========================================================================
# NAIVE BAYES - COMPARAÇÃO DE VARIANTES
# ===========================================================================
print(f"\n{'='*80}")
print(f"NAIVE BAYES - COMPARAÇÃO DE VARIANTES")
print(f"{'='*80}")
print(f"Variantes testadas:")
print(f"  1️⃣ GaussianNB: Assume distribuição Gaussiana (features contínuas)")
print(f"  2️⃣ MultinomialNB: Para features de contagem/frequência (não-negativas)")
print(f"  3️⃣ BernoulliNB: Para features binárias (0/1)")
print(f"{'='*80}")

# ===========================================================================
# ETAPA 1: CARREGAR E PREPARAR DADOS
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 1: CARREGAR DADOS")
print(f"{'='*70}")

project_id = "proj-ml-469320"
client = bigquery.Client(project=project_id)

query = """
    SELECT * FROM `proj-ml-469320.app_cittamobi.dataset-updated` 
    TABLESAMPLE SYSTEM (20 PERCENT)
    LIMIT 50000
"""

print("Carregando dados...")
df = client.query(query).to_dataframe()
print(f"✓ Dados carregados: {len(df):,} registros")

target = "target"

df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])
df = df.sort_values('event_timestamp').reset_index(drop=True)
print(f"✓ Dados ordenados temporalmente")

# Features temporais básicas
df['hour'] = df['event_timestamp'].dt.hour
df['day_of_week'] = df['event_timestamp'].dt.dayofweek
df['day_of_month'] = df['event_timestamp'].dt.day
df['month'] = df['event_timestamp'].dt.month

# Features cíclicas
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

print(f"✓ Features temporais básicas criadas")

# ===========================================================================
# ETAPA 2: EXPANDING WINDOWS
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 2: EXPANDING WINDOWS (LEAK-FREE)")
print(f"{'='*70}")

df['user_hist_conversion_rate'] = 0.0
df['stop_hist_conversion_rate'] = 0.0
df['line_hist_conversion_rate'] = 0.0
df['user_hist_count'] = 0
df['stop_hist_count'] = 0
df['line_hist_count'] = 0
df['user_recency_days'] = 999
df['stop_recency_days'] = 999

print(f"📊 Calculando expanding windows...")
start_time = time.time()
sample_size = len(df)

for i in range(sample_size):
    if i % 5000 == 0 and i > 0:
        elapsed = time.time() - start_time
        eta = (elapsed / i) * (sample_size - i)
        print(f"   {i:,}/{sample_size:,} ({100*i/sample_size:.1f}%) - ETA: {eta/60:.1f} min")
    
    if i < 100:
        continue
    
    hist_data = df.iloc[:i].copy()
    current_row = df.iloc[i]
    
    user_hist = hist_data[hist_data['user_pseudo_id'] == current_row['user_pseudo_id']]
    if len(user_hist) > 0:
        df.at[i, 'user_hist_conversion_rate'] = user_hist[target].mean()
        df.at[i, 'user_hist_count'] = len(user_hist)
        last_event = user_hist['event_timestamp'].max()
        df.at[i, 'user_recency_days'] = (current_row['event_timestamp'] - last_event).days
    
    stop_hist = hist_data[hist_data['gtfs_stop_id'] == current_row['gtfs_stop_id']]
    if len(stop_hist) > 0:
        df.at[i, 'stop_hist_conversion_rate'] = stop_hist[target].mean()
        df.at[i, 'stop_hist_count'] = len(stop_hist)
        last_event = stop_hist['event_timestamp'].max()
        df.at[i, 'stop_recency_days'] = (current_row['event_timestamp'] - last_event).days
    
    if 'gtfs_route_id' in df.columns:
        line_hist = hist_data[hist_data['gtfs_route_id'] == current_row['gtfs_route_id']]
        if len(line_hist) > 0:
            df.at[i, 'line_hist_conversion_rate'] = line_hist[target].mean()
            df.at[i, 'line_hist_count'] = len(line_hist)

elapsed_time = time.time() - start_time
print(f"✓ Expanding windows criadas em {elapsed_time/60:.1f} minutos")

df['user_stop_interaction'] = df['user_hist_conversion_rate'] * df['stop_hist_conversion_rate']
df['user_line_interaction'] = df['user_hist_conversion_rate'] * df['line_hist_conversion_rate']
df['stop_line_interaction'] = df['stop_hist_conversion_rate'] * df['line_hist_conversion_rate']
print(f"✓ Features de interação criadas")

# ===========================================================================
# ETAPA 3: FEATURE ENGINEERING
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 3: FEATURE ENGINEERING")
print(f"{'='*70}")

# Features básicas do V3
if 'device_lat' in df.columns and 'device_lon' in df.columns:
    if 'stop_lat_event' in df.columns and 'stop_lon_event' in df.columns:
        df['geo_distance'] = np.sqrt(
            (df['device_lat'] - df['stop_lat_event'])**2 + 
            (df['device_lon'] - df['stop_lon_event'])**2
        )

if 'stop_event_rate' in df.columns and 'stop_total_samples' in df.columns:
    df['stop_rate_per_sample'] = df['stop_event_rate'] / (df['stop_total_samples'] + 1)

if 'stop_event_count' in df.columns and 'stop_total_samples' in df.columns:
    df['stop_density'] = df['stop_event_count'] / (df['stop_total_samples'] + 1)

df['time_period'] = pd.cut(df['hour'], 
                           bins=[0, 6, 9, 12, 14, 17, 19, 24],
                           labels=[0, 1, 2, 3, 4, 5, 6])
df['time_period'] = df['time_period'].astype(float)

df['hour_weekday_interaction'] = df['hour'] * df['day_of_week']
df['distance_from_peak'] = df['hour'].apply(lambda h: min(abs(h-8), abs(h-18)))

df['user_recency_score'] = np.exp(-df['user_recency_days'] / 7)
df['stop_recency_score'] = np.exp(-df['stop_recency_days'] / 7)

print(f"✓ Features criadas")

# ===========================================================================
# ETAPA 4: LIMPEZA E PREPARAÇÃO
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 4: LIMPEZA E PREPARAÇÃO")
print(f"{'='*70}")

# Tratar NaN
print(f"📊 Tratando valores NaN...")
before_nan = df.isna().sum().sum()
df = df.fillna(0)
after_nan = df.isna().sum().sum()
print(f"✓ NaN tratados: {before_nan:,} → {after_nan:,}")

if 'user_frequency' in df.columns:
    df = df[df['user_frequency'] >= 2].copy()
    print(f"✓ Filtro user_frequency aplicado")

if 'device_lat' in df.columns and 'device_lon' in df.columns:
    df = df[~((df['device_lat'].isna()) | (df['device_lon'].isna()))].copy()
    df = df[~((df['device_lat'] == 0) & (df['device_lon'] == 0))].copy()
    print(f"✓ Filtro coordenadas aplicado")

if 'dist_device_stop' in df.columns:
    df = df[df['dist_device_stop'] < df['dist_device_stop'].quantile(0.99)].copy()
    print(f"✓ Filtro outliers aplicado")

print(f"✓ Dados limpos: {len(df):,} registros")

# Selecionar features numéricas
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_features.remove(target)

id_cols = ['user_pseudo_id', 'gtfs_stop_id', 'gtfs_route_id', 'session_id']
for col in id_cols:
    if col in numeric_features:
        numeric_features.remove(col)
if 'event_timestamp' in numeric_features:
    numeric_features.remove('event_timestamp')

X = df[numeric_features].copy()
y = df[target].copy()

print(f"✓ Features totais: {len(numeric_features)}")

target_dist = y.value_counts()
print(f"\n=== Distribuição do Target ===")
for classe, count in target_dist.items():
    print(f"Classe {classe}: {count:,} ({100*count/len(y):.2f}%)")

# ===========================================================================
# ETAPA 5: DIVISÃO TEMPORAL
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 5: DIVISÃO TEMPORAL")
print(f"{'='*70}")

tscv = TimeSeriesSplit(n_splits=3)
splits = list(tscv.split(X))
train_idx, test_idx = splits[-1]
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print(f"✓ Train: {len(X_train):,} | Test: {len(X_test):,}")

# ===========================================================================
# ETAPA 6: PREPARAR DADOS PARA CADA VARIANTE
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 6: PREPARAR DADOS PARA CADA VARIANTE")
print(f"{'='*70}")

# 1. GaussianNB: Features contínuas (padrão)
X_train_gaussian = X_train.copy()
X_test_gaussian = X_test.copy()
print(f"✓ GaussianNB: {X_train_gaussian.shape[1]} features contínuas")

# 2. MultinomialNB: Requer features não-negativas
# Normalizar para [0, 1] e depois escalar para contagens
scaler_multinomial = MinMaxScaler()
X_train_multinomial = scaler_multinomial.fit_transform(X_train)
X_test_multinomial = scaler_multinomial.transform(X_test)
# Multiplicar por 100 para simular "contagens"
X_train_multinomial = X_train_multinomial * 100
X_test_multinomial = X_test_multinomial * 100
print(f"✓ MultinomialNB: {X_train_multinomial.shape[1]} features escaladas [0, 100]")

# 3. BernoulliNB: Binarizar features
# Usar mediana como threshold
X_train_bernoulli = (X_train > X_train.median()).astype(int)
X_test_bernoulli = (X_test > X_train.median()).astype(int)
print(f"✓ BernoulliNB: {X_train_bernoulli.shape[1]} features binárias (0/1)")

# ===========================================================================
# ETAPA 7: TREINAR VARIANTES
# ===========================================================================
print(f"\n{'='*80}")
print(f"ETAPA 7: TREINAR E COMPARAR VARIANTES")
print(f"{'='*80}")

results = []

# ----------------------------
# 1. GAUSSIAN NB
# ----------------------------
print(f"\n{'='*70}")
print(f"1. GAUSSIAN NB (Distribuição Gaussiana)")
print(f"{'='*70}")

start_time = time.time()
model_gaussian = GaussianNB()
model_gaussian.fit(X_train_gaussian, y_train)
y_pred_proba_gaussian = model_gaussian.predict_proba(X_test_gaussian)[:, 1]
y_pred_gaussian = model_gaussian.predict(X_test_gaussian)
train_time_gaussian = time.time() - start_time

roc_auc_gaussian = roc_auc_score(y_test, y_pred_proba_gaussian)
accuracy_gaussian = accuracy_score(y_test, y_pred_gaussian)
precision_gaussian = precision_score(y_test, y_pred_gaussian, zero_division=0)
recall_gaussian = recall_score(y_test, y_pred_gaussian)
f1_gaussian = f1_score(y_test, y_pred_gaussian)
f1_macro_gaussian = f1_score(y_test, y_pred_gaussian, average='macro')
cm_gaussian = confusion_matrix(y_test, y_pred_gaussian)

print(f"\n📊 GAUSSIAN NB:")
print(f"   ROC-AUC:      {roc_auc_gaussian:.4f}")
print(f"   Accuracy:     {accuracy_gaussian:.4f}")
print(f"   Precision:    {precision_gaussian:.4f}")
print(f"   Recall:       {recall_gaussian:.4f}")
print(f"   F1-Score:     {f1_gaussian:.4f}")
print(f"   F1-Macro:     {f1_macro_gaussian:.4f}")
print(f"   Tempo:        {train_time_gaussian:.3f}s")

results.append({
    'model': 'GaussianNB',
    'roc_auc': roc_auc_gaussian,
    'accuracy': accuracy_gaussian,
    'precision': precision_gaussian,
    'recall': recall_gaussian,
    'f1': f1_gaussian,
    'f1_macro': f1_macro_gaussian,
    'train_time': train_time_gaussian,
    'confusion_matrix': cm_gaussian
})

# ----------------------------
# 2. MULTINOMIAL NB
# ----------------------------
print(f"\n{'='*70}")
print(f"2. MULTINOMIAL NB (Features de Contagem)")
print(f"{'='*70}")

start_time = time.time()
model_multinomial = MultinomialNB(alpha=1.0)  # Laplace smoothing
model_multinomial.fit(X_train_multinomial, y_train)
y_pred_proba_multinomial = model_multinomial.predict_proba(X_test_multinomial)[:, 1]
y_pred_multinomial = model_multinomial.predict(X_test_multinomial)
train_time_multinomial = time.time() - start_time

roc_auc_multinomial = roc_auc_score(y_test, y_pred_proba_multinomial)
accuracy_multinomial = accuracy_score(y_test, y_pred_multinomial)
precision_multinomial = precision_score(y_test, y_pred_multinomial, zero_division=0)
recall_multinomial = recall_score(y_test, y_pred_multinomial)
f1_multinomial = f1_score(y_test, y_pred_multinomial)
f1_macro_multinomial = f1_score(y_test, y_pred_multinomial, average='macro')
cm_multinomial = confusion_matrix(y_test, y_pred_multinomial)

print(f"\n📊 MULTINOMIAL NB:")
print(f"   ROC-AUC:      {roc_auc_multinomial:.4f}")
print(f"   Accuracy:     {accuracy_multinomial:.4f}")
print(f"   Precision:    {precision_multinomial:.4f}")
print(f"   Recall:       {recall_multinomial:.4f}")
print(f"   F1-Score:     {f1_multinomial:.4f}")
print(f"   F1-Macro:     {f1_macro_multinomial:.4f}")
print(f"   Tempo:        {train_time_multinomial:.3f}s")

results.append({
    'model': 'MultinomialNB',
    'roc_auc': roc_auc_multinomial,
    'accuracy': accuracy_multinomial,
    'precision': precision_multinomial,
    'recall': recall_multinomial,
    'f1': f1_multinomial,
    'f1_macro': f1_macro_multinomial,
    'train_time': train_time_multinomial,
    'confusion_matrix': cm_multinomial
})

# ----------------------------
# 3. BERNOULLI NB
# ----------------------------
print(f"\n{'='*70}")
print(f"3. BERNOULLI NB (Features Binárias)")
print(f"{'='*70}")

start_time = time.time()
model_bernoulli = BernoulliNB(alpha=1.0)  # Laplace smoothing
model_bernoulli.fit(X_train_bernoulli, y_train)
y_pred_proba_bernoulli = model_bernoulli.predict_proba(X_test_bernoulli)[:, 1]
y_pred_bernoulli = model_bernoulli.predict(X_test_bernoulli)
train_time_bernoulli = time.time() - start_time

roc_auc_bernoulli = roc_auc_score(y_test, y_pred_proba_bernoulli)
accuracy_bernoulli = accuracy_score(y_test, y_pred_bernoulli)
precision_bernoulli = precision_score(y_test, y_pred_bernoulli, zero_division=0)
recall_bernoulli = recall_score(y_test, y_pred_bernoulli)
f1_bernoulli = f1_score(y_test, y_pred_bernoulli)
f1_macro_bernoulli = f1_score(y_test, y_pred_bernoulli, average='macro')
cm_bernoulli = confusion_matrix(y_test, y_pred_bernoulli)

print(f"\n📊 BERNOULLI NB:")
print(f"   ROC-AUC:      {roc_auc_bernoulli:.4f}")
print(f"   Accuracy:     {accuracy_bernoulli:.4f}")
print(f"   Precision:    {precision_bernoulli:.4f}")
print(f"   Recall:       {recall_bernoulli:.4f}")
print(f"   F1-Score:     {f1_bernoulli:.4f}")
print(f"   F1-Macro:     {f1_macro_bernoulli:.4f}")
print(f"   Tempo:        {train_time_bernoulli:.3f}s")

results.append({
    'model': 'BernoulliNB',
    'roc_auc': roc_auc_bernoulli,
    'accuracy': accuracy_bernoulli,
    'precision': precision_bernoulli,
    'recall': recall_bernoulli,
    'f1': f1_bernoulli,
    'f1_macro': f1_macro_bernoulli,
    'train_time': train_time_bernoulli,
    'confusion_matrix': cm_bernoulli
})

# ===========================================================================
# ETAPA 8: COMPARAÇÃO FINAL
# ===========================================================================
print(f"\n{'='*80}")
print(f"ETAPA 8: COMPARAÇÃO FINAL")
print(f"{'='*80}")

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('roc_auc', ascending=False)

print(f"\n📊 RANKING POR ROC-AUC:")
print(f"{'='*80}")
print(f"{'Modelo':<20} | {'ROC-AUC':<10} | {'F1-Macro':<10} | {'Precision':<10} | {'Recall':<10} | {'Tempo (s)':<10}")
print(f"{'='*80}")
for _, row in results_df.iterrows():
    print(f"{row['model']:<20} | {row['roc_auc']:<10.4f} | {row['f1_macro']:<10.4f} | "
          f"{row['precision']:<10.4f} | {row['recall']:<10.4f} | {row['train_time']:<10.3f}")

best_model = results_df.iloc[0]['model']
best_roc_auc = results_df.iloc[0]['roc_auc']
print(f"\n🏆 MELHOR MODELO: {best_model} (ROC-AUC = {best_roc_auc:.4f})")

# ===========================================================================
# ETAPA 9: VISUALIZAÇÕES
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 9: VISUALIZAÇÕES")
print(f"{'='*70}")

# 1. Comparação de Métricas
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

models = [r['model'] for r in results]
colors = ['steelblue', 'orange', 'forestgreen']

# ROC-AUC
ax1 = axes[0, 0]
roc_aucs = [r['roc_auc'] for r in results]
bars = ax1.bar(models, roc_aucs, color=colors, alpha=0.7)
ax1.set_ylabel('ROC-AUC', fontweight='bold')
ax1.set_title('ROC-AUC: Comparação Naive Bayes', fontweight='bold')
ax1.set_ylim(0.5, 0.9)
ax1.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, roc_aucs):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.4f}', 
             ha='center', va='bottom', fontweight='bold', fontsize=9)

# F1-Macro
ax2 = axes[0, 1]
f1_macros = [r['f1_macro'] for r in results]
bars = ax2.bar(models, f1_macros, color=colors, alpha=0.7)
ax2.set_ylabel('F1-Macro', fontweight='bold')
ax2.set_title('F1-Macro: Comparação Naive Bayes', fontweight='bold')
ax2.set_ylim(0.5, 0.8)
ax2.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, f1_macros):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.4f}', 
             ha='center', va='bottom', fontweight='bold', fontsize=9)

# Precision vs Recall
ax3 = axes[1, 0]
precisions = [r['precision'] for r in results]
recalls = [r['recall'] for r in results]
x = np.arange(len(models))
width = 0.35
bars1 = ax3.bar(x - width/2, precisions, width, label='Precision', color='skyblue', alpha=0.7)
bars2 = ax3.bar(x + width/2, recalls, width, label='Recall', color='salmon', alpha=0.7)
ax3.set_ylabel('Score', fontweight='bold')
ax3.set_title('Precision vs Recall', fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(models)
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# Tempo de Treinamento
ax4 = axes[1, 1]
train_times = [r['train_time'] for r in results]
bars = ax4.bar(models, train_times, color=colors, alpha=0.7)
ax4.set_ylabel('Tempo (segundos)', fontweight='bold')
ax4.set_title('Tempo de Treinamento', fontweight='bold')
ax4.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, train_times):
    ax4.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}s', 
             ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('visualizations/naive_bayes_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Comparação salva: visualizations/naive_bayes_comparison.png")

# 2. ROC Curves
fpr_gaussian, tpr_gaussian, _ = roc_curve(y_test, y_pred_proba_gaussian)
fpr_multinomial, tpr_multinomial, _ = roc_curve(y_test, y_pred_proba_multinomial)
fpr_bernoulli, tpr_bernoulli, _ = roc_curve(y_test, y_pred_proba_bernoulli)

plt.figure(figsize=(10, 8))
plt.plot(fpr_gaussian, tpr_gaussian, linewidth=2, 
         label=f'GaussianNB (AUC = {roc_auc_gaussian:.4f})', color='steelblue')
plt.plot(fpr_multinomial, tpr_multinomial, linewidth=2, 
         label=f'MultinomialNB (AUC = {roc_auc_multinomial:.4f})', color='orange')
plt.plot(fpr_bernoulli, tpr_bernoulli, linewidth=2, 
         label=f'BernoulliNB (AUC = {roc_auc_bernoulli:.4f})', color='forestgreen')
plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random', alpha=0.5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
plt.title('ROC Curves: Naive Bayes Variants', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/roc_curves_naive_bayes.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ ROC Curves salvas: visualizations/roc_curves_naive_bayes.png")

# 3. Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

cms = [cm_gaussian, cm_multinomial, cm_bernoulli]
titles = ['GaussianNB', 'MultinomialNB', 'BernoulliNB']

for ax, cm, title in zip(axes, cms, titles):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_title(f'Confusion Matrix: {title}', fontweight='bold')
    ax.set_ylabel('True Label', fontweight='bold')
    ax.set_xlabel('Predicted Label', fontweight='bold')

plt.tight_layout()
plt.savefig('visualizations/confusion_matrices_naive_bayes.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Confusion Matrices salvas: visualizations/confusion_matrices_naive_bayes.png")

# ===========================================================================
# ETAPA 10: SALVAR RESULTADOS
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 10: SALVAR RESULTADOS")
print(f"{'='*70}")

# Salvar comparação
results_df.to_csv('reports/naive_bayes_comparison.csv', index=False)
print(f"✓ Comparação salva: reports/naive_bayes_comparison.csv")

# Relatório detalhado
with open('reports/naive_bayes_report.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("NAIVE BAYES - COMPARAÇÃO DE VARIANTES\n")
    f.write("="*80 + "\n\n")
    
    f.write("VARIANTES TESTADAS\n")
    f.write("-"*80 + "\n")
    f.write("1. GaussianNB:\n")
    f.write("   - Assume distribuição Gaussiana (Normal) para features contínuas\n")
    f.write("   - Melhor para features com valores reais\n")
    f.write("   - Não requer normalização\n\n")
    
    f.write("2. MultinomialNB:\n")
    f.write("   - Projetado para features de contagem/frequência\n")
    f.write("   - Requer features não-negativas\n")
    f.write("   - Features escaladas [0, 100] para simular contagens\n")
    f.write("   - Alpha=1.0 (Laplace smoothing)\n\n")
    
    f.write("3. BernoulliNB:\n")
    f.write("   - Projetado para features binárias (0/1)\n")
    f.write("   - Features binarizadas usando mediana como threshold\n")
    f.write("   - Alpha=1.0 (Laplace smoothing)\n\n")
    
    f.write("RESULTADOS COMPARATIVOS\n")
    f.write("-"*80 + "\n")
    f.write(f"{'Modelo':<20} | {'ROC-AUC':<10} | {'F1-Macro':<10} | {'Precision':<10} | {'Recall':<10} | {'Tempo':<10}\n")
    f.write("-"*80 + "\n")
    for _, row in results_df.iterrows():
        f.write(f"{row['model']:<20} | {row['roc_auc']:<10.4f} | {row['f1_macro']:<10.4f} | "
                f"{row['precision']:<10.4f} | {row['recall']:<10.4f} | {row['train_time']:<10.3f}s\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write(f"MELHOR MODELO: {best_model}\n")
    f.write("="*80 + "\n")
    best_result = results_df.iloc[0]
    f.write(f"ROC-AUC:       {best_result['roc_auc']:.4f}\n")
    f.write(f"Accuracy:      {best_result['accuracy']:.4f}\n")
    f.write(f"Precision:     {best_result['precision']:.4f}\n")
    f.write(f"Recall:        {best_result['recall']:.4f}\n")
    f.write(f"F1-Score:      {best_result['f1']:.4f}\n")
    f.write(f"F1-Macro:      {best_result['f1_macro']:.4f}\n")
    f.write(f"Tempo:         {best_result['train_time']:.3f}s\n\n")
    
    f.write("MATRIZ DE CONFUSÃO - MELHOR MODELO\n")
    f.write("-"*80 + "\n")
    best_cm = best_result['confusion_matrix']
    f.write(f"True Negatives:    {best_cm[0,0]:,}\n")
    f.write(f"False Positives:   {best_cm[0,1]:,}\n")
    f.write(f"False Negatives:   {best_cm[1,0]:,}\n")
    f.write(f"True Positives:    {best_cm[1,1]:,}\n\n")
    
    f.write("ANÁLISE\n")
    f.write("-"*80 + "\n")
    f.write("1. Performance:\n")
    if roc_auc_gaussian > roc_auc_multinomial and roc_auc_gaussian > roc_auc_bernoulli:
        f.write("   ✓ GaussianNB teve melhor desempenho - features contínuas são adequadas\n")
    elif roc_auc_multinomial > roc_auc_gaussian and roc_auc_multinomial > roc_auc_bernoulli:
        f.write("   ✓ MultinomialNB teve melhor desempenho - estrutura de contagem funciona bem\n")
    else:
        f.write("   ✓ BernoulliNB teve melhor desempenho - binarização captura padrões relevantes\n")
    
    f.write("\n2. Velocidade:\n")
    fastest = results_df.iloc[0]['model'] if results_df.iloc[0]['train_time'] == min(train_times) else results_df.iloc[-1]['model']
    f.write(f"   ✓ Todos os modelos são extremamente rápidos (<1s)\n")
    f.write(f"   ✓ Naive Bayes é ideal para prototipagem rápida\n")
    
    f.write("\n3. Trade-offs:\n")
    f.write("   - Precision vs Recall: Ajustar threshold para otimizar\n")
    f.write("   - GaussianNB: Simples mas assume independência\n")
    f.write("   - MultinomialNB: Requer transformação de features\n")
    f.write("   - BernoulliNB: Perda de informação na binarização\n")

print(f"✓ Relatório salvo: reports/naive_bayes_report.txt")

# ===========================================================================
# COMPARAÇÃO COM OUTROS MODELOS
# ===========================================================================
print(f"\n{'='*80}")
print(f"COMPARAÇÃO COM OUTROS MODELOS DO PROJETO")
print(f"{'='*80}")

# Benchmarks de outros modelos (do projeto)
other_models = {
    'KNN': 0.7542,
    'SGD V2': 0.7917,
    'SGD V3': 0.8027,
    'CatBoost': 0.8669,
    'LightGBM': 0.8642
}

print(f"\n📊 CONTEXTO:")
print(f"{'='*70}")
print(f"{'Modelo':<20} | {'ROC-AUC':<10} | {'Posição':<15}")
print(f"{'='*70}")

all_models = {**{r['model']: r['roc_auc'] for r in results}, **other_models}
sorted_models = sorted(all_models.items(), key=lambda x: x[1], reverse=True)

for i, (model, auc) in enumerate(sorted_models, 1):
    marker = "🏆" if i == 1 else "  "
    print(f"{marker} {model:<18} | {auc:<10.4f} | #{i}")

print(f"\n💡 INSIGHTS:")
print(f"   • Naive Bayes: Rápido mas performance limitada")
print(f"   • Trade-off: Velocidade vs Precisão")
print(f"   • Bom para baseline e prototipagem")

# ===========================================================================
# CONCLUSÃO
# ===========================================================================
print(f"\n{'='*80}")
print(f"✅ NAIVE BAYES - ANÁLISE CONCLUÍDA!")
print(f"{'='*80}")

print(f"\n🎯 RESULTADOS:")
print(f"   GaussianNB:     {roc_auc_gaussian:.4f} ROC-AUC")
print(f"   MultinomialNB:  {roc_auc_multinomial:.4f} ROC-AUC")
print(f"   BernoulliNB:    {roc_auc_bernoulli:.4f} ROC-AUC")

print(f"\n🏆 MELHOR: {best_model} ({best_roc_auc:.4f} ROC-AUC)")

print(f"\n⚡ VELOCIDADE:")
avg_time = np.mean(train_times)
print(f"   Média: {avg_time:.3f}s (extremamente rápido!)")

print(f"\n📁 Arquivos gerados:")
print(f"   - visualizations/naive_bayes_comparison.png")
print(f"   - visualizations/roc_curves_naive_bayes.png")
print(f"   - visualizations/confusion_matrices_naive_bayes.png")
print(f"   - reports/naive_bayes_comparison.csv")
print(f"   - reports/naive_bayes_report.txt")

print(f"\n✅ ANÁLISE COMPLETA!")
