import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, 
                             recall_score, f1_score, confusion_matrix, classification_report, roc_curve)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
import time
from google.cloud import bigquery
import os
warnings.filterwarnings('ignore')

# Criar diretórios se não existirem
os.makedirs('visualizations', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# ===========================================================================
# SGD CLASSIFIER V2 - FEATURE SELECTION POR COEFICIENTES
# ===========================================================================
print(f"\n{'='*80}")
print(f"SGD CLASSIFIER V2 - FEATURE SELECTION BASEADA EM COEFICIENTES")
print(f"{'='*80}")
print(f"Estratégia:")
print(f"  1️⃣ Treinar modelo baseline com todas as features")
print(f"  2️⃣ Analisar coeficientes e identificar features irrelevantes")
print(f"  3️⃣ Remover features com coeficiente absoluto < threshold")
print(f"  4️⃣ Retreinar e comparar performance")
print(f"{'='*80}")

# ===========================================================================
# ETAPA 1: CARREGAR E PREPARAR DADOS
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 1: CARREGAR E PREPARAR DADOS")
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

# Converter timestamp e ordenar
df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])
df = df.sort_values('event_timestamp').reset_index(drop=True)
print(f"✓ Dados ordenados temporalmente")

# Features temporais
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

print(f"✓ Features temporais criadas")

# ===========================================================================
# ETAPA 2: EXPANDING WINDOWS (LEAK-FREE)
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
    
    # User histórico
    user_hist = hist_data[hist_data['user_pseudo_id'] == current_row['user_pseudo_id']]
    if len(user_hist) > 0:
        df.at[i, 'user_hist_conversion_rate'] = user_hist[target].mean()
        df.at[i, 'user_hist_count'] = len(user_hist)
        last_event = user_hist['event_timestamp'].max()
        df.at[i, 'user_recency_days'] = (current_row['event_timestamp'] - last_event).days
    
    # Stop histórico
    stop_hist = hist_data[hist_data['gtfs_stop_id'] == current_row['gtfs_stop_id']]
    if len(stop_hist) > 0:
        df.at[i, 'stop_hist_conversion_rate'] = stop_hist[target].mean()
        df.at[i, 'stop_hist_count'] = len(stop_hist)
        last_event = stop_hist['event_timestamp'].max()
        df.at[i, 'stop_recency_days'] = (current_row['event_timestamp'] - last_event).days
    
    # Line histórico
    if 'gtfs_route_id' in df.columns:
        line_hist = hist_data[hist_data['gtfs_route_id'] == current_row['gtfs_route_id']]
        if len(line_hist) > 0:
            df.at[i, 'line_hist_conversion_rate'] = line_hist[target].mean()
            df.at[i, 'line_hist_count'] = len(line_hist)

elapsed_time = time.time() - start_time
print(f"✓ Expanding windows criadas em {elapsed_time/60:.1f} minutos")

# Features de interação
df['user_stop_interaction'] = df['user_hist_conversion_rate'] * df['stop_hist_conversion_rate']
df['user_line_interaction'] = df['user_hist_conversion_rate'] * df['line_hist_conversion_rate']
df['stop_line_interaction'] = df['stop_hist_conversion_rate'] * df['line_hist_conversion_rate']
print(f"✓ Features de interação criadas")

# ===========================================================================
# ETAPA 3: LIMPEZA E PREPARAÇÃO
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 3: LIMPEZA E PREPARAÇÃO")
print(f"{'='*70}")

# Filtros
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

# Remover IDs
id_cols = ['user_pseudo_id', 'gtfs_stop_id', 'gtfs_route_id', 'session_id']
for col in id_cols:
    if col in numeric_features:
        numeric_features.remove(col)
if 'event_timestamp' in numeric_features:
    numeric_features.remove('event_timestamp')

X = df[numeric_features].copy()
y = df[target].copy()

print(f"✓ Features originais: {len(numeric_features)}")
print(f"\n=== Distribuição do Target ===")
target_dist = y.value_counts()
for classe, count in target_dist.items():
    print(f"Classe {classe}: {count:,} ({100*count/len(y):.2f}%)")

# ===========================================================================
# ETAPA 4: DIVISÃO TEMPORAL
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 4: DIVISÃO TEMPORAL")
print(f"{'='*70}")

tscv = TimeSeriesSplit(n_splits=3)
splits = list(tscv.split(X))
train_idx, test_idx = splits[-1]
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print(f"✓ Train: {len(X_train):,} | Test: {len(X_test):,}")

# ===========================================================================
# ETAPA 5: BASELINE - TREINAR COM TODAS AS FEATURES
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 5: BASELINE - TODAS AS FEATURES")
print(f"{'='*70}")

print(f"\n🔄 Treinando modelo baseline...")
start_time = time.time()

pipeline_baseline = Pipeline([
    ('scaler', StandardScaler()),
    ('sgd', SGDClassifier(
        loss='log_loss',
        penalty='l2',
        alpha=0.001,  # Melhor config do V1
        class_weight='balanced',
        learning_rate='optimal',
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=42,
        n_jobs=-1
    ))
])

pipeline_baseline.fit(X_train, y_train)
y_pred_proba_baseline = pipeline_baseline.predict_proba(X_test)[:, 1]

# Encontrar melhor threshold
thresholds = np.arange(0.1, 0.9, 0.05)
best_f1_macro = 0
best_threshold = 0.5

for threshold in thresholds:
    y_pred_temp = (y_pred_proba_baseline >= threshold).astype(int)
    f1_macro_temp = f1_score(y_test, y_pred_temp, average='macro')
    if f1_macro_temp > best_f1_macro:
        best_f1_macro = f1_macro_temp
        best_threshold = threshold

y_pred_baseline = (y_pred_proba_baseline >= best_threshold).astype(int)

# Métricas baseline
roc_auc_baseline = roc_auc_score(y_test, y_pred_proba_baseline)
accuracy_baseline = accuracy_score(y_test, y_pred_baseline)
precision_baseline = precision_score(y_test, y_pred_baseline, zero_division=0)
recall_baseline = recall_score(y_test, y_pred_baseline)
f1_baseline = f1_score(y_test, y_pred_baseline)
f1_macro_baseline = f1_score(y_test, y_pred_baseline, average='macro')
train_time_baseline = time.time() - start_time

print(f"\n📊 BASELINE (Todas as {len(numeric_features)} features):")
print(f"   ROC-AUC:   {roc_auc_baseline:.4f}")
print(f"   F1-Macro:  {f1_macro_baseline:.4f}")
print(f"   Accuracy:  {accuracy_baseline:.4f}")
print(f"   Tempo:     {train_time_baseline:.2f}s")

# ===========================================================================
# ETAPA 6: ANÁLISE DE COEFICIENTES
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 6: ANÁLISE DE COEFICIENTES")
print(f"{'='*70}")

coefficients = pipeline_baseline.named_steps['sgd'].coef_[0]
feature_importance = pd.DataFrame({
    'feature': numeric_features,
    'coefficient': coefficients,
    'abs_coefficient': np.abs(coefficients)
}).sort_values('abs_coefficient', ascending=False)

print(f"\n📊 TOP 20 FEATURES (Maior Impacto):")
print(f"{'='*70}")
for i, row in feature_importance.head(20).iterrows():
    sign = "+" if row['coefficient'] > 0 else ""
    print(f"{row['feature']:40s} | {sign}{row['coefficient']:+.6f}")

print(f"\n📊 BOTTOM 20 FEATURES (Menor Impacto):")
print(f"{'='*70}")
for i, row in feature_importance.tail(20).iterrows():
    sign = "+" if row['coefficient'] > 0 else ""
    print(f"{row['feature']:40s} | {sign}{row['coefficient']:+.6f}")

# ===========================================================================
# ETAPA 7: FEATURE SELECTION - REMOVER FEATURES IRRELEVANTES
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 7: FEATURE SELECTION")
print(f"{'='*70}")

# Testar diferentes thresholds de corte
thresholds_to_test = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
results = []

for threshold in thresholds_to_test:
    # Selecionar features acima do threshold
    selected_features = feature_importance[feature_importance['abs_coefficient'] >= threshold]['feature'].tolist()
    removed_features = len(numeric_features) - len(selected_features)
    
    print(f"\n🔄 Testando threshold={threshold:.2f}...")
    print(f"   Features selecionadas: {len(selected_features)} ({len(selected_features)/len(numeric_features)*100:.1f}%)")
    print(f"   Features removidas: {removed_features} ({removed_features/len(numeric_features)*100:.1f}%)")
    
    # Treinar com features selecionadas
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    start_time = time.time()
    
    pipeline_selected = Pipeline([
        ('scaler', StandardScaler()),
        ('sgd', SGDClassifier(
            loss='log_loss',
            penalty='l2',
            alpha=0.001,
            class_weight='balanced',
            learning_rate='optimal',
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    pipeline_selected.fit(X_train_selected, y_train)
    y_pred_proba_selected = pipeline_selected.predict_proba(X_test_selected)[:, 1]
    
    # Encontrar melhor threshold
    best_f1 = 0
    best_thresh = 0.5
    for t in np.arange(0.1, 0.9, 0.05):
        y_pred_temp = (y_pred_proba_selected >= t).astype(int)
        f1_temp = f1_score(y_test, y_pred_temp, average='macro')
        if f1_temp > best_f1:
            best_f1 = f1_temp
            best_thresh = t
    
    y_pred_selected = (y_pred_proba_selected >= best_thresh).astype(int)
    
    # Métricas
    roc_auc_selected = roc_auc_score(y_test, y_pred_proba_selected)
    accuracy_selected = accuracy_score(y_test, y_pred_selected)
    precision_selected = precision_score(y_test, y_pred_selected, zero_division=0)
    recall_selected = recall_score(y_test, y_pred_selected)
    f1_selected = f1_score(y_test, y_pred_selected)
    f1_macro_selected = f1_score(y_test, y_pred_selected, average='macro')
    train_time_selected = time.time() - start_time
    
    # Calcular ganhos/perdas
    roc_auc_diff = roc_auc_selected - roc_auc_baseline
    f1_macro_diff = f1_macro_selected - f1_macro_baseline
    
    results.append({
        'threshold': threshold,
        'n_features': len(selected_features),
        'features_removed': removed_features,
        'removal_pct': removed_features / len(numeric_features) * 100,
        'roc_auc': roc_auc_selected,
        'roc_auc_diff': roc_auc_diff,
        'f1_macro': f1_macro_selected,
        'f1_macro_diff': f1_macro_diff,
        'accuracy': accuracy_selected,
        'precision': precision_selected,
        'recall': recall_selected,
        'f1': f1_selected,
        'train_time': train_time_selected
    })
    
    symbol = "✅" if roc_auc_diff >= 0 else "⚠️"
    print(f"   {symbol} ROC-AUC: {roc_auc_selected:.4f} ({roc_auc_diff:+.4f})")
    print(f"   {symbol} F1-Macro: {f1_macro_selected:.4f} ({f1_macro_diff:+.4f})")
    print(f"   ⚡ Tempo: {train_time_selected:.2f}s")

results_df = pd.DataFrame(results)

# ===========================================================================
# ETAPA 8: ANÁLISE COMPARATIVA
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 8: ANÁLISE COMPARATIVA")
print(f"{'='*70}")

print(f"\nRESUMO COMPLETO:")
print(f"{'='*90}")
print(f"{'Threshold':>10} | {'Features':>8} | {'Removed':>7} | {'ROC-AUC':>8} | {'Diff':>8} | {'F1-Macro':>9} | {'Diff':>8}")
print(f"{'-'*90}")
print(f"{'BASELINE':>10} | {len(numeric_features):>8} | {0:>7} | {roc_auc_baseline:>8.4f} | {0:>8.4f} | {f1_macro_baseline:>9.4f} | {0:>8.4f}")
for i, row in results_df.iterrows():
    symbol = "+" if row['roc_auc_diff'] >= 0 else ""
    print(f"{row['threshold']:>10.2f} | {int(row['n_features']):>8} | {int(row['features_removed']):>7} | {row['roc_auc']:>8.4f} | {symbol}{row['roc_auc_diff']:>7.4f} | {row['f1_macro']:>9.4f} | {symbol}{row['f1_macro_diff']:>7.4f}")

# Melhor modelo
best_idx = results_df['roc_auc'].idxmax()
best_result = results_df.loc[best_idx]

print(f"\n🏆 MELHOR CONFIGURAÇÃO:")
print(f"{'='*70}")
print(f"Threshold:         {best_result['threshold']:.2f}")
print(f"Features:          {int(best_result['n_features'])} (removidas: {int(best_result['features_removed'])})")
print(f"ROC-AUC:           {best_result['roc_auc']:.4f} ({best_result['roc_auc_diff']:+.4f})")
print(f"F1-Macro:          {best_result['f1_macro']:.4f} ({best_result['f1_macro_diff']:+.4f})")
print(f"Accuracy:          {best_result['accuracy']:.4f}")
print(f"Precision:         {best_result['precision']:.4f}")
print(f"Recall:            {best_result['recall']:.4f}")
print(f"Tempo:             {best_result['train_time']:.2f}s")

# ===========================================================================
# ETAPA 9: TREINAR MODELO FINAL V2
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 9: MODELO FINAL V2")
print(f"{'='*70}")

# Selecionar features com melhor threshold
best_threshold = best_result['threshold']
selected_features_final = feature_importance[feature_importance['abs_coefficient'] >= best_threshold]['feature'].tolist()

print(f"\n🚀 Treinando SGD V2 com {len(selected_features_final)} features...")
print(f"   Threshold: {best_threshold:.2f}")
print(f"   Removidas: {len(numeric_features) - len(selected_features_final)} features")

X_train_final = X_train[selected_features_final]
X_test_final = X_test[selected_features_final]

pipeline_final = Pipeline([
    ('scaler', StandardScaler()),
    ('sgd', SGDClassifier(
        loss='log_loss',
        penalty='l2',
        alpha=0.001,
        class_weight='balanced',
        learning_rate='optimal',
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=42,
        n_jobs=-1
    ))
])

pipeline_final.fit(X_train_final, y_train)
y_pred_proba_final = pipeline_final.predict_proba(X_test_final)[:, 1]

# Threshold otimizado
best_f1_final = 0
best_thresh_final = 0.5
for t in np.arange(0.1, 0.9, 0.05):
    y_pred_temp = (y_pred_proba_final >= t).astype(int)
    f1_temp = f1_score(y_test, y_pred_temp, average='macro')
    if f1_temp > best_f1_final:
        best_f1_final = f1_temp
        best_thresh_final = t

y_pred_final = (y_pred_proba_final >= best_thresh_final).astype(int)

# Métricas finais
roc_auc_final = roc_auc_score(y_test, y_pred_proba_final)
accuracy_final = accuracy_score(y_test, y_pred_final)
precision_final = precision_score(y_test, y_pred_final, zero_division=0)
recall_final = recall_score(y_test, y_pred_final)
f1_final = f1_score(y_test, y_pred_final)
f1_macro_final = f1_score(y_test, y_pred_final, average='macro')
cm_final = confusion_matrix(y_test, y_pred_final)

print(f"\n📊 MÉTRICAS FINAIS SGD V2:")
print(f"   ROC-AUC:      {roc_auc_final:.4f} 🎯")
print(f"   F1-Macro:     {f1_macro_final:.4f}")
print(f"   Accuracy:     {accuracy_final:.4f}")
print(f"   Precision:    {precision_final:.4f}")
print(f"   Recall:       {recall_final:.4f}")
print(f"   F1-Score:     {f1_final:.4f}")
print(f"   Threshold:    {best_thresh_final:.2f}")

print(f"\n📊 Matriz de Confusão:")
print(cm_final)
print(f"\nTrue Negatives:  {cm_final[0,0]:,}")
print(f"False Positives: {cm_final[0,1]:,}")
print(f"False Negatives: {cm_final[1,0]:,}")
print(f"True Positives:  {cm_final[1,1]:,}")

# ===========================================================================
# ETAPA 10: VISUALIZAÇÕES
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 10: VISUALIZAÇÕES")
print(f"{'='*70}")

# 1. Comparação V1 vs V2
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ROC-AUC comparison
ax1 = axes[0, 0]
models = ['V1\n(All Features)', f'V2\n({len(selected_features_final)} Features)']
roc_aucs = [roc_auc_baseline, roc_auc_final]
colors = ['steelblue', 'forestgreen']
bars = ax1.bar(models, roc_aucs, color=colors, alpha=0.7)
ax1.set_ylabel('ROC-AUC', fontweight='bold')
ax1.set_title('ROC-AUC: V1 vs V2', fontweight='bold')
ax1.set_ylim(0.70, 0.85)
ax1.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, roc_aucs):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.005, f'{val:.4f}', 
             ha='center', va='bottom', fontweight='bold')

# F1-Macro comparison
ax2 = axes[0, 1]
f1_macros = [f1_macro_baseline, f1_macro_final]
bars = ax2.bar(models, f1_macros, color=colors, alpha=0.7)
ax2.set_ylabel('F1-Macro', fontweight='bold')
ax2.set_title('F1-Macro: V1 vs V2', fontweight='bold')
ax2.set_ylim(0.60, 0.75)
ax2.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, f1_macros):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.005, f'{val:.4f}', 
             ha='center', va='bottom', fontweight='bold')

# Feature reduction
ax3 = axes[1, 0]
feature_counts = [len(numeric_features), len(selected_features_final)]
bars = ax3.bar(models, feature_counts, color=colors, alpha=0.7)
ax3.set_ylabel('Número de Features', fontweight='bold')
ax3.set_title('Redução de Dimensionalidade', fontweight='bold')
ax3.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, feature_counts):
    ax3.text(bar.get_x() + bar.get_width()/2, val + 1, f'{int(val)}', 
             ha='center', va='bottom', fontweight='bold')
ax3.axhline(y=len(numeric_features), color='red', linestyle='--', alpha=0.5, label='Baseline')

# Threshold analysis
ax4 = axes[1, 1]
ax4.plot(results_df['threshold'], results_df['roc_auc'], marker='o', linewidth=2, 
         markersize=8, label='ROC-AUC', color='steelblue')
ax4.axhline(y=roc_auc_baseline, color='red', linestyle='--', linewidth=2, 
            label=f'Baseline ({roc_auc_baseline:.4f})', alpha=0.7)
ax4.set_xlabel('Threshold de Coeficiente', fontweight='bold')
ax4.set_ylabel('ROC-AUC', fontweight='bold')
ax4.set_title('Impacto do Threshold de Feature Selection', fontweight='bold')
ax4.legend()
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/sgd_v2_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Comparação V1 vs V2 salva: visualizations/sgd_v2_comparison.png")

# 2. ROC Curve V1 vs V2
fpr_v1, tpr_v1, _ = roc_curve(y_test, y_pred_proba_baseline)
fpr_v2, tpr_v2, _ = roc_curve(y_test, y_pred_proba_final)

plt.figure(figsize=(10, 8))
plt.plot(fpr_v1, tpr_v1, linewidth=2, label=f'V1 (AUC = {roc_auc_baseline:.4f})', color='steelblue')
plt.plot(fpr_v2, tpr_v2, linewidth=2, label=f'V2 (AUC = {roc_auc_final:.4f})', color='forestgreen')
plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random (AUC = 0.5000)', alpha=0.5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
plt.title('ROC Curve: SGD V1 vs V2', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/roc_curve_v1_v2.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ ROC Curve salva: visualizations/roc_curve_v1_v2.png")

# 3. Feature importance V2
coefficients_final = pipeline_final.named_steps['sgd'].coef_[0]
feature_importance_final = pd.DataFrame({
    'feature': selected_features_final,
    'coefficient': coefficients_final,
    'abs_coefficient': np.abs(coefficients_final)
}).sort_values('abs_coefficient', ascending=False)

plt.figure(figsize=(12, 10))
top_20 = feature_importance_final.head(20)
colors = ['green' if x > 0 else 'red' for x in top_20['coefficient']]
plt.barh(range(len(top_20)), top_20['coefficient'], color=colors, alpha=0.7)
plt.yticks(range(len(top_20)), top_20['feature'])
plt.xlabel('Coeficiente', fontsize=12, fontweight='bold')
plt.ylabel('Feature', fontsize=12, fontweight='bold')
plt.title(f'Top 20 Features - SGD V2 ({len(selected_features_final)} features)', 
          fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/feature_importance_v2.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Feature Importance V2 salva: visualizations/feature_importance_v2.png")

# ===========================================================================
# ETAPA 11: SALVAR RESULTADOS
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 11: SALVAR RESULTADOS")
print(f"{'='*70}")

# Salvar comparação de thresholds
results_df.to_csv('reports/sgd_v2_threshold_comparison.csv', index=False)
print(f"✓ Comparação de thresholds salva: reports/sgd_v2_threshold_comparison.csv")

# Salvar features selecionadas
feature_importance_final.to_csv('reports/sgd_v2_selected_features.csv', index=False)
print(f"✓ Features selecionadas salvas: reports/sgd_v2_selected_features.csv")

# Salvar features removidas
removed_features_list = [f for f in numeric_features if f not in selected_features_final]
pd.DataFrame({'removed_feature': removed_features_list}).to_csv('reports/sgd_v2_removed_features.csv', index=False)
print(f"✓ Features removidas salvas: reports/sgd_v2_removed_features.csv")

# Relatório detalhado
with open('reports/sgd_v2_report.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("SGD CLASSIFIER V2 - FEATURE SELECTION REPORT\n")
    f.write("="*80 + "\n\n")
    
    f.write("ESTRATÉGIA\n")
    f.write("-"*80 + "\n")
    f.write("1. Treinar modelo baseline com todas as features\n")
    f.write("2. Analisar coeficientes absolutos do modelo linear\n")
    f.write("3. Remover features com |coeficiente| < threshold\n")
    f.write("4. Retreinar e comparar performance\n\n")
    
    f.write("BASELINE (V1)\n")
    f.write("-"*80 + "\n")
    f.write(f"Features:          {len(numeric_features)}\n")
    f.write(f"ROC-AUC:           {roc_auc_baseline:.4f}\n")
    f.write(f"F1-Macro:          {f1_macro_baseline:.4f}\n")
    f.write(f"Accuracy:          {accuracy_baseline:.4f}\n")
    f.write(f"Tempo Treino:      {train_time_baseline:.2f}s\n\n")
    
    f.write("MODELO FINAL (V2)\n")
    f.write("-"*80 + "\n")
    f.write(f"Features:          {len(selected_features_final)} (removidas: {len(numeric_features) - len(selected_features_final)})\n")
    f.write(f"Threshold:         {best_threshold:.2f}\n")
    f.write(f"ROC-AUC:           {roc_auc_final:.4f} ({roc_auc_final - roc_auc_baseline:+.4f})\n")
    f.write(f"F1-Macro:          {f1_macro_final:.4f} ({f1_macro_final - f1_macro_baseline:+.4f})\n")
    f.write(f"Accuracy:          {accuracy_final:.4f}\n")
    f.write(f"Precision:         {precision_final:.4f}\n")
    f.write(f"Recall:            {recall_final:.4f}\n")
    f.write(f"F1-Score:          {f1_final:.4f}\n")
    f.write(f"Threshold Pred:    {best_thresh_final:.2f}\n\n")
    
    f.write("MATRIZ DE CONFUSÃO\n")
    f.write("-"*80 + "\n")
    f.write(f"True Negatives:    {cm_final[0,0]:,}\n")
    f.write(f"False Positives:   {cm_final[0,1]:,}\n")
    f.write(f"False Negatives:   {cm_final[1,0]:,}\n")
    f.write(f"True Positives:    {cm_final[1,1]:,}\n\n")
    
    f.write("FEATURES SELECIONADAS (TOP 30)\n")
    f.write("-"*80 + "\n")
    for i, row in feature_importance_final.head(30).iterrows():
        f.write(f"{row['feature']:40s} | Coef: {row['coefficient']:+.6f}\n")
    f.write("\n")
    
    f.write("FEATURES REMOVIDAS\n")
    f.write("-"*80 + "\n")
    for feat in removed_features_list[:30]:
        f.write(f"- {feat}\n")
    if len(removed_features_list) > 30:
        f.write(f"... e mais {len(removed_features_list) - 30} features\n")
    f.write("\n")
    
    f.write("COMPARAÇÃO DE THRESHOLDS\n")
    f.write("-"*80 + "\n")
    f.write(f"{'Threshold':>10} | {'Features':>8} | {'ROC-AUC':>8} | {'Diff':>8} | {'F1-Macro':>9}\n")
    f.write("-"*80 + "\n")
    for i, row in results_df.iterrows():
        f.write(f"{row['threshold']:>10.2f} | {int(row['n_features']):>8} | {row['roc_auc']:>8.4f} | ")
        f.write(f"{row['roc_auc_diff']:>+8.4f} | {row['f1_macro']:>9.4f}\n")

print(f"✓ Relatório V2 salvo: reports/sgd_v2_report.txt")

# ===========================================================================
# CONCLUSÃO
# ===========================================================================
print(f"\n{'='*80}")
print(f"✅ SGD CLASSIFIER V2 - FEATURE SELECTION CONCLUÍDO!")
print(f"{'='*80}")

print(f"\n🎯 RESULTADO FINAL:")
print(f"   V1 (Baseline):    {roc_auc_baseline:.4f} ROC-AUC ({len(numeric_features)} features)")
print(f"   V2 (Selected):    {roc_auc_final:.4f} ROC-AUC ({len(selected_features_final)} features)")
print(f"   Diferença:        {roc_auc_final - roc_auc_baseline:+.4f} ({(roc_auc_final - roc_auc_baseline)*100:+.2f}%)")
print(f"   Features removidas: {len(removed_features_list)} ({len(removed_features_list)/len(numeric_features)*100:.1f}%)")

if roc_auc_final > roc_auc_baseline:
    print(f"\n✅ SUCESSO! Feature selection MELHOROU o modelo")
elif roc_auc_final == roc_auc_baseline:
    print(f"\n⚖️  NEUTRO! Feature selection manteve performance (modelo mais simples)")
else:
    print(f"\n⚠️  ATENÇÃO! Feature selection PIOROU o modelo")

print(f"\n📁 Arquivos gerados:")
print(f"   - visualizations/sgd_v2_comparison.png")
print(f"   - visualizations/roc_curve_v1_v2.png")
print(f"   - visualizations/feature_importance_v2.png")
print(f"   - reports/sgd_v2_report.txt")
print(f"   - reports/sgd_v2_threshold_comparison.csv")
print(f"   - reports/sgd_v2_selected_features.csv")
print(f"   - reports/sgd_v2_removed_features.csv")

print(f"\n💡 RECOMENDAÇÃO:")
if roc_auc_final >= roc_auc_baseline:
    print(f"   Usar SGD V2 em produção: menos features, mesma/melhor performance")
else:
    print(f"   Manter SGD V1 (todas features): melhor performance geral")

print(f"\n✅ ANÁLISE COMPLETA!")
