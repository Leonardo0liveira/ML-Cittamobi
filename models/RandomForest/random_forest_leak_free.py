import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, 
                             recall_score, f1_score, confusion_matrix, roc_curve)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
import warnings
import time
from google.cloud import bigquery
import os
warnings.filterwarnings('ignore')

os.makedirs('visualizations', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# ===========================================================================
# RANDOM FOREST CLASSIFIER - LEAK-FREE
# ===========================================================================
print(f"\n{'='*80}")
print(f"RANDOM FOREST CLASSIFIER - LEAK-FREE")
print(f"{'='*80}")
print(f"Configurações testadas:")
print(f"  1️⃣ BASELINE: n_estimators=100, parâmetros padrão")
print(f"  2️⃣ BALANCED: class_weight='balanced'")
print(f"  3️⃣ DEPTH_CONTROL: max_depth=20, min_samples_split=50")
print(f"  4️⃣ LARGE_ENSEMBLE: n_estimators=300")
print(f"  5️⃣ OPTIMIZED: Combinação dos melhores parâmetros")
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

# Features básicas
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
# ETAPA 6: TREINAR CONFIGURAÇÕES
# ===========================================================================
print(f"\n{'='*80}")
print(f"ETAPA 6: TREINAR E COMPARAR CONFIGURAÇÕES")
print(f"{'='*80}")

configs = {
    'BASELINE': {
        'n_estimators': 100,
        'criterion': 'gini',
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1
    },
    'BALANCED': {
        'n_estimators': 100,
        'criterion': 'gini',
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'sqrt',
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    },
    'DEPTH_CONTROL': {
        'n_estimators': 100,
        'criterion': 'gini',
        'max_depth': 20,
        'min_samples_split': 50,
        'min_samples_leaf': 20,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1
    },
    'LARGE_ENSEMBLE': {
        'n_estimators': 300,
        'criterion': 'gini',
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1
    },
    'OPTIMIZED': {
        'n_estimators': 300,
        'criterion': 'gini',
        'max_depth': 25,
        'min_samples_split': 30,
        'min_samples_leaf': 10,
        'max_features': 'sqrt',
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }
}

results = []

for config_name, params in configs.items():
    print(f"\n{'='*70}")
    print(f"CONFIG: {config_name}")
    print(f"{'='*70}")
    print(f"Parâmetros: n_estimators={params['n_estimators']}, max_depth={params['max_depth']}")
    
    # Treinar
    start_time = time.time()
    model = RandomForestClassifier(**params)
    
    print(f"🔄 Treinando {params['n_estimators']} árvores...")
    model.fit(X_train, y_train)
    
    # Predições
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    train_time = time.time() - start_time
    
    # Métricas
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)
    
    # Estatísticas das árvores
    n_trees = params['n_estimators']
    tree_depths = [tree.tree_.max_depth for tree in model.estimators_]
    avg_depth = np.mean(tree_depths)
    avg_leaves = np.mean([tree.tree_.n_leaves for tree in model.estimators_])
    
    print(f"\n📊 Métricas:")
    print(f"   ROC-AUC:      {roc_auc:.4f}")
    print(f"   Accuracy:     {accuracy:.4f}")
    print(f"   Precision:    {precision:.4f}")
    print(f"   Recall:       {recall:.4f}")
    print(f"   F1-Score:     {f1:.4f}")
    print(f"   F1-Macro:     {f1_macro:.4f}")
    print(f"   Tempo:        {train_time:.2f}s")
    
    print(f"\n🌳 Ensemble:")
    print(f"   N° Árvores:        {n_trees}")
    print(f"   Profundidade Média: {avg_depth:.1f}")
    print(f"   Folhas Médias:      {avg_leaves:.0f}")
    
    results.append({
        'config': config_name,
        'roc_auc': roc_auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f1_macro': f1_macro,
        'train_time': train_time,
        'n_trees': n_trees,
        'avg_depth': avg_depth,
        'avg_leaves': avg_leaves,
        'model': model,
        'y_pred_proba': y_pred_proba,
        'confusion_matrix': cm
    })

# ===========================================================================
# ETAPA 7: COMPARAÇÃO FINAL
# ===========================================================================
print(f"\n{'='*80}")
print(f"ETAPA 7: COMPARAÇÃO FINAL")
print(f"{'='*80}")

results_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['model', 'y_pred_proba', 'confusion_matrix']} 
                           for r in results])
results_df = results_df.sort_values('roc_auc', ascending=False)

print(f"\n📊 RANKING POR ROC-AUC:")
print(f"{'='*110}")
print(f"{'Config':<20} | {'ROC-AUC':<10} | {'F1-Macro':<10} | {'Trees':<8} | {'Avg Depth':<10} | {'Tempo (s)':<10}")
print(f"{'='*110}")
for _, row in results_df.iterrows():
    print(f"{row['config']:<20} | {row['roc_auc']:<10.4f} | {row['f1_macro']:<10.4f} | "
          f"{int(row['n_trees']):<8} | {row['avg_depth']:<10.1f} | {row['train_time']:<10.2f}")

best_config = results_df.iloc[0]['config']
best_roc_auc = results_df.iloc[0]['roc_auc']
print(f"\n🏆 MELHOR CONFIG: {best_config} (ROC-AUC = {best_roc_auc:.4f})")

# Pegar melhor modelo
best_result = [r for r in results if r['config'] == best_config][0]
best_model = best_result['model']

# ===========================================================================
# ETAPA 8: ANÁLISE DE FEATURE IMPORTANCE
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 8: ANÁLISE DE FEATURE IMPORTANCE (MELHOR MODELO)")
print(f"{'='*70}")

feature_importance = pd.DataFrame({
    'feature': numeric_features,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n📊 TOP 25 FEATURES:")
print(f"{'='*70}")
for i, row in feature_importance.head(25).iterrows():
    bar_length = int(row['importance'] * 50)
    bar = '█' * bar_length
    print(f"{row['feature']:45s} | {row['importance']:.4f} {bar}")

# ===========================================================================
# ETAPA 9: VISUALIZAÇÕES
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 9: VISUALIZAÇÕES")
print(f"{'='*70}")

# 1. Comparação de Configurações
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

configs_list = [r['config'] for r in results]
colors = ['steelblue', 'orange', 'forestgreen', 'purple', 'crimson']

# ROC-AUC
ax1 = axes[0, 0]
roc_aucs = [r['roc_auc'] for r in results]
bars = ax1.bar(range(len(configs_list)), roc_aucs, color=colors, alpha=0.7)
ax1.set_xticks(range(len(configs_list)))
ax1.set_xticklabels(configs_list, rotation=45, ha='right')
ax1.set_ylabel('ROC-AUC', fontweight='bold')
ax1.set_title('ROC-AUC: Comparação Random Forest', fontweight='bold')
ax1.set_ylim(0.75, 0.90)
ax1.grid(axis='y', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, roc_aucs)):
    ax1.text(i, val + 0.003, f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=8)

# F1-Macro
ax2 = axes[0, 1]
f1_macros = [r['f1_macro'] for r in results]
bars = ax2.bar(range(len(configs_list)), f1_macros, color=colors, alpha=0.7)
ax2.set_xticks(range(len(configs_list)))
ax2.set_xticklabels(configs_list, rotation=45, ha='right')
ax2.set_ylabel('F1-Macro', fontweight='bold')
ax2.set_title('F1-Macro: Comparação Random Forest', fontweight='bold')
ax2.set_ylim(0.60, 0.75)
ax2.grid(axis='y', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, f1_macros)):
    ax2.text(i, val + 0.003, f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=8)

# Tempo de Treinamento
ax3 = axes[1, 0]
train_times = [r['train_time'] for r in results]
bars = ax3.bar(range(len(configs_list)), train_times, color=colors, alpha=0.7)
ax3.set_xticks(range(len(configs_list)))
ax3.set_xticklabels(configs_list, rotation=45, ha='right')
ax3.set_ylabel('Tempo (segundos)', fontweight='bold')
ax3.set_title('Tempo de Treinamento', fontweight='bold')
ax3.grid(axis='y', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, train_times)):
    ax3.text(i, val + 1, f'{val:.1f}s', ha='center', va='bottom', fontweight='bold', fontsize=8)

# Número de Árvores vs ROC-AUC
ax4 = axes[1, 1]
n_trees = [r['n_trees'] for r in results]
ax4.scatter(n_trees, roc_aucs, c=colors[:len(n_trees)], s=200, alpha=0.7)
for i, (trees, auc, config) in enumerate(zip(n_trees, roc_aucs, configs_list)):
    ax4.annotate(config, (trees, auc), fontsize=8, ha='center', va='bottom')
ax4.set_xlabel('Número de Árvores', fontweight='bold')
ax4.set_ylabel('ROC-AUC', fontweight='bold')
ax4.set_title('N° Árvores vs ROC-AUC', fontweight='bold')
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/random_forest_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Comparação salva: visualizations/random_forest_comparison.png")

# 2. ROC Curves
plt.figure(figsize=(10, 8))
for i, result in enumerate(results):
    fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
    plt.plot(fpr, tpr, linewidth=2, label=f"{result['config']} (AUC = {result['roc_auc']:.4f})", 
             color=colors[i])

plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random', alpha=0.5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
plt.title('ROC Curves: Random Forest Configurations', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/roc_curves_random_forest.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ ROC Curves salvas: visualizations/roc_curves_random_forest.png")

# 3. Feature Importance (Top 30)
plt.figure(figsize=(12, 10))
top_30 = feature_importance.head(30)
plt.barh(range(len(top_30)), top_30['importance'], color='forestgreen', alpha=0.7)
plt.yticks(range(len(top_30)), top_30['feature'])
plt.xlabel('Importance', fontsize=12, fontweight='bold')
plt.ylabel('Feature', fontsize=12, fontweight='bold')
plt.title(f'Top 30 Features - {best_config}', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/feature_importance_random_forest.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Feature Importance salva: visualizations/feature_importance_random_forest.png")

# 4. Confusion Matrix (Melhor Modelo)
plt.figure(figsize=(8, 6))
sns.heatmap(best_result['confusion_matrix'], annot=True, fmt='d', cmap='Greens', cbar=True)
plt.title(f'Confusion Matrix: {best_config}', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontweight='bold')
plt.xlabel('Predicted Label', fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/confusion_matrix_random_forest.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Confusion Matrix salva: visualizations/confusion_matrix_random_forest.png")

# 5. Comparação Decision Tree vs Random Forest
dt_best_roc = 0.7466  # Do Decision Tree anterior
rf_best_roc = best_roc_auc

plt.figure(figsize=(10, 6))
models_comp = ['Decision Tree\n(Single)', 'Random Forest\n(Ensemble)']
roc_comp = [dt_best_roc, rf_best_roc]
colors_comp = ['steelblue', 'forestgreen']

bars = plt.bar(models_comp, roc_comp, color=colors_comp, alpha=0.7, width=0.6)
plt.ylabel('ROC-AUC', fontsize=12, fontweight='bold')
plt.title('Decision Tree vs Random Forest', fontsize=14, fontweight='bold')
plt.ylim(0.7, 0.9)
plt.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, roc_comp):
    plt.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.4f}', 
             ha='center', va='bottom', fontweight='bold', fontsize=12)

# Adicionar ganho
gain = (rf_best_roc - dt_best_roc) * 100
plt.text(0.5, (dt_best_roc + rf_best_roc) / 2, f'+{gain:.1f}%', 
         ha='center', fontsize=14, fontweight='bold', color='darkgreen')

plt.tight_layout()
plt.savefig('visualizations/decision_tree_vs_random_forest.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Comparação DT vs RF salva: visualizations/decision_tree_vs_random_forest.png")

# ===========================================================================
# ETAPA 10: SALVAR RESULTADOS
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 10: SALVAR RESULTADOS")
print(f"{'='*70}")

# Salvar comparação
results_df.to_csv('reports/random_forest_comparison.csv', index=False)
print(f"✓ Comparação salva: reports/random_forest_comparison.csv")

# Salvar feature importance
feature_importance.to_csv('reports/random_forest_feature_importance.csv', index=False)
print(f"✓ Feature Importance salva: reports/random_forest_feature_importance.csv")

# Relatório detalhado
with open('reports/random_forest_report.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("RANDOM FOREST CLASSIFIER - LEAK-FREE\n")
    f.write("="*80 + "\n\n")
    
    f.write("CONFIGURAÇÕES TESTADAS\n")
    f.write("-"*80 + "\n")
    for config_name, params in configs.items():
        f.write(f"\n{config_name}:\n")
        for param, value in params.items():
            if param != 'n_jobs':
                f.write(f"   {param}: {value}\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("RESULTADOS COMPARATIVOS\n")
    f.write("="*80 + "\n")
    f.write(f"{'Config':<20} | {'ROC-AUC':<10} | {'F1-Macro':<10} | {'Trees':<8} | {'Avg Depth':<10} | {'Tempo':<10}\n")
    f.write("-"*90 + "\n")
    for _, row in results_df.iterrows():
        f.write(f"{row['config']:<20} | {row['roc_auc']:<10.4f} | {row['f1_macro']:<10.4f} | "
                f"{int(row['n_trees']):<8} | {row['avg_depth']:<10.1f} | {row['train_time']:<10.2f}s\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write(f"MELHOR MODELO: {best_config}\n")
    f.write("="*80 + "\n")
    f.write(f"ROC-AUC:           {best_result['roc_auc']:.4f}\n")
    f.write(f"Accuracy:          {best_result['accuracy']:.4f}\n")
    f.write(f"Precision:         {best_result['precision']:.4f}\n")
    f.write(f"Recall:            {best_result['recall']:.4f}\n")
    f.write(f"F1-Score:          {best_result['f1']:.4f}\n")
    f.write(f"F1-Macro:          {best_result['f1_macro']:.4f}\n")
    f.write(f"Tempo:             {best_result['train_time']:.2f}s\n\n")
    
    f.write("CARACTERÍSTICAS DO ENSEMBLE\n")
    f.write("-"*80 + "\n")
    f.write(f"Número de Árvores:      {best_result['n_trees']}\n")
    f.write(f"Profundidade Média:     {best_result['avg_depth']:.1f}\n")
    f.write(f"Folhas Médias:          {best_result['avg_leaves']:.0f}\n\n")
    
    f.write("MATRIZ DE CONFUSÃO\n")
    f.write("-"*80 + "\n")
    cm = best_result['confusion_matrix']
    f.write(f"True Negatives:    {cm[0,0]:,}\n")
    f.write(f"False Positives:   {cm[0,1]:,}\n")
    f.write(f"False Negatives:   {cm[1,0]:,}\n")
    f.write(f"True Positives:    {cm[1,1]:,}\n\n")
    
    f.write("TOP 30 FEATURES\n")
    f.write("-"*80 + "\n")
    for i, row in feature_importance.head(30).iterrows():
        f.write(f"{row['feature']:45s} | {row['importance']:.6f}\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("COMPARAÇÃO: DECISION TREE vs RANDOM FOREST\n")
    f.write("="*80 + "\n")
    f.write(f"Decision Tree (melhor):  {dt_best_roc:.4f} ROC-AUC\n")
    f.write(f"Random Forest (melhor):  {rf_best_roc:.4f} ROC-AUC\n")
    f.write(f"Ganho:                   {(rf_best_roc - dt_best_roc)*100:+.2f}%\n\n")
    
    f.write("ANÁLISE\n")
    f.write("-"*80 + "\n")
    f.write("1. Vantagens do Ensemble:\n")
    f.write("   - Reduz overfitting via bagging\n")
    f.write("   - Mais robusto a ruído nos dados\n")
    f.write("   - Feature importance mais confiável\n\n")
    
    f.write("2. Trade-offs:\n")
    f.write(f"   - Tempo de treinamento: ~{best_result['train_time']:.0f}s (vs 0.7s Decision Tree)\n")
    f.write("   - Menos interpretável (ensemble vs single tree)\n")
    f.write(f"   - Ganho de performance: +{(rf_best_roc - dt_best_roc)*100:.1f}%\n\n")
    
    f.write("3. Recomendação:\n")
    f.write("   - Random Forest é superior para produção\n")
    f.write("   - Melhor generalização\n")
    f.write("   - Trade-off velocidade/performance aceitável\n")

print(f"✓ Relatório salvo: reports/random_forest_report.txt")

# ===========================================================================
# COMPARAÇÃO COM TODOS OS MODELOS
# ===========================================================================
print(f"\n{'='*80}")
print(f"COMPARAÇÃO COM TODOS OS MODELOS DO PROJETO")
print(f"{'='*80}")

other_models = {
    'CatBoost': 0.8669,
    'LightGBM': 0.8642,
    'SGD V3': 0.8027,
    'SGD V2': 0.7917,
    'BernoulliNB': 0.7782,
    'MultinomialNB': 0.7607,
    'KNN': 0.7542,
    'Decision Tree': 0.7466,
    'GaussianNB': 0.7241
}

print(f"\n📊 RANKING GERAL:")
print(f"{'='*70}")
print(f"{'Modelo':<25} | {'ROC-AUC':<10} | {'Posição':<15}")
print(f"{'='*70}")

all_models = {best_config: best_roc_auc, **other_models}
sorted_models = sorted(all_models.items(), key=lambda x: x[1], reverse=True)

for i, (model, auc) in enumerate(sorted_models, 1):
    marker = "🏆" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
    print(f"{marker} {model:<23} | {auc:<10.4f} | #{i}")

# ===========================================================================
# CONCLUSÃO
# ===========================================================================
print(f"\n{'='*80}")
print(f"✅ RANDOM FOREST - ANÁLISE CONCLUÍDA!")
print(f"{'='*80}")

print(f"\n🎯 RESULTADOS:")
for result in results:
    print(f"   {result['config']:<20}: {result['roc_auc']:.4f} ROC-AUC ({result['n_trees']} trees)")

print(f"\n🏆 MELHOR: {best_config} ({best_roc_auc:.4f} ROC-AUC)")

print(f"\n🌳 CARACTERÍSTICAS:")
print(f"   Árvores:           {best_result['n_trees']}")
print(f"   Profundidade Média: {best_result['avg_depth']:.1f}")
print(f"   Folhas Médias:      {best_result['avg_leaves']:.0f}")

print(f"\n📈 GANHOS:")
print(f"   vs Decision Tree:  +{(best_roc_auc - dt_best_roc)*100:.2f}%")
print(f"   vs KNN:            +{(best_roc_auc - 0.7542)*100:.2f}%")
print(f"   Gap vs CatBoost:   {(0.8669 - best_roc_auc)*100:.2f}%")

print(f"\n⚡ VELOCIDADE:")
print(f"   Treinamento: {best_result['train_time']:.2f}s")

print(f"\n📁 Arquivos gerados:")
print(f"   - visualizations/random_forest_comparison.png")
print(f"   - visualizations/roc_curves_random_forest.png")
print(f"   - visualizations/feature_importance_random_forest.png")
print(f"   - visualizations/confusion_matrix_random_forest.png")
print(f"   - visualizations/decision_tree_vs_random_forest.png")
print(f"   - reports/random_forest_comparison.csv")
print(f"   - reports/random_forest_feature_importance.csv")
print(f"   - reports/random_forest_report.txt")

print(f"\n✅ ANÁLISE COMPLETA!")
