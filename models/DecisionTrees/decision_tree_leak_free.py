import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, 
                             recall_score, f1_score, confusion_matrix, roc_curve)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler
import warnings
import time
from google.cloud import bigquery
import os
warnings.filterwarnings('ignore')

os.makedirs('visualizations', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# ===========================================================================
# DECISION TREE CLASSIFIER - LEAK-FREE
# ===========================================================================
print(f"\n{'='*80}")
print(f"DECISION TREE CLASSIFIER - LEAK-FREE")
print(f"{'='*80}")
print(f"Configurações testadas:")
print(f"  1️⃣ BASELINE: Parâmetros padrão")
print(f"  2️⃣ MAX_DEPTH_10: Limitar profundidade (overfitting control)")
print(f"  3️⃣ MIN_SAMPLES_SPLIT_100: Mínimo de amostras para split")
print(f"  4️⃣ BALANCED: class_weight='balanced'")
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
        'criterion': 'gini',
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42
    },
    'MAX_DEPTH_10': {
        'criterion': 'gini',
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42
    },
    'MIN_SAMPLES_SPLIT_100': {
        'criterion': 'gini',
        'max_depth': None,
        'min_samples_split': 100,
        'min_samples_leaf': 50,
        'random_state': 42
    },
    'BALANCED': {
        'criterion': 'gini',
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'class_weight': 'balanced',
        'random_state': 42
    },
    'OPTIMIZED': {
        'criterion': 'gini',
        'max_depth': 15,
        'min_samples_split': 50,
        'min_samples_leaf': 20,
        'max_features': 'sqrt',
        'class_weight': 'balanced',
        'random_state': 42
    }
}

results = []

for config_name, params in configs.items():
    print(f"\n{'='*70}")
    print(f"CONFIG: {config_name}")
    print(f"{'='*70}")
    
    # Treinar
    start_time = time.time()
    model = DecisionTreeClassifier(**params)
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
    
    # Complexidade da árvore
    n_nodes = model.tree_.node_count
    max_depth_actual = model.tree_.max_depth
    n_leaves = model.tree_.n_leaves
    
    print(f"\n📊 Métricas:")
    print(f"   ROC-AUC:      {roc_auc:.4f}")
    print(f"   Accuracy:     {accuracy:.4f}")
    print(f"   Precision:    {precision:.4f}")
    print(f"   Recall:       {recall:.4f}")
    print(f"   F1-Score:     {f1:.4f}")
    print(f"   F1-Macro:     {f1_macro:.4f}")
    print(f"   Tempo:        {train_time:.3f}s")
    
    print(f"\n🌳 Complexidade:")
    print(f"   Nós:          {n_nodes:,}")
    print(f"   Profundidade: {max_depth_actual}")
    print(f"   Folhas:       {n_leaves:,}")
    
    results.append({
        'config': config_name,
        'roc_auc': roc_auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f1_macro': f1_macro,
        'train_time': train_time,
        'n_nodes': n_nodes,
        'max_depth': max_depth_actual,
        'n_leaves': n_leaves,
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
print(f"{'='*100}")
print(f"{'Config':<25} | {'ROC-AUC':<10} | {'F1-Macro':<10} | {'Nodes':<10} | {'Depth':<8} | {'Tempo (s)':<10}")
print(f"{'='*100}")
for _, row in results_df.iterrows():
    print(f"{row['config']:<25} | {row['roc_auc']:<10.4f} | {row['f1_macro']:<10.4f} | "
          f"{row['n_nodes']:<10,} | {row['max_depth']:<8} | {row['train_time']:<10.3f}")

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

print(f"\n📊 TOP 20 FEATURES:")
print(f"{'='*70}")
for i, row in feature_importance.head(20).iterrows():
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
ax1.set_title('ROC-AUC: Comparação de Configurações', fontweight='bold')
ax1.set_ylim(0.7, 0.9)
ax1.grid(axis='y', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, roc_aucs)):
    ax1.text(i, val + 0.005, f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=8)

# F1-Macro
ax2 = axes[0, 1]
f1_macros = [r['f1_macro'] for r in results]
bars = ax2.bar(range(len(configs_list)), f1_macros, color=colors, alpha=0.7)
ax2.set_xticks(range(len(configs_list)))
ax2.set_xticklabels(configs_list, rotation=45, ha='right')
ax2.set_ylabel('F1-Macro', fontweight='bold')
ax2.set_title('F1-Macro: Comparação de Configurações', fontweight='bold')
ax2.set_ylim(0.6, 0.8)
ax2.grid(axis='y', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, f1_macros)):
    ax2.text(i, val + 0.005, f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=8)

# Complexidade (Nós)
ax3 = axes[1, 0]
n_nodes = [r['n_nodes'] for r in results]
bars = ax3.bar(range(len(configs_list)), n_nodes, color=colors, alpha=0.7)
ax3.set_xticks(range(len(configs_list)))
ax3.set_xticklabels(configs_list, rotation=45, ha='right')
ax3.set_ylabel('Número de Nós', fontweight='bold')
ax3.set_title('Complexidade: Número de Nós', fontweight='bold')
ax3.grid(axis='y', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, n_nodes)):
    ax3.text(i, val + 100, f'{val:,}', ha='center', va='bottom', fontweight='bold', fontsize=8)

# Profundidade
ax4 = axes[1, 1]
depths = [r['max_depth'] for r in results]
bars = ax4.bar(range(len(configs_list)), depths, color=colors, alpha=0.7)
ax4.set_xticks(range(len(configs_list)))
ax4.set_xticklabels(configs_list, rotation=45, ha='right')
ax4.set_ylabel('Profundidade Máxima', fontweight='bold')
ax4.set_title('Complexidade: Profundidade', fontweight='bold')
ax4.grid(axis='y', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, depths)):
    ax4.text(i, val + 0.5, f'{val}', ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('visualizations/decision_tree_config_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Comparação salva: visualizations/decision_tree_config_comparison.png")

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
plt.title('ROC Curves: Decision Tree Configurations', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/roc_curves_decision_tree.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ ROC Curves salvas: visualizations/roc_curves_decision_tree.png")

# 3. Feature Importance (Melhor Modelo)
plt.figure(figsize=(12, 10))
top_20 = feature_importance.head(20)
plt.barh(range(len(top_20)), top_20['importance'], color='steelblue', alpha=0.7)
plt.yticks(range(len(top_20)), top_20['feature'])
plt.xlabel('Importance', fontsize=12, fontweight='bold')
plt.ylabel('Feature', fontsize=12, fontweight='bold')
plt.title(f'Top 20 Features - {best_config}', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/feature_importance_decision_tree.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Feature Importance salva: visualizations/feature_importance_decision_tree.png")

# 4. Confusion Matrix (Melhor Modelo)
plt.figure(figsize=(8, 6))
sns.heatmap(best_result['confusion_matrix'], annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title(f'Confusion Matrix: {best_config}', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontweight='bold')
plt.xlabel('Predicted Label', fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/confusion_matrix_decision_tree.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Confusion Matrix salva: visualizations/confusion_matrix_decision_tree.png")

# 5. Árvore Visualizada (primeiros 3 níveis apenas - para não ficar muito grande)
if best_result['max_depth'] <= 5:
    max_depth_plot = best_result['max_depth']
else:
    max_depth_plot = 3

plt.figure(figsize=(20, 12))
plot_tree(best_model, 
          max_depth=max_depth_plot,
          feature_names=numeric_features,
          class_names=['No Convert', 'Convert'],
          filled=True,
          rounded=True,
          fontsize=8)
plt.title(f'Decision Tree Structure: {best_config} (Top {max_depth_plot} levels)', 
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/tree_structure_decision_tree.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Estrutura da Árvore salva: visualizations/tree_structure_decision_tree.png")

# ===========================================================================
# ETAPA 10: SALVAR RESULTADOS
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 10: SALVAR RESULTADOS")
print(f"{'='*70}")

# Salvar comparação
results_df.to_csv('reports/decision_tree_comparison.csv', index=False)
print(f"✓ Comparação salva: reports/decision_tree_comparison.csv")

# Salvar feature importance
feature_importance.to_csv('reports/decision_tree_feature_importance.csv', index=False)
print(f"✓ Feature Importance salva: reports/decision_tree_feature_importance.csv")

# Relatório detalhado
with open('reports/decision_tree_report.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("DECISION TREE CLASSIFIER - LEAK-FREE\n")
    f.write("="*80 + "\n\n")
    
    f.write("CONFIGURAÇÕES TESTADAS\n")
    f.write("-"*80 + "\n")
    for config_name, params in configs.items():
        f.write(f"\n{config_name}:\n")
        for param, value in params.items():
            f.write(f"   {param}: {value}\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("RESULTADOS COMPARATIVOS\n")
    f.write("="*80 + "\n")
    f.write(f"{'Config':<25} | {'ROC-AUC':<10} | {'F1-Macro':<10} | {'Nodes':<10} | {'Depth':<8} | {'Tempo':<10}\n")
    f.write("-"*100 + "\n")
    for _, row in results_df.iterrows():
        f.write(f"{row['config']:<25} | {row['roc_auc']:<10.4f} | {row['f1_macro']:<10.4f} | "
                f"{int(row['n_nodes']):<10,} | {int(row['max_depth']):<8} | {row['train_time']:<10.3f}s\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write(f"MELHOR MODELO: {best_config}\n")
    f.write("="*80 + "\n")
    f.write(f"ROC-AUC:           {best_result['roc_auc']:.4f}\n")
    f.write(f"Accuracy:          {best_result['accuracy']:.4f}\n")
    f.write(f"Precision:         {best_result['precision']:.4f}\n")
    f.write(f"Recall:            {best_result['recall']:.4f}\n")
    f.write(f"F1-Score:          {best_result['f1']:.4f}\n")
    f.write(f"F1-Macro:          {best_result['f1_macro']:.4f}\n")
    f.write(f"Tempo:             {best_result['train_time']:.3f}s\n\n")
    
    f.write("COMPLEXIDADE DA ÁRVORE\n")
    f.write("-"*80 + "\n")
    f.write(f"Número de Nós:     {best_result['n_nodes']:,}\n")
    f.write(f"Profundidade:      {best_result['max_depth']}\n")
    f.write(f"Número de Folhas:  {best_result['n_leaves']:,}\n\n")
    
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
    f.write("ANÁLISE\n")
    f.write("="*80 + "\n")
    f.write("1. Overfitting:\n")
    baseline_nodes = [r for r in results if r['config'] == 'BASELINE'][0]['n_nodes']
    optimized_nodes = best_result['n_nodes']
    f.write(f"   - BASELINE: {baseline_nodes:,} nós (possível overfitting)\n")
    f.write(f"   - {best_config}: {optimized_nodes:,} nós (controle de complexidade)\n\n")
    
    f.write("2. Interpretabilidade:\n")
    f.write(f"   - Profundidade: {best_result['max_depth']} níveis\n")
    f.write(f"   - Árvores mais rasas são mais interpretáveis\n\n")
    
    f.write("3. Performance vs Complexidade:\n")
    f.write(f"   - Trade-off: Simplicidade vs Acurácia\n")
    f.write(f"   - {best_config} oferece melhor equilíbrio\n")

print(f"✓ Relatório salvo: reports/decision_tree_report.txt")

# ===========================================================================
# COMPARAÇÃO COM OUTROS MODELOS
# ===========================================================================
print(f"\n{'='*80}")
print(f"COMPARAÇÃO COM OUTROS MODELOS DO PROJETO")
print(f"{'='*80}")

other_models = {
    'CatBoost': 0.8669,
    'LightGBM': 0.8642,
    'SGD V3': 0.8027,
    'SGD V2': 0.7917,
    'BernoulliNB': 0.7782,
    'MultinomialNB': 0.7607,
    'KNN': 0.7542,
    'GaussianNB': 0.7241
}

print(f"\n📊 CONTEXTO:")
print(f"{'='*70}")
print(f"{'Modelo':<20} | {'ROC-AUC':<10} | {'Posição':<15}")
print(f"{'='*70}")

all_models = {best_config: best_roc_auc, **other_models}
sorted_models = sorted(all_models.items(), key=lambda x: x[1], reverse=True)

for i, (model, auc) in enumerate(sorted_models, 1):
    marker = "🏆" if i == 1 else "  "
    print(f"{marker} {model:<18} | {auc:<10.4f} | #{i}")

# ===========================================================================
# CONCLUSÃO
# ===========================================================================
print(f"\n{'='*80}")
print(f"✅ DECISION TREE - ANÁLISE CONCLUÍDA!")
print(f"{'='*80}")

print(f"\n🎯 RESULTADOS:")
for result in results:
    print(f"   {result['config']:<25}: {result['roc_auc']:.4f} ROC-AUC")

print(f"\n🏆 MELHOR: {best_config} ({best_roc_auc:.4f} ROC-AUC)")

print(f"\n🌳 COMPLEXIDADE DO MELHOR MODELO:")
print(f"   Nós:          {best_result['n_nodes']:,}")
print(f"   Profundidade: {best_result['max_depth']}")
print(f"   Folhas:       {best_result['n_leaves']:,}")

print(f"\n⚡ VELOCIDADE:")
print(f"   Treinamento: {best_result['train_time']:.3f}s")

print(f"\n📁 Arquivos gerados:")
print(f"   - visualizations/decision_tree_config_comparison.png")
print(f"   - visualizations/roc_curves_decision_tree.png")
print(f"   - visualizations/feature_importance_decision_tree.png")
print(f"   - visualizations/confusion_matrix_decision_tree.png")
print(f"   - visualizations/tree_structure_decision_tree.png")
print(f"   - reports/decision_tree_comparison.csv")
print(f"   - reports/decision_tree_feature_importance.csv")
print(f"   - reports/decision_tree_report.txt")

print(f"\n✅ ANÁLISE COMPLETA!")
