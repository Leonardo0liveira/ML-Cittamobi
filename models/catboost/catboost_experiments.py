from google.cloud import bigquery
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, 
                             recall_score, f1_score, confusion_matrix, classification_report, roc_curve)
from sklearn.model_selection import TimeSeriesSplit
from catboost import CatBoostClassifier, Pool
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
import time

# ===========================================================================
# EXPERIMENTO: OTIMIZAÇÃO DE HIPERPARÂMETROS CATBOOST
# ===========================================================================
print(f"\n{'='*80}")
print(f"EXPERIMENTO V6 - OTIMIZAÇÃO DE HIPERPARÂMETROS CATBOOST")
print(f"{'='*80}")
print(f"Objetivo: Testar diferentes configurações para maximizar ROC-AUC")
print(f"{'='*80}")

# Carregar dados
df = pd.read_csv('/Users/leonardooliveira/Downloads/Projeto Machine Learning/data/sampled_dataset.csv')
print(f"✓ Dados carregados: {len(df):,} registros")

# Feature Engineering rápido (igual ao modelo anterior)
target = "target"

# Agregar por usuário
if 'user_pseudo_id' in df.columns:
    user_agg = df.groupby('user_pseudo_id').agg({
        target: ['mean', 'sum', 'count'],
        'dist_device_stop': ['mean', 'std', 'min', 'max'] if 'dist_device_stop' in df.columns else ['mean'],
        'time_hour': ['mean', 'std'] if 'time_hour' in df.columns else ['mean']
    }).reset_index()
    
    user_agg.columns = ['user_pseudo_id', 'user_conversion_rate', 'user_total_conversions', 
                        'user_total_events', 'user_avg_dist', 'user_std_dist', 
                        'user_min_dist', 'user_max_dist', 'user_avg_hour', 'user_std_hour']
    
    df = df.merge(user_agg, on='user_pseudo_id', how='left')

# Agregar por parada
if 'gtfs_stop_id' in df.columns:
    stop_agg = df.groupby('gtfs_stop_id').agg({
        target: ['mean', 'count'],
        'user_frequency': ['mean', 'median'] if 'user_frequency' in df.columns else ['mean']
    }).reset_index()
    
    stop_agg.columns = ['gtfs_stop_id', 'stop_conversion_rate', 'stop_event_count_agg',
                        'stop_user_freq_mean', 'stop_user_freq_median']
    
    df = df.merge(stop_agg, on='gtfs_stop_id', how='left')

# Features de interação
if 'user_conversion_rate' in df.columns and 'stop_conversion_rate' in df.columns:
    df['conversion_interaction'] = df['user_conversion_rate'] * df['stop_conversion_rate']
    
if 'user_avg_dist' in df.columns and 'dist_device_stop' in df.columns:
    df['dist_deviation'] = abs(df['dist_device_stop'] - df['user_avg_dist'])
    df['dist_ratio'] = df['dist_device_stop'] / (df['user_avg_dist'] + 1)

if 'user_frequency' in df.columns and 'stop_event_count_agg' in df.columns:
    df['user_stop_affinity'] = df['user_frequency'] * df['stop_event_count_agg']

# Limpeza rápida
df_original_size = len(df)
if 'user_frequency' in df.columns:
    user_freq_threshold = df['user_frequency'].quantile(0.10)
    df = df[df['user_frequency'] >= user_freq_threshold]

if 'device_lat' in df.columns and 'device_lon' in df.columns:
    df = df[~((df['device_lat'].isna()) | (df['device_lon'].isna()))]
    df = df[~((df['device_lat'] == 0) & (df['device_lon'] == 0))]

if 'dist_device_stop' in df.columns:
    dist_threshold = df['dist_device_stop'].quantile(0.98)
    df = df[df['dist_device_stop'] <= dist_threshold]

print(f"✓ Dados limpos: {len(df):,} registros mantidos")

# Preparar features
features_to_drop = ['y_pred', 'y_pred_proba', 'ctm_service_route', 'direction', 'lotacao_proxy_binaria']
X = df.drop(columns=[target] + features_to_drop, errors='ignore')
y = df[target]

# Processar timestamp
if 'event_timestamp' in X.columns:
    X['event_timestamp'] = pd.to_datetime(X['event_timestamp'], format='ISO8601')
    X['year'] = X['event_timestamp'].dt.year
    X['month'] = X['event_timestamp'].dt.month
    X['day'] = X['event_timestamp'].dt.day
    X['hour'] = X['event_timestamp'].dt.hour
    X['dayofweek'] = X['event_timestamp'].dt.dayofweek
    X['minute'] = X['event_timestamp'].dt.minute
    X['week_of_year'] = X['event_timestamp'].dt.isocalendar().week
    X = X.drop(columns=['event_timestamp'])

# Features cíclicas
if 'time_day_of_month' in X.columns:
    X['day_of_month_sin'] = np.sin(2 * np.pi * X['time_day_of_month'] / 31)
    X['day_of_month_cos'] = np.cos(2 * np.pi * X['time_day_of_month'] / 31)

if 'week_of_year' in X.columns:
    X['week_sin'] = np.sin(2 * np.pi * X['week_of_year'] / 52)
    X['week_cos'] = np.cos(2 * np.pi * X['week_of_year'] / 52)

if 'hour' in X.columns:
    X['hour_sin'] = np.sin(2 * np.pi * X['hour'] / 24)
    X['hour_cos'] = np.cos(2 * np.pi * X['hour'] / 24)

# Identificar categóricas
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
X = X.replace([np.inf, -np.inf], np.nan)

print(f"✓ Features totais: {X.shape[1]}")
print(f"✓ Features categóricas: {len(categorical_cols)}")

# Divisão temporal
tscv = TimeSeriesSplit(n_splits=3)
for fold, (train_index, test_index) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    if fold == 2:
        break

print(f"✓ Train: {len(X_train):,} | Test: {len(X_test):,}")

# Criar pools
train_pool = Pool(data=X_train, label=y_train, cat_features=categorical_cols if categorical_cols else None)
test_pool = Pool(data=X_test, label=y_test, cat_features=categorical_cols if categorical_cols else None)

# ===========================================================================
# EXPERIMENTOS COM DIFERENTES CONFIGURAÇÕES
# ===========================================================================
print(f"\n{'='*80}")
print(f"INICIANDO EXPERIMENTOS")
print(f"{'='*80}")

# Definir configurações para testar
experiments = {
    "BASELINE": {
        "iterations": 50,
        "learning_rate": 0.05,
        "depth": 12,
        "auto_class_weights": "Balanced",
        "l2_leaf_reg": 1.0,
        "border_count": 128,
        "subsample": 0.85,
        "rsm": 0.85,
    },
    
    "HIGH_ITERATIONS": {
        "iterations": 150,  # 3x mais iterações
        "learning_rate": 0.03,  # LR menor para compensar
        "depth": 12,
        "auto_class_weights": "Balanced",
        "l2_leaf_reg": 1.0,
        "border_count": 128,
        "subsample": 0.85,
        "rsm": 0.85,
    },
    
    "DEEP_TREES": {
        "iterations": 50,
        "learning_rate": 0.05,
        "depth": 16,  # Máxima profundidade
        "auto_class_weights": "Balanced",
        "l2_leaf_reg": 2.0,  # Mais regularização para compensar
        "border_count": 128,
        "subsample": 0.85,
        "rsm": 0.85,
    },
    
    "HIGH_LEARNING_RATE": {
        "iterations": 50,
        "learning_rate": 0.1,  # LR alto
        "depth": 10,  # Profundidade menor para compensar
        "auto_class_weights": "Balanced",
        "l2_leaf_reg": 1.0,
        "border_count": 128,
        "subsample": 0.85,
        "rsm": 0.85,
    },
    
    "MORE_BORDERS": {
        "iterations": 50,
        "learning_rate": 0.05,
        "depth": 12,
        "auto_class_weights": "Balanced",
        "l2_leaf_reg": 1.0,
        "border_count": 254,  # Mais splits
        "subsample": 0.85,
        "rsm": 0.85,
    },
    
    "LESS_REGULARIZATION": {
        "iterations": 50,
        "learning_rate": 0.05,
        "depth": 12,
        "auto_class_weights": "Balanced",
        "l2_leaf_reg": 0.1,  # Menos regularização
        "border_count": 128,
        "subsample": 0.9,  # Mais dados
        "rsm": 0.9,  # Mais features
    },
    
    "MORE_REGULARIZATION": {
        "iterations": 50,
        "learning_rate": 0.05,
        "depth": 12,
        "auto_class_weights": "Balanced",
        "l2_leaf_reg": 3.0,  # Mais regularização
        "border_count": 128,
        "subsample": 0.7,  # Menos dados
        "rsm": 0.7,  # Menos features
    },
    
    "BOOTSTRAP_BAYESIAN": {
        "iterations": 50,
        "learning_rate": 0.05,
        "depth": 12,
        "auto_class_weights": "Balanced",
        "l2_leaf_reg": 1.0,
        "border_count": 128,
        "bootstrap_type": "Bayesian",  # Diferente tipo de bootstrap
        "bagging_temperature": 1.0,  # Disponível só com Bayesian
    },
    
    "OPTIMAL_COMBO": {
        "iterations": 100,  # Médio
        "learning_rate": 0.04,  # Médio-baixo
        "depth": 14,  # Médio-alto
        "auto_class_weights": "Balanced",
        "l2_leaf_reg": 1.5,  # Médio
        "border_count": 200,  # Médio-alto
        "subsample": 0.8,
        "rsm": 0.8,
        "random_strength": 0.5,  # Menos randomização
    }
}

results = []

for exp_name, params in experiments.items():
    print(f"\n🔬 EXPERIMENTO: {exp_name}")
    print(f"{'='*40}")
    
    start_time = time.time()
    
    # Criar modelo com parâmetros específicos
    model_params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "task_type": "CPU",
        "verbose": False,  # Silenciar logs
        "early_stopping_rounds": 10,  # Parar cedo para acelerar
        "random_seed": 42,
        "thread_count": -1,
        **params  # Adicionar parâmetros específicos do experimento
    }
    
    model = CatBoostClassifier(**model_params)
    
    # Treinar
    model.fit(train_pool, eval_set=test_pool, verbose=False, plot=False)
    
    # Predizer
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calcular métricas
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Otimizar threshold
    best_f1_macro = 0
    best_threshold = 0.5
    
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        y_pred_temp = (y_pred_proba >= threshold).astype(int)
        f1_macro = f1_score(y_test, y_pred_temp, average='macro', zero_division=0)
        if f1_macro > best_f1_macro:
            best_f1_macro = f1_macro
            best_threshold = threshold
    
    y_pred = (y_pred_proba >= best_threshold).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    train_time = time.time() - start_time
    best_iteration = model.get_best_iteration()
    
    # Armazenar resultados
    result = {
        "experiment": exp_name,
        "roc_auc": roc_auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "f1_macro": best_f1_macro,
        "best_threshold": best_threshold,
        "best_iteration": best_iteration,
        "train_time": train_time,
        **params  # Incluir todos os parâmetros
    }
    results.append(result)
    
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"F1-Macro: {best_f1_macro:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Tempo: {train_time:.1f}s")
    print(f"Iterações: {best_iteration}")

# ===========================================================================
# ANÁLISE DOS RESULTADOS
# ===========================================================================
print(f"\n{'='*80}")
print(f"ANÁLISE COMPARATIVA DOS RESULTADOS")
print(f"{'='*80}")

# Converter para DataFrame
results_df = pd.DataFrame(results)

# Ordenar por ROC-AUC
results_df = results_df.sort_values('roc_auc', ascending=False)

print(f"\nRANKING POR ROC-AUC:")
print(f"{'='*60}")
for i, row in results_df.iterrows():
    print(f"{row['experiment']:20s} | ROC-AUC: {row['roc_auc']:.4f} | F1-Macro: {row['f1_macro']:.4f} | Tempo: {row['train_time']:.1f}s")

# Identificar melhor configuração
best_config = results_df.iloc[0]
print(f"\n🏆 MELHOR CONFIGURAÇÃO: {best_config['experiment']}")
print(f"{'='*60}")
print(f"ROC-AUC:      {best_config['roc_auc']:.4f}")
print(f"F1-Macro:     {best_config['f1_macro']:.4f}")
print(f"Accuracy:     {best_config['accuracy']:.4f}")
print(f"Precision:    {best_config['precision']:.4f}")
print(f"Recall:       {best_config['recall']:.4f}")
print(f"Threshold:    {best_config['best_threshold']:.2f}")
print(f"Tempo:        {best_config['train_time']:.1f}s")

print(f"\nPARÂMETROS DA MELHOR CONFIGURAÇÃO:")
print(f"{'='*40}")
param_cols = ['iterations', 'learning_rate', 'depth', 'l2_leaf_reg', 'border_count', 'subsample', 'rsm']
for param in param_cols:
    if param in best_config:
        print(f"{param:20s}: {best_config[param]}")

# ===========================================================================
# ANÁLISE DE IMPACTO DOS PARÂMETROS
# ===========================================================================
print(f"\n{'='*80}")
print(f"ANÁLISE DE IMPACTO DOS PARÂMETROS")
print(f"{'='*80}")

baseline_auc = results_df[results_df['experiment'] == 'BASELINE']['roc_auc'].iloc[0]

print(f"BASELINE ROC-AUC: {baseline_auc:.4f}")
print(f"\nIMPACTO RELATIVO (vs BASELINE):")
print(f"{'='*50}")

for i, row in results_df.iterrows():
    if row['experiment'] != 'BASELINE':
        impact = row['roc_auc'] - baseline_auc
        impact_pct = (impact / baseline_auc) * 100
        symbol = "📈" if impact > 0 else "📉" if impact < 0 else "➡️"
        print(f"{symbol} {row['experiment']:20s}: {impact:+.4f} ({impact_pct:+.2f}%)")

# ===========================================================================
# SALVAR RESULTADOS
# ===========================================================================
print(f"\n{'='*80}")
print(f"SALVANDO RESULTADOS")
print(f"{'='*80}")

# Salvar CSV com todos os resultados
results_df.to_csv('../../reports/catboost_experiments.csv', index=False)
print("✓ Resultados salvos: reports/catboost_experiments.csv")

# Salvar relatório detalhado
with open('../../reports/catboost_experiments_report.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("EXPERIMENTO V6 - OTIMIZAÇÃO DE HIPERPARÂMETROS CATBOOST\n")
    f.write("="*80 + "\n\n")
    
    f.write("RANKING POR ROC-AUC:\n")
    f.write("="*60 + "\n")
    for i, row in results_df.iterrows():
        f.write(f"{row['experiment']:20s} | ROC-AUC: {row['roc_auc']:.4f} | F1-Macro: {row['f1_macro']:.4f} | Tempo: {row['train_time']:.1f}s\n")
    
    f.write(f"\nMELHOR CONFIGURAÇÃO: {best_config['experiment']}\n")
    f.write("="*60 + "\n")
    f.write(f"ROC-AUC:      {best_config['roc_auc']:.4f}\n")
    f.write(f"F1-Macro:     {best_config['f1_macro']:.4f}\n")
    f.write(f"Accuracy:     {best_config['accuracy']:.4f}\n")
    f.write(f"Precision:    {best_config['precision']:.4f}\n")
    f.write(f"Recall:       {best_config['recall']:.4f}\n")
    f.write(f"Threshold:    {best_config['best_threshold']:.2f}\n")
    f.write(f"Tempo:        {best_config['train_time']:.1f}s\n\n")
    
    f.write("PARÂMETROS DA MELHOR CONFIGURAÇÃO:\n")
    f.write("="*40 + "\n")
    for param in param_cols:
        if param in best_config:
            f.write(f"{param:20s}: {best_config[param]}\n")
    
    f.write(f"\nIMPACTO RELATIVO (vs BASELINE {baseline_auc:.4f}):\n")
    f.write("="*50 + "\n")
    for i, row in results_df.iterrows():
        if row['experiment'] != 'BASELINE':
            impact = row['roc_auc'] - baseline_auc
            impact_pct = (impact / baseline_auc) * 100
            f.write(f"{row['experiment']:20s}: {impact:+.4f} ({impact_pct:+.2f}%)\n")

print("✓ Relatório salvo: reports/catboost_experiments_report.txt")

# Visualização
plt.figure(figsize=(12, 8))
plt.barh(range(len(results_df)), results_df['roc_auc'], color='green', alpha=0.7)
plt.yticks(range(len(results_df)), results_df['experiment'])
plt.xlabel('ROC-AUC')
plt.title('Comparação de Configurações CatBoost - ROC-AUC')
plt.axvline(x=baseline_auc, color='red', linestyle='--', label=f'Baseline: {baseline_auc:.4f}')
plt.legend()
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('../../visualizations/v6/catboost_experiments_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Visualização salva: visualizations/v6/catboost_experiments_comparison.png")

print(f"\n{'='*80}")
print(f"✅ EXPERIMENTOS CONCLUÍDOS!")
print(f"{'='*80}")
print(f"\nRECOMENDAÇÃO:")
print(f"Use a configuração '{best_config['experiment']}' para máxima performance!")
print(f"ROC-AUC esperado: {best_config['roc_auc']:.4f}")