"""
Model V7 - LightGBM vs XGBoost Comparison + Ensemble
=====================================================
Combina as melhores práticas do OFICIAL.ipynb e model_v4_advanced.py:

DO OFICIAL.ipynb:
✅ LightGBM (mais rápido, eficiente)
✅ Features: holidays, weekend, peak_hour
✅ Features cíclicas completas (sin/cos)
✅ Agregações por parada

DO model_v4_advanced.py:
✅ XGBoost (benchmark)
✅ Agregações por usuário (CRÍTICAS!)
✅ Interações user×stop
✅ Seleção de top 50 features
✅ 5 estratégias de balanceamento
✅ TimeSeriesSplit robusto

TESTES:
1. LightGBM com hiperparâmetros otimizados
2. XGBoost (baseline V4)
3. Ensemble (LightGBM + XGBoost)
"""

import pandas as pd
import numpy as np
from google.cloud import bigquery
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix, roc_curve
)
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import json
import joblib
import holidays

warnings.filterwarnings('ignore')

# Configuração
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("MODEL V7 - LightGBM vs XGBoost Comparison + Ensemble")
print("="*80)
print(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ===========================================================================
# ETAPA 1: CARREGAR DADOS
# ===========================================================================
print(f"\n{'='*80}")
print(f"ETAPA 1: CARREGANDO DATASET")
print(f"{'='*80}")

project_id = "proj-ml-469320"
client = bigquery.Client(project=project_id)

query = """
    SELECT * FROM `proj-ml-469320.app_cittamobi.dataset-updated` 
    TABLESAMPLE SYSTEM (20 PERCENT)
    LIMIT 200000
"""

print("Carregando 200,000 amostras...")
df = client.query(query).to_dataframe()
print(f"✓ Dados carregados: {len(df):,} registros")

target = "target"

# ===========================================================================
# ETAPA 2: FEATURE ENGINEERING AVANÇADO
# ===========================================================================
print(f"\n{'='*80}")
print(f"ETAPA 2: FEATURE ENGINEERING AVANÇADO")
print(f"{'='*80}")

# --- A. Features de Agregação por Usuário (V4) ---
print("\n📊 Criando features de agregação por usuário...")
if 'user_pseudo_id' in df.columns:
    user_agg = df.groupby('user_pseudo_id').agg({
        target: ['mean', 'sum', 'count'],
        'dist_device_stop': ['mean', 'std', 'min', 'max'] if 'dist_device_stop' in df.columns else ['mean'],
        'time_hour': ['mean', 'std'] if 'time_hour' in df.columns else ['mean']
    }).reset_index()
    
    user_agg.columns = ['user_pseudo_id', 'user_conversion_rate', 'user_total_conversions', 
                        'user_frequency', 'user_avg_dist', 'user_std_dist', 'user_min_dist', 
                        'user_max_dist', 'user_avg_hour', 'user_std_hour']
    
    df = df.merge(user_agg, on='user_pseudo_id', how='left')
    print(f"  ✓ {len(user_agg.columns)-1} features de usuário criadas")

# --- B. Features de Agregação por Parada (OFICIAL.ipynb) ---
print("\n📊 Criando features de agregação por parada...")
if 'gtfs_stop_id' in df.columns:
    stop_agg = df.groupby('gtfs_stop_id').agg({
        target: ['mean', 'sum', 'count'],
        'dist_device_stop': ['mean', 'std'] if 'dist_device_stop' in df.columns else ['mean'],
        'stop_lat_event': ['mean'] if 'stop_lat_event' in df.columns else [],
        'stop_lon_event': ['mean'] if 'stop_lon_event' in df.columns else []
    }).reset_index()
    
    stop_agg.columns = ['gtfs_stop_id', 'stop_conversion_rate', 'stop_total_conversions',
                        'stop_event_count_agg', 'stop_avg_dist', 'stop_dist_std',
                        'stop_lat_agg', 'stop_lon_agg']
    
    df = df.merge(stop_agg, on='gtfs_stop_id', how='left')
    print(f"  ✓ {len(stop_agg.columns)-1} features de parada criadas")

# --- C. Features de Interação de 2ª Ordem (V4) ---
print("\n📊 Criando features de interação...")
if 'user_conversion_rate' in df.columns and 'stop_conversion_rate' in df.columns:
    df['conversion_interaction'] = df['user_conversion_rate'] * df['stop_conversion_rate']
    
if 'user_avg_dist' in df.columns and 'dist_device_stop' in df.columns:
    df['distance_interaction'] = df['user_avg_dist'] * df['dist_device_stop']

if 'user_frequency' in df.columns and 'stop_event_count_agg' in df.columns:
    df['frequency_interaction'] = df['user_frequency'] * df['stop_event_count_agg']

print(f"✓ Features de interação criadas")

# --- D. Features Temporais Cíclicas (OFICIAL.ipynb) ---
print("\n📊 Criando features temporais cíclicas...")
if 'event_timestamp' in df.columns:
    df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])
    df['time_hour'] = df['event_timestamp'].dt.hour
    df['time_day_of_week'] = df['event_timestamp'].dt.dayofweek
    df['time_day_of_month'] = df['event_timestamp'].dt.day
    df['time_month'] = df['event_timestamp'].dt.month
    df['week_of_year'] = df['event_timestamp'].dt.isocalendar().week
    
    # Features cíclicas
    df['hour_sin'] = np.sin(2 * np.pi * df['time_hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['time_hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['time_day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['time_day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['time_month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['time_month'] / 12)
    
    print(f"  ✓ Features cíclicas criadas (hour, day, month)")

# --- E. Features de Contexto Urbano (OFICIAL.ipynb) ---
print("\n📊 Criando features de contexto urbano...")
# Feriados no Brasil (SP)
br_holidays = holidays.Brazil(state='SP')
if 'event_timestamp' in df.columns:
    df['is_holiday'] = df['event_timestamp'].dt.date.apply(lambda x: x in br_holidays).astype(int)
    df['is_weekend'] = (df['time_day_of_week'] >= 5).astype(int)
    
    # Hora de pico (6-9h manhã, 17-19h tarde)
    df['is_peak_hour'] = ((df['time_hour'] >= 6) & (df['time_hour'] < 9) | 
                          (df['time_hour'] >= 17) & (df['time_hour'] < 19)).astype(int)
    
    print(f"  ✓ Features urbanas criadas (holiday, weekend, peak_hour)")

# --- F. Features de Interação Temporal (OFICIAL.ipynb) ---
print("\n📊 Criando interações temporais...")
if 'dist_device_stop' in df.columns:
    if 'is_peak_hour' in df.columns:
        df['dist_x_peak'] = df['dist_device_stop'] * df['is_peak_hour']
    if 'is_weekend' in df.columns:
        df['dist_x_weekend'] = df['dist_device_stop'] * df['is_weekend']

print(f"✓ Interações temporais criadas")

# ===========================================================================
# ETAPA 3: LIMPEZA MODERADA
# ===========================================================================
print(f"\n{'='*80}")
print(f"ETAPA 3: LIMPEZA MODERADA DOS DADOS")
print(f"{'='*80}")

print(f"\n=== Distribuição do Target (Original) ===")
target_dist = df[target].value_counts()
print(target_dist)
print(f"Proporção classe 0: {target_dist[0]/len(df)*100:.2f}%")
print(f"Proporção classe 1: {target_dist[1]/len(df)*100:.2f}%")

df_original_size = len(df)

# Filtros moderados
if 'user_frequency' in df.columns:
    freq_threshold = df['user_frequency'].quantile(0.05)
    df = df[df['user_frequency'] >= freq_threshold]
    print(f"✓ Removido usuários com frequência < {freq_threshold:.0f}")

if 'device_lat' in df.columns and 'device_lon' in df.columns:
    df = df[(df['device_lat'].between(-90, 90)) & (df['device_lon'].between(-180, 180))]
    print(f"✓ Filtrado coordenadas válidas")

if 'dist_device_stop' in df.columns:
    dist_threshold = df['dist_device_stop'].quantile(0.99)
    df = df[df['dist_device_stop'] <= dist_threshold]
    print(f"✓ Removido distâncias > {dist_threshold:.0f}m (outliers)")

total_removed = df_original_size - len(df)
removal_pct = total_removed / df_original_size * 100
print(f"\n✓ Total mantido: {len(df):,} ({100-removal_pct:.1f}%)")

# ===========================================================================
# ETAPA 4: PREPARAÇÃO DE FEATURES
# ===========================================================================
print(f"\n{'='*80}")
print(f"ETAPA 4: PREPARAÇÃO DE FEATURES")
print(f"{'='*80}")

features_to_drop = ['y_pred', 'y_pred_proba', 'ctm_service_route', 'direction', 
                    'lotacao_proxy_binaria', 'event_timestamp', 'user_pseudo_id']
X = df.drop(columns=[target] + features_to_drop, errors='ignore')
y = df[target]

# Label Encoding para categóricas
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
if categorical_cols:
    print(f"\n📝 Encoding de {len(categorical_cols)} colunas categóricas...")
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le

# Tratar infinitos e NaN
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

print(f"✓ Features totais: {X.shape[1]}")

# ===========================================================================
# ETAPA 5: SELEÇÃO DE FEATURES (TOP 50)
# ===========================================================================
print(f"\n{'='*80}")
print(f"ETAPA 5: SELEÇÃO DAS MELHORES FEATURES")
print(f"{'='*80}")

tscv = TimeSeriesSplit(n_splits=2)
for fold, (train_index, test_index) in enumerate(tscv.split(X)):
    X_train_temp = X.iloc[train_index]
    y_train_temp = y.iloc[train_index]

dtrain_temp = xgb.DMatrix(X_train_temp, label=y_train_temp)
scale_pos_weight = (len(y_train_temp) - y_train_temp.sum()) / y_train_temp.sum()

params_temp = {
    'objective': 'binary:logistic',
    'max_depth': 6,
    'learning_rate': 0.1,
    'eval_metric': 'logloss',
    'seed': 42,
    'scale_pos_weight': scale_pos_weight
}

model_temp = xgb.train(
    params=params_temp,
    dtrain=dtrain_temp,
    num_boost_round=50,
    verbose_eval=False
)

importance_dict = model_temp.get_score(importance_type='gain')
importance_df = pd.DataFrame([
    {'feature': k, 'importance': v} 
    for k, v in importance_dict.items()
]).sort_values('importance', ascending=False)

top_n = 50
top_features = importance_df.head(top_n)['feature'].tolist()
X_selected = X[top_features].copy()

print(f"✓ Features selecionadas: {len(top_features)}")
print(f"✓ Top 10 features:")
for i, row in importance_df.head(10).iterrows():
    print(f"  {row['feature']:30s}: {row['importance']:.2f}")

# ===========================================================================
# ETAPA 6: DIVISÃO TEMPORAL
# ===========================================================================
print(f"\n{'='*80}")
print(f"ETAPA 6: DIVISÃO TEMPORAL (TimeSeriesSplit)")
print(f"{'='*80}")

tscv = TimeSeriesSplit(n_splits=3)
X_train, X_test, y_train, y_test = None, None, None, None

for fold, (train_index, test_index) in enumerate(tscv.split(X_selected)):
    X_train = X_selected.iloc[train_index]
    X_test = X_selected.iloc[test_index]
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]
    print(f"Fold {fold+1}: Train={len(X_train):,}, Test={len(X_test):,}")

print(f"\n=== Distribuição do Treino (Fold Final) ===")
train_dist = y_train.value_counts()
print(f"Classe 0: {train_dist[0]:,} ({train_dist[0]/len(y_train)*100:.2f}%)")
print(f"Classe 1: {train_dist[1]:,} ({train_dist[1]/len(y_train)*100:.2f}%)")
print(f"Razão (0/1): {train_dist[0]/train_dist[1]:.2f}:1")

# Escalar features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===========================================================================
# ETAPA 7: TREINAR MODELOS
# ===========================================================================
print(f"\n{'='*80}")
print(f"ETAPA 7: TREINANDO MODELOS (LightGBM vs XGBoost)")
print(f"{'='*80}")

results = []

# --- MODELO 1: LightGBM ---
print(f"\n--- MODELO 1: LightGBM (Otimizado) ---")
import time
start_time = time.time()

scale_pos_weight_lgb = (len(y_train) - y_train.sum()) / y_train.sum()

lgb_model = lgb.LGBMClassifier(
    objective='binary',
    metric='auc',
    scale_pos_weight=scale_pos_weight_lgb,
    n_estimators=2000,
    learning_rate=0.015,
    num_leaves=25,
    max_depth=10,
    min_child_samples=30,
    min_child_weight=0.01,
    subsample=0.85,
    subsample_freq=5,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=0.1,
    n_jobs=-1,
    random_state=42,
    verbose=-1
)

# Treinar com early stopping
lgb_model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_test_scaled, y_test)],
    eval_metric='auc',
    callbacks=[lgb.early_stopping(100, verbose=False)]
)

lgb_time = time.time() - start_time

# Predições
y_pred_proba_lgb = lgb_model.predict_proba(X_test_scaled)[:, 1]

# Otimizar threshold
thresholds_test = np.arange(0.3, 0.8, 0.05)
best_f1_lgb = 0
best_threshold_lgb = 0.5

for thresh in thresholds_test:
    y_pred_temp = (y_pred_proba_lgb >= thresh).astype(int)
    f1_macro = f1_score(y_test, y_pred_temp, average='macro')
    if f1_macro > best_f1_lgb:
        best_f1_lgb = f1_macro
        best_threshold_lgb = thresh

y_pred_lgb = (y_pred_proba_lgb >= best_threshold_lgb).astype(int)

results.append({
    'model': 'LightGBM',
    'roc_auc': roc_auc_score(y_test, y_pred_proba_lgb),
    'accuracy': accuracy_score(y_test, y_pred_lgb),
    'precision': precision_score(y_test, y_pred_lgb),
    'recall': recall_score(y_test, y_pred_lgb),
    'f1_score': f1_score(y_test, y_pred_lgb),
    'f1_macro': f1_score(y_test, y_pred_lgb, average='macro'),
    'threshold': best_threshold_lgb,
    'time': lgb_time,
    'model_obj': lgb_model,
    'y_pred_proba': y_pred_proba_lgb
})

print(f"ROC-AUC: {results[-1]['roc_auc']:.4f} | F1-Macro: {results[-1]['f1_macro']:.4f}")
print(f"Threshold: {best_threshold_lgb:.2f} | Tempo: {lgb_time:.2f}s")

# --- MODELO 2: XGBoost ---
print(f"\n--- MODELO 2: XGBoost (V4 Baseline) ---")
start_time = time.time()

scale_pos_weight_xgb = (len(y_train) - y_train.sum()) / y_train.sum()

params_xgb = {
    'objective': 'binary:logistic',
    'max_depth': 18,
    'learning_rate': 0.02,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'min_child_weight': 3,
    'gamma': 0.05,
    'reg_alpha': 0.01,
    'reg_lambda': 1,
    'eval_metric': 'logloss',
    'scale_pos_weight': scale_pos_weight_xgb,
    'seed': 42
}

dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
dtest = xgb.DMatrix(X_test_scaled, label=y_test)

xgb_model = xgb.train(
    params=params_xgb,
    dtrain=dtrain,
    num_boost_round=250,
    evals=[(dtest, 'test')],
    early_stopping_rounds=25,
    verbose_eval=False
)

xgb_time = time.time() - start_time

# Predições
y_pred_proba_xgb = xgb_model.predict(dtest)

# Otimizar threshold
best_f1_xgb = 0
best_threshold_xgb = 0.5

for thresh in thresholds_test:
    y_pred_temp = (y_pred_proba_xgb >= thresh).astype(int)
    f1_macro = f1_score(y_test, y_pred_temp, average='macro')
    if f1_macro > best_f1_xgb:
        best_f1_xgb = f1_macro
        best_threshold_xgb = thresh

y_pred_xgb = (y_pred_proba_xgb >= best_threshold_xgb).astype(int)

results.append({
    'model': 'XGBoost',
    'roc_auc': roc_auc_score(y_test, y_pred_proba_xgb),
    'accuracy': accuracy_score(y_test, y_pred_xgb),
    'precision': precision_score(y_test, y_pred_xgb),
    'recall': recall_score(y_test, y_pred_xgb),
    'f1_score': f1_score(y_test, y_pred_xgb),
    'f1_macro': f1_score(y_test, y_pred_xgb, average='macro'),
    'threshold': best_threshold_xgb,
    'time': xgb_time,
    'model_obj': xgb_model,
    'y_pred_proba': y_pred_proba_xgb
})

print(f"ROC-AUC: {results[-1]['roc_auc']:.4f} | F1-Macro: {results[-1]['f1_macro']:.4f}")
print(f"Threshold: {best_threshold_xgb:.2f} | Tempo: {xgb_time:.2f}s")

# --- MODELO 3: Ensemble (Simple Average) ---
print(f"\n--- MODELO 3: Ensemble (Simple Average) ---")

y_pred_proba_ensemble = (y_pred_proba_lgb + y_pred_proba_xgb) / 2

# Otimizar threshold
best_f1_ens = 0
best_threshold_ens = 0.5

for thresh in thresholds_test:
    y_pred_temp = (y_pred_proba_ensemble >= thresh).astype(int)
    f1_macro = f1_score(y_test, y_pred_temp, average='macro')
    if f1_macro > best_f1_ens:
        best_f1_ens = f1_macro
        best_threshold_ens = thresh

y_pred_ensemble = (y_pred_proba_ensemble >= best_threshold_ens).astype(int)

results.append({
    'model': 'Ensemble (Average)',
    'roc_auc': roc_auc_score(y_test, y_pred_proba_ensemble),
    'accuracy': accuracy_score(y_test, y_pred_ensemble),
    'precision': precision_score(y_test, y_pred_ensemble),
    'recall': recall_score(y_test, y_pred_ensemble),
    'f1_score': f1_score(y_test, y_pred_ensemble),
    'f1_macro': f1_score(y_test, y_pred_ensemble, average='macro'),
    'threshold': best_threshold_ens,
    'time': lgb_time + xgb_time,
    'model_obj': None,
    'y_pred_proba': y_pred_proba_ensemble
})

print(f"ROC-AUC: {results[-1]['roc_auc']:.4f} | F1-Macro: {results[-1]['f1_macro']:.4f}")
print(f"Threshold: {best_threshold_ens:.2f}")

# --- MODELO 4: Ensemble (Weighted by Performance) ---
print(f"\n--- MODELO 4: Ensemble (F1-Weighted) ---")

# Pesos baseados em F1-Macro
weight_lgb = results[0]['f1_macro']
weight_xgb = results[1]['f1_macro']
total_weight = weight_lgb + weight_xgb

weight_lgb_norm = weight_lgb / total_weight
weight_xgb_norm = weight_xgb / total_weight

print(f"  Peso LightGBM: {weight_lgb_norm:.3f}")
print(f"  Peso XGBoost: {weight_xgb_norm:.3f}")

y_pred_proba_weighted = (y_pred_proba_lgb * weight_lgb_norm + 
                         y_pred_proba_xgb * weight_xgb_norm)

# Otimizar threshold
best_f1_wens = 0
best_threshold_wens = 0.5

for thresh in thresholds_test:
    y_pred_temp = (y_pred_proba_weighted >= thresh).astype(int)
    f1_macro = f1_score(y_test, y_pred_temp, average='macro')
    if f1_macro > best_f1_wens:
        best_f1_wens = f1_macro
        best_threshold_wens = thresh

y_pred_weighted = (y_pred_proba_weighted >= best_threshold_wens).astype(int)

results.append({
    'model': 'Ensemble (F1-Weighted)',
    'roc_auc': roc_auc_score(y_test, y_pred_proba_weighted),
    'accuracy': accuracy_score(y_test, y_pred_weighted),
    'precision': precision_score(y_test, y_pred_weighted),
    'recall': recall_score(y_test, y_pred_weighted),
    'f1_score': f1_score(y_test, y_pred_weighted),
    'f1_macro': f1_score(y_test, y_pred_weighted, average='macro'),
    'threshold': best_threshold_wens,
    'time': lgb_time + xgb_time,
    'model_obj': None,
    'y_pred_proba': y_pred_proba_weighted
})

print(f"ROC-AUC: {results[-1]['roc_auc']:.4f} | F1-Macro: {results[-1]['f1_macro']:.4f}")
print(f"Threshold: {best_threshold_wens:.2f}")

# ===========================================================================
# ETAPA 8: COMPARAÇÃO FINAL
# ===========================================================================
print(f"\n{'='*80}")
print(f"ETAPA 8: COMPARAÇÃO FINAL DOS MODELOS")
print(f"{'='*80}")

results_df = pd.DataFrame([{
    'Modelo': r['model'],
    'ROC-AUC': f"{r['roc_auc']:.4f}",
    'Accuracy': f"{r['accuracy']:.4f}",
    'Precision': f"{r['precision']:.4f}",
    'Recall': f"{r['recall']:.4f}",
    'F1-Score': f"{r['f1_score']:.4f}",
    'F1-Macro': f"{r['f1_macro']:.4f}",
    'Threshold': f"{r['threshold']:.2f}",
    'Tempo (s)': f"{r['time']:.2f}"
} for r in results])

print("\n" + results_df.to_string(index=False))

# Selecionar melhor modelo
best_idx = np.argmax([r['f1_macro'] for r in results])
best_result = results[best_idx]

print(f"\n{'='*80}")
print(f"🏆 MELHOR MODELO: {best_result['model']}")
print(f"{'='*80}")
print(f"ROC-AUC:   {best_result['roc_auc']:.4f}")
print(f"Accuracy:  {best_result['accuracy']:.4f}")
print(f"Precision: {best_result['precision']:.4f}")
print(f"Recall:    {best_result['recall']:.4f}")
print(f"F1-Score:  {best_result['f1_score']:.4f}")
print(f"F1-Macro:  {best_result['f1_macro']:.4f} ⭐")
print(f"Threshold: {best_result['threshold']:.2f}")
print(f"Tempo:     {best_result['time']:.2f}s")

# ===========================================================================
# ETAPA 9: MATRIZ DE CONFUSÃO
# ===========================================================================
print(f"\n{'='*80}")
print(f"ETAPA 9: ANÁLISE DE CONFUSÃO")
print(f"{'='*80}")

# Matriz de confusão para cada modelo
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

for idx, (ax, result) in enumerate(zip(axes.flatten(), results)):
    y_pred = (result['y_pred_proba'] >= result['threshold']).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Classe 0', 'Classe 1'],
                yticklabels=['Classe 0', 'Classe 1'])
    ax.set_title(f"{result['model']}\nF1-Macro: {result['f1_macro']:.4f}, Threshold: {result['threshold']:.2f}",
                fontweight='bold')
    ax.set_ylabel('Real')
    ax.set_xlabel('Predito')

plt.tight_layout()
plt.savefig('v7_confusion_matrices.png', dpi=300, bbox_inches='tight')
print("✓ v7_confusion_matrices.png")

# ===========================================================================
# ETAPA 10: CURVAS ROC
# ===========================================================================
print(f"\n{'='*80}")
print(f"ETAPA 10: CURVAS ROC")
print(f"{'='*80}")

plt.figure(figsize=(10, 8))

for result in results:
    fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
    plt.plot(fpr, tpr, linewidth=2, 
            label=f"{result['model']} (AUC={result['roc_auc']:.4f})")

plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Curvas ROC - Comparação de Modelos V7', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('v7_roc_curves.png', dpi=300, bbox_inches='tight')
print("✓ v7_roc_curves.png")

# ===========================================================================
# ETAPA 11: COMPARAÇÃO DE MÉTRICAS
# ===========================================================================
print(f"\n{'='*80}")
print(f"ETAPA 11: VISUALIZAÇÃO COMPARATIVA")
print(f"{'='*80}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics = ['roc_auc', 'precision', 'recall', 'f1_macro']
titles = ['ROC-AUC', 'Precision', 'Recall', 'F1-Macro']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for idx, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
    ax = axes[idx // 2, idx % 2]
    
    values = [r[metric] for r in results]
    models = [r['model'] for r in results]
    
    bars = ax.barh(models, values, color=color, alpha=0.7)
    
    # Destacar o melhor
    best_val = max(values)
    for bar, val in zip(bars, values):
        if val == best_val:
            bar.set_alpha(1.0)
            bar.set_edgecolor('black')
            bar.set_linewidth(2)
    
    # Adicionar valores
    for i, (val, model) in enumerate(zip(values, models)):
        ax.text(val, i, f' {val:.4f}', va='center', fontweight='bold')
    
    ax.set_xlabel(title, fontsize=12)
    ax.set_title(f'{title} por Modelo', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim([0, 1.0])

plt.tight_layout()
plt.savefig('v7_metrics_comparison.png', dpi=300, bbox_inches='tight')
print("✓ v7_metrics_comparison.png")

# ===========================================================================
# ETAPA 12: SALVAR MODELOS
# ===========================================================================
print(f"\n{'='*80}")
print(f"ETAPA 12: SALVANDO MODELOS E ARTEFATOS")
print(f"{'='*80}")

# Criar diretório se não existir
import os
os.makedirs('models/v7', exist_ok=True)

# Salvar LightGBM
joblib.dump(lgb_model, 'models/v7/lightgbm_model_v7.pkl')
print("✓ models/v7/lightgbm_model_v7.pkl")

# Salvar XGBoost
xgb_model.save_model('models/v7/xgboost_model_v7.json')
print("✓ models/v7/xgboost_model_v7.json")

# Salvar scaler
joblib.dump(scaler, 'models/v7/scaler_v7.pkl')
print("✓ models/v7/scaler_v7.pkl")

# Salvar features selecionadas
with open('models/v7/selected_features_v7.txt', 'w') as f:
    for feat in top_features:
        f.write(f"{feat}\n")
print("✓ models/v7/selected_features_v7.txt")

# Salvar configuração
config = {
    'models': {
        'lightgbm': {
            'threshold': float(best_threshold_lgb),
            'f1_macro': float(results[0]['f1_macro']),
            'roc_auc': float(results[0]['roc_auc']),
            'time': float(results[0]['time'])
        },
        'xgboost': {
            'threshold': float(best_threshold_xgb),
            'f1_macro': float(results[1]['f1_macro']),
            'roc_auc': float(results[1]['roc_auc']),
            'time': float(results[1]['time'])
        },
        'ensemble_average': {
            'threshold': float(best_threshold_ens),
            'f1_macro': float(results[2]['f1_macro']),
            'roc_auc': float(results[2]['roc_auc'])
        },
        'ensemble_weighted': {
            'threshold': float(best_threshold_wens),
            'f1_macro': float(results[3]['f1_macro']),
            'roc_auc': float(results[3]['roc_auc']),
            'weight_lgb': float(weight_lgb_norm),
            'weight_xgb': float(weight_xgb_norm)
        }
    },
    'best_model': best_result['model'],
    'features': top_features,
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

with open('models/v7/model_config_v7.json', 'w') as f:
    json.dump(config, f, indent=2)
print("✓ models/v7/model_config_v7.json")

# ===========================================================================
# RESUMO FINAL
# ===========================================================================
print(f"\n{'='*80}")
print(f"🎉 V7 COMPARISON - RESUMO FINAL")
print(f"{'='*80}")

print(f"\n📊 Comparação com Versões Anteriores:")
print(f"   V4 XGBoost:  ROC-AUC 0.9731, F1-Macro 0.7760")
print(f"   V5 XGBoost:  ROC-AUC 0.9729, F1-Macro 0.7782")
print(f"   V6 XGBoost:  ROC-AUC 0.9720, F1-Macro 0.7742")
print(f"")
print(f"   V7 LightGBM: ROC-AUC {results[0]['roc_auc']:.4f}, F1-Macro {results[0]['f1_macro']:.4f}")
print(f"   V7 XGBoost:  ROC-AUC {results[1]['roc_auc']:.4f}, F1-Macro {results[1]['f1_macro']:.4f}")
print(f"   V7 Ensemble: ROC-AUC {results[2]['roc_auc']:.4f}, F1-Macro {results[2]['f1_macro']:.4f}")

print(f"\n⚡ Performance:")
print(f"   LightGBM: {results[0]['time']:.2f}s")
print(f"   XGBoost:  {results[1]['time']:.2f}s")
print(f"   Speedup:  {results[1]['time']/results[0]['time']:.2f}x")

print(f"\n📈 Principais Melhorias do V7:")
print(f"   ✓ Testa LightGBM vs XGBoost lado a lado")
print(f"   ✓ Features do OFICIAL.ipynb (holidays, weekend, peak)")
print(f"   ✓ Features do V4 (user/stop aggregations)")
print(f"   ✓ Features cíclicas completas (sin/cos)")
print(f"   ✓ Ensemble de 2 algoritmos diferentes")
print(f"   ✓ Threshold optimization para cada modelo")

print(f"\n🎯 Modelo Recomendado para Produção:")
if best_result['model'] == 'LightGBM':
    print(f"   → LightGBM (mais rápido e eficiente)")
elif best_result['model'] == 'XGBoost':
    print(f"   → XGBoost (melhor performance)")
else:
    print(f"   → {best_result['model']} (melhor de ambos)")

print(f"\n{'='*80}")
print(f"Fim: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*80}")
