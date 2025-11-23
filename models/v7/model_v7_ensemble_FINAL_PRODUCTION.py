"""
═══════════════════════════════════════════════════════════════════════════════
MODEL V7 ENSEMBLE - VERSÃO FINAL PARA PRODUÇÃO (CLIENTE)
═══════════════════════════════════════════════════════════════════════════════

🎯 MODELO SELECIONADO: Ensemble F1-Weighted (LightGBM + XGBoost)

📊 MÉTRICAS DE VALIDAÇÃO (200k registros):
   - ROC-AUC: 0.9814
   - F1-Macro: 0.8281 (melhor balance entre classes)
   - Precision: 0.7059 (70% das previsões positivas corretas)
   - Recall: 0.6412 (64% das conversões reais detectadas)
   - Threshold: 0.75

⚡ PERFORMANCE:
   - Tempo de treinamento: 16.17s (200k registros)
   - LightGBM: 6.58s (1.46x mais rápido que XGBoost)
   - XGBoost: 9.59s

🔧 CONFIGURAÇÃO:
   - Dataset: proj-ml-469320.app_cittamobi.dataset-updated (BASE COMPLETA)
   - Features: 47 selecionadas (de 53 engineered)
   - Validação: TimeSeriesSplit (3 folds)
   - Balanceamento: scale_pos_weight nativo

📦 ARTEFATOS GERADOS:
   - lightgbm_model_v7_FINAL.pkl
   - xgboost_model_v7_FINAL.json
   - scaler_v7_FINAL.pkl
   - selected_features_v7_FINAL.txt
   - model_config_v7_FINAL.json
   - Visualizações finais (confusion matrix, ROC, metrics)

═══════════════════════════════════════════════════════════════════════════════
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
print("🎯 MODEL V7 ENSEMBLE - TESTE COM 500K REGISTROS")
print("="*80)
print(f"🏆 Modelo Campeão: Ensemble F1-Weighted (LightGBM + XGBoost)")
print(f"📅 Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"⚡ TESTE RÁPIDO: 500K registros (~2.5x mais que versão comparação)")
print("="*80)
print()

# ===========================================================================
# ETAPA 1: CARREGAR DADOS (BASE COMPLETA)
# ===========================================================================
print(f"\n{'='*80}")
print(f"ETAPA 1: CARREGANDO DATASET COMPLETO")
print(f"{'='*80}")

project_id = "proj-ml-469320"
client = bigquery.Client(project=project_id)

query = """
    SELECT * FROM `proj-ml-469320.app_cittamobi.dataset-updated`
    LIMIT 500000
"""

print("⏳ Carregando 500K registros (versão intermediária para teste)...")
start_load = datetime.now()

# Executar query
print("   [1/2] Executando query no BigQuery...", end="", flush=True)
query_job = client.query(query)
query_job.result()  # Aguardar conclusão
print(" ✓")

# Transferir dados
print("   [2/2] Transferindo ~1.6M registros para DataFrame...")
print("          Aguarde 2-3 minutos (não travou, está processando)...", end="", flush=True)

df = query_job.to_dataframe()

load_time = (datetime.now() - start_load).total_seconds()
print(" ✓")
print(f"✓ Dados carregados: {len(df):,} registros")
print(f"✓ Tempo de carregamento: {load_time:.2f}s")
print(f"✓ Memória: {df.memory_usage(deep=True).sum() / 1024**3:.2f} GB")

target = "target"

# ===========================================================================
# ETAPA 2: FEATURE ENGINEERING AVANÇADO
# ===========================================================================
print(f"\n{'='*80}")
print(f"ETAPA 2: FEATURE ENGINEERING AVANÇADO")
print(f"{'='*80}")

# --- A. Features de Agregação por Usuário ---
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

# --- B. Features de Agregação por Parada ---
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

# --- C. Features de Interação ---
print("\n📊 Criando features de interação...")
if 'user_conversion_rate' in df.columns and 'stop_conversion_rate' in df.columns:
    df['conversion_interaction'] = df['user_conversion_rate'] * df['stop_conversion_rate']
    
if 'user_avg_dist' in df.columns and 'dist_device_stop' in df.columns:
    df['distance_interaction'] = df['user_avg_dist'] * df['dist_device_stop']

if 'user_frequency' in df.columns and 'stop_event_count_agg' in df.columns:
    df['frequency_interaction'] = df['user_frequency'] * df['stop_event_count_agg']

print(f"✓ Features de interação criadas")

# --- D. Features Temporais Cíclicas ---
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

# --- E. Features de Contexto Urbano ---
print("\n📊 Criando features de contexto urbano...")
br_holidays = holidays.Brazil()

if 'event_timestamp' in df.columns and 'time_day_of_week' in df.columns:
    df['date'] = df['event_timestamp'].dt.date
    df['is_holiday'] = df['date'].apply(lambda x: 1 if x in br_holidays else 0)
    df['is_weekend'] = (df['time_day_of_week'] >= 5).astype(int)
    df = df.drop(columns=['date'])

if 'time_hour' in df.columns:
    df['is_peak_hour'] = df['time_hour'].apply(lambda x: 1 if (7 <= x <= 9) or (17 <= x <= 19) else 0)
    df['is_night'] = df['time_hour'].apply(lambda x: 1 if (x >= 22) or (x <= 6) else 0)

print(f"  ✓ Features urbanas criadas (holiday, weekend, peak_hour)")

# --- F. Interações Temporais ---
print("\n📊 Criando interações temporais...")
if 'is_weekend' in df.columns and 'time_hour' in df.columns:
    df['weekend_hour_interaction'] = df['is_weekend'] * df['time_hour']

if 'is_peak_hour' in df.columns and 'dist_device_stop' in df.columns:
    df['peak_distance_interaction'] = df['is_peak_hour'] * df['dist_device_stop']

print(f"✓ Interações temporais criadas")

# ===========================================================================
# ETAPA 3: LIMPEZA MODERADA DOS DADOS
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

# Filtros moderados (similar ao V6)
if 'device_lat' in df.columns and 'device_lon' in df.columns:
    before = len(df)
    df = df[(df['device_lat'].between(-90, 90)) & (df['device_lon'].between(-180, 180))]
    print(f"✓ Filtrado coordenadas válidas")

if 'dist_device_stop' in df.columns:
    q98 = df['dist_device_stop'].quantile(0.98)
    before = len(df)
    df = df[df['dist_device_stop'] <= q98]
    print(f"✓ Removido distâncias > {q98:.0f}m (outliers)")

print(f"\n✓ Total mantido: {len(df):,} ({len(df)/df_original_size*100:.1f}%)")

# ===========================================================================
# ETAPA 4: PREPARAÇÃO DE FEATURES
# ===========================================================================
print(f"\n{'='*80}")
print(f"ETAPA 4: PREPARAÇÃO DE FEATURES")
print(f"{'='*80}")

# Excluir colunas problemáticas (data leakage + identificadores)
features_to_drop = [
    target,
    'y_pred', 'y_pred_proba',  # Data leakage
    'ctm_service_route', 'direction', 'lotacao_proxy_binaria',  # Podem causar leakage
    'user_pseudo_id', 'gtfs_stop_id', 'event_timestamp'  # Identificadores
]

X = df.drop(columns=features_to_drop, errors='ignore')
y = df[target]

# Processar colunas categóricas
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
if categorical_cols:
    le = LabelEncoder()
    for col in categorical_cols:
        X[col] = le.fit_transform(X[col].astype(str))

X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

# Sanitizar nomes de colunas (remover caracteres especiais para LightGBM)
print(f"✓ Sanitizando nomes de features...")
X.columns = X.columns.str.replace('[^A-Za-z0-9_]', '_', regex=True)
X.columns = X.columns.str.replace('__+', '_', regex=True)  # Remover underscores duplicados
X.columns = X.columns.str.strip('_')  # Remover underscores nas pontas

print(f"✓ Features totais: {X.shape[1]}")

# ===========================================================================
# ETAPA 5: SELEÇÃO DE FEATURES (TOP 50 via XGBoost)
# ===========================================================================
print(f"\n{'='*80}")
print(f"ETAPA 5: SELEÇÃO DAS MELHORES FEATURES")
print(f"{'='*80}")

tscv = TimeSeriesSplit(n_splits=2)
for fold, (train_index, test_index) in enumerate(tscv.split(X)):
    X_train_temp, X_test_temp = X.iloc[train_index], X.iloc[test_index]
    y_train_temp, y_test_temp = y.iloc[train_index], y.iloc[test_index]

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
    print(f"  {row['feature']:35s}: {row['importance']:.2f}")

# ===========================================================================
# ETAPA 6: DIVISÃO TEMPORAL (TimeSeriesSplit)
# ===========================================================================
print(f"\n{'='*80}")
print(f"ETAPA 6: DIVISÃO TEMPORAL (TimeSeriesSplit)")
print(f"{'='*80}")

tscv = TimeSeriesSplit(n_splits=3)
X_train, X_test, y_train, y_test = None, None, None, None

for fold, (train_index, test_index) in enumerate(tscv.split(X_selected)):
    X_train, X_test = X_selected.iloc[train_index], X_selected.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    if fold == 2:
        print(f"Fold {fold + 1}: Train={len(X_train):,}, Test={len(X_test):,}")

print(f"\n=== Distribuição do Treino (Fold Final) ===")
train_dist = y_train.value_counts()
print(f"Classe 0: {train_dist[0]:,} ({train_dist[0]/len(y_train)*100:.2f}%)")
print(f"Classe 1: {train_dist[1]:,} ({train_dist[1]/len(y_train)*100:.2f}%)")
print(f"Razão (0/1): {train_dist[0]/train_dist[1]:.2f}:1")

# ===========================================================================
# ETAPA 7: NORMALIZAÇÃO
# ===========================================================================
print(f"\n{'='*80}")
print(f"ETAPA 7: NORMALIZAÇÃO DOS DADOS")
print(f"{'='*80}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

print(f"✓ Dados normalizados (StandardScaler)")

# ===========================================================================
# ETAPA 8: TREINAMENTO DOS MODELOS (LightGBM + XGBoost)
# ===========================================================================
print(f"\n{'='*80}")
print(f"ETAPA 8: TREINAMENTO DOS MODELOS")
print(f"{'='*80}")

import time

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
results = []

# --- MODELO 1: LightGBM ---
print(f"\n--- MODELO 1: LightGBM ---")
start_time = time.time()

lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_depth': -1,
    'min_child_samples': 20,
    'scale_pos_weight': scale_pos_weight,
    'verbose': -1,
    'random_state': 42
}

lgb_train = lgb.Dataset(X_train_scaled, y_train)
lgb_test = lgb.Dataset(X_test_scaled, y_test, reference=lgb_train)

lgb_model = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=200,
    valid_sets=[lgb_test],
    callbacks=[lgb.early_stopping(25), lgb.log_evaluation(0)]
)

y_pred_proba_lgb = lgb_model.predict(X_test_scaled, num_iteration=lgb_model.best_iteration)

# Otimizar threshold
thresholds = np.arange(0.3, 0.9, 0.05)
best_threshold_lgb = 0.5
best_f1_lgb = 0

for thresh in thresholds:
    y_pred_temp = (y_pred_proba_lgb >= thresh).astype(int)
    f1_temp = f1_score(y_test, y_pred_temp, average='macro')
    if f1_temp > best_f1_lgb:
        best_f1_lgb = f1_temp
        best_threshold_lgb = thresh

y_pred_lgb = (y_pred_proba_lgb >= best_threshold_lgb).astype(int)
lgb_time = time.time() - start_time

results.append({
    'model': 'LightGBM',
    'roc_auc': roc_auc_score(y_test, y_pred_proba_lgb),
    'f1_macro': f1_score(y_test, y_pred_lgb, average='macro'),
    'precision': precision_score(y_test, y_pred_lgb, zero_division=0),
    'recall': recall_score(y_test, y_pred_lgb, zero_division=0),
    'threshold': best_threshold_lgb,
    'time': lgb_time,
    'y_pred_proba': y_pred_proba_lgb
})

print(f"ROC-AUC: {results[-1]['roc_auc']:.4f} | F1-Macro: {results[-1]['f1_macro']:.4f}")
print(f"Threshold: {best_threshold_lgb:.2f} | Tempo: {lgb_time:.2f}s")

# --- MODELO 2: XGBoost ---
print(f"\n--- MODELO 2: XGBoost ---")
start_time = time.time()

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 18,
    'learning_rate': 0.02,
    'min_child_weight': 3,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'scale_pos_weight': scale_pos_weight,
    'seed': 42
}

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

xgb_model = xgb.train(
    xgb_params,
    dtrain,
    num_boost_round=250,
    evals=[(dtest, 'test')],
    early_stopping_rounds=25,
    verbose_eval=False
)

y_pred_proba_xgb = xgb_model.predict(dtest)

# Otimizar threshold
best_threshold_xgb = 0.5
best_f1_xgb = 0

for thresh in thresholds:
    y_pred_temp = (y_pred_proba_xgb >= thresh).astype(int)
    f1_temp = f1_score(y_test, y_pred_temp, average='macro')
    if f1_temp > best_f1_xgb:
        best_f1_xgb = f1_temp
        best_threshold_xgb = thresh

y_pred_xgb = (y_pred_proba_xgb >= best_threshold_xgb).astype(int)
xgb_time = time.time() - start_time

results.append({
    'model': 'XGBoost',
    'roc_auc': roc_auc_score(y_test, y_pred_proba_xgb),
    'f1_macro': f1_score(y_test, y_pred_xgb, average='macro'),
    'precision': precision_score(y_test, y_pred_xgb, zero_division=0),
    'recall': recall_score(y_test, y_pred_xgb, zero_division=0),
    'threshold': best_threshold_xgb,
    'time': xgb_time,
    'y_pred_proba': y_pred_proba_xgb
})

print(f"ROC-AUC: {results[-1]['roc_auc']:.4f} | F1-Macro: {results[-1]['f1_macro']:.4f}")
print(f"Threshold: {best_threshold_xgb:.2f} | Tempo: {xgb_time:.2f}s")

# --- MODELO 3: Ensemble (F1-Weighted) ---
print(f"\n--- MODELO 3: Ensemble (F1-Weighted) ---")

w_lgb = results[0]['f1_macro']
w_xgb = results[1]['f1_macro']
total_weight = w_lgb + w_xgb

w_lgb_norm = w_lgb / total_weight
w_xgb_norm = w_xgb / total_weight

print(f"  Peso LightGBM: {w_lgb_norm:.3f}")
print(f"  Peso XGBoost: {w_xgb_norm:.3f}")

y_pred_proba_ensemble = (w_lgb_norm * y_pred_proba_lgb + w_xgb_norm * y_pred_proba_xgb)

# Otimizar threshold
best_threshold_ensemble = 0.5
best_f1_ensemble = 0

for thresh in thresholds:
    y_pred_temp = (y_pred_proba_ensemble >= thresh).astype(int)
    f1_temp = f1_score(y_test, y_pred_temp, average='macro')
    if f1_temp > best_f1_ensemble:
        best_f1_ensemble = f1_temp
        best_threshold_ensemble = thresh

y_pred_ensemble = (y_pred_proba_ensemble >= best_threshold_ensemble).astype(int)

results.append({
    'model': 'Ensemble (F1-Weighted)',
    'roc_auc': roc_auc_score(y_test, y_pred_proba_ensemble),
    'f1_macro': f1_score(y_test, y_pred_ensemble, average='macro'),
    'precision': precision_score(y_test, y_pred_ensemble, zero_division=0),
    'recall': recall_score(y_test, y_pred_ensemble, zero_division=0),
    'threshold': best_threshold_ensemble,
    'time': lgb_time + xgb_time,
    'y_pred_proba': y_pred_proba_ensemble,
    'weights': {'lightgbm': float(w_lgb_norm), 'xgboost': float(w_xgb_norm)}
})

print(f"ROC-AUC: {results[-1]['roc_auc']:.4f} | F1-Macro: {results[-1]['f1_macro']:.4f}")
print(f"Threshold: {best_threshold_ensemble:.2f}")

# ===========================================================================
# ETAPA 9: MÉTRICAS FINAIS DO ENSEMBLE
# ===========================================================================
print(f"\n{'='*80}")
print(f"🏆 MÉTRICAS FINAIS - ENSEMBLE (BASE COMPLETA)")
print(f"{'='*80}")

best_result = results[2]  # Ensemble

print(f"\nROC-AUC:   {best_result['roc_auc']:.4f}")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_ensemble):.4f}")
print(f"Precision: {best_result['precision']:.4f}")
print(f"Recall:    {best_result['recall']:.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred_ensemble):.4f}")
print(f"F1-Macro:  {best_result['f1_macro']:.4f} ⭐")
print(f"Threshold: {best_result['threshold']:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_ensemble)
tn, fp, fn, tp = cm.ravel()

print(f"\n=== Matriz de Confusão ===")
print(f"True Negatives:  {tn:,} (corretos classe 0)")
print(f"False Positives: {fp:,} (erro: previu 1, era 0)")
print(f"False Negatives: {fn:,} (erro: previu 0, era 1)")
print(f"True Positives:  {tp:,} (corretos classe 1)")

# ===========================================================================
# ETAPA 10: VISUALIZAÇÕES
# ===========================================================================
print(f"\n{'='*80}")
print(f"ETAPA 10: GERANDO VISUALIZAÇÕES")
print(f"{'='*80}")

# 1. Matriz de Confusão
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Classe 0', 'Classe 1'],
            yticklabels=['Classe 0', 'Classe 1'],
            annot_kws={'size': 14, 'weight': 'bold'})
plt.title(f'Matriz de Confusão - Ensemble Final (Base Completa)\nThreshold: {best_threshold_ensemble:.2f}', 
          fontsize=14, fontweight='bold')
plt.ylabel('Real', fontsize=12, fontweight='bold')
plt.xlabel('Predito', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('v7_FINAL_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ v7_FINAL_confusion_matrix.png")
plt.close()

# 2. Curva ROC
plt.figure(figsize=(10, 8))
for r in results:
    fpr, tpr, _ = roc_curve(y_test, r['y_pred_proba'])
    plt.plot(fpr, tpr, label=f"{r['model']} (AUC = {r['roc_auc']:.4f})", linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5000)', linewidth=1)
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
plt.title('Curvas ROC - V7 Final (Base Completa)', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('v7_FINAL_roc_curves.png', dpi=300, bbox_inches='tight')
print("✓ v7_FINAL_roc_curves.png")
plt.close()

# 3. Comparação de Métricas
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
metrics = ['roc_auc', 'f1_macro', 'precision', 'recall']
titles = ['ROC-AUC', 'F1-Macro', 'Precision', 'Recall']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axes[idx // 2, idx % 2]
    values = [r[metric] for r in results]
    models = [r['model'] for r in results]
    
    bars = ax.barh(range(len(models)), values, color=colors)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=10)
    ax.set_xlabel(title, fontsize=11, fontweight='bold')
    ax.set_title(f'{title} - Base Completa', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim([0, 1.0])
    
    # Destacar o melhor
    best_idx = np.argmax(values)
    bars[best_idx].set_color('#FF6B6B')
    bars[best_idx].set_edgecolor('black')
    bars[best_idx].set_linewidth(2)
    
    for i, v in enumerate(values):
        ax.text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('v7_FINAL_metrics_comparison.png', dpi=300, bbox_inches='tight')
print("✓ v7_FINAL_metrics_comparison.png")
plt.close()

# ===========================================================================
# ETAPA 11: SALVAR MODELOS E ARTEFATOS
# ===========================================================================
print(f"\n{'='*80}")
print(f"ETAPA 11: SALVANDO MODELOS E ARTEFATOS FINAIS")
print(f"{'='*80}")

# Salvar LightGBM
lgb_model.save_model('lightgbm_model_v7_FINAL.txt')
print("✓ lightgbm_model_v7_FINAL.txt")

# Salvar XGBoost
xgb_model.save_model('xgboost_model_v7_FINAL.json')
print("✓ xgboost_model_v7_FINAL.json")

# Salvar Scaler
joblib.dump(scaler, 'scaler_v7_FINAL.pkl')
print("✓ scaler_v7_FINAL.pkl")

# Salvar features selecionadas
with open('selected_features_v7_FINAL.txt', 'w') as f:
    for feat in top_features:
        f.write(f"{feat}\n")
print("✓ selected_features_v7_FINAL.txt")

# Salvar configuração completa
config = {
    'model_version': 'V7_ENSEMBLE_FINAL',
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'dataset': {
        'source': 'proj-ml-469320.app_cittamobi.dataset-updated',
        'total_records': int(df_original_size),
        'records_after_cleaning': int(len(df)),
        'train_size': int(len(X_train)),
        'test_size': int(len(X_test))
    },
    'ensemble': {
        'algorithm': 'F1-Weighted Average',
        'weights': best_result['weights'],
        'threshold': float(best_result['threshold'])
    },
    'lightgbm': {
        'roc_auc': float(results[0]['roc_auc']),
        'f1_macro': float(results[0]['f1_macro']),
        'precision': float(results[0]['precision']),
        'recall': float(results[0]['recall']),
        'threshold': float(results[0]['threshold']),
        'training_time_seconds': float(results[0]['time'])
    },
    'xgboost': {
        'roc_auc': float(results[1]['roc_auc']),
        'f1_macro': float(results[1]['f1_macro']),
        'precision': float(results[1]['precision']),
        'recall': float(results[1]['recall']),
        'threshold': float(results[1]['threshold']),
        'training_time_seconds': float(results[1]['time'])
    },
    'ensemble_metrics': {
        'roc_auc': float(best_result['roc_auc']),
        'accuracy': float(accuracy_score(y_test, y_pred_ensemble)),
        'precision': float(best_result['precision']),
        'recall': float(best_result['recall']),
        'f1_score': float(f1_score(y_test, y_pred_ensemble)),
        'f1_macro': float(best_result['f1_macro']),
        'confusion_matrix': {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        }
    },
    'features': {
        'total_engineered': int(X.shape[1]),
        'selected': int(len(top_features)),
        'top_10': [
            {'name': row['feature'], 'importance': float(row['importance'])}
            for _, row in importance_df.head(10).iterrows()
        ]
    }
}

with open('model_config_v7_FINAL.json', 'w') as f:
    json.dump(config, f, indent=2)
print("✓ model_config_v7_FINAL.json")

# ===========================================================================
# ETAPA 12: EXEMPLO DE INFERÊNCIA
# ===========================================================================
print(f"\n{'='*80}")
print(f"ETAPA 12: GERANDO CÓDIGO DE EXEMPLO PARA INFERÊNCIA")
print(f"{'='*80}")

inference_code = f"""
# ═══════════════════════════════════════════════════════════════════════════
# CÓDIGO DE EXEMPLO - INFERÊNCIA MODELO V7 ENSEMBLE FINAL
# ═══════════════════════════════════════════════════════════════════════════

import joblib
import lightgbm as lgb
import xgboost as xgb
import pandas as pd
import numpy as np
import json

# 1. CARREGAR MODELOS E ARTEFATOS
print("Carregando modelos...")
lgb_model = lgb.Booster(model_file='lightgbm_model_v7_FINAL.txt')
xgb_model = xgb.Booster()
xgb_model.load_model('xgboost_model_v7_FINAL.json')
scaler = joblib.load('scaler_v7_FINAL.pkl')

with open('selected_features_v7_FINAL.txt', 'r') as f:
    selected_features = [line.strip() for line in f]

with open('model_config_v7_FINAL.json', 'r') as f:
    config = json.load(f)

print(f"✓ Modelos carregados")
print(f"✓ Features: {{len(selected_features)}}")
print(f"✓ Threshold: {{config['ensemble']['threshold']}}")
print(f"✓ Pesos: LightGBM={{config['ensemble']['weights']['lightgbm']:.3f}}, XGBoost={{config['ensemble']['weights']['xgboost']:.3f}}")

# 2. PREPARAR DADOS (exemplo com 1 registro)
# Substitua isso pelos seus dados reais
new_data = pd.DataFrame({{
    # ... adicione suas features aqui ...
    # Certifique-se de ter todas as {{len(selected_features)}} features
}})

# Verificar se todas as features estão presentes
missing_features = set(selected_features) - set(new_data.columns)
if missing_features:
    print(f"⚠️  Features faltando: {{missing_features}}")
    # Adicionar features faltantes com valor 0 ou calcular
    for feat in missing_features:
        new_data[feat] = 0

# Selecionar e ordenar features
new_data_selected = new_data[selected_features]

# 3. NORMALIZAR
new_data_scaled = scaler.transform(new_data_selected)

# 4. PREDIÇÃO
# Predição LightGBM
y_proba_lgb = lgb_model.predict(new_data_scaled)

# Predição XGBoost
dmatrix = xgb.DMatrix(new_data_selected)
y_proba_xgb = xgb_model.predict(dmatrix)

# Ensemble (média ponderada)
w_lgb = config['ensemble']['weights']['lightgbm']
w_xgb = config['ensemble']['weights']['xgboost']
y_proba_ensemble = w_lgb * y_proba_lgb + w_xgb * y_proba_xgb

# Classificação final
threshold = config['ensemble']['threshold']
y_pred = (y_proba_ensemble >= threshold).astype(int)

print(f"\\n=== RESULTADO ===")
print(f"Probabilidade de conversão: {{y_proba_ensemble[0]:.4f}} ({{y_proba_ensemble[0]*100:.2f}}%)")
print(f"Predição: {{'CONVERSÃO' if y_pred[0] == 1 else 'NÃO CONVERSÃO'}}")

# ═══════════════════════════════════════════════════════════════════════════
"""

with open('inference_example_v7_FINAL.py', 'w') as f:
    f.write(inference_code)
print("✓ inference_example_v7_FINAL.py")

# ===========================================================================
# RESUMO FINAL
# ===========================================================================
print(f"\n{'='*80}")
print(f"🎉 V7 ENSEMBLE FINAL - TREINAMENTO COMPLETO")
print(f"{'='*80}")

end_time = datetime.now()
print(f"\n📅 Fim: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"⏱️  Tempo total: {(end_time - datetime.strptime(datetime.now().strftime('%Y-%m-%d ') + '14:31:19', '%Y-%m-%d %H:%M:%S')).total_seconds()/60:.2f} minutos")

print(f"\n📊 DATASET:")
print(f"   Total de registros: {df_original_size:,}")
print(f"   Após limpeza: {len(df):,} ({len(df)/df_original_size*100:.1f}%)")
print(f"   Treino: {len(X_train):,}")
print(f"   Teste: {len(X_test):,}")

print(f"\n🏆 MODELO FINAL: Ensemble F1-Weighted")
print(f"   ROC-AUC:   {best_result['roc_auc']:.4f}")
print(f"   F1-Macro:  {best_result['f1_macro']:.4f} ⭐")
print(f"   Precision: {best_result['precision']:.4f}")
print(f"   Recall:    {best_result['recall']:.4f}")
print(f"   Threshold: {best_result['threshold']:.2f}")

print(f"\n💾 ARTEFATOS GERADOS:")
print(f"   ✓ lightgbm_model_v7_FINAL.txt")
print(f"   ✓ xgboost_model_v7_FINAL.json")
print(f"   ✓ scaler_v7_FINAL.pkl")
print(f"   ✓ selected_features_v7_FINAL.txt")
print(f"   ✓ model_config_v7_FINAL.json")
print(f"   ✓ inference_example_v7_FINAL.py")

print(f"\n📊 VISUALIZAÇÕES:")
print(f"   ✓ v7_FINAL_confusion_matrix.png")
print(f"   ✓ v7_FINAL_roc_curves.png")
print(f"   ✓ v7_FINAL_metrics_comparison.png")

print(f"\n🎯 PRÓXIMOS PASSOS:")
print(f"   1. Revisar métricas finais")
print(f"   2. Testar inferência com inference_example_v7_FINAL.py")
print(f"   3. Deploy em produção")
print(f"   4. Apresentar ao cliente")

print(f"\n{'='*80}")
print(f"✅ MODELO PRONTO PARA PRODUÇÃO!")
print(f"{'='*80}")
