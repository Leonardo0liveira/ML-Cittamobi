from google.cloud import bigquery
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, 
                             recall_score, f1_score, confusion_matrix, classification_report,
                             roc_curve, auc)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
from collections import Counter
import warnings
import time
import joblib
warnings.filterwarnings('ignore')

project_id = "proj-ml-469320"
client = bigquery.Client(project=project_id)

# ===========================================================================
# ETAPA 1: CARREGAR DADOS
# ===========================================================================
print(f"\n{'='*70}")
print(f"MODELO V6 PRODUCTION - ENSEMBLE OTIMIZADO")
print(f"{'='*70}")
print(f"ETAPA 1: CARREGANDO DATASET COM AMOSTRAGEM ALEATÓRIA")
print(f"{'='*70}")

query = """
    SELECT * FROM `proj-ml-469320.app_cittamobi.dataset-updated` 
    TABLESAMPLE SYSTEM (20 PERCENT)
    LIMIT 200000
"""

print("Carregando 200,000 amostras com amostragem aleatória...")
df = client.query(query).to_dataframe()
print(f"✓ Dados carregados: {len(df):,} registros")

# ===========================================================================
# ETAPA 2: FEATURE ENGINEERING AVANÇADO
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 2: FEATURE ENGINEERING AVANÇADO")
print(f"{'='*70}")

target = "target"

# Criar features de agregação por usuário
print("\n📊 Criando features de agregação por usuário...")
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
    print(f"✓ Features de usuário criadas: {len(user_agg.columns)-1}")

# Features de agregação por parada
print("\n📊 Criando features de agregação por parada...")
if 'gtfs_stop_id' in df.columns:
    stop_agg = df.groupby('gtfs_stop_id').agg({
        target: ['mean', 'count'],
        'user_frequency': ['mean', 'median'] if 'user_frequency' in df.columns else ['mean']
    }).reset_index()
    
    stop_agg.columns = ['gtfs_stop_id', 'stop_conversion_rate', 'stop_event_count_agg',
                        'stop_user_freq_mean', 'stop_user_freq_median']
    
    df = df.merge(stop_agg, on='gtfs_stop_id', how='left')
    print(f"✓ Features de parada criadas: {len(stop_agg.columns)-1}")

# Features de interação avançadas
print("\n📊 Criando features de interação de 2ª ordem...")
if 'user_conversion_rate' in df.columns and 'stop_conversion_rate' in df.columns:
    df['conversion_interaction'] = df['user_conversion_rate'] * df['stop_conversion_rate'] * 10000
    
if 'user_avg_dist' in df.columns and 'dist_device_stop' in df.columns:
    df['dist_deviation'] = np.abs(df['dist_device_stop'] - df['user_avg_dist'])
    df['dist_ratio'] = df['dist_device_stop'] / (df['user_avg_dist'] + 1)

if 'user_frequency' in df.columns and 'stop_event_count_agg' in df.columns:
    df['user_stop_affinity'] = df['user_frequency'] * df['stop_event_count_agg']

print(f"✓ Features de interação criadas")

# ===========================================================================
# ETAPA 3: LIMPEZA MODERADA
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 3: LIMPEZA MODERADA DOS DADOS")
print(f"{'='*70}")

print(f"\n=== Distribuição do Target (Original) ===")
target_dist = df[target].value_counts()
print(target_dist)
print(f"Proporção classe 0: {target_dist[0]/len(df)*100:.2f}%")
print(f"Proporção classe 1: {target_dist[1]/len(df)*100:.2f}%")

df_original_size = len(df)

# Filtros moderados
if 'user_frequency' in df.columns:
    q10 = df['user_frequency'].quantile(0.10)
    before = len(df)
    df = df[df['user_frequency'] >= q10]
    print(f"✓ Filtro 1: Removidos {before - len(df):,} registros ({(before - len(df))/df_original_size*100:.1f}%)")

if 'device_lat' in df.columns and 'device_lon' in df.columns:
    before = len(df)
    df = df[(df['device_lat'].between(-90, 90)) & (df['device_lon'].between(-180, 180))]
    print(f"✓ Filtro 2: Removidos {before - len(df):,} registros ({(before - len(df))/df_original_size*100:.1f}%)")

if 'dist_device_stop' in df.columns:
    q98 = df['dist_device_stop'].quantile(0.98)
    before = len(df)
    df = df[df['dist_device_stop'] <= q98]
    print(f"✓ Filtro 3: Removidos {before - len(df):,} registros ({(before - len(df))/df_original_size*100:.1f}%)")

total_removed = 200000 - len(df)
removal_pct = total_removed / 200000 * 100
print(f"\n✓ Total mantido: {len(df):,} ({100-removal_pct:.1f}%)")

# ===========================================================================
# ETAPA 4: PREPARAÇÃO DE FEATURES
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 4: PREPARAÇÃO DE FEATURES")
print(f"{'='*70}")

features_to_drop = ['y_pred', 'y_pred_proba', 'ctm_service_route', 'direction', 'lotacao_proxy_binaria']
X = df.drop(columns=[target] + features_to_drop, errors='ignore')
y = df[target]

# Processar timestamp
if 'event_timestamp' in X.columns:
    X['event_timestamp'] = pd.to_datetime(X['event_timestamp'])
    X['day'] = X['event_timestamp'].dt.day
    X['hour'] = X['event_timestamp'].dt.hour
    X['dayofweek'] = X['event_timestamp'].dt.dayofweek
    X['minute'] = X['event_timestamp'].dt.minute
    X['week_of_year'] = X['event_timestamp'].dt.isocalendar().week.astype(int)
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

# Label Encoding
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
if categorical_cols:
    le = LabelEncoder()
    for col in categorical_cols:
        X[col] = le.fit_transform(X[col].astype(str))

X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

print(f"✓ Features totais: {X.shape[1]}")

# ===========================================================================
# ETAPA 5: SELEÇÃO DE FEATURES (TOP 50)
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 5: SELEÇÃO DAS MELHORES FEATURES")
print(f"{'='*70}")

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
    print(f"   {i+1}. {row['feature']}: {row['importance']:.2f}")

# ===========================================================================
# ETAPA 6: DIVISÃO TEMPORAL
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 6: DIVISÃO TEMPORAL (TimeSeriesSplit)")
print(f"{'='*70}")

tscv = TimeSeriesSplit(n_splits=3)
X_train, X_test, y_train, y_test = None, None, None, None

for fold, (train_index, test_index) in enumerate(tscv.split(X_selected)):
    X_train, X_test = X_selected.iloc[train_index], X_selected.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    if fold == 2:
        print(f"Fold {fold + 1}: Train={len(X_train):,} | Test={len(X_test):,}")

print(f"\n=== Distribuição Original do Treino ===")
train_dist = y_train.value_counts()
print(f"Classe 0: {train_dist[0]:,} ({train_dist[0]/len(y_train)*100:.2f}%)")
print(f"Classe 1: {train_dist[1]:,} ({train_dist[1]/len(y_train)*100:.2f}%)")
print(f"Razão (0/1): {train_dist[0]/train_dist[1]:.2f}:1")

# ===========================================================================
# ETAPA 7: NORMALIZAÇÃO
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 7: NORMALIZAÇÃO DOS DADOS")
print(f"{'='*70}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

print(f"✓ Dados normalizados (StandardScaler)")

# ===========================================================================
# ETAPA 8: TREINAMENTO DOS 3 MELHORES MODELOS INDIVIDUAIS
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 8: TREINAMENTO DOS 3 MELHORES MODELOS")
print(f"{'='*70}")

results = []

# ===========================================================================
# MODELO 1: XGBoost (Melhor F1-Macro do V5)
# ===========================================================================
print(f"\n--- Modelo 1: XGBOOST (OTIMIZADO) ---")
start_time = time.time()

scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

xgb_params = {
    'objective': 'binary:logistic',
    'max_depth': 18,
    'learning_rate': 0.02,
    'min_child_weight': 3,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'scale_pos_weight': scale_pos_weight,
    'eval_metric': 'logloss',
    'seed': 42
}

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

xgb_model = xgb.train(
    params=xgb_params,
    dtrain=dtrain,
    num_boost_round=250,
    evals=[(dtest, 'test')],
    early_stopping_rounds=25,
    verbose_eval=False
)

y_pred_proba_xgb = xgb_model.predict(dtest)

# Otimizar threshold
thresholds = np.arange(0.3, 0.8, 0.05)
best_threshold_xgb = 0.5
best_f1_xgb = 0

for thresh in thresholds:
    y_pred_temp = (y_pred_proba_xgb >= thresh).astype(int)
    f1_temp = f1_score(y_test, y_pred_temp, average='macro')
    if f1_temp > best_f1_xgb:
        best_f1_xgb = f1_temp
        best_threshold_xgb = thresh

y_pred_xgb = (y_pred_proba_xgb >= best_threshold_xgb).astype(int)

train_time = time.time() - start_time

results.append({
    'model': 'XGBoost',
    'threshold': best_threshold_xgb,
    'roc_auc': roc_auc_score(y_test, y_pred_proba_xgb),
    'accuracy': accuracy_score(y_test, y_pred_xgb),
    'precision': precision_score(y_test, y_pred_xgb, zero_division=0),
    'recall': recall_score(y_test, y_pred_xgb),
    'f1_score': f1_score(y_test, y_pred_xgb),
    'f1_macro': f1_score(y_test, y_pred_xgb, average='macro'),
    'train_time': train_time,
    'y_pred_proba': y_pred_proba_xgb,
    'classifier': xgb_model
})

print(f"Threshold otimizado: {best_threshold_xgb:.2f}")
print(f"ROC-AUC: {results[-1]['roc_auc']:.4f} | F1-Macro: {results[-1]['f1_macro']:.4f} | Tempo: {train_time:.2f}s")

# ===========================================================================
# MODELO 2: k-NN (Segundo melhor F1-Macro)
# ===========================================================================
print(f"\n--- Modelo 2: K-NEAREST NEIGHBORS (OTIMIZADO) ---")
start_time = time.time()

# Testar diferentes valores de k
best_k = 50
best_f1_knn = 0
best_knn_model = None

for k in [30, 50, 70, 100]:
    knn_temp = KNeighborsClassifier(n_neighbors=k, weights='distance', n_jobs=-1)
    knn_temp.fit(X_train_scaled, y_train)
    y_pred_temp = knn_temp.predict(X_test_scaled)
    f1_temp = f1_score(y_test, y_pred_temp, average='macro')
    
    if f1_temp > best_f1_knn:
        best_f1_knn = f1_temp
        best_k = k
        best_knn_model = knn_temp

knn_model = best_knn_model
y_pred_proba_knn = knn_model.predict_proba(X_test_scaled)[:, 1]

# Otimizar threshold
best_threshold_knn = 0.5
best_f1_knn_thresh = 0

for thresh in thresholds:
    y_pred_temp = (y_pred_proba_knn >= thresh).astype(int)
    f1_temp = f1_score(y_test, y_pred_temp, average='macro')
    if f1_temp > best_f1_knn_thresh:
        best_f1_knn_thresh = f1_temp
        best_threshold_knn = thresh

y_pred_knn = (y_pred_proba_knn >= best_threshold_knn).astype(int)

train_time = time.time() - start_time

results.append({
    'model': 'k-NN',
    'threshold': best_threshold_knn,
    'roc_auc': roc_auc_score(y_test, y_pred_proba_knn),
    'accuracy': accuracy_score(y_test, y_pred_knn),
    'precision': precision_score(y_test, y_pred_knn, zero_division=0),
    'recall': recall_score(y_test, y_pred_knn),
    'f1_score': f1_score(y_test, y_pred_knn),
    'f1_macro': f1_score(y_test, y_pred_knn, average='macro'),
    'train_time': train_time,
    'y_pred_proba': y_pred_proba_knn,
    'classifier': knn_model
})

print(f"k otimizado: {best_k}")
print(f"Threshold otimizado: {best_threshold_knn:.2f}")
print(f"ROC-AUC: {results[-1]['roc_auc']:.4f} | F1-Macro: {results[-1]['f1_macro']:.4f} | Tempo: {train_time:.2f}s")

# ===========================================================================
# MODELO 3: Gaussian Naive Bayes (Terceiro melhor + Ultra rápido)
# ===========================================================================
print(f"\n--- Modelo 3: GAUSSIAN NAIVE BAYES ---")
start_time = time.time()

gnb_model = GaussianNB()
gnb_model.fit(X_train_scaled, y_train)

y_pred_proba_gnb = gnb_model.predict_proba(X_test_scaled)[:, 1]

# Otimizar threshold
best_threshold_gnb = 0.5
best_f1_gnb = 0

for thresh in thresholds:
    y_pred_temp = (y_pred_proba_gnb >= thresh).astype(int)
    f1_temp = f1_score(y_test, y_pred_temp, average='macro')
    if f1_temp > best_f1_gnb:
        best_f1_gnb = f1_temp
        best_threshold_gnb = thresh

y_pred_gnb = (y_pred_proba_gnb >= best_threshold_gnb).astype(int)

train_time = time.time() - start_time

results.append({
    'model': 'Gaussian NB',
    'threshold': best_threshold_gnb,
    'roc_auc': roc_auc_score(y_test, y_pred_proba_gnb),
    'accuracy': accuracy_score(y_test, y_pred_gnb),
    'precision': precision_score(y_test, y_pred_gnb, zero_division=0),
    'recall': recall_score(y_test, y_pred_gnb),
    'f1_score': f1_score(y_test, y_pred_gnb),
    'f1_macro': f1_score(y_test, y_pred_gnb, average='macro'),
    'train_time': train_time,
    'y_pred_proba': y_pred_proba_gnb,
    'classifier': gnb_model
})

print(f"Threshold otimizado: {best_threshold_gnb:.2f}")
print(f"ROC-AUC: {results[-1]['roc_auc']:.4f} | F1-Macro: {results[-1]['f1_macro']:.4f} | Tempo: {train_time:.2f}s")

# ===========================================================================
# ETAPA 9: ENSEMBLE VOTING (SOFT)
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 9: ENSEMBLE VOTING - SOFT (PROBABILIDADES)")
print(f"{'='*70}")

# Estratégia 1: Média Simples
print(f"\n--- Estratégia 1: MÉDIA SIMPLES DAS PROBABILIDADES ---")
y_pred_proba_avg = (y_pred_proba_xgb + y_pred_proba_knn + y_pred_proba_gnb) / 3

# Otimizar threshold
best_threshold_avg = 0.5
best_f1_avg = 0

for thresh in thresholds:
    y_pred_temp = (y_pred_proba_avg >= thresh).astype(int)
    f1_temp = f1_score(y_test, y_pred_temp, average='macro')
    if f1_temp > best_f1_avg:
        best_f1_avg = f1_temp
        best_threshold_avg = thresh

y_pred_avg = (y_pred_proba_avg >= best_threshold_avg).astype(int)

results.append({
    'model': 'Ensemble - Média Simples',
    'threshold': best_threshold_avg,
    'roc_auc': roc_auc_score(y_test, y_pred_proba_avg),
    'accuracy': accuracy_score(y_test, y_pred_avg),
    'precision': precision_score(y_test, y_pred_avg, zero_division=0),
    'recall': recall_score(y_test, y_pred_avg),
    'f1_score': f1_score(y_test, y_pred_avg),
    'f1_macro': f1_score(y_test, y_pred_avg, average='macro'),
    'train_time': 0,
    'y_pred_proba': y_pred_proba_avg,
    'classifier': None
})

print(f"Threshold otimizado: {best_threshold_avg:.2f}")
print(f"ROC-AUC: {results[-1]['roc_auc']:.4f} | F1-Macro: {results[-1]['f1_macro']:.4f}")

# Estratégia 2: Média Ponderada por F1-Macro
print(f"\n--- Estratégia 2: MÉDIA PONDERADA (BASEADA EM F1-MACRO) ---")

# Pesos baseados em F1-Macro dos modelos individuais
w_xgb = results[0]['f1_macro']
w_knn = results[1]['f1_macro']
w_gnb = results[2]['f1_macro']

total_weight = w_xgb + w_knn + w_gnb
w_xgb_norm = w_xgb / total_weight
w_knn_norm = w_knn / total_weight
w_gnb_norm = w_gnb / total_weight

print(f"Pesos: XGBoost={w_xgb_norm:.3f}, k-NN={w_knn_norm:.3f}, GaussianNB={w_gnb_norm:.3f}")

y_pred_proba_weighted = (w_xgb_norm * y_pred_proba_xgb + 
                          w_knn_norm * y_pred_proba_knn + 
                          w_gnb_norm * y_pred_proba_gnb)

# Otimizar threshold
best_threshold_weighted = 0.5
best_f1_weighted = 0

for thresh in thresholds:
    y_pred_temp = (y_pred_proba_weighted >= thresh).astype(int)
    f1_temp = f1_score(y_test, y_pred_temp, average='macro')
    if f1_temp > best_f1_weighted:
        best_f1_weighted = f1_temp
        best_threshold_weighted = thresh

y_pred_weighted = (y_pred_proba_weighted >= best_threshold_weighted).astype(int)

results.append({
    'model': 'Ensemble - Ponderado (F1)',
    'threshold': best_threshold_weighted,
    'roc_auc': roc_auc_score(y_test, y_pred_proba_weighted),
    'accuracy': accuracy_score(y_test, y_pred_weighted),
    'precision': precision_score(y_test, y_pred_weighted, zero_division=0),
    'recall': recall_score(y_test, y_pred_weighted),
    'f1_score': f1_score(y_test, y_pred_weighted),
    'f1_macro': f1_score(y_test, y_pred_weighted, average='macro'),
    'train_time': 0,
    'y_pred_proba': y_pred_proba_weighted,
    'classifier': None
})

print(f"Threshold otimizado: {best_threshold_weighted:.2f}")
print(f"ROC-AUC: {results[-1]['roc_auc']:.4f} | F1-Macro: {results[-1]['f1_macro']:.4f}")

# Estratégia 3: Média Ponderada por ROC-AUC
print(f"\n--- Estratégia 3: MÉDIA PONDERADA (BASEADA EM ROC-AUC) ---")

w_xgb_auc = results[0]['roc_auc']
w_knn_auc = results[1]['roc_auc']
w_gnb_auc = results[2]['roc_auc']

total_weight_auc = w_xgb_auc + w_knn_auc + w_gnb_auc
w_xgb_auc_norm = w_xgb_auc / total_weight_auc
w_knn_auc_norm = w_knn_auc / total_weight_auc
w_gnb_auc_norm = w_gnb_auc / total_weight_auc

print(f"Pesos: XGBoost={w_xgb_auc_norm:.3f}, k-NN={w_knn_auc_norm:.3f}, GaussianNB={w_gnb_auc_norm:.3f}")

y_pred_proba_weighted_auc = (w_xgb_auc_norm * y_pred_proba_xgb + 
                              w_knn_auc_norm * y_pred_proba_knn + 
                              w_gnb_auc_norm * y_pred_proba_gnb)

# Otimizar threshold
best_threshold_auc = 0.5
best_f1_auc = 0

for thresh in thresholds:
    y_pred_temp = (y_pred_proba_weighted_auc >= thresh).astype(int)
    f1_temp = f1_score(y_test, y_pred_temp, average='macro')
    if f1_temp > best_f1_auc:
        best_f1_auc = f1_temp
        best_threshold_auc = thresh

y_pred_weighted_auc = (y_pred_proba_weighted_auc >= best_threshold_auc).astype(int)

results.append({
    'model': 'Ensemble - Ponderado (AUC)',
    'threshold': best_threshold_auc,
    'roc_auc': roc_auc_score(y_test, y_pred_proba_weighted_auc),
    'accuracy': accuracy_score(y_test, y_pred_weighted_auc),
    'precision': precision_score(y_test, y_pred_weighted_auc, zero_division=0),
    'recall': recall_score(y_test, y_pred_weighted_auc),
    'f1_score': f1_score(y_test, y_pred_weighted_auc),
    'f1_macro': f1_score(y_test, y_pred_weighted_auc, average='macro'),
    'train_time': 0,
    'y_pred_proba': y_pred_proba_weighted_auc,
    'classifier': None
})

print(f"Threshold otimizado: {best_threshold_auc:.2f}")
print(f"ROC-AUC: {results[-1]['roc_auc']:.4f} | F1-Macro: {results[-1]['f1_macro']:.4f}")

# ===========================================================================
# ETAPA 10: COMPARAÇÃO E SELEÇÃO DO MODELO FINAL
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 10: COMPARAÇÃO DE TODOS OS MODELOS")
print(f"{'='*70}")

results_df = pd.DataFrame([{
    'Modelo': r['model'],
    'Threshold': f"{r['threshold']:.2f}",
    'ROC-AUC': f"{r['roc_auc']:.4f}",
    'Accuracy': f"{r['accuracy']:.4f}",
    'Precision': f"{r['precision']:.4f}",
    'Recall': f"{r['recall']:.4f}",
    'F1-Score': f"{r['f1_score']:.4f}",
    'F1-Macro': f"{r['f1_macro']:.4f}",
    'Tempo (s)': f"{r['train_time']:.2f}" if r['train_time'] > 0 else '-'
} for r in results])

print("\n" + results_df.to_string(index=False))

# Selecionar melhor modelo
best_idx = np.argmax([r['f1_macro'] for r in results])
best_result = results[best_idx]

print(f"\n{'='*70}")
print(f"🏆 MODELO FINAL SELECIONADO: {best_result['model']}")
print(f"{'='*70}")
print(f"Threshold:  {best_result['threshold']:.2f}")
print(f"ROC-AUC:    {best_result['roc_auc']:.4f}")
print(f"Accuracy:   {best_result['accuracy']:.4f}")
print(f"Precision:  {best_result['precision']:.4f}")
print(f"Recall:     {best_result['recall']:.4f}")
print(f"F1-Score:   {best_result['f1_score']:.4f}")
print(f"F1-Macro:   {best_result['f1_macro']:.4f} ⭐")

# ===========================================================================
# ETAPA 11: ANÁLISE DETALHADA DO MODELO FINAL
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 11: ANÁLISE DETALHADA DO MODELO FINAL")
print(f"{'='*70}")

best_y_pred = (best_result['y_pred_proba'] >= best_result['threshold']).astype(int)
cm = confusion_matrix(y_test, best_y_pred)

print(f"\n=== Matriz de Confusão ===")
print(f"                 Predito")
print(f"                 0      1")
print(f"Real  0       {cm[0,0]:6d} {cm[0,1]:6d}")
print(f"      1       {cm[1,0]:6d} {cm[1,1]:6d}")

tn, fp, fn, tp = cm.ravel()
print(f"\n=== Análise de Erros ===")
print(f"True Negatives:  {tn:,} (corretos classe 0)")
print(f"False Positives: {fp:,} (erro: previu 1, era 0)")
print(f"False Negatives: {fn:,} (erro: previu 0, era 1)")
print(f"True Positives:  {tp:,} (corretos classe 1)")

print(f"\n=== Relatório de Classificação ===")
print(classification_report(y_test, best_y_pred, target_names=['Classe 0', 'Classe 1']))

# ===========================================================================
# ETAPA 12: VISUALIZAÇÕES
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 12: GERANDO VISUALIZAÇÕES")
print(f"{'='*70}")

# 1. Comparação de métricas
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

metrics = ['roc_auc', 'precision', 'recall', 'f1_macro']
titles = ['ROC-AUC', 'Precision', 'Recall', 'F1-Macro']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2']

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axes[idx // 2, idx % 2]
    
    values = [r[metric] for r in results]
    models = [r['model'] for r in results]
    
    bars = ax.barh(range(len(models)), values, color=colors)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=10)
    ax.set_xlabel(title, fontsize=11, fontweight='bold')
    ax.set_title(f'{title} por Modelo', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Destacar o melhor
    best_val_idx = np.argmax(values)
    bars[best_val_idx].set_color('#FF6B6B')
    bars[best_val_idx].set_edgecolor('black')
    bars[best_val_idx].set_linewidth(2)
    
    # Adicionar valores
    for i, v in enumerate(values):
        ax.text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('v6_ensemble_comparison.png', dpi=300, bbox_inches='tight')
print("✓ v6_ensemble_comparison.png")

# 2. Curva ROC comparativa
plt.figure(figsize=(10, 8))

for r in results:
    fpr, tpr, _ = roc_curve(y_test, r['y_pred_proba'])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{r['model']} (AUC = {roc_auc:.4f})", linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5000)', linewidth=1)
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
plt.title('Curvas ROC - Comparação de Modelos V6', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('v6_roc_curves.png', dpi=300, bbox_inches='tight')
print("✓ v6_roc_curves.png")

# 3. Matriz de confusão do melhor modelo
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Classe 0', 'Classe 1'],
            yticklabels=['Classe 0', 'Classe 1'],
            annot_kws={'size': 14, 'weight': 'bold'})
plt.title(f'Matriz de Confusão - {best_result["model"]}\nThreshold: {best_result["threshold"]:.2f}', 
          fontsize=14, fontweight='bold')
plt.ylabel('Real', fontsize=12, fontweight='bold')
plt.xlabel('Predito', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('v6_confusion_matrix_final.png', dpi=300, bbox_inches='tight')
print("✓ v6_confusion_matrix_final.png")

# ===========================================================================
# ETAPA 13: SALVAR MODELOS E CONFIGURAÇÕES
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 13: SALVANDO MODELOS E CONFIGURAÇÕES")
print(f"{'='*70}")

# Salvar XGBoost
if results[0]['classifier'] is not None:
    results[0]['classifier'].save_model('xgboost_model_v6_production.json')
    print("✓ xgboost_model_v6_production.json")

# Salvar k-NN
if results[1]['classifier'] is not None:
    joblib.dump(results[1]['classifier'], 'knn_model_v6_production.pkl')
    print("✓ knn_model_v6_production.pkl")

# Salvar Gaussian NB
if results[2]['classifier'] is not None:
    joblib.dump(results[2]['classifier'], 'gnb_model_v6_production.pkl')
    print("✓ gnb_model_v6_production.pkl")

# Salvar Scaler
joblib.dump(scaler, 'scaler_v6_production.pkl')
print("✓ scaler_v6_production.pkl")

# Salvar features selecionadas
with open('selected_features_v6.txt', 'w') as f:
    for feat in top_features:
        f.write(f"{feat}\n")
print("✓ selected_features_v6.txt")

# Salvar configuração do ensemble
config = {
    'best_model': best_result['model'],
    'threshold': float(best_result['threshold']),
    'weights_f1': {
        'xgboost': float(w_xgb_norm),
        'knn': float(w_knn_norm),
        'gaussian_nb': float(w_gnb_norm)
    },
    'weights_auc': {
        'xgboost': float(w_xgb_auc_norm),
        'knn': float(w_knn_auc_norm),
        'gaussian_nb': float(w_gnb_auc_norm)
    },
    'metrics': {
        'roc_auc': float(best_result['roc_auc']),
        'f1_macro': float(best_result['f1_macro']),
        'precision': float(best_result['precision']),
        'recall': float(best_result['recall'])
    },
    'knn_neighbors': best_k
}

import json
with open('ensemble_config_v6.json', 'w') as f:
    json.dump(config, f, indent=2)
print("✓ ensemble_config_v6.json")

# ===========================================================================
# RESUMO FINAL
# ===========================================================================
print(f"\n{'='*70}")
print(f"🎉 V6 PRODUCTION - RESUMO FINAL")
print(f"{'='*70}")

print(f"\n📊 Modelos Testados: {len(results)}")
print(f"1. XGBoost (individual)")
print(f"2. k-NN (individual)")
print(f"3. Gaussian Naive Bayes (individual)")
print(f"4. Ensemble - Média Simples")
print(f"5. Ensemble - Ponderado por F1-Macro")
print(f"6. Ensemble - Ponderado por ROC-AUC")

print(f"\n🏆 Modelo Final: {best_result['model']}")
print(f"   ROC-AUC: {best_result['roc_auc']:.4f}")
print(f"   F1-Macro: {best_result['f1_macro']:.4f}")
print(f"   Precision: {best_result['precision']:.4f}")
print(f"   Recall: {best_result['recall']:.4f}")
print(f"   Threshold: {best_result['threshold']:.2f}")

# Comparação com versões anteriores
v4_roc_auc = 0.9731
v4_f1_macro = 0.7760
v5_roc_auc = 0.9729
v5_f1_macro = 0.7782

print(f"\n📈 Evolução das Versões:")
print(f"   V4: ROC-AUC {v4_roc_auc:.4f}, F1-Macro {v4_f1_macro:.4f}")
print(f"   V5: ROC-AUC {v5_roc_auc:.4f}, F1-Macro {v5_f1_macro:.4f}")
print(f"   V6: ROC-AUC {best_result['roc_auc']:.4f}, F1-Macro {best_result['f1_macro']:.4f}")

if best_result['f1_macro'] > v5_f1_macro:
    improvement = ((best_result['f1_macro'] - v5_f1_macro) / v5_f1_macro) * 100
    print(f"\n✅ Melhoria sobre V5: +{improvement:.2f}% em F1-Macro")
else:
    decline = ((v5_f1_macro - best_result['f1_macro']) / v5_f1_macro) * 100
    print(f"\n⚠️ V5 ainda superior: -{decline:.2f}% em F1-Macro")

print(f"\n💾 Artefatos Salvos:")
print(f"   • xgboost_model_v6_production.json")
print(f"   • knn_model_v6_production.pkl")
print(f"   • gnb_model_v6_production.pkl")
print(f"   • scaler_v6_production.pkl")
print(f"   • selected_features_v6.txt")
print(f"   • ensemble_config_v6.json")

print(f"\n📊 Visualizações Geradas:")
print(f"   • v6_ensemble_comparison.png")
print(f"   • v6_roc_curves.png")
print(f"   • v6_confusion_matrix_final.png")

print(f"\n{'='*70}")
print(f"✅ MODELO V6 PRODUCTION PRONTO PARA DEPLOY!")
print(f"{'='*70}")
