"""
═══════════════════════════════════════════════════════════════════════════════
MODEL V8 - FASE 2A: MELHORIAS RÁPIDAS PARA F1 CLASSE 1
═══════════════════════════════════════════════════════════════════════════════

🎯 OBJETIVO: Aumentar F1 Classe 1 de 0.42 → 0.50+ (~20% melhoria)

🔧 MELHORIAS FASE 2A:
   ✅ 1. Threshold dinâmico por faixa de conversão
   ✅ 2. Sample weights dinâmicos
   ✅ 3. Features de contexto adicional (5 novas)
   ✅ 4. Interações geográfico-temporais

⏱️ Tempo estimado: 2-3 horas de treinamento

═══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
from google.cloud import bigquery
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import json
import joblib

warnings.filterwarnings('ignore')

print("="*80)
print("🚀 MODEL V8 - FASE 2A: MELHORIAS RÁPIDAS F1")
print("="*80)
print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# ===========================================================================
# CARREGAR DADOS (IGUAL À FASE 1)
# ===========================================================================
print(f"\n{'='*80}")
print("ETAPA 1: CARREGANDO DADOS")
print(f"{'='*80}")

project_id = "proj-ml-469320"
client = bigquery.Client(project=project_id)

query = """
    SELECT * FROM `proj-ml-469320.app_cittamobi.dataset-updated`
    LIMIT 300000
"""

print("⏳ Carregando 300K registros...")
start_time = datetime.now()
df = client.query(query).to_dataframe()
print(f"✅ {len(df):,} registros em {(datetime.now()-start_time).total_seconds():.1f}s")

target = "target"

# ===========================================================================
# FEATURE ENGINEERING BÁSICO + FASE 1
# ===========================================================================
print(f"\n{'='*80}")
print("ETAPA 2: FEATURE ENGINEERING (FASE 1 + NOVAS)")
print(f"{'='*80}")

# Temporal básico
if 'event_timestamp' in df.columns:
    df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])
    df['time_hour'] = df['event_timestamp'].dt.hour
    df['time_day_of_week'] = df['event_timestamp'].dt.dayofweek
    df['time_month'] = df['event_timestamp'].dt.month
    
    df['hour_sin'] = np.sin(2 * np.pi * df['time_hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['time_hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['time_day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['time_day_of_week'] / 7)
    
    df['is_weekend'] = (df['time_day_of_week'] >= 5).astype(int)
    df['is_peak_hour'] = df['time_hour'].apply(lambda x: 1 if (7 <= x <= 9) or (17 <= x <= 19) else 0)

# Geográficas FASE 1
print("\n[2.1] Features geográficas (Fase 1)...")

if 'gtfs_stop_id' in df.columns:
    stop_conversion = df.groupby('gtfs_stop_id').agg({
        target: ['mean', 'sum', 'std', 'count']
    })
    stop_conversion.columns = ['stop_historical_conversion', 'stop_total_conversions',
                                'stop_conversion_std', 'stop_event_count']
    df = df.merge(stop_conversion, left_on='gtfs_stop_id', right_index=True, how='left')
    df['stop_volatility'] = df['stop_conversion_std'] / (df['stop_historical_conversion'] + 0.01)

if 'stop_lat_event' in df.columns and 'stop_lon_event' in df.columns:
    coords_df = df[['stop_lat_event', 'stop_lon_event']].drop_duplicates().dropna()
    coords = coords_df.values
    
    if len(coords) > 10:
        nn = NearestNeighbors(n_neighbors=min(11, len(coords)))
        nn.fit(coords)
        distances, _ = nn.kneighbors(df[['stop_lat_event', 'stop_lon_event']].values)
        df['stop_density'] = 1 / (distances[:, 1:].mean(axis=1) + 0.001)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

cbds = {
    'sp': (-23.550520, -46.633308),
    'rio': (-22.906847, -43.172896),
    'salvador': (-12.971599, -38.501362),
    'recife': (-8.063173, -34.871016),
}

if 'stop_lat_event' in df.columns:
    dists = [haversine(df['stop_lat_event'], df['stop_lon_event'], lat, lon) 
             for lat, lon in cbds.values()]
    df['dist_to_nearest_cbd'] = np.min(dists, axis=0)

if len(coords) > 50:
    clustering = DBSCAN(eps=0.01, min_samples=5)
    coords_df['cluster'] = clustering.fit_predict(coords)
    df = df.merge(coords_df, on=['stop_lat_event', 'stop_lon_event'], how='left')
    df['stop_cluster'] = df['cluster'].fillna(-1).astype(int)
    
    cluster_stats = df.groupby('stop_cluster')[target].agg(['mean', 'count'])
    cluster_stats.columns = ['cluster_conversion_rate', 'cluster_size']
    df = df.merge(cluster_stats, left_on='stop_cluster', right_index=True, how='left')

if 'gtfs_stop_id' in df.columns:
    df['stop_unique_users'] = df.groupby('gtfs_stop_id')['user_pseudo_id'].transform('nunique')
    df['stop_peak_ratio'] = df.groupby('gtfs_stop_id')['is_peak_hour'].transform('mean')

print("✅ Features Fase 1 criadas")

# ===========================================================================
# NOVAS FEATURES FASE 2A
# ===========================================================================
print("\n[2.2] 🆕 NOVAS FEATURES FASE 2A...")

# 1. Conversão por hora (padrão temporal)
if 'time_hour' in df.columns:
    hour_conversion = df.groupby('time_hour')[target].mean()
    df['hour_conversion_rate'] = df['time_hour'].map(hour_conversion)
    print("  ✅ hour_conversion_rate")

# 2. Conversão por dia da semana
if 'time_day_of_week' in df.columns:
    dow_conversion = df.groupby('time_day_of_week')[target].mean()
    df['dow_conversion_rate'] = df['time_day_of_week'].map(dow_conversion)
    print("  ✅ dow_conversion_rate")

# 3. Conversão por parada + hora (padrão específico)
if 'gtfs_stop_id' in df.columns and 'time_hour' in df.columns:
    stop_hour_conv = df.groupby(['gtfs_stop_id', 'time_hour'])[target].mean()
    df['stop_hour_conversion'] = df.set_index(['gtfs_stop_id', 'time_hour']).index.map(stop_hour_conv)
    df['stop_hour_conversion'] = df['stop_hour_conversion'].fillna(df['stop_historical_conversion'])
    print("  ✅ stop_hour_conversion (importante!)")

# 4. Ratio usuário vs parada
if 'user_pseudo_id' in df.columns and 'gtfs_stop_id' in df.columns:
    user_conversion = df.groupby('user_pseudo_id')[target].mean()
    df['user_conversion_rate'] = df['user_pseudo_id'].map(user_conversion)
    df['user_vs_stop_ratio'] = df['user_conversion_rate'] / (df['stop_historical_conversion'] + 0.01)
    print("  ✅ user_vs_stop_ratio")

# 5. Desvio de distância
if 'dist_device_stop' in df.columns and 'gtfs_stop_id' in df.columns:
    stop_dist_mean = df.groupby('gtfs_stop_id')['dist_device_stop'].mean()
    stop_dist_std = df.groupby('gtfs_stop_id')['dist_device_stop'].std()
    df['stop_avg_dist'] = df['gtfs_stop_id'].map(stop_dist_mean)
    df['stop_dist_std'] = df['gtfs_stop_id'].map(stop_dist_std).fillna(1)
    df['dist_deviation'] = (df['dist_device_stop'] - df['stop_avg_dist']) / (df['stop_dist_std'] + 0.01)
    print("  ✅ dist_deviation")

# 6. Interações geográfico-temporais
if 'dist_to_nearest_cbd' in df.columns and 'is_peak_hour' in df.columns:
    df['geo_temporal'] = df['dist_to_nearest_cbd'] * df['is_peak_hour']
    df['density_peak'] = df['stop_density'] * df['is_peak_hour']
    print("  ✅ geo_temporal, density_peak")

# 7. Features de raridade
if 'stop_event_count' in df.columns:
    df['stop_rarity'] = 1 / (df['stop_event_count'] + 1)
if 'user_pseudo_id' in df.columns:
    user_freq = df.groupby('user_pseudo_id').size()
    df['user_frequency'] = df['user_pseudo_id'].map(user_freq)
    df['user_rarity'] = 1 / (df['user_frequency'] + 1)
    print("  ✅ stop_rarity, user_rarity")

print(f"\n✅ {df.shape[1]} features totais (incluindo novas)")

# ===========================================================================
# PREPARAÇÃO
# ===========================================================================
print(f"\n{'='*80}")
print("ETAPA 3: PREPARAÇÃO")
print(f"{'='*80}")

features_to_drop = [
    target, 'user_pseudo_id', 'gtfs_stop_id', 'event_timestamp',
    'y_pred', 'y_pred_proba', 'cluster', 'date'
]

X = df.drop(columns=[col for col in features_to_drop if col in df.columns], errors='ignore')
y = df[target]

X = X.select_dtypes(include=[np.number])
X = X.fillna(X.median())

print(f"✅ Features finais: {X.shape[1]} (+ {X.shape[1] - 43} novas)")

selected_features = X.columns.tolist()
with open('selected_features_v8_phase2a.txt', 'w') as f:
    f.write('\n'.join(selected_features))

# Split
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"✅ Treino: {len(X_train):,} | Teste: {len(X_test):,}")

# Normalizar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, 'scaler_v8_phase2a.pkl')

# ===========================================================================
# SAMPLE WEIGHTS DINÂMICOS (NOVO!)
# ===========================================================================
print(f"\n{'='*80}")
print("ETAPA 4: 🆕 SAMPLE WEIGHTS DINÂMICOS")
print(f"{'='*80}")

def get_sample_weights(X_df, y):
    """Pesos dinâmicos baseados em conversão histórica"""
    weights = np.ones(len(y))
    
    if 'stop_historical_conversion' in X_df.columns:
        stop_conv = X_df['stop_historical_conversion'].values
        
        # Paradas de alta conversão (>50%)
        high_mask = stop_conv > 0.5
        weights[high_mask & (y == 1)] = 3.0  # Conversões: peso 3x
        weights[high_mask & (y == 0)] = 0.5  # Não-conversões: peso 0.5x
        
        # Paradas médias (20-50%)
        med_mask = (stop_conv > 0.2) & (stop_conv <= 0.5)
        weights[med_mask & (y == 1)] = 2.0
        weights[med_mask & (y == 0)] = 0.8
        
        # Paradas baixas (<20%)
        low_mask = stop_conv <= 0.2
        weights[low_mask & (y == 1)] = 1.5
        weights[low_mask & (y == 0)] = 1.0
    
    return weights

sample_weights_train = get_sample_weights(X_train, y_train.values)

print(f"✅ Sample weights criados:")
print(f"   Min: {sample_weights_train.min():.2f}")
print(f"   Max: {sample_weights_train.max():.2f}")
print(f"   Média: {sample_weights_train.mean():.2f}")

# ===========================================================================
# TREINAR MODELOS
# ===========================================================================
print(f"\n{'='*80}")
print("ETAPA 5: TREINANDO MODELOS COM MELHORIAS")
print(f"{'='*80}")

scale_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
print(f"\n🔧 scale_pos_weight: {scale_weight * 1.3:.1f}")

# LightGBM com sample weights
print("\n[1/2] LightGBM com sample weights...")
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'scale_pos_weight': scale_weight * 1.3,
    'verbose': -1,
    'random_state': 42
}

dtrain_lgb = lgb.Dataset(X_train_scaled, y_train, weight=sample_weights_train)
lgb_model = lgb.train(lgb_params, dtrain_lgb, num_boost_round=200)

pred_lgb_test = lgb_model.predict(X_test_scaled)
print(f"✅ LightGBM AUC: {roc_auc_score(y_test, pred_lgb_test):.4f}")

# XGBoost com sample weights
print("\n[2/2] XGBoost com sample weights...")
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 8,
    'learning_rate': 0.05,
    'colsample_bytree': 0.8,
    'scale_pos_weight': scale_weight * 1.3,
    'random_state': 42
}

dtrain_xgb = xgb.DMatrix(X_train, y_train, weight=sample_weights_train, feature_names=selected_features)
dtest_xgb = xgb.DMatrix(X_test, y_test, feature_names=selected_features)

xgb_model = xgb.train(xgb_params, dtrain_xgb, num_boost_round=200)
pred_xgb_test = xgb_model.predict(dtest_xgb)

print(f"✅ XGBoost AUC: {roc_auc_score(y_test, pred_xgb_test):.4f}")

# Ensemble
pred_ensemble_test = 0.485 * pred_lgb_test + 0.515 * pred_xgb_test
print(f"✅ Ensemble AUC: {roc_auc_score(y_test, pred_ensemble_test):.4f}")

# ===========================================================================
# THRESHOLD DINÂMICO (NOVO!)
# ===========================================================================
print(f"\n{'='*80}")
print("ETAPA 6: 🆕 THRESHOLD DINÂMICO POR FAIXA")
print(f"{'='*80}")

def get_dynamic_threshold(stop_conv):
    """Threshold adaptativo baseado em conversão histórica"""
    if stop_conv > 0.7:
        return 0.40  # Agressivo para paradas muito altas
    elif stop_conv > 0.5:
        return 0.50
    elif stop_conv > 0.3:
        return 0.60
    else:
        return 0.75  # Conservador para paradas baixas

X_test_eval = X_test.copy()
X_test_eval['y_true'] = y_test.values
X_test_eval['y_pred_prob'] = pred_ensemble_test

if 'stop_historical_conversion' in X_test_eval.columns:
    X_test_eval['threshold_custom'] = X_test_eval['stop_historical_conversion'].apply(get_dynamic_threshold)
    X_test_eval['y_pred_dynamic'] = (X_test_eval['y_pred_prob'] > X_test_eval['threshold_custom']).astype(int)
    
    print("\n📊 THRESHOLDS APLICADOS:")
    print(X_test_eval.groupby('threshold_custom').size())
else:
    X_test_eval['y_pred_dynamic'] = (X_test_eval['y_pred_prob'] > 0.60).astype(int)

# Comparar threshold fixo vs dinâmico
y_pred_fixed = (pred_ensemble_test > 0.60).astype(int)
y_pred_dynamic = X_test_eval['y_pred_dynamic'].values

print(f"\n📊 COMPARAÇÃO THRESHOLD:")
print(f"   Fixo (0.60):")
print(f"      F1-Macro: {f1_score(y_test, y_pred_fixed, average='macro'):.4f}")
print(f"      F1 Classe 1: {f1_score(y_test, y_pred_fixed, pos_label=1):.4f}")

print(f"   Dinâmico:")
print(f"      F1-Macro: {f1_score(y_test, y_pred_dynamic, average='macro'):.4f}")
print(f"      F1 Classe 1: {f1_score(y_test, y_pred_dynamic, pos_label=1):.4f}")

# Usar dinâmico
y_pred_final = y_pred_dynamic

# ===========================================================================
# MÉTRICAS FINAIS
# ===========================================================================
print(f"\n{'='*80}")
print("ETAPA 7: MÉTRICAS FINAIS")
print(f"{'='*80}")

print("\n📊 CLASSIFICATION REPORT:")
print("="*80)
print(classification_report(y_test, y_pred_final, 
                          target_names=['Não Conversão', 'Conversão'],
                          digits=4))

cm = confusion_matrix(y_test, y_pred_final)
tn, fp, fn, tp = cm.ravel()

print(f"\n📊 CONFUSION MATRIX:")
print(f"  TN: {tn:,} | FP: {fp:,}")
print(f"  FN: {fn:,} | TP: {tp:,}")

# Salvar modelos
lgb_model.save_model('lightgbm_model_v8_phase2a.txt')
xgb_model.save_model('xgboost_model_v8_phase2a.json')

config = {
    'version': 'v8_phase2a',
    'date': datetime.now().isoformat(),
    'n_features': X.shape[1],
    'improvements': [
        'threshold_dinamico',
        'sample_weights_dinamicos',
        'stop_hour_conversion',
        'user_vs_stop_ratio',
        'dist_deviation',
        'geo_temporal',
        'raridade_features'
    ],
    'metrics': {
        'roc_auc': float(roc_auc_score(y_test, pred_ensemble_test)),
        'f1_macro': float(f1_score(y_test, y_pred_final, average='macro')),
        'f1_class_1': float(f1_score(y_test, y_pred_final, pos_label=1)),
        'precision': float(precision_score(y_test, y_pred_final)),
        'recall': float(recall_score(y_test, y_pred_final)),
    }
}

with open('model_config_v8_phase2a.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"\n{'='*80}")
print("✅ V8 FASE 2A COMPLETO!")
print(f"{'='*80}")
print(f"\n📊 COMPARAÇÃO FASE 1 → FASE 2A:")
print(f"   F1 Classe 1: 0.4206 → {f1_score(y_test, y_pred_final, pos_label=1):.4f}")
print(f"   Precision:   0.3023 → {precision_score(y_test, y_pred_final):.4f}")
print(f"   Recall:      0.6910 → {recall_score(y_test, y_pred_final):.4f}")
print("="*80)
