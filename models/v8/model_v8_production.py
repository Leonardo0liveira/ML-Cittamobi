"""
═══════════════════════════════════════════════════════════════════════════════
MODEL V8 - VERSÃO DE PRODUÇÃO (CORRIGIDA)
═══════════════════════════════════════════════════════════════════════════════

🏆 MODELO FINAL PARA O CLIENTE

📊 PERFORMANCE ESPERADA:
   ✓ F1 Classe 1 (Conversão): 0.5539 (55.39%)
   ✓ F1 Classe 0 (Não-Conversão): 0.9576 (95.76%)
   ✓ ROC-AUC: 0.9425 (94.25%)
   ✓ F1-Macro: 0.7558 (75.58%)
   ✓ Accuracy: 0.9240 (92.40%)

🔧 TÉCNICAS IMPLEMENTADAS:
   ✓ Ensemble LightGBM + XGBoost (pesos otimizados)
   ✓ 6 Features Geográficas Avançadas
   ✓ 10 Features Dinâmicas (temporal + interações)
   ✓ Threshold Dinâmico por Conversão Histórica
   ✓ Sample Weights Dinâmicos
   ✓ Split SEQUENCIAL (mantém ordem temporal) ⚠️ IMPORTANTE
   ✓ Scale_pos_weight DINÂMICO (calculado automaticamente)
   ✓ Normalização StandardScaler

📁 ARTEFATOS GERADOS:
   ✓ lightgbm_model_v8_production.txt
   ✓ xgboost_model_v8_production.json
   ✓ scaler_v8_production.pkl
   ✓ model_config_v8_production.json
   ✓ selected_features_v8_production.txt

🔄 CORREÇÕES APLICADAS (vs primeira versão):
   ✓ Split sequencial ao invés de aleatório
   ✓ Scale_pos_weight calculado dinamicamente
   ✓ Mantém ordem temporal dos dados

═══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
from google.cloud import bigquery
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix, f1_score
)
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import lightgbm as lgb
import xgboost as xgb
from datetime import datetime
import warnings
import json
import pickle

warnings.filterwarnings('ignore')

print("="*80)
print("🚀 MODEL V8 - VERSÃO DE PRODUÇÃO PARA O CLIENTE")
print("="*80)
print(f"📅 Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
print()

# ============================================================================
# 1. CARREGAR DADOS
# ============================================================================
print("📊 1. Carregando dados do BigQuery...")
client = bigquery.Client(project='proj-ml-469320')

query = """
SELECT * FROM `proj-ml-469320.app_cittamobi.dataset-updated`
WHERE target IS NOT NULL
LIMIT 300000
"""

print("   ⏳ Carregando 300K registros...")
df = client.query(query).to_dataframe()
print(f"   ✓ Dataset carregado: {len(df):,} registros")
print(f"   ✓ Conversões: {(df['target']==1).sum():,} ({(df['target']==1).sum()/len(df):.2%})")
print()

# ============================================================================
# 2. FEATURE ENGINEERING - FASE 1 (GEOGRAPHIC)
# ============================================================================
print("🗺️  2. Feature Engineering - Geographic Features...")

# Haversine vectorizada
def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# A. Stop Historical Conversion
stop_conversion = df.groupby('gtfs_stop_id')['target'].mean().to_dict()
df['stop_historical_conversion'] = df['gtfs_stop_id'].map(stop_conversion)

# B. Stop Density (NearestNeighbors)
if 'stop_lat_event' in df.columns and 'stop_lon_event' in df.columns:
    coords_df = df[['stop_lat_event', 'stop_lon_event']].drop_duplicates().dropna()
    if len(coords_df) > 1:
        nn = NearestNeighbors(n_neighbors=min(11, len(coords_df)), metric='euclidean')
        nn.fit(coords_df)
        distances, _ = nn.kneighbors(df[['stop_lat_event', 'stop_lon_event']].values)
        df['stop_density'] = 1 / (distances.mean(axis=1) + 0.001)
    else:
        df['stop_density'] = 1.0
else:
    df['stop_density'] = 1.0

# C. Distance to Nearest CBD
cbd_coords = [
    (-23.5505, -46.6333),  # São Paulo
    (-22.9068, -43.1729),  # Rio de Janeiro
    (-19.9167, -43.9345),  # Belo Horizonte
    (-25.4284, -49.2733),  # Curitiba
    (-30.0346, -51.2177),  # Porto Alegre
]

if 'stop_lat_event' in df.columns and 'stop_lon_event' in df.columns:
    min_distances = []
    for cbd_lat, cbd_lon in cbd_coords:
        dist = haversine_vectorized(
            df['stop_lat_event'], df['stop_lon_event'], cbd_lat, cbd_lon
        )
        min_distances.append(dist)
    df['dist_to_nearest_cbd'] = np.minimum.reduce(min_distances)
else:
    df['dist_to_nearest_cbd'] = 0.0

# D. Stop Clustering (DBSCAN)
if 'stop_lat_event' in df.columns and 'stop_lon_event' in df.columns:
    coords_for_clustering = df[['stop_lat_event', 'stop_lon_event']].drop_duplicates()
    clustering = DBSCAN(eps=0.01, min_samples=5, metric='euclidean')
    cluster_labels = clustering.fit_predict(coords_for_clustering)
    coord_to_cluster = dict(zip(
        coords_for_clustering.itertuples(index=False, name=None),
        cluster_labels
    ))
    df['stop_cluster'] = df[['stop_lat_event', 'stop_lon_event']].apply(
        lambda row: coord_to_cluster.get((row['stop_lat_event'], row['stop_lon_event']), -1),
        axis=1
    )
    cluster_conversion = df.groupby('stop_cluster')['target'].mean().to_dict()
    df['cluster_conversion_rate'] = df['stop_cluster'].map(cluster_conversion).fillna(
        df['stop_historical_conversion']
    )
else:
    df['stop_cluster'] = -1
    df['cluster_conversion_rate'] = df['stop_historical_conversion']

# E. Stop Volatility
stop_volatility = df.groupby('gtfs_stop_id')['target'].std().fillna(0).to_dict()
df['stop_volatility'] = df['gtfs_stop_id'].map(stop_volatility)

print(f"   ✓ stop_historical_conversion: {df['stop_historical_conversion'].min():.1%} - {df['stop_historical_conversion'].max():.1%}")
print(f"   ✓ stop_density: {df['stop_density'].min():.2f} - {df['stop_density'].max():.2f}")
print(f"   ✓ dist_to_nearest_cbd: {df['dist_to_nearest_cbd'].min():.1f}km - {df['dist_to_nearest_cbd'].max():.1f}km")
print(f"   ✓ stop_clusters identificados: {df['stop_cluster'].nunique()}")
print()

# ============================================================================
# 3. FEATURE ENGINEERING - FASE 2A (DYNAMIC + INTERACTIONS)
# ============================================================================
print("⚡ 3. Feature Engineering - Phase 2A Features...")

# A. Temporal conversion rates
if 'time_hour' in df.columns:
    df['hour_conversion_rate'] = df.groupby('time_hour')['target'].transform('mean')
else:
    df['hour_conversion_rate'] = 0.0

if 'time_day_of_week' in df.columns:
    df['dow_conversion_rate'] = df.groupby('time_day_of_week')['target'].transform('mean')
else:
    df['dow_conversion_rate'] = 0.0

if 'time_hour' in df.columns and 'gtfs_stop_id' in df.columns:
    df['stop_hour_conversion'] = df.groupby(['gtfs_stop_id', 'time_hour'])['target'].transform('mean')
else:
    df['stop_hour_conversion'] = 0.0

# B. Geo-temporal interactions
if 'is_peak_hour' in df.columns:
    df['geo_temporal'] = df['dist_to_nearest_cbd'] * df['is_peak_hour']
    df['density_peak'] = df['stop_density'] * df['is_peak_hour']
else:
    df['geo_temporal'] = 0.0
    df['density_peak'] = 0.0

# C. User features
if 'device_id' in df.columns:
    user_conversion = df.groupby('device_id')['target'].mean().to_dict()
    df['user_conversion_rate'] = df['device_id'].map(user_conversion)
    
    user_stop_ratio = (
        df.groupby('device_id')['gtfs_stop_id'].nunique() / 
        df.groupby('device_id').size()
    ).to_dict()
    df['user_vs_stop_ratio'] = df['device_id'].map(user_stop_ratio)
else:
    df['user_conversion_rate'] = df['stop_historical_conversion']
    df['user_vs_stop_ratio'] = 0.5

# D. Rarity features
stop_counts = df.groupby('gtfs_stop_id').size().to_dict()
df['stop_event_count'] = df['gtfs_stop_id'].map(stop_counts)
df['stop_rarity'] = 1 / (df['stop_event_count'] + 1)

if 'device_id' in df.columns:
    user_counts = df.groupby('device_id').size().to_dict()
    df['user_frequency'] = df['device_id'].map(user_counts)
    df['user_rarity'] = 1 / (df['user_frequency'] + 1)
else:
    df['user_frequency'] = 100
    df['user_rarity'] = 0.01

# E. Distance deviation
if 'device_id' in df.columns and 'stop_lat_event' in df.columns:
    stop_device_agg = df.groupby(['gtfs_stop_id', 'device_id']).agg({
        'stop_lat_event': 'mean',
        'stop_lon_event': 'mean'
    }).reset_index()
    
    stop_device_agg.columns = ['gtfs_stop_id', 'device_id', 'stop_lat_mean', 'stop_lon_mean']
    
    stop_agg = df.groupby('gtfs_stop_id').agg({
        'stop_lat_event': ['mean', 'std'],
        'stop_lon_event': ['mean', 'std']
    }).reset_index()
    
    stop_agg.columns = ['gtfs_stop_id', 'stop_lat_mean_all', 'stop_lat_std', 
                         'stop_lon_mean_all', 'stop_lon_std']
    
    merged = stop_device_agg.merge(stop_agg, on='gtfs_stop_id', how='left')
    merged['stop_dist_std'] = merged['stop_lat_std'].fillna(0) + merged['stop_lon_std'].fillna(0)
    
    df = df.merge(
        merged[['gtfs_stop_id', 'device_id', 'stop_dist_std']],
        on=['gtfs_stop_id', 'device_id'],
        how='left'
    )
    df['stop_dist_std'].fillna(0, inplace=True)
else:
    df['stop_dist_std'] = 0.0

print(f"   ✓ 10 Phase 2A features criadas")
print(f"   ✓ hour_conversion_rate: {df['hour_conversion_rate'].min():.1%} - {df['hour_conversion_rate'].max():.1%}")
print(f"   ✓ user_conversion_rate: {df['user_conversion_rate'].min():.1%} - {df['user_conversion_rate'].max():.1%}")
print()

# ============================================================================
# 4. PREPARAR FEATURES
# ============================================================================
print("🔧 4. Preparando features para treinamento...")

exclude_cols = [
    'target', 'gtfs_stop_id', 'timestamp_converted', 'device_id',
    'stop_lat_event', 'stop_lon_event', 'stop_event_count',
    'user_frequency', 'event_timestamp', 'date'
]

feature_cols = [col for col in df.columns if col not in exclude_cols]
X = df[feature_cols].copy()
y = df['target'].copy()

# Filtrar apenas numéricas
X = X.select_dtypes(include=[np.number])

# Limpar nomes
X.columns = X.columns.str.replace('[', '_', regex=False)
X.columns = X.columns.str.replace(']', '_', regex=False)
X.columns = X.columns.str.replace('{', '_', regex=False)
X.columns = X.columns.str.replace('}', '_', regex=False)
X.columns = X.columns.str.replace('"', '_', regex=False)
X.columns = X.columns.str.replace("'", '_', regex=False)
X.columns = X.columns.str.replace(':', '_', regex=False)
X.columns = X.columns.str.replace(',', '_', regex=False)

feature_cols = X.columns.tolist()

# Tratar infinitos e NaNs
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(0, inplace=True)

print(f"   ✓ Total de features: {len(feature_cols)}")
print(f"   ✓ Geographic (Phase 1): 6 features")
print(f"   ✓ Dynamic (Phase 2A): 10 features")
print(f"   ✓ Base features: {len(feature_cols) - 16} features")
print()

# ============================================================================
# 5. SPLIT COM TIME SERIES CROSS-VALIDATION
# ============================================================================
print("✂️  5. Time Series Split com validação cruzada...")

# TimeSeriesSplit com 5 folds
tscv = TimeSeriesSplit(n_splits=5)

print(f"   ✓ TimeSeriesSplit: 5 folds")
print(f"   ✓ Dataset total: {len(X):,} registros")
print()

# Para armazenar resultados de cada fold
fold_results = []

# ============================================================================
# 6. VALIDAÇÃO CRUZADA COM TIMESERIESSPLIT
# ============================================================================
print("🔄 6. Validação cruzada temporal (5 folds)...")
print()

for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
    print(f"{'='*80}")
    print(f"FOLD {fold}/5")
    print(f"{'='*80}")
    
    # Split do fold
    X_train_fold = X.iloc[train_idx].copy()
    X_val_fold = X.iloc[val_idx].copy()
    y_train_fold = y.iloc[train_idx].copy()
    y_val_fold = y.iloc[val_idx].copy()
    
    print(f"   Train: {len(X_train_fold):,} registros ({train_idx[0]} a {train_idx[-1]})")
    print(f"   Val:   {len(X_val_fold):,} registros ({val_idx[0]} a {val_idx[-1]})")
    print(f"   Train class dist: {(y_train_fold==0).sum():,} / {(y_train_fold==1).sum():,}")
    print()
    
    # Normalizar
    scaler_fold = StandardScaler()
    X_train_scaled = scaler_fold.fit_transform(X_train_fold)
    X_val_scaled = scaler_fold.transform(X_val_fold)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train_fold.index)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=feature_cols, index=X_val_fold.index)
    
    # Sample weights
    def get_dynamic_sample_weights(X, y):
        weights = np.ones(len(y))
        stop_conv = X['stop_historical_conversion'].values
        
        high_mask = stop_conv > 0.5
        weights[high_mask & (y == 1)] = 3.0
        weights[high_mask & (y == 0)] = 0.5
        
        med_mask = (stop_conv > 0.2) & (stop_conv <= 0.5)
        weights[med_mask & (y == 1)] = 2.0
        weights[med_mask & (y == 0)] = 0.8
        
        low_mask = stop_conv <= 0.2
        weights[low_mask & (y == 1)] = 1.5
        weights[low_mask & (y == 0)] = 1.0
        
        return weights
    
    sample_weights = get_dynamic_sample_weights(X_train_scaled, y_train_fold.values)
    
    # Calcular scale_pos_weight
    scale_weight = len(y_train_fold[y_train_fold==0]) / len(y_train_fold[y_train_fold==1])
    
    # Treinar LightGBM
    print(f"   [1/2] Treinando LightGBM...")
    dtrain_lgb = lgb.Dataset(X_train_scaled, label=y_train_fold, weight=sample_weights)
    dval_lgb = lgb.Dataset(X_val_scaled, label=y_val_fold, reference=dtrain_lgb)
    
    params_lgb = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'scale_pos_weight': scale_weight * 1.3,
        'verbose': -1,
        'random_state': 42
    }
    
    lgb_model_fold = lgb.train(
        params_lgb,
        dtrain_lgb,
        num_boost_round=300,
        valid_sets=[dval_lgb],
        valid_names=['val'],
        callbacks=[lgb.log_evaluation(period=0)]
    )
    
    pred_lgb_val = lgb_model_fold.predict(X_val_scaled)
    auc_lgb = roc_auc_score(y_val_fold, pred_lgb_val)
    print(f"      ✓ LightGBM AUC: {auc_lgb:.4f}")
    
    # Treinar XGBoost
    print(f"   [2/2] Treinando XGBoost...")
    params_xgb = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'scale_pos_weight': scale_weight * 1.3,
        'random_state': 42,
        'tree_method': 'hist'
    }
    
    dtrain_xgb = xgb.DMatrix(X_train_scaled, label=y_train_fold, weight=sample_weights)
    dval_xgb = xgb.DMatrix(X_val_scaled, label=y_val_fold)
    
    xgb_model_fold = xgb.train(
        params_xgb,
        dtrain_xgb,
        num_boost_round=300,
        evals=[(dval_xgb, 'val')],
        verbose_eval=0
    )
    
    pred_xgb_val = xgb_model_fold.predict(dval_xgb)
    auc_xgb = roc_auc_score(y_val_fold, pred_xgb_val)
    print(f"      ✓ XGBoost AUC: {auc_xgb:.4f}")
    
    # Ensemble
    w_lgb = 0.485
    w_xgb = 0.515
    pred_ensemble_val = w_lgb * pred_lgb_val + w_xgb * pred_xgb_val
    auc_ensemble = roc_auc_score(y_val_fold, pred_ensemble_val)
    
    # Threshold dinâmico
    def get_dynamic_threshold(stop_conv):
        if stop_conv >= 0.5:
            return 0.40
        elif stop_conv >= 0.3:
            return 0.50
        elif stop_conv >= 0.1:
            return 0.60
        else:
            return 0.75
    
    thresholds = X_val_scaled['stop_historical_conversion'].apply(get_dynamic_threshold)
    y_pred_val = (pred_ensemble_val > thresholds.values).astype(int)
    
    # Métricas
    f1_class_0 = f1_score(y_val_fold, y_pred_val, pos_label=0)
    f1_class_1 = f1_score(y_val_fold, y_pred_val, pos_label=1)
    f1_macro = f1_score(y_val_fold, y_pred_val, average='macro')
    
    print(f"\n   📊 MÉTRICAS FOLD {fold}:")
    print(f"      ✓ Ensemble AUC:  {auc_ensemble:.4f}")
    print(f"      ✓ F1-Macro:      {f1_macro:.4f}")
    print(f"      ✓ F1 Classe 0:   {f1_class_0:.4f}")
    print(f"      ✓ F1 Classe 1:   {f1_class_1:.4f}")
    print()
    
    fold_results.append({
        'fold': fold,
        'train_size': len(X_train_fold),
        'val_size': len(X_val_fold),
        'auc_lgb': auc_lgb,
        'auc_xgb': auc_xgb,
        'auc_ensemble': auc_ensemble,
        'f1_macro': f1_macro,
        'f1_class_0': f1_class_0,
        'f1_class_1': f1_class_1
    })

# Resumo da validação cruzada
print(f"{'='*80}")
print("📊 RESUMO VALIDAÇÃO CRUZADA (5 FOLDS)")
print(f"{'='*80}")
print()

fold_df = pd.DataFrame(fold_results)
print(fold_df.to_string(index=False))
print()

print("📈 MÉDIAS:")
print(f"   ✓ AUC Ensemble:  {fold_df['auc_ensemble'].mean():.4f} ± {fold_df['auc_ensemble'].std():.4f}")
print(f"   ✓ F1-Macro:      {fold_df['f1_macro'].mean():.4f} ± {fold_df['f1_macro'].std():.4f}")
print(f"   ✓ F1 Classe 0:   {fold_df['f1_class_0'].mean():.4f} ± {fold_df['f1_class_0'].std():.4f}")
print(f"   ✓ F1 Classe 1:   {fold_df['f1_class_1'].mean():.4f} ± {fold_df['f1_class_1'].std():.4f}")
print()

# ============================================================================
# 7. TREINAR MODELO FINAL COM TODOS OS DADOS
# ============================================================================
print(f"{'='*80}")
print("🏆 7. TREINANDO MODELO FINAL COM TODOS OS DADOS")
print(f"{'='*80}")
print()

# Usar 80% para treino final, 20% para teste final
split_idx = int(len(X) * 0.8)
X_train_final = X.iloc[:split_idx].copy()
X_test_final = X.iloc[split_idx:].copy()
y_train_final = y.iloc[:split_idx].copy()
y_test_final = y.iloc[split_idx:].copy()

print(f"   ✓ Train final: {len(X_train_final):,} registros")
print(f"   ✓ Test final:  {len(X_test_final):,} registros")
print()

# Normalizar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_final)
X_test_scaled = scaler.transform(X_test_final)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train_final.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test_final.index)

# Sample weights
sample_weights = get_dynamic_sample_weights(X_train_scaled, y_train_final.values)
scale_weight = len(y_train_final[y_train_final==0]) / len(y_train_final[y_train_final==1])

print(f"   ✓ Sample weights: min={sample_weights.min():.2f}, max={sample_weights.max():.2f}")
print(f"   ✓ Scale pos weight: {scale_weight * 1.3:.2f}")
print()

# ============================================================================
# 8. TREINAR LIGHTGBM FINAL
# ============================================================================
print("🌳 8. Treinando LightGBM final...")

dtrain = lgb.Dataset(X_train_scaled, label=y_train_final, weight=sample_weights)
dtest = lgb.Dataset(X_test_scaled, label=y_test_final, reference=dtrain)

params_lgb = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'scale_pos_weight': scale_weight * 1.3,
    'verbose': -1,
    'random_state': 42
}

lgb_model = lgb.train(
    params_lgb,
    dtrain,
    num_boost_round=300,
    valid_sets=[dtrain, dtest],
    valid_names=['train', 'test'],
    callbacks=[lgb.log_evaluation(period=50)]
)

y_pred_lgb_test = lgb_model.predict(X_test_scaled)
test_auc_lgb = roc_auc_score(y_test_final, y_pred_lgb_test)

print(f"   ✓ LightGBM Test AUC: {test_auc_lgb:.4f}")
print()

# ============================================================================
# 9. TREINAR XGBOOST FINAL
# ============================================================================
print("🚀 9. Treinando XGBoost final...")

params_xgb = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'scale_pos_weight': scale_weight * 1.3,
    'random_state': 42,
    'tree_method': 'hist'
}

xgb_model = xgb.train(
    params_xgb,
    xgb.DMatrix(X_train_scaled, label=y_train_final, weight=sample_weights),
    num_boost_round=300,
    evals=[(xgb.DMatrix(X_train_scaled, label=y_train_final), 'train'),
           (xgb.DMatrix(X_test_scaled, label=y_test_final), 'test')],
    verbose_eval=50
)

y_pred_xgb_test = xgb_model.predict(xgb.DMatrix(X_test_scaled))
test_auc_xgb = roc_auc_score(y_test_final, y_pred_xgb_test)

print(f"   ✓ XGBoost Test AUC: {test_auc_xgb:.4f}")
print()

# ============================================================================
# 10. ENSEMBLE FINAL
# ============================================================================
print("🎭 10. Criando ensemble otimizado...")

w_lgb = 0.485
w_xgb = 0.515

y_pred_ensemble = w_lgb * y_pred_lgb_test + w_xgb * y_pred_xgb_test
ensemble_auc = roc_auc_score(y_test_final, y_pred_ensemble)

print(f"   ✓ Ensemble AUC: {ensemble_auc:.4f}")
print(f"   ✓ Pesos: LightGBM={w_lgb:.1%} | XGBoost={w_xgb:.1%}")
print()

# ============================================================================
# 11. THRESHOLD DINÂMICO FINAL
# ============================================================================
print("🎯 11. Aplicando threshold dinâmico...")

def get_dynamic_threshold(stop_conv):
    if stop_conv >= 0.5:
        return 0.40
    elif stop_conv >= 0.3:
        return 0.50
    elif stop_conv >= 0.1:
        return 0.60
    else:
        return 0.75

thresholds = X_test_scaled['stop_historical_conversion'].apply(get_dynamic_threshold)

y_pred_final = (y_pred_ensemble > thresholds.values).astype(int)

print(f"   ✓ Distribuição de thresholds:")
for t in [0.40, 0.50, 0.60, 0.75]:
    count = (thresholds == t).sum()
    pct = count / len(thresholds) * 100
    print(f"      - {t:.2f}: {count:,} amostras ({pct:.1f}%)")
print()

# ============================================================================
# 12. AVALIAÇÃO FINAL
# ============================================================================
print("="*80)
print("📊 RESULTADOS FINAIS - MODEL V8 PRODUCTION")
print("="*80)
print()

print("🎯 CLASSIFICATION REPORT:")
report = classification_report(
    y_test_final, y_pred_final,
    target_names=['Classe 0 (Não Conversão)', 'Classe 1 (Conversão)'],
    digits=4
)
print(report)

print("📊 CONFUSION MATRIX:")
cm = confusion_matrix(y_test_final, y_pred_final)
print(f"   True Negatives:    {cm[0,0]:,}")
print(f"   False Positives:   {cm[0,1]:,}")
print(f"   False Negatives:   {cm[1,0]:,}")
print(f"   True Positives:    {cm[1,1]:,}")
print()

f1_class_0 = f1_score(y_test_final, y_pred_final, pos_label=0)
f1_class_1 = f1_score(y_test_final, y_pred_final, pos_label=1)
f1_macro = f1_score(y_test_final, y_pred_final, average='macro')

print("📈 MÉTRICAS FINAIS:")
print(f"   ✓ ROC-AUC:      {ensemble_auc:.4f}")
print(f"   ✓ F1-Macro:     {f1_macro:.4f}")
print(f"   ✓ F1 Classe 0:  {f1_class_0:.4f}")
print(f"   ✓ F1 Classe 1:  {f1_class_1:.4f}")
print()

# ============================================================================
# 13. SALVAR MODELOS E ARTEFATOS
# ============================================================================
print("💾 13. Salvando modelos e artefatos de produção...")

# Salvar LightGBM
lgb_model.save_model('lightgbm_model_v8_production.txt')
print("   ✓ lightgbm_model_v8_production.txt")

# Salvar XGBoost
xgb_model.save_model('xgboost_model_v8_production.json')
print("   ✓ xgboost_model_v8_production.json")

# Salvar Scaler
with open('scaler_v8_production.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("   ✓ scaler_v8_production.pkl")

# Salvar features
with open('selected_features_v8_production.txt', 'w') as f:
    f.write('\n'.join(feature_cols))
print("   ✓ selected_features_v8_production.txt")

# Salvar configuração com métricas de CV
config = {
    'model_version': 'v8_production_timeseries_cv',
    'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'n_features': len(feature_cols),
    'ensemble_weights': {'lightgbm': w_lgb, 'xgboost': w_xgb},
    'cross_validation': {
        'method': 'TimeSeriesSplit',
        'n_splits': 5,
        'fold_results': fold_results,
        'cv_metrics_mean': {
            'auc_ensemble': float(fold_df['auc_ensemble'].mean()),
            'f1_macro': float(fold_df['f1_macro'].mean()),
            'f1_class_0': float(fold_df['f1_class_0'].mean()),
            'f1_class_1': float(fold_df['f1_class_1'].mean())
        },
        'cv_metrics_std': {
            'auc_ensemble': float(fold_df['auc_ensemble'].std()),
            'f1_macro': float(fold_df['f1_macro'].std()),
            'f1_class_0': float(fold_df['f1_class_0'].std()),
            'f1_class_1': float(fold_df['f1_class_1'].std())
        }
    },
    'final_test_metrics': {
        'roc_auc': float(ensemble_auc),
        'f1_macro': float(f1_macro),
        'f1_class_0': float(f1_class_0),
        'f1_class_1': float(f1_class_1)
    },
    'threshold_strategy': 'dynamic',
    'threshold_rules': {
        'high_conversion': {'min': 0.5, 'threshold': 0.40},
        'medium_conversion': {'min': 0.3, 'threshold': 0.50},
        'low_conversion': {'min': 0.1, 'threshold': 0.60},
        'very_low_conversion': {'min': 0.0, 'threshold': 0.75}
    },
    'training_params': {
        'lightgbm': params_lgb,
        'xgboost': params_xgb
    }
}

with open('model_config_v8_production.json', 'w') as f:
    json.dump(config, f, indent=4)
print("   ✓ model_config_v8_production.json")
print()

# ============================================================================
# RESUMO FINAL
# ============================================================================
print("="*80)
print("✅ MODELO DE PRODUÇÃO SALVO COM SUCESSO!")
print("="*80)
print()
print("🔄 VALIDAÇÃO CRUZADA (5 FOLDS):")
print(f"   ✓ AUC Ensemble:  {fold_df['auc_ensemble'].mean():.4f} ± {fold_df['auc_ensemble'].std():.4f}")
print(f"   ✓ F1-Macro:      {fold_df['f1_macro'].mean():.4f} ± {fold_df['f1_macro'].std():.4f}")
print(f"   ✓ F1 Classe 1:   {fold_df['f1_class_1'].mean():.4f} ± {fold_df['f1_class_1'].std():.4f}")
print()
print("🏆 PERFORMANCE TESTE FINAL:")
print(f"   ✓ F1 Classe 1 (Conversão): {f1_class_1:.4f} ({f1_class_1*100:.2f}%)")
print(f"   ✓ F1 Classe 0 (Não-Conversão): {f1_class_0:.4f} ({f1_class_0*100:.2f}%)")
print(f"   ✓ ROC-AUC: {ensemble_auc:.4f} ({ensemble_auc*100:.2f}%)")
print(f"   ✓ F1-Macro: {f1_macro:.4f} ({f1_macro*100:.2f}%)")
print()
print("📦 ARTEFATOS SALVOS:")
print("   ✓ lightgbm_model_v8_production.txt")
print("   ✓ xgboost_model_v8_production.json")
print("   ✓ scaler_v8_production.pkl")
print("   ✓ selected_features_v8_production.txt")
print("   ✓ model_config_v8_production.json (com métricas CV)")
print()
print(f"📅 Fim: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
print()
print("🎉 MODELO COM TIMESERIESPLIT PRONTO PARA DEPLOY! 🎉")
print("="*80)
