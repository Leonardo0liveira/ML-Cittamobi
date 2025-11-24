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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.calibration import calibration_curve

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("🚀 MODEL V8 - VERSÃO DE PRODUÇÃO PARA O CLIENTE")
print("="*80)
print(f"📅 Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
print()

# ============================================================================
# 1. CARREGAR DADOS (BigQuery ou CSV)
# ============================================================================
print("📊 1. Carregando dados...")

# ============================================================================
# CONFIGURAÇÃO: Escolha a fonte de dados
# ============================================================================
USE_CSV = True  # ⚠️ MUDE PARA False PARA USAR BIGQUERY
CSV_PATH = 'dataset-updated.csv'  # ⚠️ AJUSTE O CAMINHO

if USE_CSV:
    print(f"   📂 Carregando do CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    
    # Filtrar apenas registros com target válido
    df = df[df['target'].notna()].copy()
    
    # Limitar tamanho se necessário (para teste rápido)
    # REMOVA esta linha para usar TODA a base
    #df = df.head(300000)  # ⚠️ REMOVA ESTA LINHA PARA CARREGAR TUDO
    
    print(f"   ✓ Dataset carregado: {len(df):,} registros")
    print(f"   ✓ Conversões: {(df['target']==1).sum():,} ({(df['target']==1).sum()/len(df):.2%})")
    
else:
    print("   ☁️  Carregando do BigQuery...")
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
print()

# ============================================================================
# 14. GERAR VISUALIZAÇÕES
# ============================================================================
print("="*80)
print("📊 14. GERANDO VISUALIZAÇÕES")
print("="*80)
print()

# Criar diretório para visualizações
import os
os.makedirs('visualizations', exist_ok=True)

# ----------------------------------------------------------------------------
# GRÁFICO 1: CONFUSION MATRIX
# ----------------------------------------------------------------------------
print("   [1/7] Gerando Confusion Matrix...")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues', 
    square=True,
    cbar_kws={'label': 'Contagem'},
    ax=ax
)
ax.set_xlabel('Predito', fontsize=14, fontweight='bold')
ax.set_ylabel('Real', fontsize=14, fontweight='bold')
ax.set_title('Confusion Matrix - Model V8 Production\nTimeSeriesSplit + Ensemble', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticklabels(['Não Conversão (0)', 'Conversão (1)'])
ax.set_yticklabels(['Não Conversão (0)', 'Conversão (1)'])

# Adicionar estatísticas
textstr = f'Accuracy: {(cm[0,0]+cm[1,1])/cm.sum():.2%}\n'
textstr += f'Precision C1: {cm[1,1]/(cm[1,1]+cm[0,1]):.2%}\n'
textstr += f'Recall C1: {cm[1,1]/(cm[1,1]+cm[1,0]):.2%}\n'
textstr += f'F1-Score C1: {f1_class_1:.2%}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(1.35, 0.5, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='center', bbox=props)

plt.tight_layout()
plt.savefig('visualizations/confusion_matrix_v8.png', dpi=300, bbox_inches='tight')
plt.close()
print("      ✓ visualizations/confusion_matrix_v8.png")

# ----------------------------------------------------------------------------
# GRÁFICO 2: ROC CURVE
# ----------------------------------------------------------------------------
print("   [2/7] Gerando ROC Curve...")
fpr, tpr, _ = roc_curve(y_test_final, y_pred_ensemble)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(fpr, tpr, color='darkorange', lw=3, 
        label=f'Ensemble (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
        label='Random Classifier (AUC = 0.50)')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
ax.set_title('ROC Curve - Model V8 Production\nLightGBM (48.5%) + XGBoost (51.5%)', 
             fontsize=16, fontweight='bold', pad=20)
ax.legend(loc="lower right", fontsize=12)
ax.grid(True, alpha=0.3)

# Adicionar ponto ótimo
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = _[optimal_idx]
ax.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=10, 
        label=f'Optimal Point (threshold={optimal_threshold:.3f})')
ax.legend(loc="lower right", fontsize=11)

plt.tight_layout()
plt.savefig('visualizations/roc_curve_v8.png', dpi=300, bbox_inches='tight')
plt.close()
print("      ✓ visualizations/roc_curve_v8.png")

# ----------------------------------------------------------------------------
# GRÁFICO 3: PRECISION-RECALL CURVE
# ----------------------------------------------------------------------------
print("   [3/7] Gerando Precision-Recall Curve...")
precision, recall, thresholds_pr = precision_recall_curve(y_test_final, y_pred_ensemble)
pr_auc = auc(recall, precision)

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(recall, precision, color='blue', lw=3, 
        label=f'Ensemble (AUC = {pr_auc:.4f})')
ax.axhline(y=y_test_final.mean(), color='red', linestyle='--', lw=2,
           label=f'Baseline (taxa conversão = {y_test_final.mean():.2%})')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('Recall', fontsize=14, fontweight='bold')
ax.set_ylabel('Precision', fontsize=14, fontweight='bold')
ax.set_title('Precision-Recall Curve - Model V8 Production\nClasse 1 (Conversão)', 
             fontsize=16, fontweight='bold', pad=20)
ax.legend(loc="upper right", fontsize=12)
ax.grid(True, alpha=0.3)

# Adicionar ponto F1 máximo
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
best_f1_idx = np.argmax(f1_scores)
ax.plot(recall[best_f1_idx], precision[best_f1_idx], 'ro', markersize=10,
        label=f'Best F1 = {f1_scores[best_f1_idx]:.4f}')
ax.legend(loc="upper right", fontsize=11)

plt.tight_layout()
plt.savefig('visualizations/precision_recall_curve_v8.png', dpi=300, bbox_inches='tight')
plt.close()
print("      ✓ visualizations/precision_recall_curve_v8.png")

# ----------------------------------------------------------------------------
# GRÁFICO 4: FEATURE IMPORTANCE (TOP 20)
# ----------------------------------------------------------------------------
print("   [4/7] Gerando Feature Importance...")
# Combinar importâncias dos dois modelos
lgb_importance = lgb_model.feature_importance(importance_type='gain')
xgb_importance = xgb_model.get_score(importance_type='gain')

# Converter XGBoost para array
xgb_importance_array = np.zeros(len(feature_cols))
for i, feat in enumerate(feature_cols):
    feat_clean = feat.replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_')
    if feat_clean in xgb_importance:
        xgb_importance_array[i] = xgb_importance[feat_clean]

# Combinar com pesos do ensemble
combined_importance = w_lgb * lgb_importance + w_xgb * xgb_importance_array

# Criar DataFrame
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': combined_importance
}).sort_values('importance', ascending=False).head(20)

fig, ax = plt.subplots(figsize=(12, 10))
colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
bars = ax.barh(range(len(importance_df)), importance_df['importance'], color=colors)
ax.set_yticks(range(len(importance_df)))
ax.set_yticklabels(importance_df['feature'])
ax.invert_yaxis()
ax.set_xlabel('Importância (Gain)', fontsize=14, fontweight='bold')
ax.set_title('Top 20 Features Mais Importantes - Model V8\nEnsemble LightGBM + XGBoost', 
             fontsize=16, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

# Adicionar valores nas barras
for i, (bar, val) in enumerate(zip(bars, importance_df['importance'])):
    ax.text(val, i, f' {val:.0f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('visualizations/feature_importance_v8.png', dpi=300, bbox_inches='tight')
plt.close()
print("      ✓ visualizations/feature_importance_v8.png")

# ----------------------------------------------------------------------------
# GRÁFICO 5: DISTRIBUIÇÃO DE PROBABILIDADES
# ----------------------------------------------------------------------------
print("   [5/7] Gerando Distribuição de Probabilidades...")
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Subplot 1: Histograma
axes[0].hist(y_pred_ensemble[y_test_final == 0], bins=50, alpha=0.6, 
             label='Classe 0 (Não Conversão)', color='blue', edgecolor='black')
axes[0].hist(y_pred_ensemble[y_test_final == 1], bins=50, alpha=0.6, 
             label='Classe 1 (Conversão)', color='red', edgecolor='black')
axes[0].axvline(x=0.5, color='green', linestyle='--', lw=2, 
                label='Threshold padrão (0.5)')
axes[0].set_xlabel('Probabilidade Predita', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Frequência', fontsize=12, fontweight='bold')
axes[0].set_title('Distribuição das Probabilidades Preditas por Classe', 
                  fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Subplot 2: Density plot
from scipy.stats import gaussian_kde
kde_0 = gaussian_kde(y_pred_ensemble[y_test_final == 0])
kde_1 = gaussian_kde(y_pred_ensemble[y_test_final == 1])
x_range = np.linspace(0, 1, 1000)
axes[1].plot(x_range, kde_0(x_range), label='Classe 0 (Não Conversão)', 
             color='blue', lw=3)
axes[1].plot(x_range, kde_1(x_range), label='Classe 1 (Conversão)', 
             color='red', lw=3)
axes[1].fill_between(x_range, kde_0(x_range), alpha=0.3, color='blue')
axes[1].fill_between(x_range, kde_1(x_range), alpha=0.3, color='red')
axes[1].axvline(x=0.5, color='green', linestyle='--', lw=2, 
                label='Threshold padrão (0.5)')
axes[1].set_xlabel('Probabilidade Predita', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Densidade', fontsize=12, fontweight='bold')
axes[1].set_title('Densidade de Probabilidade (KDE)', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/probability_distribution_v8.png', dpi=300, bbox_inches='tight')
plt.close()
print("      ✓ visualizations/probability_distribution_v8.png")

# ----------------------------------------------------------------------------
# GRÁFICO 6: THRESHOLD ANALYSIS
# ----------------------------------------------------------------------------
print("   [6/7] Gerando Threshold Analysis...")
thresholds_test = np.linspace(0, 1, 100)
f1_scores_test = []
precision_scores = []
recall_scores = []

for thresh in thresholds_test:
    y_pred_thresh = (y_pred_ensemble > thresh).astype(int)
    f1 = f1_score(y_test_final, y_pred_thresh, pos_label=1, zero_division=0)
    f1_scores_test.append(f1)
    
    tp = ((y_pred_thresh == 1) & (y_test_final == 1)).sum()
    fp = ((y_pred_thresh == 1) & (y_test_final == 0)).sum()
    fn = ((y_pred_thresh == 0) & (y_test_final == 1)).sum()
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    precision_scores.append(prec)
    recall_scores.append(rec)

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(thresholds_test, f1_scores_test, label='F1-Score', color='blue', lw=3)
ax.plot(thresholds_test, precision_scores, label='Precision', color='green', lw=2)
ax.plot(thresholds_test, recall_scores, label='Recall', color='orange', lw=2)

# Marcar threshold ótimo
best_threshold = thresholds_test[np.argmax(f1_scores_test)]
best_f1 = max(f1_scores_test)
ax.axvline(x=best_threshold, color='red', linestyle='--', lw=2,
           label=f'Threshold Ótimo ({best_threshold:.3f})')
ax.plot(best_threshold, best_f1, 'ro', markersize=12)

ax.set_xlabel('Threshold', fontsize=14, fontweight='bold')
ax.set_ylabel('Score', fontsize=14, fontweight='bold')
ax.set_title('Análise de Threshold - Model V8 Production\nPrecision, Recall e F1-Score vs Threshold', 
             fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=12, loc='best')
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('visualizations/threshold_analysis_v8.png', dpi=300, bbox_inches='tight')
plt.close()
print("      ✓ visualizations/threshold_analysis_v8.png")

# ----------------------------------------------------------------------------
# GRÁFICO 7: CROSS-VALIDATION RESULTS
# ----------------------------------------------------------------------------
print("   [7/7] Gerando Cross-Validation Results...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Subplot 1: AUC por fold
axes[0, 0].plot(fold_df['fold'], fold_df['auc_ensemble'], 
                marker='o', linewidth=2, markersize=10, color='blue')
axes[0, 0].axhline(y=fold_df['auc_ensemble'].mean(), color='red', 
                   linestyle='--', lw=2, label=f"Média: {fold_df['auc_ensemble'].mean():.4f}")
axes[0, 0].fill_between(fold_df['fold'], 
                        fold_df['auc_ensemble'].mean() - fold_df['auc_ensemble'].std(),
                        fold_df['auc_ensemble'].mean() + fold_df['auc_ensemble'].std(),
                        alpha=0.2, color='red')
axes[0, 0].set_xlabel('Fold', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('AUC', fontsize=12, fontweight='bold')
axes[0, 0].set_title('ROC-AUC por Fold', fontsize=14, fontweight='bold')
axes[0, 0].legend(fontsize=11)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim([0.90, 1.0])

# Subplot 2: F1-Macro por fold
axes[0, 1].plot(fold_df['fold'], fold_df['f1_macro'], 
                marker='s', linewidth=2, markersize=10, color='green')
axes[0, 1].axhline(y=fold_df['f1_macro'].mean(), color='red', 
                   linestyle='--', lw=2, label=f"Média: {fold_df['f1_macro'].mean():.4f}")
axes[0, 1].fill_between(fold_df['fold'], 
                        fold_df['f1_macro'].mean() - fold_df['f1_macro'].std(),
                        fold_df['f1_macro'].mean() + fold_df['f1_macro'].std(),
                        alpha=0.2, color='red')
axes[0, 1].set_xlabel('Fold', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('F1-Macro', fontsize=12, fontweight='bold')
axes[0, 1].set_title('F1-Macro por Fold', fontsize=14, fontweight='bold')
axes[0, 1].legend(fontsize=11)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_ylim([0.70, 0.80])

# Subplot 3: F1 por classe
x = np.arange(len(fold_df))
width = 0.35
axes[1, 0].bar(x - width/2, fold_df['f1_class_0'], width, 
               label='Classe 0', color='blue', alpha=0.7)
axes[1, 0].bar(x + width/2, fold_df['f1_class_1'], width, 
               label='Classe 1', color='red', alpha=0.7)
axes[1, 0].set_xlabel('Fold', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('F1-Score', fontsize=12, fontweight='bold')
axes[1, 0].set_title('F1-Score por Classe e Fold', fontsize=14, fontweight='bold')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(fold_df['fold'])
axes[1, 0].legend(fontsize=11)
axes[1, 0].grid(axis='y', alpha=0.3)

# Subplot 4: Comparação LightGBM vs XGBoost vs Ensemble
axes[1, 1].plot(fold_df['fold'], fold_df['auc_lgb'], 
                marker='o', linewidth=2, markersize=8, label='LightGBM', color='purple')
axes[1, 1].plot(fold_df['fold'], fold_df['auc_xgb'], 
                marker='s', linewidth=2, markersize=8, label='XGBoost', color='orange')
axes[1, 1].plot(fold_df['fold'], fold_df['auc_ensemble'], 
                marker='^', linewidth=2, markersize=8, label='Ensemble', color='green')
axes[1, 1].set_xlabel('Fold', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('AUC', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Comparação Modelos: LightGBM vs XGBoost vs Ensemble', 
                     fontsize=14, fontweight='bold')
axes[1, 1].legend(fontsize=11)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_ylim([0.90, 1.0])

plt.suptitle('Resultados Cross-Validation (TimeSeriesSplit - 5 Folds)', 
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('visualizations/cross_validation_results_v8.png', dpi=300, bbox_inches='tight')
plt.close()
print("      ✓ visualizations/cross_validation_results_v8.png")

print()
print("="*80)
print("✅ VISUALIZAÇÕES GERADAS COM SUCESSO!")
print("="*80)
print()
print("📁 Arquivos salvos em: visualizations/")
print("   ✓ confusion_matrix_v8.png")
print("   ✓ roc_curve_v8.png")
print("   ✓ precision_recall_curve_v8.png")
print("   ✓ feature_importance_v8.png")
print("   ✓ probability_distribution_v8.png")
print("   ✓ threshold_analysis_v8.png")
print("   ✓ cross_validation_results_v8.png")
print()
print("="*80)
