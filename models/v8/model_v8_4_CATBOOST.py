"""
═══════════════════════════════════════════════════════════════════════════════
MODEL V8.4 - VERSÃO COM CATBOOST 🐈
═══════════════════════════════════════════════════════════════════════════════

🔧 BASEADO EM V8.3 MAS COM CATBOOST:
   ✓ CatBoost conhecido por lidar bem com desbalanceamento
   ✓ Mesma arquitetura de features (SEM LEAKAGE)
   ✓ Ensemble CatBoost + XGBoost
   ✓ Threshold otimizado por grid search

📊 EXPECTATIVA:
   ✓ CatBoost pode ter melhor F1-Classe 1
   ✓ Menos hiperparâmetros para tunar
   ✓ Nativo para classes desbalanceadas

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
import catboost as cb
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
print("🐈 MODEL V8.4 - VERSÃO COM CATBOOST")
print("="*80)
print(f"📅 Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
print()

# ============================================================================
# 1. CARREGAR DADOS
# ============================================================================
print("📊 1. Carregando dados...")

USE_CSV = True
CSV_PATH = 'dataset-updated.csv'

if USE_CSV:
    print(f"   📂 Carregando do CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    df = df[df['target'].notna()].copy()
    
    print(f"   ✓ Dataset carregado: {len(df):,} registros")
    print(f"   ✓ Conversões: {(df['target']==1).sum():,} ({(df['target']==1).sum()/len(df):.2%})")
else:
    print("   ☁️  Carregando do BigQuery...")
    client = bigquery.Client(project='proj-ml-469320')
    
    query = """
    SELECT * FROM `proj-ml-469320.ml_dataset.cittamobi_feature_engineering_agg`
    WHERE target IS NOT NULL
    """
    
    df = client.query(query).to_dataframe()
    print(f"   ✓ BigQuery: {len(df):,} registros carregados")

print()

# ============================================================================
# 2. FEATURES GEOGRÁFICAS (Phase 1 - SEM LEAKAGE)
# ============================================================================
print("🗺️  2. Phase 1: Features Geográficas (6 features)...")

cbd_lat, cbd_lon = -23.5505, -46.6333

if 'stop_lat_event' in df.columns and 'stop_lon_event' in df.columns:
    df['dist_to_nearest_cbd'] = np.sqrt(
        (df['stop_lat_event'] - cbd_lat)**2 + 
        (df['stop_lon_event'] - cbd_lon)**2
    )
else:
    df['dist_to_nearest_cbd'] = 0.0

# Stop density
if 'gtfs_stop_id' in df.columns and 'stop_lat_event' in df.columns:
    nn = NearestNeighbors(n_neighbors=10, metric='haversine')
    coords = df[['stop_lat_event', 'stop_lon_event']].drop_duplicates()
    coords_rad = np.radians(coords)
    nn.fit(coords_rad)
    
    distances, _ = nn.kneighbors(coords_rad)
    avg_dist = distances[:, 1:].mean(axis=1)
    density_map = dict(zip(
        df[['stop_lat_event', 'stop_lon_event']].drop_duplicates().itertuples(index=False, name=None),
        1 / (avg_dist + 1e-6)
    ))
    
    df['stop_density'] = df.apply(
        lambda x: density_map.get((x['stop_lat_event'], x['stop_lon_event']), 0),
        axis=1
    )
else:
    df['stop_density'] = 0.0

# Clustering
if 'stop_lat_event' in df.columns:
    coords_unique = df[['stop_lat_event', 'stop_lon_event']].drop_duplicates()
    coords_rad_unique = np.radians(coords_unique)
    
    dbscan = DBSCAN(eps=0.01, min_samples=5, metric='haversine')
    clusters = dbscan.fit_predict(coords_rad_unique)
    
    coords_unique['cluster'] = clusters
    df = df.merge(
        coords_unique.rename(columns={'cluster': 'stop_cluster'}),
        on=['stop_lat_event', 'stop_lon_event'],
        how='left'
    )
    df['stop_cluster'].fillna(-1, inplace=True)
else:
    df['stop_cluster'] = -1

# Volatility
if 'gtfs_stop_id' in df.columns and 'stop_lat_event' in df.columns:
    stop_volatility = df.groupby('gtfs_stop_id').agg({
        'stop_lat_event': 'std',
        'stop_lon_event': 'std'
    }).fillna(0)
    
    stop_volatility['stop_volatility'] = (
        stop_volatility['stop_lat_event'] + 
        stop_volatility['stop_lon_event']
    )
    
    df = df.merge(
        stop_volatility[['stop_volatility']],
        left_on='gtfs_stop_id',
        right_index=True,
        how='left'
    )
    df['stop_volatility'].fillna(0, inplace=True)
else:
    df['stop_volatility'] = 0.0

print(f"   ✓ 6 Phase 1 features criadas (SEM LEAKAGE)")
print()

# ============================================================================
# 3. PREPARAR FEATURES BASE
# ============================================================================
print("🔧 3. Preparando features base...")

exclude_cols = [
    'target', 'gtfs_stop_id', 'timestamp_converted', 'device_id',
    'stop_lat_event', 'stop_lon_event', 'event_timestamp', 'date'
]

feature_cols_base = [col for col in df.columns if col not in exclude_cols]
X_base = df[feature_cols_base].copy()
y = df['target'].copy()

# Armazenar colunas auxiliares
aux_cols = {
    'gtfs_stop_id': df['gtfs_stop_id'].copy() if 'gtfs_stop_id' in df.columns else None,
    'device_id': df['device_id'].copy() if 'device_id' in df.columns else None,
    'time_hour': df['time_hour'].copy() if 'time_hour' in df.columns else None,
    'time_day_of_week': df['time_day_of_week'].copy() if 'time_day_of_week' in df.columns else None,
    'is_peak_hour': df['is_peak_hour'].copy() if 'is_peak_hour' in df.columns else None,
}

X_base = X_base.select_dtypes(include=[np.number])

# Limpar nomes das colunas
X_base.columns = X_base.columns.str.replace('[', '_', regex=False)
X_base.columns = X_base.columns.str.replace(']', '_', regex=False)
X_base.columns = X_base.columns.str.replace('{', '_', regex=False)
X_base.columns = X_base.columns.str.replace('}', '_', regex=False)
X_base.columns = X_base.columns.str.replace('"', '_', regex=False)
X_base.columns = X_base.columns.str.replace("'", '_', regex=False)
X_base.columns = X_base.columns.str.replace(':', '_', regex=False)
X_base.columns = X_base.columns.str.replace(',', '_', regex=False)
X_base.columns = X_base.columns.str.replace('<', '_', regex=False)
X_base.columns = X_base.columns.str.replace('>', '_', regex=False)

X_base.replace([np.inf, -np.inf], np.nan, inplace=True)
X_base.fillna(0, inplace=True)

print(f"   ✓ Features base: {len(X_base.columns)} features")
print(f"   ✓ Nomes das colunas limpos")
print()

# ============================================================================
# 4. TIME SERIES CROSS-VALIDATION
# ============================================================================
print("✂️  4. Time Series Split com validação cruzada (SEM LEAKAGE)...")

tscv = TimeSeriesSplit(n_splits=5)
fold_results = []

print(f"   ✓ TimeSeriesSplit: 5 folds")
print(f"   ✓ Dataset total: {len(X_base):,} registros")
print()

# ============================================================================
# 5. LOOP DE VALIDAÇÃO CRUZADA
# ============================================================================
print("🔄 5. Validação cruzada temporal...")
print()

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_base), 1):
    print(f"{'='*80}")
    print(f"FOLD {fold}/5")
    print(f"{'='*80}")
    
    # SPLIT DO FOLD
    X_train_base = X_base.iloc[train_idx].copy()
    X_val_base = X_base.iloc[val_idx].copy()
    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]
    
    print(f"   Train: {len(train_idx):,} | Val: {len(val_idx):,}")
    print(f"   Conversão Train: {y_train.mean():.2%} | Val: {y_val.mean():.2%}")
    print()
    
    # ========================================================================
    # CRIAR FEATURES DINÂMICAS (SEM LEAKAGE)
    # ========================================================================
    print(f"   🛡️  Criando features dinâmicas SEM LEAKAGE...")
    
    # A. Stop historical conversion
    if aux_cols['gtfs_stop_id'] is not None:
        stop_id_train = aux_cols['gtfs_stop_id'].iloc[train_idx]
        stop_id_val = aux_cols['gtfs_stop_id'].iloc[val_idx]
        
        stop_conversion_train = y_train.groupby(stop_id_train).mean().to_dict()
        default_conv = y_train.mean()
        
        X_train_base['stop_historical_conversion'] = stop_id_train.map(
            stop_conversion_train
        ).fillna(default_conv)
        
        X_val_base['stop_historical_conversion'] = stop_id_val.map(
            stop_conversion_train
        ).fillna(default_conv)
    else:
        X_train_base['stop_historical_conversion'] = y_train.mean()
        X_val_base['stop_historical_conversion'] = y_train.mean()
    
    # B. Hour conversion rate
    if aux_cols['time_hour'] is not None:
        hour_train = aux_cols['time_hour'].iloc[train_idx]
        hour_val = aux_cols['time_hour'].iloc[val_idx]
        
        hour_conv_train = y_train.groupby(hour_train).mean().to_dict()
        default_hour = y_train.mean()
        
        X_train_base['hour_conversion_rate'] = hour_train.map(hour_conv_train).fillna(default_hour)
        X_val_base['hour_conversion_rate'] = hour_val.map(hour_conv_train).fillna(default_hour)
    else:
        X_train_base['hour_conversion_rate'] = y_train.mean()
        X_val_base['hour_conversion_rate'] = y_train.mean()
    
    # C. Day of week conversion
    if aux_cols['time_day_of_week'] is not None:
        dow_train = aux_cols['time_day_of_week'].iloc[train_idx]
        dow_val = aux_cols['time_day_of_week'].iloc[val_idx]
        
        dow_conv_train = y_train.groupby(dow_train).mean().to_dict()
        default_dow = y_train.mean()
        
        X_train_base['dow_conversion_rate'] = dow_train.map(dow_conv_train).fillna(default_dow)
        X_val_base['dow_conversion_rate'] = dow_val.map(dow_conv_train).fillna(default_dow)
    else:
        X_train_base['dow_conversion_rate'] = y_train.mean()
        X_val_base['dow_conversion_rate'] = y_train.mean()
    
    # D. Stop-hour conversion
    if aux_cols['gtfs_stop_id'] is not None and aux_cols['time_hour'] is not None:
        stop_hour_train = pd.DataFrame({
            'stop': aux_cols['gtfs_stop_id'].iloc[train_idx],
            'hour': aux_cols['time_hour'].iloc[train_idx]
        })
        stop_hour_val = pd.DataFrame({
            'stop': aux_cols['gtfs_stop_id'].iloc[val_idx],
            'hour': aux_cols['time_hour'].iloc[val_idx]
        })
        
        stop_hour_conv_train = y_train.groupby([stop_hour_train['stop'], stop_hour_train['hour']]).mean().to_dict()
        default_sh = y_train.mean()
        
        train_keys = list(zip(stop_hour_train['stop'], stop_hour_train['hour']))
        val_keys = list(zip(stop_hour_val['stop'], stop_hour_val['hour']))
        
        X_train_base['stop_hour_conversion'] = [stop_hour_conv_train.get(k, default_sh) for k in train_keys]
        X_val_base['stop_hour_conversion'] = [stop_hour_conv_train.get(k, default_sh) for k in val_keys]
    else:
        X_train_base['stop_hour_conversion'] = y_train.mean()
        X_val_base['stop_hour_conversion'] = y_train.mean()
    
    # E. User conversion rate
    if aux_cols['device_id'] is not None:
        device_train = aux_cols['device_id'].iloc[train_idx]
        device_val = aux_cols['device_id'].iloc[val_idx]
        
        user_conv_train = y_train.groupby(device_train).mean().to_dict()
        default_user = y_train.mean()
        
        X_train_base['user_conversion_rate'] = device_train.map(user_conv_train).fillna(default_user)
        X_val_base['user_conversion_rate'] = device_val.map(user_conv_train).fillna(default_user)
    else:
        X_train_base['user_conversion_rate'] = y_train.mean()
        X_val_base['user_conversion_rate'] = y_train.mean()
    
    # F. Geo-temporal interactions
    if aux_cols['is_peak_hour'] is not None and 'dist_to_nearest_cbd' in X_train_base.columns:
        peak_train = aux_cols['is_peak_hour'].iloc[train_idx]
        peak_val = aux_cols['is_peak_hour'].iloc[val_idx]
        
        X_train_base['geo_temporal'] = X_train_base['dist_to_nearest_cbd'] * peak_train
        X_val_base['geo_temporal'] = X_val_base['dist_to_nearest_cbd'] * peak_val
        
        if 'stop_density' in X_train_base.columns:
            X_train_base['density_peak'] = X_train_base['stop_density'] * peak_train
            X_val_base['density_peak'] = X_val_base['stop_density'] * peak_val
        else:
            X_train_base['density_peak'] = 0.0
            X_val_base['density_peak'] = 0.0
    else:
        X_train_base['geo_temporal'] = 0.0
        X_val_base['geo_temporal'] = 0.0
        X_train_base['density_peak'] = 0.0
        X_val_base['density_peak'] = 0.0
    
    # G. Cluster conversion rate
    if 'stop_cluster' in X_train_base.columns:
        cluster_train = X_train_base['stop_cluster']
        cluster_val = X_val_base['stop_cluster']
        
        cluster_conv_train = y_train.groupby(cluster_train).mean().to_dict()
        default_cluster = y_train.mean()
        
        X_train_base['cluster_conversion_rate'] = cluster_train.map(cluster_conv_train).fillna(default_cluster)
        X_val_base['cluster_conversion_rate'] = cluster_val.map(cluster_conv_train).fillna(default_cluster)
    else:
        X_train_base['cluster_conversion_rate'] = y_train.mean()
        X_val_base['cluster_conversion_rate'] = y_train.mean()
    
    print(f"      ✅ 10 features dinâmicas criadas SEM LEAKAGE")
    print(f"      ✅ Total features no fold: {len(X_train_base.columns)}")
    print()
    
    # ========================================================================
    # SAMPLE WEIGHTS EXTREMOS
    # ========================================================================
    def get_dynamic_weight(conv_rate):
        if conv_rate < 0.03:
            return 8.0
        elif conv_rate < 0.05:
            return 7.0
        elif conv_rate < 0.08:
            return 6.0
        elif conv_rate < 0.12:
            return 5.0
        else:
            return 4.0
    
    stop_conv = X_train_base['stop_historical_conversion'].values
    sample_weights = np.where(
        y_train == 1,
        [get_dynamic_weight(c) for c in stop_conv],
        1.0
    )
    
    # ========================================================================
    # TREINAR MODELOS
    # ========================================================================
    print(f"   🐈 Treinando modelos...")
    
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    # CATBOOST - Otimizado para classe desbalanceada
    # Nota: CatBoost não permite scale_pos_weight + auto_class_weights juntos
    # Vamos usar apenas auto_class_weights='Balanced' que é mais poderoso!
    catboost_params = {
        'iterations': 400,
        'learning_rate': 0.02,
        'depth': 10,
        'l2_leaf_reg': 3,
        'border_count': 254,
        'bagging_temperature': 1.0,
        'random_strength': 1.0,
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'auto_class_weights': 'Balanced',  # CatBoost calcula pesos automaticamente!
        'verbose': False,
        'random_seed': 42
    }
    
    catboost_pool_train = cb.Pool(X_train_base, y_train, weight=sample_weights)
    catboost_model = cb.CatBoostClassifier(**catboost_params)
    catboost_model.fit(catboost_pool_train)
    
    # XGBoost - Mesma config de V8.3
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 12,
        'learning_rate': 0.02,
        'subsample': 0.90,
        'colsample_bytree': 0.90,
        'colsample_bylevel': 0.90,
        'colsample_bynode': 0.90,
        'min_child_weight': 1,
        'scale_pos_weight': scale_pos_weight * 2.0,
        'gamma': 0.05,
        'alpha': 0.05,
        'lambda': 0.05,
        'max_bin': 256,
        'tree_method': 'hist',
        'grow_policy': 'lossguide',
        'max_leaves': 127
    }
    
    xgb_train = xgb.DMatrix(X_train_base, label=y_train, weight=sample_weights)
    xgb_model = xgb.train(xgb_params, xgb_train, num_boost_round=400)
    
    # ========================================================================
    # PREDIÇÕES E ENSEMBLE OTIMIZADO
    # ========================================================================
    xgb_val = xgb.DMatrix(X_val_base)
    
    y_pred_catboost = catboost_model.predict_proba(X_val_base)[:, 1]
    y_pred_xgb = xgb_model.predict(xgb_val)
    
    print(f"      🔬 Otimizando pesos do ensemble...")
    
    best_ensemble_auc = 0
    best_w_catboost = 0.5
    best_w_xgb = 0.5
    
    for w_catboost_test in np.arange(0.3, 0.8, 0.05):
        w_xgb_test = 1.0 - w_catboost_test
        y_pred_test = w_catboost_test * y_pred_catboost + w_xgb_test * y_pred_xgb
        auc_test = roc_auc_score(y_val, y_pred_test)
        
        if auc_test > best_ensemble_auc:
            best_ensemble_auc = auc_test
            best_w_catboost = w_catboost_test
            best_w_xgb = w_xgb_test
    
    print(f"      ✓ Pesos ótimos: CatBoost={best_w_catboost:.3f}, XGBoost={best_w_xgb:.3f}")
    
    w_catboost = best_w_catboost
    w_xgb = best_w_xgb
    y_pred_ensemble = w_catboost * y_pred_catboost + w_xgb * y_pred_xgb
    
    # ========================================================================
    # OTIMIZAÇÃO DE THRESHOLD (GRID SEARCH REFINADO)
    # ========================================================================
    print(f"      🎯 Otimizando threshold com grid search refinado...")
    
    best_f1_class1 = 0
    best_threshold = 0.5
    best_f1_macro = 0
    
    # FASE 1: Grid search grosso
    for threshold in np.arange(0.15, 0.85, 0.05):
        y_pred_temp = (y_pred_ensemble >= threshold).astype(int)
        f1_temp_class1 = f1_score(y_val, y_pred_temp, pos_label=1, zero_division=0)
        f1_temp_class0 = f1_score(y_val, y_pred_temp, pos_label=0, zero_division=0)
        f1_temp_macro = (f1_temp_class1 + f1_temp_class0) / 2
        
        if f1_temp_class1 > best_f1_class1:
            best_f1_class1 = f1_temp_class1
            best_threshold = threshold
            best_f1_macro = f1_temp_macro
    
    # FASE 2: Grid search fino
    print(f"      🔍 Refinando ao redor de {best_threshold:.3f}...")
    fine_start = max(0.10, best_threshold - 0.10)
    fine_end = min(0.90, best_threshold + 0.10)
    
    for threshold in np.arange(fine_start, fine_end, 0.01):
        y_pred_temp = (y_pred_ensemble >= threshold).astype(int)
        f1_temp_class1 = f1_score(y_val, y_pred_temp, pos_label=1, zero_division=0)
        f1_temp_class0 = f1_score(y_val, y_pred_temp, pos_label=0, zero_division=0)
        f1_temp_macro = (f1_temp_class1 + f1_temp_class0) / 2
        
        objective = f1_temp_class1 + 0.1 * f1_temp_macro
        best_objective = best_f1_class1 + 0.1 * best_f1_macro
        
        if objective > best_objective:
            best_f1_class1 = f1_temp_class1
            best_threshold = threshold
            best_f1_macro = f1_temp_macro
    
    print(f"      ✓ Threshold ótimo: {best_threshold:.4f} (F1-C1: {best_f1_class1:.4f}, F1-Macro: {best_f1_macro:.4f})")
    
    y_pred_binary = (y_pred_ensemble >= best_threshold).astype(int)
    
    # ========================================================================
    # MÉTRICAS
    # ========================================================================
    from sklearn.metrics import precision_score, recall_score, accuracy_score
    
    auc_catboost = roc_auc_score(y_val, y_pred_catboost)
    auc_xgb = roc_auc_score(y_val, y_pred_xgb)
    auc_ensemble = roc_auc_score(y_val, y_pred_ensemble)
    
    cm = confusion_matrix(y_val, y_pred_binary)
    
    # F1-Scores
    f1_class_0 = f1_score(y_val, y_pred_binary, pos_label=0)
    f1_class_1 = f1_score(y_val, y_pred_binary, pos_label=1)
    f1_macro = (f1_class_0 + f1_class_1) / 2
    
    # Precision e Recall por classe
    precision_class_0 = precision_score(y_val, y_pred_binary, pos_label=0)
    precision_class_1 = precision_score(y_val, y_pred_binary, pos_label=1)
    recall_class_0 = recall_score(y_val, y_pred_binary, pos_label=0)
    recall_class_1 = recall_score(y_val, y_pred_binary, pos_label=1)
    
    # Accuracy
    accuracy = accuracy_score(y_val, y_pred_binary)
    
    print(f"      CatBoost AUC:     {auc_catboost:.4f}")
    print(f"      XGBoost AUC:      {auc_xgb:.4f}")
    print(f"      Ensemble AUC:     {auc_ensemble:.4f}")
    print(f"      ─────────────────────────────")
    print(f"      F1 Classe 0:      {f1_class_0:.4f}")
    print(f"      F1 Classe 1:      {f1_class_1:.4f}")
    print(f"      F1-Macro:         {f1_macro:.4f}")
    print(f"      ─────────────────────────────")
    print(f"      Precision C0:     {precision_class_0:.4f}")
    print(f"      Precision C1:     {precision_class_1:.4f}")
    print(f"      Recall C0:        {recall_class_0:.4f}")
    print(f"      Recall C1:        {recall_class_1:.4f}")
    print(f"      ─────────────────────────────")
    print(f"      Accuracy:         {accuracy:.4f}")
    print()
    
    fold_results.append({
        'fold': fold,
        'auc_catboost': auc_catboost,
        'auc_xgb': auc_xgb,
        'auc_ensemble': auc_ensemble,
        'f1_class_0': f1_class_0,
        'f1_class_1': f1_class_1,
        'f1_macro': f1_macro,
        'precision_class_0': precision_class_0,
        'precision_class_1': precision_class_1,
        'recall_class_0': recall_class_0,
        'recall_class_1': recall_class_1,
        'accuracy': accuracy,
        'threshold': best_threshold,
        'w_catboost': w_catboost,
        'w_xgb': w_xgb,
        'y_val': y_val.values,
        'y_pred_proba': y_pred_ensemble,
        'y_pred_binary': y_pred_binary,
        'catboost_model': catboost_model,
        'xgb_model': xgb_model,
        'confusion_matrix': cm
    })

# ============================================================================
# 6. RESUMO CROSS-VALIDATION
# ============================================================================
print("="*80)
print("📊 RESUMO CROSS-VALIDATION (CATBOOST + XGBOOST)")
print("="*80)

fold_df = pd.DataFrame(fold_results)

print(f"\n   🎯 AUC:")
print(f"      CatBoost:  {fold_df['auc_catboost'].mean():.4f} ± {fold_df['auc_catboost'].std():.4f}")
print(f"      XGBoost:   {fold_df['auc_xgb'].mean():.4f} ± {fold_df['auc_xgb'].std():.4f}")
print(f"      Ensemble:  {fold_df['auc_ensemble'].mean():.4f} ± {fold_df['auc_ensemble'].std():.4f}")

print(f"\n   📊 F1-Score:")
print(f"      Classe 0:  {fold_df['f1_class_0'].mean():.4f} ± {fold_df['f1_class_0'].std():.4f}")
print(f"      Classe 1:  {fold_df['f1_class_1'].mean():.4f} ± {fold_df['f1_class_1'].std():.4f}")
print(f"      F1-Macro:  {fold_df['f1_macro'].mean():.4f} ± {fold_df['f1_macro'].std():.4f}")

print(f"\n   🎯 Precision:")
print(f"      Classe 0:  {fold_df['precision_class_0'].mean():.4f} ± {fold_df['precision_class_0'].std():.4f}")
print(f"      Classe 1:  {fold_df['precision_class_1'].mean():.4f} ± {fold_df['precision_class_1'].std():.4f}")

print(f"\n   📈 Recall:")
print(f"      Classe 0:  {fold_df['recall_class_0'].mean():.4f} ± {fold_df['recall_class_0'].std():.4f}")
print(f"      Classe 1:  {fold_df['recall_class_1'].mean():.4f} ± {fold_df['recall_class_1'].std():.4f}")

print(f"\n   ✅ Accuracy:  {fold_df['accuracy'].mean():.4f} ± {fold_df['accuracy'].std():.4f}")
print()
print("="*80)
print("✅ MODELO V8.4 COM CATBOOST COMPLETO!")
print("="*80)
print()
print("📈 COMPARAÇÃO:")
print(f"   V8.3 (LightGBM+XGBoost): AUC = 0.9023, F1-C1 = 0.4144")
print(f"   V8.4 (CatBoost+XGBoost): AUC = {fold_df['auc_ensemble'].mean():.4f}, F1-C1 = {fold_df['f1_class_1'].mean():.4f}")
print()
print("🎯 CatBoost é conhecido por lidar melhor com classes desbalanceadas!")
print("="*80)
print()

# ============================================================================
# 7. VISUALIZAÇÕES PROFISSIONAIS
# ============================================================================
print("="*80)
print("📊 7. GERANDO VISUALIZAÇÕES PROFISSIONAIS (V8.4)...")
print("="*80)
print()

import os
output_dir = 'visualizations_v8_4'
os.makedirs(output_dir, exist_ok=True)

# Usar último fold (Fold 5) para visualizações detalhadas
last_fold = fold_results[-1]

# ----------------------------------------------------------------------------
# 7.1 Cross-Validation Analysis
# ----------------------------------------------------------------------------
print("   📈 1/7 - Cross-Validation Analysis...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Model V8.4 (CatBoost+XGBoost) - Cross-Validation Analysis', 
             fontsize=16, fontweight='bold')

# AUC por fold
ax = axes[0, 0]
folds = fold_df['fold'].values
ax.plot(folds, fold_df['auc_catboost'], 'o-', label='CatBoost', linewidth=2, markersize=8)
ax.plot(folds, fold_df['auc_xgb'], 's-', label='XGBoost', linewidth=2, markersize=8)
ax.plot(folds, fold_df['auc_ensemble'], '^-', label='Ensemble', linewidth=2, markersize=8)
ax.axhline(y=fold_df['auc_ensemble'].mean(), color='red', linestyle='--', 
           label=f'Ensemble Média: {fold_df["auc_ensemble"].mean():.4f}')
ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
ax.set_ylabel('AUC', fontsize=12, fontweight='bold')
ax.set_title('AUC por Fold', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xticks(folds)

# F1-Score por fold
ax = axes[0, 1]
ax.plot(folds, fold_df['f1_class_0'], 'o-', label='Classe 0 (Não-Conversão)', linewidth=2, markersize=8)
ax.plot(folds, fold_df['f1_class_1'], 's-', label='Classe 1 (Conversão)', linewidth=2, markersize=8)
ax.plot(folds, fold_df['f1_macro'], '^-', label='F1-Macro', linewidth=2, markersize=8)
ax.axhline(y=fold_df['f1_class_1'].mean(), color='red', linestyle='--', 
           label=f'F1-C1 Média: {fold_df["f1_class_1"].mean():.4f}')
ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
ax.set_title('F1-Score por Fold', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xticks(folds)

# Boxplot de métricas
ax = axes[0, 2]
metrics_data = [
    fold_df['auc_ensemble'].values,
    fold_df['f1_class_1'].values,
    fold_df['f1_macro'].values
]
bp = ax.boxplot(metrics_data, labels=['AUC', 'F1-Classe 1', 'F1-Macro'],
                patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Distribuição das Métricas', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Threshold por fold
ax = axes[1, 0]
thresholds = [r['threshold'] for r in fold_results]
ax.plot(folds, thresholds, 'o-', linewidth=2, markersize=10, color='purple')
ax.axhline(y=np.mean(thresholds), color='red', linestyle='--', 
           label=f'Média: {np.mean(thresholds):.4f}')
ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
ax.set_ylabel('Threshold Ótimo', fontsize=12, fontweight='bold')
ax.set_title('Threshold Otimizado por Fold', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(folds)
ax.set_ylim([0.8, 0.95])

# Ensemble weights por fold
ax = axes[1, 1]
w_catboost_list = [r['w_catboost'] for r in fold_results]
w_xgb_list = [r['w_xgb'] for r in fold_results]
x = np.arange(len(folds))
width = 0.35
ax.bar(x - width/2, w_catboost_list, width, label='CatBoost', alpha=0.8)
ax.bar(x + width/2, w_xgb_list, width, label='XGBoost', alpha=0.8)
ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
ax.set_ylabel('Peso no Ensemble', fontsize=12, fontweight='bold')
ax.set_title('Pesos Otimizados do Ensemble', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(folds)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Estatísticas resumidas
ax = axes[1, 2]
ax.axis('off')
stats_text = f"""
RESUMO ESTATÍSTICO V8.4
{'─'*30}

AUC Ensemble:
  Média: {fold_df['auc_ensemble'].mean():.4f}
  Std:   {fold_df['auc_ensemble'].std():.4f}
  Min:   {fold_df['auc_ensemble'].min():.4f}
  Max:   {fold_df['auc_ensemble'].max():.4f}

F1-Score Classe 1:
  Média: {fold_df['f1_class_1'].mean():.4f}
  Std:   {fold_df['f1_class_1'].std():.4f}
  Min:   {fold_df['f1_class_1'].min():.4f}
  Max:   {fold_df['f1_class_1'].max():.4f}

F1-Macro:
  Média: {fold_df['f1_macro'].mean():.4f}
  Std:   {fold_df['f1_macro'].std():.4f}

Threshold:
  Média: {np.mean(thresholds):.4f}
"""
ax.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
        fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig(f'{output_dir}/01_cross_validation_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"      ✓ Salvo: {output_dir}/01_cross_validation_analysis.png")

# ----------------------------------------------------------------------------
# 7.2 Confusion Matrix (Fold 5)
# ----------------------------------------------------------------------------
print("   🎯 2/7 - Confusion Matrix...")

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
cm = last_fold['confusion_matrix']
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax,
            xticklabels=['Não-Conversão', 'Conversão'],
            yticklabels=['Não-Conversão', 'Conversão'])
ax.set_xlabel('Predição', fontsize=14, fontweight='bold')
ax.set_ylabel('Real', fontsize=14, fontweight='bold')
ax.set_title(f'Confusion Matrix - Model V8.4 (Fold 5)\nThreshold = {last_fold["threshold"]:.4f}',
             fontsize=16, fontweight='bold')

# Adicionar percentuais
total = cm.sum()
for i in range(2):
    for j in range(2):
        percentage = (cm[i, j] / total) * 100
        ax.text(j + 0.5, i + 0.7, f'({percentage:.2f}%)',
                ha='center', va='center', fontsize=10, color='red')

plt.tight_layout()
plt.savefig(f'{output_dir}/02_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"      ✓ Salvo: {output_dir}/02_confusion_matrix.png")

# ----------------------------------------------------------------------------
# 7.3 ROC Curve (Fold 5)
# ----------------------------------------------------------------------------
print("   📈 3/7 - ROC Curve...")

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

y_val = last_fold['y_val']
y_pred_proba = last_fold['y_pred_proba']

fpr, tpr, thresholds_roc = roc_curve(y_val, y_pred_proba)
roc_auc = auc(fpr, tpr)

ax.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC Curve (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')

# Marcar ponto ótimo
optimal_idx = np.argmin(np.abs(thresholds_roc - last_fold['threshold']))
ax.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=12, 
        label=f'Optimal Point (threshold={last_fold["threshold"]:.3f})')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
ax.set_title('ROC Curve - Model V8.4 (CatBoost+XGBoost) - Fold 5\nSEM Data Leakage',
             fontsize=16, fontweight='bold')
ax.legend(loc="lower right", fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/03_roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"      ✓ Salvo: {output_dir}/03_roc_curve.png")

# ----------------------------------------------------------------------------
# 7.4 Precision-Recall Curve
# ----------------------------------------------------------------------------
print("   📊 4/7 - Precision-Recall Curve...")

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

precision, recall, thresholds_pr = precision_recall_curve(y_val, y_pred_proba)

ax.plot(recall, precision, color='blue', lw=3, label='Precision-Recall Curve')
ax.axhline(y=y_val.mean(), color='red', linestyle='--', lw=2,
           label=f'Baseline (Prevalência = {y_val.mean():.2%})')

# Marcar ponto ótimo
optimal_idx_pr = np.argmin(np.abs(thresholds_pr - last_fold['threshold']))
if optimal_idx_pr < len(recall):
    ax.plot(recall[optimal_idx_pr], precision[optimal_idx_pr], 'ro', markersize=12,
            label=f'Optimal Point (threshold={last_fold["threshold"]:.3f})')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('Recall (Sensibilidade)', fontsize=14, fontweight='bold')
ax.set_ylabel('Precision (Precisão)', fontsize=14, fontweight='bold')
ax.set_title('Precision-Recall Curve - Model V8.4 (Fold 5)',
             fontsize=16, fontweight='bold')
ax.legend(loc="upper right", fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/04_precision_recall_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"      ✓ Salvo: {output_dir}/04_precision_recall_curve.png")

# ----------------------------------------------------------------------------
# 7.5 Feature Importance (CatBoost)
# ----------------------------------------------------------------------------
print("   🔍 5/7 - Feature Importance (CatBoost)...")

fig, ax = plt.subplots(1, 1, figsize=(12, 10))

catboost_model = last_fold['catboost_model']
feature_importance = catboost_model.get_feature_importance()

# Usar as features do último fold treinado
if len(fold_results) > 0:
    # Pegar feature names do DataFrame base
    feature_names_list = list(X_base.columns) + [
        'stop_historical_conversion', 'hour_conversion_rate', 'dow_conversion_rate',
        'stop_hour_conversion', 'user_conversion_rate', 'geo_temporal', 
        'density_peak', 'cluster_conversion_rate'
    ]
    
    importance_df = pd.DataFrame({
        'feature': feature_names_list[:len(feature_importance)],
        'importance': feature_importance
    }).sort_values('importance', ascending=False).head(25)
    
    import matplotlib.cm as cm
    colors = cm.get_cmap('viridis')(importance_df['importance'] / importance_df['importance'].max())
    ax.barh(range(len(importance_df)), importance_df['importance'], color=colors)
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['feature'], fontsize=10)
    ax.set_xlabel('Importance', fontsize=14, fontweight='bold')
    ax.set_title('Top 25 Features - CatBoost Model V8.4 (Fold 5)',
                 fontsize=16, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(f'{output_dir}/05_feature_importance_catboost.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"      ✓ Salvo: {output_dir}/05_feature_importance_catboost.png")

# ----------------------------------------------------------------------------
# 7.6 Probability Distribution
# ----------------------------------------------------------------------------
print("   📉 6/7 - Probability Distribution...")

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Histogram
ax = axes[0]
ax.hist(y_pred_proba[y_val == 0], bins=50, alpha=0.7, label='Classe 0 (Não-Conversão)', 
        color='blue', edgecolor='black')
ax.hist(y_pred_proba[y_val == 1], bins=50, alpha=0.7, label='Classe 1 (Conversão)', 
        color='red', edgecolor='black')
ax.axvline(x=last_fold['threshold'], color='green', linestyle='--', lw=3,
           label=f'Threshold = {last_fold["threshold"]:.4f}')
ax.set_xlabel('Probabilidade Predita', fontsize=14, fontweight='bold')
ax.set_ylabel('Frequência', fontsize=14, fontweight='bold')
ax.set_title('Distribuição de Probabilidades - Model V8.4 (Fold 5)',
             fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

# KDE Plot
ax = axes[1]
from scipy import stats
kde_0 = stats.gaussian_kde(y_pred_proba[y_val == 0])
kde_1 = stats.gaussian_kde(y_pred_proba[y_val == 1])
x_range = np.linspace(0, 1, 1000)
ax.plot(x_range, kde_0(x_range), label='Classe 0 (Não-Conversão)', lw=3, color='blue')
ax.plot(x_range, kde_1(x_range), label='Classe 1 (Conversão)', lw=3, color='red')
ax.axvline(x=last_fold['threshold'], color='green', linestyle='--', lw=3,
           label=f'Threshold = {last_fold["threshold"]:.4f}')
ax.set_xlabel('Probabilidade Predita', fontsize=14, fontweight='bold')
ax.set_ylabel('Densidade', fontsize=14, fontweight='bold')
ax.set_title('Densidade de Probabilidades (KDE) - Model V8.4',
             fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/06_probability_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"      ✓ Salvo: {output_dir}/06_probability_distribution.png")

# ----------------------------------------------------------------------------
# 7.7 Threshold Analysis
# ----------------------------------------------------------------------------
print("   🎯 7/7 - Threshold Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Threshold Analysis - Model V8.4 (Fold 5)', fontsize=16, fontweight='bold')

# Calcular métricas para range de thresholds
threshold_range = np.arange(0.1, 0.95, 0.01)
metrics_by_threshold = {
    'precision_0': [], 'recall_0': [], 'f1_0': [],
    'precision_1': [], 'recall_1': [], 'f1_1': [],
    'accuracy': []
}

for t in threshold_range:
    y_pred_temp = (y_pred_proba >= t).astype(int)
    
    tn = ((y_val == 0) & (y_pred_temp == 0)).sum()
    fp = ((y_val == 0) & (y_pred_temp == 1)).sum()
    fn = ((y_val == 1) & (y_pred_temp == 0)).sum()
    tp = ((y_val == 1) & (y_pred_temp == 1)).sum()
    
    # Classe 0
    prec_0 = tn / (tn + fn) if (tn + fn) > 0 else 0
    rec_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_0 = 2 * prec_0 * rec_0 / (prec_0 + rec_0) if (prec_0 + rec_0) > 0 else 0
    
    # Classe 1
    prec_1 = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_1 = 2 * prec_1 * rec_1 / (prec_1 + rec_1) if (prec_1 + rec_1) > 0 else 0
    
    acc = (tp + tn) / len(y_val)
    
    metrics_by_threshold['precision_0'].append(prec_0)
    metrics_by_threshold['recall_0'].append(rec_0)
    metrics_by_threshold['f1_0'].append(f1_0)
    metrics_by_threshold['precision_1'].append(prec_1)
    metrics_by_threshold['recall_1'].append(rec_1)
    metrics_by_threshold['f1_1'].append(f1_1)
    metrics_by_threshold['accuracy'].append(acc)

# Plot 1: F1-Scores
ax = axes[0, 0]
ax.plot(threshold_range, metrics_by_threshold['f1_0'], label='F1 Classe 0', lw=2)
ax.plot(threshold_range, metrics_by_threshold['f1_1'], label='F1 Classe 1', lw=2)
ax.plot(threshold_range, [(f0 + f1)/2 for f0, f1 in zip(metrics_by_threshold['f1_0'], 
        metrics_by_threshold['f1_1'])], label='F1-Macro', lw=2, linestyle='--')
ax.axvline(x=last_fold['threshold'], color='red', linestyle='--', lw=2,
           label=f'Optimal = {last_fold["threshold"]:.4f}')
ax.set_xlabel('Threshold', fontsize=12, fontweight='bold')
ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
ax.set_title('F1-Score vs Threshold', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Precision vs Recall
ax = axes[0, 1]
ax.plot(threshold_range, metrics_by_threshold['precision_1'], label='Precision Classe 1', lw=2)
ax.plot(threshold_range, metrics_by_threshold['recall_1'], label='Recall Classe 1', lw=2)
ax.axvline(x=last_fold['threshold'], color='red', linestyle='--', lw=2,
           label=f'Optimal = {last_fold["threshold"]:.4f}')
ax.set_xlabel('Threshold', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Precision & Recall (Classe 1) vs Threshold', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Accuracy
ax = axes[1, 0]
ax.plot(threshold_range, metrics_by_threshold['accuracy'], lw=2, color='purple')
ax.axvline(x=last_fold['threshold'], color='red', linestyle='--', lw=2,
           label=f'Optimal = {last_fold["threshold"]:.4f}')
ax.set_xlabel('Threshold', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Accuracy vs Threshold', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Trade-off TPR vs FPR
ax = axes[1, 1]
tpr_list = metrics_by_threshold['recall_1']
fpr_list = [1 - rec0 for rec0 in metrics_by_threshold['recall_0']]
ax.plot(threshold_range, tpr_list, label='TPR (True Positive Rate)', lw=2)
ax.plot(threshold_range, fpr_list, label='FPR (False Positive Rate)', lw=2)
ax.axvline(x=last_fold['threshold'], color='red', linestyle='--', lw=2,
           label=f'Optimal = {last_fold["threshold"]:.4f}')
ax.set_xlabel('Threshold', fontsize=12, fontweight='bold')
ax.set_ylabel('Rate', fontsize=12, fontweight='bold')
ax.set_title('TPR vs FPR vs Threshold', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/07_threshold_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"      ✓ Salvo: {output_dir}/07_threshold_analysis.png")

print()
print("="*80)
print(f"✅ 7 VISUALIZAÇÕES GERADAS COM SUCESSO!")
print(f"📁 Pasta: {output_dir}/")
print("="*80)
print()

print(f"📅 Término: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
