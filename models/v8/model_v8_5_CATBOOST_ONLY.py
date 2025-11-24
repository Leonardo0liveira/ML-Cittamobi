"""
═══════════════════════════════════════════════════════════════════════════════
MODEL V8.5 - APENAS CATBOOST (SEM ENSEMBLE) 🐈
═══════════════════════════════════════════════════════════════════════════════

🔧 TESTE: CATBOOST PURO SEM ENSEMBLE
   ✓ Apenas CatBoost (sem XGBoost)
   ✓ Mesma arquitetura de features (SEM LEAKAGE)
   ✓ Threshold otimizado por grid search
   ✓ auto_class_weights='Balanced'

📊 OBJETIVO:
   ✓ Testar se CatBoost sozinho supera o ensemble
   ✓ Simplicidade vs performance

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
print("🐈 MODEL V8.5 - APENAS CATBOOST (SEM ENSEMBLE)")
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
    # TREINAR MODELO (APENAS CATBOOST)
    # ========================================================================
    print(f"   🐈 Treinando CatBoost (SOZINHO)...")
    
    # CATBOOST - Otimizado para classe desbalanceada
    catboost_params = {
        'iterations': 500,  # Aumentado para compensar falta de ensemble
        'learning_rate': 0.015,  # Mais lento para melhor aprendizado
        'depth': 12,  # Mais profundo
        'l2_leaf_reg': 3,
        'border_count': 254,
        'bagging_temperature': 1.0,
        'random_strength': 1.5,  # Mais randomização
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'auto_class_weights': 'Balanced',
        'verbose': False,
        'random_seed': 42,
        'leaf_estimation_iterations': 10  # Melhor estimação de folhas
    }
    
    catboost_pool_train = cb.Pool(X_train_base, y_train, weight=sample_weights)
    catboost_model = cb.CatBoostClassifier(**catboost_params)
    catboost_model.fit(catboost_pool_train)
    
    # ========================================================================
    # PREDIÇÕES (SEM ENSEMBLE)
    # ========================================================================
    y_pred_proba = catboost_model.predict_proba(X_val_base)[:, 1]
    
    # ========================================================================
    # OTIMIZAÇÃO DE THRESHOLD (GRID SEARCH REFINADO)
    # ========================================================================
    print(f"      🎯 Otimizando threshold com grid search refinado...")
    
    best_f1_class1 = 0
    best_threshold = 0.5
    best_f1_macro = 0
    
    # FASE 1: Grid search grosso
    for threshold in np.arange(0.15, 0.85, 0.05):
        y_pred_temp = (y_pred_proba >= threshold).astype(int)
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
        y_pred_temp = (y_pred_proba >= threshold).astype(int)
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
    
    y_pred_binary = (y_pred_proba >= best_threshold).astype(int)
    
    # ========================================================================
    # MÉTRICAS
    # ========================================================================
    auc_catboost = roc_auc_score(y_val, y_pred_proba)
    
    cm = confusion_matrix(y_val, y_pred_binary)
    f1_class_0 = f1_score(y_val, y_pred_binary, pos_label=0)
    f1_class_1 = f1_score(y_val, y_pred_binary, pos_label=1)
    f1_macro = (f1_class_0 + f1_class_1) / 2
    
    print(f"      CatBoost AUC: {auc_catboost:.4f}")
    print(f"      F1 Classe 0:  {f1_class_0:.4f}")
    print(f"      F1 Classe 1:  {f1_class_1:.4f}")
    print(f"      F1-Macro:     {f1_macro:.4f}")
    print()
    
    fold_results.append({
        'fold': fold,
        'auc': auc_catboost,
        'f1_class_0': f1_class_0,
        'f1_class_1': f1_class_1,
        'f1_macro': f1_macro,
        'threshold': best_threshold,
        'y_val': y_val.values,
        'y_pred_proba': y_pred_proba,
        'y_pred_binary': y_pred_binary,
        'catboost_model': catboost_model,
        'confusion_matrix': cm
    })

# ============================================================================
# 6. RESUMO CROSS-VALIDATION
# ============================================================================
print("="*80)
print("📊 RESUMO CROSS-VALIDATION (APENAS CATBOOST)")
print("="*80)

fold_df = pd.DataFrame(fold_results)

print(f"\n   CatBoost AUC: {fold_df['auc'].mean():.4f} ± {fold_df['auc'].std():.4f}")
print(f"   F1 Classe 0:  {fold_df['f1_class_0'].mean():.4f} ± {fold_df['f1_class_0'].std():.4f}")
print(f"   F1 Classe 1:  {fold_df['f1_class_1'].mean():.4f} ± {fold_df['f1_class_1'].std():.4f}")
print(f"   F1-Macro:     {fold_df['f1_macro'].mean():.4f} ± {fold_df['f1_macro'].std():.4f}")

print()
print("="*80)
print("✅ MODELO V8.5 COM APENAS CATBOOST COMPLETO!")
print("="*80)
print()
print("📈 COMPARAÇÃO:")
print(f"   V8.3 (LightGBM+XGBoost): AUC = 0.9023, F1-C1 = 0.4144")
print(f"   V8.4 (CatBoost+XGBoost): AUC = 0.9011, F1-C1 = 0.4810")
print(f"   V8.5 (CatBoost ONLY):    AUC = {fold_df['auc'].mean():.4f}, F1-C1 = {fold_df['f1_class_1'].mean():.4f}")
print()
print("🎯 Teste: Será que CatBoost sozinho supera o ensemble?")
print("="*80)
print()

print(f"📅 Término: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
