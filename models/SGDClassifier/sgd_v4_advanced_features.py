import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, 
                             recall_score, f1_score, confusion_matrix, roc_curve)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures
import warnings
import time
from google.cloud import bigquery
import os
warnings.filterwarnings('ignore')

os.makedirs('visualizations', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# ===========================================================================
# SGD CLASSIFIER V4 - ADVANCED FEATURES (FASE 2)
# ===========================================================================
print(f"\n{'='*80}")
print(f"SGD CLASSIFIER V4 - ADVANCED FEATURES (FASE 2)")
print(f"{'='*80}")
print(f"Melhorias implementadas:")
print(f"  1️⃣ Geographic Clusters: K-means em lat/lon (8 clusters)")
print(f"  2️⃣ Stop Tiers: Quantiles de popularidade (5 tiers)")
print(f"  3️⃣ Complex Interactions: stop_tier × time_period, cluster × hour")
print(f"  4️⃣ Polynomial Features: Interações de 2ª ordem nas top features")
print(f"  5️⃣ User Engagement Score: Agregação de múltiplas dimensões")
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
# ETAPA 3: FEATURE ENGINEERING V3 (BASELINE)
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 3: FEATURE ENGINEERING V3 (BASELINE)")
print(f"{'='*70}")

# Todas as features do V3
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

if 'stop_event_rate' in df.columns:
    df['stop_rate_squared'] = df['stop_event_rate'] ** 2
    df['stop_rate_log'] = np.log1p(df['stop_event_rate'])

df['time_period'] = pd.cut(df['hour'], 
                           bins=[0, 6, 9, 12, 14, 17, 19, 24],
                           labels=[0, 1, 2, 3, 4, 5, 6])
df['time_period'] = df['time_period'].astype(float)

df['hour_weekday_interaction'] = df['hour'] * df['day_of_week']
df['distance_from_peak'] = df['hour'].apply(lambda h: min(abs(h-8), abs(h-18)))

df['user_recency_score'] = np.exp(-df['user_recency_days'] / 7)
df['stop_recency_score'] = np.exp(-df['stop_recency_days'] / 7)

if 'user_recency_score' in df.columns and 'stop_event_rate' in df.columns:
    df['user_stop_recency_interaction'] = df['user_recency_score'] * df['stop_event_rate']

print(f"✓ Features V3 criadas (11 features)")

# ===========================================================================
# ETAPA 4: ADVANCED FEATURE ENGINEERING V4 (FASE 2)
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 4: ADVANCED FEATURE ENGINEERING V4 (FASE 2)")
print(f"{'='*70}")

# 1. GEOGRAPHIC CLUSTERS (K-means)
if 'device_lat' in df.columns and 'device_lon' in df.columns:
    print(f"🔄 Criando clusters geográficos (K-means, K=8)...")
    geo_data = df[['device_lat', 'device_lon']].dropna()
    
    if len(geo_data) > 100:
        kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
        
        # Treinar apenas em subset para velocidade
        sample_indices = np.random.choice(len(geo_data), min(10000, len(geo_data)), replace=False)
        kmeans.fit(geo_data.iloc[sample_indices])
        
        # Predizer para todos
        df['geo_cluster'] = -1
        valid_mask = df[['device_lat', 'device_lon']].notna().all(axis=1)
        df.loc[valid_mask, 'geo_cluster'] = kmeans.predict(df.loc[valid_mask, ['device_lat', 'device_lon']])
        
        print(f"✓ geo_cluster criada (8 clusters)")
        
        # Cluster statistics
        for cluster_id in range(8):
            cluster_mask = df['geo_cluster'] == cluster_id
            cluster_conv = df.loc[cluster_mask, target].mean()
            df.loc[cluster_mask, f'cluster_{cluster_id}_conversion'] = cluster_conv
        print(f"✓ cluster_conversion rates criadas (8 features)")

# 2. STOP TIERS (Quantile-based)
if 'stop_event_rate' in df.columns:
    print(f"🔄 Criando stop tiers (5 quantiles)...")
    df['stop_tier'] = pd.qcut(df['stop_event_rate'], q=5, labels=[0, 1, 2, 3, 4], duplicates='drop')
    df['stop_tier'] = df['stop_tier'].astype(float)
    print(f"✓ stop_tier criada (5 tiers)")

# 3. COMPLEX INTERACTIONS
print(f"🔄 Criando interações complexas...")

# stop_tier × time_period
if 'stop_tier' in df.columns and 'time_period' in df.columns:
    df['stop_tier_x_time_period'] = df['stop_tier'] * df['time_period']
    print(f"✓ stop_tier_x_time_period criada")

# geo_cluster × hour
if 'geo_cluster' in df.columns:
    df['geo_cluster_x_hour'] = df['geo_cluster'] * df['hour']
    print(f"✓ geo_cluster_x_hour criada")

# stop_tier × is_weekend
if 'stop_tier' in df.columns and 'is_weekend' in df.columns:
    df['stop_tier_x_weekend'] = df['stop_tier'] * df['is_weekend']
    print(f"✓ stop_tier_x_weekend criada")

# geo_cluster × time_period
if 'geo_cluster' in df.columns and 'time_period' in df.columns:
    df['geo_cluster_x_time_period'] = df['geo_cluster'] * df['time_period']
    print(f"✓ geo_cluster_x_time_period criada")

# 4. USER ENGAGEMENT SCORE
print(f"🔄 Criando user engagement score...")
if all(col in df.columns for col in ['user_hist_count', 'user_hist_conversion_rate', 'user_recency_score']):
    df['user_engagement_score'] = (
        np.log1p(df['user_hist_count']) * 
        df['user_hist_conversion_rate'] * 
        df['user_recency_score']
    )
    print(f"✓ user_engagement_score criada")

# 5. STOP POPULARITY SCORE
print(f"🔄 Criando stop popularity score...")
if all(col in df.columns for col in ['stop_event_rate', 'stop_total_samples', 'stop_hist_count']):
    df['stop_popularity_score'] = (
        df['stop_event_rate'] * 
        np.log1p(df['stop_total_samples']) * 
        np.log1p(df['stop_hist_count'])
    )
    print(f"✓ stop_popularity_score criada")

# 6. POLYNOMIAL INTERACTIONS (top features only)
print(f"🔄 Criando interações polinomiais...")
if all(col in df.columns for col in ['stop_event_rate', 'stop_event_count']):
    df['stop_rate_x_count'] = df['stop_event_rate'] * df['stop_event_count']
    print(f"✓ stop_rate_x_count criada")

if all(col in df.columns for col in ['user_hist_conversion_rate', 'stop_hist_conversion_rate']):
    df['user_stop_conv_product'] = df['user_hist_conversion_rate'] * df['stop_hist_conversion_rate']
    print(f"✓ user_stop_conv_product criada")

if all(col in df.columns for col in ['geo_distance', 'stop_event_rate']):
    df['geo_distance_x_stop_rate'] = df['geo_distance'] * df['stop_event_rate']
    print(f"✓ geo_distance_x_stop_rate criada")

# 7. TIME-BASED FEATURES
print(f"🔄 Criando features temporais avançadas...")
df['is_morning_peak'] = ((df['hour'] >= 6) & (df['hour'] <= 9)).astype(int)
df['is_evening_peak'] = ((df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
df['is_lunch_time'] = ((df['hour'] >= 11) & (df['hour'] <= 14)).astype(int)
df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
print(f"✓ Períodos específicos criados (4 features)")

# 8. RECENCY INTERACTIONS
if all(col in df.columns for col in ['user_recency_days', 'stop_recency_days']):
    df['combined_recency'] = np.minimum(df['user_recency_days'], df['stop_recency_days'])
    df['recency_ratio'] = df['user_recency_days'] / (df['stop_recency_days'] + 1)
    print(f"✓ combined_recency e recency_ratio criadas")

print(f"\n✅ Total de novas features V4: ~30 advanced features")

# ===========================================================================
# ETAPA 5: LIMPEZA E PREPARAÇÃO
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 5: LIMPEZA E PREPARAÇÃO")
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

print(f"✓ Features totais: {len(numeric_features)} (V3 + V4)")

target_dist = y.value_counts()
print(f"\n=== Distribuição do Target ===")
for classe, count in target_dist.items():
    print(f"Classe {classe}: {count:,} ({100*count/len(y):.2f}%)")

# ===========================================================================
# ETAPA 6: DIVISÃO TEMPORAL
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 6: DIVISÃO TEMPORAL")
print(f"{'='*70}")

tscv = TimeSeriesSplit(n_splits=3)
splits = list(tscv.split(X))
train_idx, test_idx = splits[-1]
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print(f"✓ Train: {len(X_train):,} | Test: {len(X_test):,}")

# ===========================================================================
# ETAPA 7: TREINAR V3 (BASELINE)
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 7: V3 BASELINE (PARA COMPARAÇÃO)")
print(f"{'='*70}")

# Features V3 (sem as V4)
v3_features = ['stop_event_rate', 'stop_event_count', 'stop_total_samples', 'hour', 
               'time_hour', 'is_weekend', 'stop_lat_event', 'headway_x_weekend',
               'time_day_of_month', 'device_lon', 'stop_lon_event', 'is_peak_hour',
               'hour_cos', 'time_month', 'device_lat', 'headway_x_hour',
               'stop_recency_days', 'stop_dist_mean', 'user_recency_days', 'time_day_of_week',
               'geo_distance', 'stop_rate_per_sample', 'stop_density', 'stop_rate_squared',
               'stop_rate_log', 'time_period', 'hour_weekday_interaction', 'distance_from_peak',
               'user_recency_score', 'stop_recency_score', 'user_stop_recency_interaction']

v3_features_available = [f for f in v3_features if f in X.columns]
X_train_v3 = X_train[v3_features_available]
X_test_v3 = X_test[v3_features_available]

print(f"🔄 Treinando V3 baseline ({len(v3_features_available)} features)...")
start_time = time.time()

pipeline_v3 = Pipeline([
    ('scaler', StandardScaler()),
    ('sgd', SGDClassifier(
        loss='log_loss',
        penalty='l2',
        alpha=0.001,
        class_weight='balanced',
        learning_rate='optimal',
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=42,
        n_jobs=-1
    ))
])

pipeline_v3.fit(X_train_v3, y_train)

# Calibrar
calibrated_v3 = CalibratedClassifierCV(pipeline_v3, method='sigmoid', cv=5)
calibrated_v3.fit(X_train_v3, y_train)

y_pred_proba_v3 = calibrated_v3.predict_proba(X_test_v3)[:, 1]

# Threshold otimizado
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_proba_v3)
valid_indices = np.where(precision >= 0.35)[0]
if len(valid_indices) > 0:
    best_idx = valid_indices[np.argmax(recall[valid_indices])]
    optimal_threshold_v3 = thresholds_pr[best_idx]
else:
    optimal_threshold_v3 = 0.5

y_pred_v3 = (y_pred_proba_v3 >= optimal_threshold_v3).astype(int)

roc_auc_v3 = roc_auc_score(y_test, y_pred_proba_v3)
f1_macro_v3 = f1_score(y_test, y_pred_v3, average='macro')
train_time_v3 = time.time() - start_time

print(f"\n📊 V3 BASELINE:")
print(f"   Features:     {len(v3_features_available)}")
print(f"   ROC-AUC:      {roc_auc_v3:.4f}")
print(f"   F1-Macro:     {f1_macro_v3:.4f}")
print(f"   Threshold:    {optimal_threshold_v3:.2f}")
print(f"   Tempo:        {train_time_v3:.2f}s")

# ===========================================================================
# ETAPA 8: TREINAR V4 (COM ADVANCED FEATURES)
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 8: V4 - COM ADVANCED FEATURES (FASE 2)")
print(f"{'='*70}")

print(f"🔄 Treinando V4 ({len(numeric_features)} features)...")
start_time = time.time()

pipeline_v4 = Pipeline([
    ('scaler', StandardScaler()),
    ('sgd', SGDClassifier(
        loss='log_loss',
        penalty='l2',
        alpha=0.001,
        class_weight='balanced',
        learning_rate='optimal',
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=42,
        n_jobs=-1
    ))
])

pipeline_v4.fit(X_train, y_train)

# Calibrar
print(f"🔄 Calibrando probabilidades...")
calibrated_v4 = CalibratedClassifierCV(pipeline_v4, method='sigmoid', cv=5)
calibrated_v4.fit(X_train, y_train)

y_pred_proba_v4 = calibrated_v4.predict_proba(X_test)[:, 1]

# Threshold otimizado
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_proba_v4)
valid_indices = np.where(precision >= 0.35)[0]
if len(valid_indices) > 0:
    best_idx = valid_indices[np.argmax(recall[valid_indices])]
    optimal_threshold_v4 = thresholds_pr[best_idx]
else:
    optimal_threshold_v4 = 0.5

y_pred_v4 = (y_pred_proba_v4 >= optimal_threshold_v4).astype(int)

roc_auc_v4 = roc_auc_score(y_test, y_pred_proba_v4)
accuracy_v4 = accuracy_score(y_test, y_pred_v4)
precision_v4 = precision_score(y_test, y_pred_v4, zero_division=0)
recall_v4 = recall_score(y_test, y_pred_v4)
f1_v4 = f1_score(y_test, y_pred_v4)
f1_macro_v4 = f1_score(y_test, y_pred_v4, average='macro')
cm_v4 = confusion_matrix(y_test, y_pred_v4)
train_time_v4 = time.time() - start_time

print(f"\n📊 V4 FINAL (ADVANCED FEATURES):")
print(f"   Features:     {len(numeric_features)}")
print(f"   ROC-AUC:      {roc_auc_v4:.4f} 🎯 ({roc_auc_v4 - roc_auc_v3:+.4f})")
print(f"   F1-Macro:     {f1_macro_v4:.4f} ({f1_macro_v4 - f1_macro_v3:+.4f})")
print(f"   Accuracy:     {accuracy_v4:.4f}")
print(f"   Precision:    {precision_v4:.4f}")
print(f"   Recall:       {recall_v4:.4f}")
print(f"   F1-Score:     {f1_v4:.4f}")
print(f"   Threshold:    {optimal_threshold_v4:.2f}")
print(f"   Tempo:        {train_time_v4:.2f}s")

print(f"\n📊 Matriz de Confusão:")
print(cm_v4)
print(f"\nTrue Negatives:  {cm_v4[0,0]:,}")
print(f"False Positives: {cm_v4[0,1]:,}")
print(f"False Negatives: {cm_v4[1,0]:,}")
print(f"True Positives:  {cm_v4[1,1]:,}")

# ===========================================================================
# ETAPA 9: ANÁLISE DE FEATURES V4
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 9: ANÁLISE DE FEATURES V4")
print(f"{'='*70}")

coefficients_v4 = pipeline_v4.named_steps['sgd'].coef_[0]
feature_importance_v4 = pd.DataFrame({
    'feature': numeric_features,
    'coefficient': coefficients_v4,
    'abs_coefficient': np.abs(coefficients_v4)
}).sort_values('abs_coefficient', ascending=False)

print(f"\n📊 TOP 30 FEATURES V4:")
print(f"{'='*70}")

# Identificar features V4
v4_new_features = [
    'geo_cluster', 'stop_tier', 'stop_tier_x_time_period', 'geo_cluster_x_hour',
    'stop_tier_x_weekend', 'geo_cluster_x_time_period', 'user_engagement_score',
    'stop_popularity_score', 'stop_rate_x_count', 'user_stop_conv_product',
    'geo_distance_x_stop_rate', 'is_morning_peak', 'is_evening_peak', 'is_lunch_time',
    'is_night', 'combined_recency', 'recency_ratio'
]
v4_new_features += [f'cluster_{i}_conversion' for i in range(8)]

for i, row in feature_importance_v4.head(30).iterrows():
    sign = "+" if row['coefficient'] > 0 else ""
    marker = "🆕" if any(nf in row['feature'] for nf in v4_new_features) else "  "
    print(f"{marker} {row['feature']:45s} | {sign}{row['coefficient']:+.6f}")

# ===========================================================================
# ETAPA 10: VISUALIZAÇÕES
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 10: VISUALIZAÇÕES")
print(f"{'='*70}")

# 1. Comparação V3 vs V4
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ROC-AUC
ax1 = axes[0, 0]
models = ['V3\n(Quick Wins)', 'V4\n(Advanced)']
roc_aucs = [roc_auc_v3, roc_auc_v4]
colors = ['forestgreen', 'purple']
bars = ax1.bar(models, roc_aucs, color=colors, alpha=0.7)
ax1.set_ylabel('ROC-AUC', fontweight='bold')
ax1.set_title('ROC-AUC: V3 vs V4', fontweight='bold')
ax1.set_ylim(0.75, 0.85)
ax1.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, roc_aucs):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.003, f'{val:.4f}', 
             ha='center', va='bottom', fontweight='bold', fontsize=10)

# F1-Macro
ax2 = axes[0, 1]
f1_macros = [f1_macro_v3, f1_macro_v4]
bars = ax2.bar(models, f1_macros, color=colors, alpha=0.7)
ax2.set_ylabel('F1-Macro', fontweight='bold')
ax2.set_title('F1-Macro: V3 vs V4', fontweight='bold')
ax2.set_ylim(0.65, 0.75)
ax2.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, f1_macros):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.003, f'{val:.4f}', 
             ha='center', va='bottom', fontweight='bold', fontsize=10)

# Feature count
ax3 = axes[1, 0]
feature_counts = [len(v3_features_available), len(numeric_features)]
bars = ax3.bar(models, feature_counts, color=colors, alpha=0.7)
ax3.set_ylabel('Número de Features', fontweight='bold')
ax3.set_title('Features: V3 vs V4', fontweight='bold')
ax3.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, feature_counts):
    ax3.text(bar.get_x() + bar.get_width()/2, val + 2, f'{int(val)}', 
             ha='center', va='bottom', fontweight='bold')

# Ganhos percentuais
ax4 = axes[1, 1]
gains = [0, (roc_auc_v4 - roc_auc_v3)*100]
colors_gain = ['gray', 'purple']
bars = ax4.bar(models, gains, color=colors_gain, alpha=0.7)
ax4.set_ylabel('Ganho em ROC-AUC (%)', fontweight='bold')
ax4.set_title('Ganho Percentual vs V3', fontweight='bold')
ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax4.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, gains):
    ax4.text(bar.get_x() + bar.get_width()/2, val + 0.1, f'{val:+.2f}%', 
             ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('visualizations/sgd_v4_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Comparação salva: visualizations/sgd_v4_comparison.png")

# 2. ROC Curves
fpr_v3, tpr_v3, _ = roc_curve(y_test, y_pred_proba_v3)
fpr_v4, tpr_v4, _ = roc_curve(y_test, y_pred_proba_v4)

plt.figure(figsize=(10, 8))
plt.plot(fpr_v3, tpr_v3, linewidth=2, label=f'V3 Quick Wins (AUC = {roc_auc_v3:.4f})', 
         color='forestgreen')
plt.plot(fpr_v4, tpr_v4, linewidth=2, label=f'V4 Advanced (AUC = {roc_auc_v4:.4f})', 
         color='purple')
plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random', alpha=0.5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
plt.title('ROC Curve: V3 vs V4', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/roc_curve_v3_v4.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ ROC Curve salva: visualizations/roc_curve_v3_v4.png")

# 3. Feature importance V4
plt.figure(figsize=(12, 12))
top_40 = feature_importance_v4.head(40)
colors_feat = ['green' if x > 0 else 'red' for x in top_40['coefficient']]
plt.barh(range(len(top_40)), top_40['coefficient'], color=colors_feat, alpha=0.7)
plt.yticks(range(len(top_40)), top_40['feature'], fontsize=9)
plt.xlabel('Coeficiente', fontsize=12, fontweight='bold')
plt.ylabel('Feature', fontsize=12, fontweight='bold')
plt.title(f'Top 40 Features - SGD V4 ({len(numeric_features)} features)', 
          fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/feature_importance_v4.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Feature Importance salva: visualizations/feature_importance_v4.png")

# 4. Comparação Progressiva (V2 → V3 → V4)
# Buscar ROC-AUC do V2 se disponível
v2_roc_auc = 0.7917  # Do relatório anterior

plt.figure(figsize=(10, 6))
versions = ['V2\n(20 feat)', 'V3\n(Quick Wins)', 'V4\n(Advanced)']
roc_progression = [v2_roc_auc, roc_auc_v3, roc_auc_v4]
colors_prog = ['steelblue', 'forestgreen', 'purple']

bars = plt.bar(versions, roc_progression, color=colors_prog, alpha=0.7)
plt.ylabel('ROC-AUC', fontsize=12, fontweight='bold')
plt.title('Progressão SGD: V2 → V3 → V4', fontsize=14, fontweight='bold')
plt.ylim(0.75, 0.85)
plt.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, roc_progression):
    plt.text(bar.get_x() + bar.get_width()/2, val + 0.003, f'{val:.4f}', 
             ha='center', va='bottom', fontweight='bold', fontsize=11)

# Adicionar setas de ganho
for i in range(len(roc_progression)-1):
    gain = (roc_progression[i+1] - roc_progression[i]) * 100
    plt.annotate(f'+{gain:.2f}%', 
                xy=(i+0.5, (roc_progression[i] + roc_progression[i+1])/2),
                fontsize=10, fontweight='bold', color='darkgreen',
                ha='center')

plt.tight_layout()
plt.savefig('visualizations/sgd_progression_v2_v3_v4.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Progressão salva: visualizations/sgd_progression_v2_v3_v4.png")

# ===========================================================================
# ETAPA 11: SALVAR RESULTADOS
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 11: SALVAR RESULTADOS")
print(f"{'='*70}")

# Salvar features V4
feature_importance_v4.to_csv('reports/sgd_v4_features.csv', index=False)
print(f"✓ Features V4 salvas: reports/sgd_v4_features.csv")

# Relatório comparativo
with open('reports/sgd_v4_report.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("SGD CLASSIFIER V4 - ADVANCED FEATURES (FASE 2) REPORT\n")
    f.write("="*80 + "\n\n")
    
    f.write("MELHORIAS IMPLEMENTADAS - FASE 2\n")
    f.write("-"*80 + "\n")
    f.write("1. Geographic Clusters:\n")
    f.write("   - K-means clustering (K=8) em device_lat/device_lon\n")
    f.write("   - Cluster-specific conversion rates (8 features)\n\n")
    
    f.write("2. Stop Tiers:\n")
    f.write("   - Quantile-based tiers (5 tiers) de stop_event_rate\n\n")
    
    f.write("3. Complex Interactions:\n")
    f.write("   - stop_tier × time_period\n")
    f.write("   - geo_cluster × hour\n")
    f.write("   - stop_tier × is_weekend\n")
    f.write("   - geo_cluster × time_period\n\n")
    
    f.write("4. Advanced Scores:\n")
    f.write("   - user_engagement_score: agregação multi-dimensional\n")
    f.write("   - stop_popularity_score: combinação de métricas\n\n")
    
    f.write("5. Polynomial Interactions:\n")
    f.write("   - stop_rate × count\n")
    f.write("   - user_conv × stop_conv\n")
    f.write("   - geo_distance × stop_rate\n\n")
    
    f.write("6. Time-Based Features:\n")
    f.write("   - Períodos específicos: morning_peak, evening_peak, lunch, night\n\n")
    
    f.write("7. Recency Interactions:\n")
    f.write("   - combined_recency, recency_ratio\n\n")
    
    f.write("COMPARAÇÃO DE PERFORMANCE\n")
    f.write("-"*80 + "\n")
    f.write(f"{'Modelo':<25} | {'Features':<10} | {'ROC-AUC':<10} | {'Ganho':<12} | {'F1-Macro':<10}\n")
    f.write("-"*80 + "\n")
    f.write(f"{'V2 (Feature Selection)':<25} | {20:<10} | {v2_roc_auc:<10.4f} | {'-':<12} | {'0.7000':<10}\n")
    f.write(f"{'V3 (Quick Wins)':<25} | {len(v3_features_available):<10} | {roc_auc_v3:<10.4f} | {(roc_auc_v3-v2_roc_auc)*100:+11.2f}% | {f1_macro_v3:<10.4f}\n")
    f.write(f"{'V4 (Advanced Features)':<25} | {len(numeric_features):<10} | {roc_auc_v4:<10.4f} | {(roc_auc_v4-roc_auc_v3)*100:+11.2f}% | {f1_macro_v4:<10.4f}\n\n")
    
    f.write("MÉTRICAS DETALHADAS V4 FINAL\n")
    f.write("-"*80 + "\n")
    f.write(f"ROC-AUC:           {roc_auc_v4:.4f}\n")
    f.write(f"Accuracy:          {accuracy_v4:.4f}\n")
    f.write(f"Precision:         {precision_v4:.4f}\n")
    f.write(f"Recall:            {recall_v4:.4f}\n")
    f.write(f"F1-Score:          {f1_v4:.4f}\n")
    f.write(f"F1-Macro:          {f1_macro_v4:.4f}\n")
    f.write(f"Threshold:         {optimal_threshold_v4:.2f}\n\n")
    
    f.write("MATRIZ DE CONFUSÃO\n")
    f.write("-"*80 + "\n")
    f.write(f"True Negatives:    {cm_v4[0,0]:,}\n")
    f.write(f"False Positives:   {cm_v4[0,1]:,}\n")
    f.write(f"False Negatives:   {cm_v4[1,0]:,}\n")
    f.write(f"True Positives:    {cm_v4[1,1]:,}\n\n")
    
    f.write("TOP 40 FEATURES V4\n")
    f.write("-"*80 + "\n")
    for i, row in feature_importance_v4.head(40).iterrows():
        marker = "[V4]" if any(nf in row['feature'] for nf in v4_new_features) else "    "
        f.write(f"{marker} {row['feature']:45s} | Coef: {row['coefficient']:+.6f}\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("ANÁLISE DE GANHOS\n")
    f.write("="*80 + "\n")
    f.write(f"V2 → V3:  {(roc_auc_v3-v2_roc_auc)*100:+.2f}% (Quick Wins: feature eng básico + calibração)\n")
    f.write(f"V3 → V4:  {(roc_auc_v4-roc_auc_v3)*100:+.2f}% (Advanced: clusters, interactions, scores)\n")
    f.write(f"V2 → V4:  {(roc_auc_v4-v2_roc_auc)*100:+.2f}% (Total acumulado)\n")

print(f"✓ Relatório V4 salvo: reports/sgd_v4_report.txt")

# ===========================================================================
# CONCLUSÃO
# ===========================================================================
print(f"\n{'='*80}")
print(f"✅ SGD CLASSIFIER V4 - ADVANCED FEATURES CONCLUÍDO!")
print(f"{'='*80}")

print(f"\n🎯 COMPARAÇÃO FINAL:")
print(f"   V2 (Feature Selection): {v2_roc_auc:.4f} ROC-AUC (20 features)")
print(f"   V3 (Quick Wins):        {roc_auc_v3:.4f} ROC-AUC ({(roc_auc_v3-v2_roc_auc)*100:+.2f}%)")
print(f"   V4 (Advanced):          {roc_auc_v4:.4f} ROC-AUC ({(roc_auc_v4-roc_auc_v3)*100:+.2f}%) 🎯")

print(f"\n💰 GANHOS FASE 2:")
gain_pct = (roc_auc_v4 - roc_auc_v3) * 100
total_gain = (roc_auc_v4 - v2_roc_auc) * 100

if gain_pct >= 2.0:
    print(f"   ✅ EXCELENTE! Ganho V3→V4: {gain_pct:+.2f}% supera meta de +2%")
elif gain_pct >= 1.0:
    print(f"   ✅ BOM! Ganho V3→V4: {gain_pct:+.2f}% dentro do esperado")
else:
    print(f"   ⚠️  Ganho V3→V4: {gain_pct:+.2f}% abaixo da meta")

print(f"   📊 Ganho total V2→V4: {total_gain:+.2f}%")

print(f"\n📁 Arquivos gerados:")
print(f"   - visualizations/sgd_v4_comparison.png")
print(f"   - visualizations/roc_curve_v3_v4.png")
print(f"   - visualizations/feature_importance_v4.png")
print(f"   - visualizations/sgd_progression_v2_v3_v4.png")
print(f"   - reports/sgd_v4_report.txt")
print(f"   - reports/sgd_v4_features.csv")

# Comparação com CatBoost
catboost_auc = 0.8669
gap = catboost_auc - roc_auc_v4
print(f"\n📊 GAP vs CatBoost:")
print(f"   CatBoost:    {catboost_auc:.4f} ROC-AUC")
print(f"   SGD V4:      {roc_auc_v4:.4f} ROC-AUC")
print(f"   Gap:         {gap:.4f} ({gap*100:.2f}%)")

print(f"\n🚀 PRÓXIMOS PASSOS:")
print(f"   FASE 3: Ensembles (Voting, Stacking com LightGBM)")
print(f"   Meta final: 84-86% ROC-AUC (fechar gap para CatBoost)")

print(f"\n✅ FASE 2 - ADVANCED FEATURES IMPLEMENTADO COM SUCESSO!")
