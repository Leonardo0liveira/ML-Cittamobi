"""
═══════════════════════════════════════════════════════════════════════════════
VISUALIZAÇÃO COM PREDIÇÕES REAIS DO MODELO V8
═══════════════════════════════════════════════════════════════════════════════

✅ ABORDAGEM CORRETA:
   1. Carregar dados REAIS do BigQuery
   2. Aplicar EXATAMENTE as mesmas transformações do treinamento
   3. Fazer predições com o modelo V8 treinado
   4. Comparar Real vs Predito

⚠️  NÃO usa hard-coding - Confia 100% no modelo!

═══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
from google.cloud import bigquery
import folium
from folium.plugins import TimestampedGeoJson
import lightgbm as lgb
import xgboost as xgb
import joblib
import json
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import warnings

warnings.filterwarnings('ignore')

print("="*80)
print("🎯 VISUALIZAÇÃO COM PREDIÇÕES REAIS DO MODELO V8")
print("="*80)
print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# ===========================================================================
# 1. CARREGAR MODELO TREINADO
# ===========================================================================
print("\n[1/6] Carregando modelo V8 treinado...")

lgb_model = lgb.Booster(model_file='lightgbm_model_v8_production.txt')
xgb_model = xgb.Booster()
xgb_model.load_model('xgboost_model_v8_production.json')
scaler = joblib.load('scaler_v8_production.pkl')

with open('model_config_v8_production.json', 'r') as f:
    config = json.load(f)

with open('selected_features_v8_production.txt', 'r') as f:
    selected_features = [line.strip() for line in f if line.strip()]

ensemble_weights = config['ensemble_weights']

print(f"✅ Modelo carregado!")
print(f"   - Ensemble: LightGBM ({ensemble_weights['lightgbm']:.3f}) + XGBoost ({ensemble_weights['xgboost']:.3f})")

# ===========================================================================
# 2. CARREGAR DADOS REAIS DO BIGQUERY
# ===========================================================================
print("\n[2/6] Carregando dados REAIS do BigQuery...")

client = bigquery.Client(project='proj-ml-469320')

# Query para dados reais com sample estratificado por hora
query = """
WITH sampled_data AS (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY time_hour ORDER BY RAND()) as rn
    FROM `proj-ml-469320.app_cittamobi.dataset-updated`
    WHERE target IS NOT NULL
      AND stop_lat_event IS NOT NULL
      AND stop_lon_event IS NOT NULL
)
SELECT * EXCEPT(rn)
FROM sampled_data
WHERE rn <= 500  -- 500 registros por hora = ~12K total
"""

print("   ⏳ Carregando amostra estratificada...")
df = client.query(query).to_dataframe()

print(f"✅ {len(df):,} registros carregados")
print(f"   - Conversão real: {df['target'].mean():.2%}")
print(f"   - Paradas: {df['gtfs_stop_id'].nunique()}")
print(f"   - Horas: {sorted(df['time_hour'].unique())}")

# ===========================================================================
# 3. CRIAR FEATURES EXATAMENTE COMO NO TREINAMENTO
# ===========================================================================
print("\n[3/6] Criando features (Phase 1: Geographic)...")

# Haversine
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

# B. Stop Density
if 'stop_lat_event' in df.columns:
    coords_df = df[['stop_lat_event', 'stop_lon_event']].drop_duplicates().dropna()
    if len(coords_df) > 1:
        nn = NearestNeighbors(n_neighbors=min(11, len(coords_df)), metric='euclidean')
        nn.fit(coords_df)
        distances, _ = nn.kneighbors(df[['stop_lat_event', 'stop_lon_event']].values)
        df['stop_density'] = 1 / (distances.mean(axis=1) + 0.001)
    else:
        df['stop_density'] = 1.0

# C. Distance to CBD
cbd_coords = [
    (-23.5505, -46.6333), (-22.9068, -43.1729), (-19.9167, -43.9345),
    (-25.4284, -49.2733), (-30.0346, -51.2177),
]
min_distances = []
for cbd_lat, cbd_lon in cbd_coords:
    dist = haversine_vectorized(df['stop_lat_event'], df['stop_lon_event'], cbd_lat, cbd_lon)
    min_distances.append(dist)
df['dist_to_nearest_cbd'] = np.minimum.reduce(min_distances)

# D. Stop Clustering
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

# E. Stop Volatility
stop_volatility = df.groupby('gtfs_stop_id')['target'].std().fillna(0).to_dict()
df['stop_volatility'] = df['gtfs_stop_id'].map(stop_volatility)

print(f"✅ Features geográficas criadas")

# ===========================================================================
# 4. CRIAR FEATURES PHASE 2A (DYNAMIC)
# ===========================================================================
print("\n[4/6] Criando features (Phase 2A: Dynamic)...")

# Temporal conversion rates (calculados dos DADOS REAIS)
df['hour_conversion_rate'] = df.groupby('time_hour')['target'].transform('mean')
df['dow_conversion_rate'] = df.groupby('time_day_of_week')['target'].transform('mean')
df['stop_hour_conversion'] = df.groupby(['gtfs_stop_id', 'time_hour'])['target'].transform('mean')

# Geo-temporal interactions
df['geo_temporal'] = df['dist_to_nearest_cbd'] * df['is_peak_hour']
df['density_peak'] = df['stop_density'] * df['is_peak_hour']

# User features
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

# Rarity features
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

# Distance deviation
if 'device_id' in df.columns:
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

print(f"✅ Features dinâmicas criadas")

# ===========================================================================
# 5. FAZER PREDIÇÕES
# ===========================================================================
print("\n[5/6] Fazendo predições com o modelo treinado...")

# Preparar features
exclude_cols = [
    'target', 'gtfs_stop_id', 'timestamp_converted', 'device_id',
    'stop_lat_event', 'stop_lon_event', 'stop_event_count',
    'user_frequency', 'event_timestamp', 'date', 'user_pseudo_id',
    'ctm_service_route', 'direction', 'lotacao_proxy_binaria',
    'y_pred_proba', 'y_pred'
]

feature_cols = [col for col in df.columns if col not in exclude_cols]
X = df[feature_cols].copy()
X = X.select_dtypes(include=[np.number])

# Limpar nomes
X.columns = X.columns.str.replace('[', '_', regex=False)
X.columns = X.columns.str.replace(']', '_', regex=False)

# Verificar features
missing_features = set(selected_features) - set(X.columns)
if missing_features:
    print(f"⚠️  Features faltando: {missing_features}")
    for feat in missing_features:
        X[feat] = 0

# Selecionar apenas as features do modelo
X = X[selected_features].copy()

# Limpar dados
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(0, inplace=True)

# Normalizar
X_scaled = scaler.transform(X)

# Predições
print("   ⏳ LightGBM predicting...")
lgb_pred = lgb_model.predict(X_scaled)

print("   ⏳ XGBoost predicting...")
X_df = pd.DataFrame(X_scaled, columns=selected_features)
xgb_pred = xgb_model.predict(xgb.DMatrix(X_df))

# Ensemble
df['predicted_conversion'] = (
    ensemble_weights['lightgbm'] * lgb_pred +
    ensemble_weights['xgboost'] * xgb_pred
)

print(f"✅ Predições geradas!")
print(f"   - Real: {df['target'].mean():.2%}")
print(f"   - Previsto: {df['predicted_conversion'].mean():.2%}")
print(f"   - Erro médio absoluto: {abs(df['target'] - df['predicted_conversion']).mean():.2%}")

# ===========================================================================
# 6. CRIAR MAPA ANIMADO
# ===========================================================================
print("\n[6/6] Criando mapa interativo...")

# Agregar por parada e hora
df_map = df.groupby(['gtfs_stop_id', 'time_hour', 'stop_lat_event', 'stop_lon_event']).agg({
    'predicted_conversion': 'mean',
    'target': 'mean'
}).reset_index()

df_map.columns = ['gtfs_stop_id', 'time_hour', 'stop_lat_event', 'stop_lon_event',
                  'predicted_conversion', 'actual_conversion']

# Filtrar paradas com múltiplas horas
stop_counts = df_map.groupby('gtfs_stop_id')['time_hour'].count()
valid_stops = stop_counts[stop_counts >= 10].index
df_map = df_map[df_map['gtfs_stop_id'].isin(valid_stops)]

print(f"   - Registros para mapa: {len(df_map):,}")
print(f"   - Paradas: {df_map['gtfs_stop_id'].nunique()}")

def get_color_by_conversion(conversion):
    if conversion < 0.20:
        return 'green'
    elif conversion < 0.40:
        return 'yellow'
    elif conversion < 0.60:
        return 'orange'
    else:
        return 'red'

# Criar mapa
m = folium.Map(
    location=[df_map['stop_lat_event'].mean(), df_map['stop_lon_event'].mean()],
    zoom_start=12,
    tiles='OpenStreetMap'
)

# Preparar features GeoJSON
features = []
for idx, row in df_map.iterrows():
    hour = int(row['time_hour'])
    pred = float(row['predicted_conversion'])
    actual = float(row['actual_conversion'])
    
    feature = {
        'type': 'Feature',
        'geometry': {
            'type': 'Point',
            'coordinates': [float(row['stop_lon_event']), float(row['stop_lat_event'])]
        },
        'properties': {
            'time': f"2024-01-01T{hour:02d}:00:00",
            'popup': f"""
                <b>Parada:</b> {row['gtfs_stop_id']}<br>
                <b>Hora:</b> {hour}h<br>
                <b>📊 Previsto:</b> {pred:.1%}<br>
                <b>🎯 Real:</b> {actual:.1%}<br>
                <b>❌ Erro:</b> {abs(pred-actual):.1%}<br>
            """,
            'icon': 'circle',
            'iconstyle': {
                'fillColor': get_color_by_conversion(pred),
                'color': 'black',
                'fillOpacity': 0.7,
                'weight': 1,
                'radius': 6 + (pred * 15)
            }
        }
    }
    features.append(feature)

TimestampedGeoJson(
    {'type': 'FeatureCollection', 'features': features},
    period='PT1H',
    add_last_point=True,
    auto_play=False,
    loop=True,
    max_speed=1,
    loop_button=True,
    date_options='HH:mm',
    time_slider_drag_update=True,
    duration='PT1H'
).add_to(m)

output_file = 'mapa_modelo_real_v8.html'
m.save(output_file)

# Estatísticas
hourly_stats = df_map.groupby('time_hour').agg({
    'predicted_conversion': ['mean', 'std'],
    'actual_conversion': ['mean', 'std']
}).round(4)
hourly_stats.columns = ['pred_mean', 'pred_std', 'actual_mean', 'actual_std']
hourly_stats['mae'] = abs(hourly_stats['pred_mean'] - hourly_stats['actual_mean'])

print("\n📊 COMPARAÇÃO REAL VS PREVISTO POR HORA:")
print("="*80)
print(hourly_stats.to_string())

hourly_stats.to_csv('comparacao_real_vs_previsto_v8.csv')

print("\n" + "="*80)
print("✅ VISUALIZAÇÃO COM PREDIÇÕES REAIS CONCLUÍDA!")
print("="*80)
print(f"📁 {output_file}")
print(f"📊 {len(df_map):,} predições reais do modelo")
print(f"🎯 Erro médio: {hourly_stats['mae'].mean():.2%}")
print("\n🎬 Este mapa mostra as PREDIÇÕES REAIS do modelo V8!")
print("="*80)
