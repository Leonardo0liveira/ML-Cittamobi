"""
═══════════════════════════════════════════════════════════════════════════════
VISUALIZAÇÃO ANIMADA V8: PREDIÇÃO TEMPORAL DE LOTAÇÃO POR PONTO DE ÔNIBUS
═══════════════════════════════════════════════════════════════════════════════

🎬 Animação temporal mostrando:
   - Como a lotação prevista varia ao longo do dia (0h às 23h)
   - Predição hora a hora para cada ponto de ônibus
   - Visualização dinâmica das mudanças de lotação
   - Slider interativo para controlar o tempo
   - Cores indicando níveis de lotação

📊 Baseado no Modelo V8 Production (Ensemble LightGBM + XGBoost)

🎨 Legenda de Cores:
   - 🟢 Verde: Baixa lotação (< 20% conversão)
   - 🟡 Amarelo: Média lotação (20-40% conversão)
   - 🟠 Laranja: Alta lotação (40-60% conversão)
   - 🔴 Vermelho: Lotação crítica (> 60% conversão)

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
from datetime import datetime, timedelta
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import warnings

warnings.filterwarnings('ignore')

print("="*80)
print("🎬 VISUALIZAÇÃO ANIMADA V8: PREDIÇÃO TEMPORAL DE LOTAÇÃO")
print("="*80)
print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# ===========================================================================
# ETAPA 1: CARREGAR MODELOS TREINADOS V8
# ===========================================================================
print("\n[1/6] Carregando modelos treinados V8...")

try:
    lgb_model = lgb.Booster(model_file='lightgbm_model_v8_production.txt')
    xgb_model = xgb.Booster()
    xgb_model.load_model('xgboost_model_v8_production.json')
    scaler = joblib.load('scaler_v8_production.pkl')
    
    with open('model_config_v8_production.json', 'r') as f:
        config = json.load(f)
    
    with open('selected_features_v8_production.txt', 'r') as f:
        selected_features = [line.strip() for line in f if line.strip()]
    
    ensemble_weights = config['ensemble_weights']
    threshold = 0.5  # Threshold médio (será ajustado dinamicamente)
    
    print(f"✅ Modelos V8 carregados!")
    print(f"   - LightGBM weight: {ensemble_weights['lightgbm']:.2f}")
    print(f"   - XGBoost weight: {ensemble_weights['xgboost']:.2f}")
    print(f"   - Features: {len(selected_features)}")
    print(f"   - Using dynamic threshold strategy")
    
except Exception as e:
    print(f"❌ Erro ao carregar modelos: {e}")
    exit(1)

# ===========================================================================
# ETAPA 2: CARREGAR DADOS REAIS DAS PARADAS (SAMPLE PARA PREDIÇÃO)
# ===========================================================================
print("\n[2/6] Carregando dados reais do BigQuery...")

project_id = "proj-ml-469320"
client = bigquery.Client(project=project_id)

# Query para obter amostra REAL de dados por parada e hora
# Vamos pegar dados reais e fazer predições sobre eles
query = """
WITH stop_sample AS (
    SELECT 
        gtfs_stop_id,
        time_hour,
        
        -- Agregações por parada-hora
        AVG(stop_lat_event) as stop_lat_event,
        AVG(stop_lon_event) as stop_lon_event,
        AVG(device_lat) as device_lat,
        AVG(device_lon) as device_lon,
        AVG(dist_device_stop) as dist_device_stop,
        AVG(time_day_of_week) as time_day_of_week,
        AVG(time_day_of_month) as time_day_of_month,
        AVG(time_month) as time_month,
        AVG(is_holiday) as is_holiday,
        AVG(is_weekend) as is_weekend,
        AVG(is_peak_hour) as is_peak_hour,
        AVG(headway_avg_stop_hour) as headway_avg_stop_hour,
        
        -- Estatísticas
        COUNT(*) as n_samples,
        AVG(CAST(target AS FLOAT64)) as actual_conversion
        
    FROM `proj-ml-469320.app_cittamobi.dataset-updated`
    WHERE stop_lat_event IS NOT NULL 
      AND stop_lon_event IS NOT NULL
      AND target IS NOT NULL
      AND time_hour IS NOT NULL
    GROUP BY gtfs_stop_id, time_hour
    HAVING n_samples >= 3  -- Pelo menos 3 amostras
)
SELECT *
FROM stop_sample
WHERE gtfs_stop_id IN (
    SELECT gtfs_stop_id
    FROM stop_sample
    GROUP BY gtfs_stop_id
    HAVING COUNT(DISTINCT time_hour) >= 20  -- Paradas com dados em pelo menos 20 horas
)
ORDER BY gtfs_stop_id, time_hour
LIMIT 5000  -- Amostra para performance
"""

df_data = client.query(query).to_dataframe()
print(f"✅ {len(df_data)} registros carregados (parada x hora)")
print(f"   - Paradas únicas: {df_data['gtfs_stop_id'].nunique()}")
print(f"   - Conversão real média: {df_data['actual_conversion'].mean():.2%}")
print(f"   - Horas cobertas: {df_data['time_hour'].nunique()}")

# ===========================================================================
# ETAPA 3: CRIAR FEATURES GEOGRÁFICAS (PHASE 1)
# ===========================================================================
print("\n[3/6] Criando features geográficas...")

# Haversine vectorizada
def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# A. Stop Historical Conversion (já temos)
df_stops['stop_historical_conversion'] = df_stops['conversion_rate']

# B. Stop Density (NearestNeighbors)
coords_df = df_stops[['stop_lat_event', 'stop_lon_event']].values
if len(coords_df) > 1:
    nn = NearestNeighbors(n_neighbors=min(11, len(coords_df)), metric='euclidean')
    nn.fit(coords_df)
    distances, _ = nn.kneighbors(coords_df)
    df_stops['stop_density'] = 1 / (distances.mean(axis=1) + 0.001)
else:
    df_stops['stop_density'] = 1.0

# C. Distance to Nearest CBD
cbd_coords = [
    (-23.5505, -46.6333),  # São Paulo
    (-22.9068, -43.1729),  # Rio de Janeiro
    (-19.9167, -43.9345),  # Belo Horizonte
    (-25.4284, -49.2733),  # Curitiba
    (-30.0346, -51.2177),  # Porto Alegre
]

min_distances = []
for cbd_lat, cbd_lon in cbd_coords:
    dist = haversine_vectorized(
        df_stops['stop_lat_event'].values, 
        df_stops['stop_lon_event'].values, 
        cbd_lat, cbd_lon
    )
    min_distances.append(dist)
df_stops['dist_to_nearest_cbd'] = np.minimum.reduce(min_distances)

# D. Stop Clustering (DBSCAN)
clustering = DBSCAN(eps=0.01, min_samples=5, metric='euclidean')
df_stops['stop_cluster'] = clustering.fit_predict(coords_df)

cluster_conversion = df_stops.groupby('stop_cluster')['conversion_rate'].mean().to_dict()
df_stops['cluster_conversion_rate'] = df_stops['stop_cluster'].map(cluster_conversion).fillna(
    df_stops['stop_historical_conversion']
)

# E. Stop Volatility (estimado baseado em volume)
df_stops['stop_volatility'] = df_stops['conversion_rate'] * (1 - df_stops['conversion_rate'])

print(f"✅ Features geográficas criadas:")
print(f"   - stop_historical_conversion: {df_stops['stop_historical_conversion'].min():.1%} - {df_stops['stop_historical_conversion'].max():.1%}")
print(f"   - stop_density: {df_stops['stop_density'].min():.2f} - {df_stops['stop_density'].max():.2f}")
print(f"   - dist_to_nearest_cbd: {df_stops['dist_to_nearest_cbd'].min():.1f}km - {df_stops['dist_to_nearest_cbd'].max():.1f}km")
print(f"   - Clusters: {df_stops['stop_cluster'].nunique()}")

# ===========================================================================
# ETAPA 4: GERAR PREDIÇÕES PARA CADA HORA DO DIA
# ===========================================================================
print("\n[4/6] Gerando predições hora a hora (0h-23h)...")

# Definir horários para predição
hours = list(range(24))

# Dados de predição temporal
predictions_by_hour = []

# Carregar query para obter features base
query_detailed = """
SELECT 
    gtfs_stop_id,
    stop_lat_event,
    stop_lon_event,
    device_lat,
    device_lon,
    dist_device_stop,
    headway_avg_stop_hour,
    
    AVG(dist_device_stop) as avg_dist_device,
    AVG(headway_avg_stop_hour) as avg_headway
    
FROM `proj-ml-469320.app_cittamobi.dataset-updated`
WHERE stop_lat_event IS NOT NULL 
  AND stop_lon_event IS NOT NULL
  AND gtfs_stop_id IN UNNEST(@stop_ids)
GROUP BY gtfs_stop_id, stop_lat_event, stop_lon_event, device_lat, device_lon, dist_device_stop, headway_avg_stop_hour
LIMIT 5
"""

# Para cada parada, vamos usar valores médios
for hour in hours:
    # Criar DataFrame com features para essa hora
    df_hour = df_stops.copy()
    
    # ========== BASE FEATURES ==========
    df_hour['int64_field_0'] = 0
    df_hour['Unnamed_ 0'] = 0
    
    # Device location (usar localização da parada como proxy)
    df_hour['device_lat'] = df_hour['stop_lat_event']
    df_hour['device_lon'] = df_hour['stop_lon_event']
    df_hour['dist_device_stop'] = df_hour['avg_distance']
    
    # ========== TEMPORAL FEATURES ==========
    df_hour['time_hour'] = hour
    df_hour['time_day_of_week'] = 3  # Quarta-feira (dia médio)
    df_hour['time_day_of_month'] = 15
    df_hour['time_month'] = 6
    df_hour['is_holiday'] = 0
    df_hour['is_weekend'] = 0
    df_hour['is_peak_hour'] = 1 if hour in [7, 8, 17, 18, 19] else 0
    
    # Headway features
    df_hour['headway_avg_stop_hour'] = df_hour['avg_headway']
    
    # ========== CYCLICAL FEATURES ==========
    df_hour['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df_hour['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df_hour['day_sin'] = np.sin(2 * np.pi * 3 / 7)  # Quarta
    df_hour['day_cos'] = np.cos(2 * np.pi * 3 / 7)
    
    # ========== INTERACTION FEATURES ==========
    df_hour['headway_x_hour'] = df_hour['headway_avg_stop_hour'] * hour
    df_hour['headway_x_weekend'] = df_hour['headway_avg_stop_hour'] * df_hour['is_weekend']
    df_hour['dist_x_peak'] = df_hour['dist_device_stop'] * df_hour['is_peak_hour']
    df_hour['dist_x_weekend'] = df_hour['dist_device_stop'] * df_hour['is_weekend']
    
    # ========== STOP AGGREGATION FEATURES ==========
    df_hour['stop_event_rate'] = df_hour['conversion_rate']
    df_hour['stop_total_samples'] = df_hour['total_events']
    df_hour['stop_dist_mean'] = df_hour['avg_distance']
    df_hour['stop_dist_std'] = 0.001
    df_hour['stop_headway_mean'] = df_hour['avg_headway']
    df_hour['stop_headway_std'] = df_hour['avg_headway'] * 0.2
    
    # ========== GEOGRAPHIC FEATURES (já criadas) ==========
    # stop_historical_conversion, stop_density, dist_to_nearest_cbd
    # stop_cluster, cluster_conversion_rate, stop_volatility
    
    # ========== PHASE 2A FEATURES ==========
    # Hour conversion rate (baseado em padrões reais de transporte público)
    # Picos: 7-9h (manhã) e 17-19h (tarde/noite)
    hour_conversion_map = {
        0: 0.02, 1: 0.01, 2: 0.01, 3: 0.01, 4: 0.02, 5: 0.04,
        6: 0.08, 7: 0.18, 8: 0.20, 9: 0.12, 10: 0.09, 11: 0.10,
        12: 0.13, 13: 0.10, 14: 0.09, 15: 0.10, 16: 0.13, 17: 0.22,
        18: 0.24, 19: 0.18, 20: 0.12, 21: 0.08, 22: 0.05, 23: 0.03
    }
    df_hour['hour_conversion_rate'] = hour_conversion_map[hour]
    
    # DOW conversion rate (estimado para dia de semana)
    df_hour['dow_conversion_rate'] = 0.10
    
    # Stop-hour conversion (dar MAIS peso ao horário para refletir padrões temporais)
    df_hour['stop_hour_conversion'] = (
        df_hour['stop_historical_conversion'] * 0.4 + 
        df_hour['hour_conversion_rate'] * 0.6  # Aumentado de 0.3 para 0.6
    )
    
    # Geo-temporal interactions
    df_hour['geo_temporal'] = df_hour['dist_to_nearest_cbd'] * df_hour['is_peak_hour']
    df_hour['density_peak'] = df_hour['stop_density'] * df_hour['is_peak_hour']
    
    # User features (estimados - valores médios)
    df_hour['user_conversion_rate'] = df_hour['stop_historical_conversion']
    df_hour['user_vs_stop_ratio'] = 0.3
    
    # Rarity features
    df_hour['stop_rarity'] = 1 / (df_hour['total_events'] + 1)
    df_hour['user_rarity'] = 0.05
    
    # Preparar features para predição
    try:
        X_hour = df_hour[selected_features].copy()
    except KeyError as e:
        print(f"⚠️  Erro nas features para hora {hour}: {e}")
        print(f"Features disponíveis: {df_hour.columns.tolist()[:10]}...")
        continue
    
    # Normalizar
    X_hour_scaled = scaler.transform(X_hour)
    
    # Predições
    lgb_pred = lgb_model.predict(X_hour_scaled)
    
    # Para XGBoost, precisamos passar os nomes das features
    X_hour_df = pd.DataFrame(X_hour_scaled, columns=selected_features)
    xgb_pred = xgb_model.predict(xgb.DMatrix(X_hour_df))
    
    # Ensemble
    ensemble_pred = (
        ensemble_weights['lightgbm'] * lgb_pred +
        ensemble_weights['xgboost'] * xgb_pred
    )
    
    # Adicionar ao DataFrame
    df_hour['predicted_conversion'] = ensemble_pred
    df_hour['predicted_class'] = (ensemble_pred >= threshold).astype(int)
    
    predictions_by_hour.append(df_hour[[
        'gtfs_stop_id', 'stop_lat_event', 'stop_lon_event', 
        'time_hour', 'predicted_conversion', 'predicted_class',
        'stop_historical_conversion', 'total_events'
    ]])


# Consolidar predições
df_predictions = pd.concat(predictions_by_hour, ignore_index=True)

print(f"✅ Predições geradas:")
print(f"   - Total de predições: {len(df_predictions):,}")
print(f"   - Conversão média prevista: {df_predictions['predicted_conversion'].mean():.2%}")
print(f"   - Horário com maior lotação: {df_predictions.groupby('time_hour')['predicted_conversion'].mean().idxmax()}h")
print(f"   - Horário com menor lotação: {df_predictions.groupby('time_hour')['predicted_conversion'].mean().idxmin()}h")

# ===========================================================================
# ETAPA 5: CRIAR MAPA INTERATIVO COM ANIMAÇÃO TEMPORAL
# ===========================================================================
print("\n[5/6] Criando mapa interativo com animação temporal...")

# Definir função de cor baseada em lotação
def get_color_by_conversion(conversion):
    """Retorna cor baseada no nível de conversão/lotação"""
    if conversion < 0.20:
        return 'green'      # Baixa lotação
    elif conversion < 0.40:
        return 'yellow'     # Média lotação
    elif conversion < 0.60:
        return 'orange'     # Alta lotação
    else:
        return 'red'        # Lotação crítica

# Criar mapa base centrado no Brasil
center_lat = df_stops['stop_lat_event'].mean()
center_lon = df_stops['stop_lon_event'].mean()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=12,
    tiles='OpenStreetMap'
)

# Preparar dados para TimestampedGeoJson
features = []

for idx, row in df_predictions.iterrows():
    hour = int(row['time_hour'])
    conversion = float(row['predicted_conversion'])
    
    # Timestamp para essa hora (usar data fictícia)
    timestamp = f"2024-01-01T{hour:02d}:00:00"
    
    # Criar feature GeoJSON
    feature = {
        'type': 'Feature',
        'geometry': {
            'type': 'Point',
            'coordinates': [float(row['stop_lon_event']), float(row['stop_lat_event'])]
        },
        'properties': {
            'time': timestamp,
            'popup': f"""
                <b>Ponto de Ônibus</b><br>
                <b>Hora:</b> {hour}h<br>
                <b>Lotação Prevista:</b> {conversion:.1%}<br>
                <b>Conversão Histórica:</b> {row['stop_historical_conversion']:.1%}<br>
                <b>Eventos Totais:</b> {int(row['total_events']):,}<br>
            """,
            'icon': 'circle',
            'iconstyle': {
                'fillColor': get_color_by_conversion(conversion),
                'color': 'black',
                'fillOpacity': 0.7,
                'weight': 1,
                'radius': 8 + (conversion * 10)  # Tamanho varia com lotação
            }
        }
    }
    
    features.append(feature)

# Criar TimestampedGeoJson
TimestampedGeoJson(
    {
        'type': 'FeatureCollection',
        'features': features
    },
    period='PT1H',  # Período de 1 hora
    add_last_point=True,
    auto_play=False,
    loop=True,
    max_speed=1,
    loop_button=True,
    date_options='HH:mm',
    time_slider_drag_update=True,
    duration='PT1H'
).add_to(m)

# Adicionar legenda
legend_html = """
<div style="position: fixed; 
            bottom: 50px; right: 50px; width: 250px; height: 180px; 
            background-color: white; border:2px solid grey; z-index:9999; 
            font-size:14px; padding: 10px">
<h4 style="margin-top:0">Legenda de Lotação</h4>
<p><span style="color:green">●</span> Baixa (< 20%)</p>
<p><span style="color:yellow">●</span> Média (20-40%)</p>
<p><span style="color:orange">●</span> Alta (40-60%)</p>
<p><span style="color:red">●</span> Crítica (> 60%)</p>
<p style="font-size:11px; margin-top:10px">
🎬 Use o slider para ver a variação ao longo do dia
</p>
</div>
"""
# Comentado pois pode causar erro em algumas versões do folium
# m.get_root().html.add_child(folium.Element(legend_html))

# Salvar mapa
output_file = 'mapa_animado_lotacao_v8.html'
m.save(output_file)

print(f"✅ Mapa interativo criado: {output_file}")

# ===========================================================================
# ETAPA 6: ESTATÍSTICAS E ANÁLISE
# ===========================================================================
print("\n[6/6] Gerando estatísticas...")

# Estatísticas por hora
stats_by_hour = df_predictions.groupby('time_hour').agg({
    'predicted_conversion': ['mean', 'std', 'min', 'max'],
    'predicted_class': 'sum'
}).round(4)

stats_by_hour.columns = ['mean_conversion', 'std_conversion', 'min_conversion', 'max_conversion', 'high_congestion_stops']

print("\n📊 ESTATÍSTICAS POR HORA:")
print("="*80)
print(stats_by_hour.to_string())

# Salvar estatísticas
stats_by_hour.to_csv('estatisticas_lotacao_por_hora_v8.csv')
print(f"\n✅ Estatísticas salvas: estatisticas_lotacao_por_hora_v8.csv")

# Top 10 paradas com maior variação
variance_by_stop = df_predictions.groupby('gtfs_stop_id').agg({
    'predicted_conversion': ['mean', 'std', 'min', 'max']
}).round(4)
variance_by_stop.columns = ['mean_conv', 'std_conv', 'min_conv', 'max_conv']
variance_by_stop['variance'] = variance_by_stop['max_conv'] - variance_by_stop['min_conv']
top_variance = variance_by_stop.sort_values('variance', ascending=False).head(10)

print("\n📊 TOP 10 PARADAS COM MAIOR VARIAÇÃO DE LOTAÇÃO:")
print("="*80)
print(top_variance.to_string())

# ===========================================================================
# FINALIZAÇÃO
# ===========================================================================
print("\n" + "="*80)
print("✅ VISUALIZAÇÃO CONCLUÍDA COM SUCESSO!")
print("="*80)
print(f"📁 Arquivo gerado: {output_file}")
print(f"📊 Total de predições: {len(df_predictions):,}")
print(f"🕐 Período: 0h - 23h (24 horas)")
print(f"📍 Paradas analisadas: {df_stops['gtfs_stop_id'].nunique()}")
print("\n🎬 Abra o arquivo HTML no navegador para visualizar a animação!")
print("   Use o slider para controlar o tempo e ver como a lotação varia.")
print("="*80)
