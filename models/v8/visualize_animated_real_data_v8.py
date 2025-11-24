"""
═══════════════════════════════════════════════════════════════════════════════
VISUALIZAÇÃO ANIMADA V8: USANDO DADOS REAIS DO BIGQUERY
═══════════════════════════════════════════════════════════════════════════════

✅ ABORDAGEM: Confiar 100% no modelo
   - Carregar dados REAIS do BigQuery
   - Fazer predições usando o modelo V8 treinado
   - NÃO usar hard-coding de valores

🎬 Animação temporal mostrando lotação prevista hora a hora

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
import warnings

warnings.filterwarnings('ignore')

print("="*80)
print("🎬 VISUALIZAÇÃO ANIMADA V8: DADOS REAIS + PREDIÇÃO DO MODELO")
print("="*80)
print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# ===========================================================================
# ETAPA 1: CARREGAR MODELOS TREINADOS
# ===========================================================================
print("\n[1/5] Carregando modelos treinados V8...")

lgb_model = lgb.Booster(model_file='lightgbm_model_v8_production.txt')
xgb_model = xgb.Booster()
xgb_model.load_model('xgboost_model_v8_production.json')
scaler = joblib.load('scaler_v8_production.pkl')

with open('model_config_v8_production.json', 'r') as f:
    config = json.load(f)

with open('selected_features_v8_production.txt', 'r') as f:
    selected_features = [line.strip() for line in f if line.strip()]

ensemble_weights = config['ensemble_weights']

print(f"✅ Modelos carregados!")
print(f"   - LightGBM weight: {ensemble_weights['lightgbm']:.3f}")
print(f"   - XGBoost weight: {ensemble_weights['xgboost']:.3f}")
print(f"   - Features: {len(selected_features)}")

# ===========================================================================
# ETAPA 2: CARREGAR DADOS REAIS DO BIGQUERY
# ===========================================================================
print("\n[2/5] Carregando dados reais do BigQuery...")

project_id = "proj-ml-469320"
client = bigquery.Client(project=project_id)

# Carregar uma amostra REAL de dados para fazer predições
query = """
SELECT *
FROM `proj-ml-469320.app_cittamobi.dataset-updated`
WHERE target IS NOT NULL
  AND stop_lat_event IS NOT NULL
  AND stop_lon_event IS NOT NULL
  AND time_hour IS NOT NULL
ORDER BY RAND()
LIMIT 10000
"""

print("   ⏳ Carregando amostra de 10K registros reais...")
df_raw = client.query(query).to_dataframe()

print(f"✅ {len(df_raw):,} registros carregados")
print(f"   - Paradas únicas: {df_raw['gtfs_stop_id'].nunique()}")
print(f"   - Conversão real: {df_raw['target'].mean():.2%}")
print(f"   - Horas cobertas: {sorted(df_raw['time_hour'].unique())}")

# ===========================================================================
# ETAPA 3: PREPARAR FEATURES E FAZER PREDIÇÕES
# ===========================================================================
print("\n[3/5] Preparando features e gerando predições...")

# Selecionar apenas as features necessárias
X = df_raw[selected_features].copy()

# Limpar dados
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(0, inplace=True)

# Normalizar
X_scaled = scaler.transform(X)

# Fazer predições
print("   ⏳ Gerando predições com ensemble...")
lgb_pred = lgb_model.predict(X_scaled)

X_df = pd.DataFrame(X_scaled, columns=selected_features)
xgb_pred = xgb_model.predict(xgb.DMatrix(X_df))

# Ensemble
df_raw['predicted_conversion'] = (
    ensemble_weights['lightgbm'] * lgb_pred +
    ensemble_weights['xgboost'] * xgb_pred
)

print(f"✅ Predições geradas!")
print(f"   - Conversão prevista média: {df_raw['predicted_conversion'].mean():.2%}")
print(f"   - Min: {df_raw['predicted_conversion'].min():.2%}")
print(f"   - Max: {df_raw['predicted_conversion'].max():.2%}")

# ===========================================================================
# ETAPA 4: AGREGAR POR PARADA E HORA
# ===========================================================================
print("\n[4/5] Agregando por parada e hora...")

# Agregar por parada e hora para ter uma predição média
df_agg = df_raw.groupby(['gtfs_stop_id', 'time_hour', 'stop_lat_event', 'stop_lon_event']).agg({
    'predicted_conversion': 'mean',
    'target': 'mean'
}).reset_index()

df_agg.columns = ['gtfs_stop_id', 'time_hour', 'stop_lat_event', 'stop_lon_event', 
                  'predicted_conversion', 'actual_conversion']

# Filtrar paradas com dados em múltiplas horas
stop_hours = df_agg.groupby('gtfs_stop_id')['time_hour'].nunique()
valid_stops = stop_hours[stop_hours >= 10].index
df_agg = df_agg[df_agg['gtfs_stop_id'].isin(valid_stops)]

print(f"✅ Dados agregados:")
print(f"   - Total registros: {len(df_agg):,}")
print(f"   - Paradas válidas: {df_agg['gtfs_stop_id'].nunique()}")
print(f"   - Horas cobertas por parada (média): {df_agg.groupby('gtfs_stop_id')['time_hour'].nunique().mean():.1f}")

# Estatísticas por hora
hourly_stats = df_agg.groupby('time_hour').agg({
    'predicted_conversion': ['mean', 'std', 'count']
}).round(4)
hourly_stats.columns = ['mean_pred', 'std_pred', 'n_stops']

print("\n📊 PREDIÇÕES MÉDIAS POR HORA:")
print(hourly_stats.to_string())

# ===========================================================================
# ETAPA 5: CRIAR MAPA INTERATIVO COM ANIMAÇÃO
# ===========================================================================
print("\n[5/5] Criando mapa interativo...")

def get_color_by_conversion(conversion):
    """Cor baseada no nível de conversão"""
    if conversion < 0.20:
        return 'green'
    elif conversion < 0.40:
        return 'yellow'
    elif conversion < 0.60:
        return 'orange'
    else:
        return 'red'

# Criar mapa base
center_lat = df_agg['stop_lat_event'].mean()
center_lon = df_agg['stop_lon_event'].mean()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=12,
    tiles='OpenStreetMap'
)

# Preparar dados para TimestampedGeoJson
features = []

for idx, row in df_agg.iterrows():
    hour = int(row['time_hour'])
    conversion = float(row['predicted_conversion'])
    actual = float(row['actual_conversion'])
    
    timestamp = f"2024-01-01T{hour:02d}:00:00"
    
    feature = {
        'type': 'Feature',
        'geometry': {
            'type': 'Point',
            'coordinates': [float(row['stop_lon_event']), float(row['stop_lat_event'])]
        },
        'properties': {
            'time': timestamp,
            'popup': f"""
                <b>Ponto ID:</b> {row['gtfs_stop_id']}<br>
                <b>Hora:</b> {hour}h<br>
                <b>Lotação Prevista:</b> {conversion:.1%}<br>
                <b>Conversão Real:</b> {actual:.1%}<br>
                <b>Erro:</b> {abs(conversion - actual):.1%}<br>
            """,
            'icon': 'circle',
            'iconstyle': {
                'fillColor': get_color_by_conversion(conversion),
                'color': 'black',
                'fillOpacity': 0.7,
                'weight': 1,
                'radius': 6 + (conversion * 15)
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

# Salvar
output_file = 'mapa_animado_real_data_v8.html'
m.save(output_file)

# Salvar estatísticas
hourly_stats.to_csv('estatisticas_reais_por_hora_v8.csv')

print(f"\n✅ Mapa criado: {output_file}")
print(f"✅ Estatísticas salvas: estatisticas_reais_por_hora_v8.csv")

# ===========================================================================
# FINALIZAÇÃO
# ===========================================================================
print("\n" + "="*80)
print("✅ VISUALIZAÇÃO CONCLUÍDA!")
print("="*80)
print(f"📁 {output_file}")
print(f"📊 {len(df_agg):,} predições (dados reais)")
print(f"📍 {df_agg['gtfs_stop_id'].nunique()} paradas")
print(f"🕐 {df_agg['time_hour'].nunique()} horas")
print("\n🎬 Abra o HTML no navegador para ver a animação!")
print("="*80)
