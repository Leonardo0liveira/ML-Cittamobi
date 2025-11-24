"""
═══════════════════════════════════════════════════════════════════════════════
VISUALIZAÇÃO DAS PREDIÇÕES DO MODELO V8
═══════════════════════════════════════════════════════════════════════════════

🎯 OBJETIVO: Mostrar APENAS o que o modelo está prevendo
   - Dados reais do BigQuery
   - Features criadas como no treinamento
   - Predições do modelo V8
   - Foco: Visualizar lotação prevista ao longo do dia

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
print("🎯 VISUALIZAÇÃO DAS PREDIÇÕES DO MODELO V8")
print("="*80)
print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# ===========================================================================
# 1. CARREGAR MODELO TREINADO
# ===========================================================================
print("\n[1/6] Carregando modelo V8...")

lgb_model = lgb.Booster(model_file='lightgbm_model_v8_production.txt')
xgb_model = xgb.Booster()
xgb_model.load_model('xgboost_model_v8_production.json')
scaler = joblib.load('scaler_v8_production.pkl')

with open('model_config_v8_production.json', 'r') as f:
    config = json.load(f)

with open('selected_features_v8_production.txt', 'r') as f:
    selected_features = [line.strip() for line in f if line.strip()]

ensemble_weights = config['ensemble_weights']

print(f"✅ Modelo V8 Production carregado!")

# ===========================================================================
# 2. CARREGAR DADOS DO BIGQUERY
# ===========================================================================
print("\n[2/6] Carregando dados do BigQuery...")

# ===========================================================================
# 2. CARREGAR DADOS (BigQuery ou CSV)
# ===========================================================================
print("\n[2/6] Carregando dados...")

# ============================================================================
# CONFIGURAÇÃO: Escolha a fonte de dados
# ============================================================================
USE_CSV = True  # ⚠️ MUDE PARA False PARA USAR BIGQUERY
CSV_PATH = '../OFICIAL-20251112T122637Z-1-001/dataset_completo.csv'  # ⚠️ AJUSTE O CAMINHO

if USE_CSV:
    print(f"   📂 Carregando do CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    
    # Filtrar apenas horário comercial + dias úteis (se colunas existirem)
    if 'time_hour' in df.columns:
        df = df[
            (df['time_hour'] >= 5) & 
            (df['time_hour'] <= 23)
        ].copy()
    
    if 'time_day_of_week' in df.columns:
        df = df[df['time_day_of_week'] <= 4].copy()
    
    # Filtrar apenas registros com target válido
    if 'target' in df.columns:
        df = df[df['target'].notna()].copy()
    
    # Filtrar apenas registros com coordenadas válidas
    if 'stop_lat_event' in df.columns and 'stop_lon_event' in df.columns:
        df = df[
            df['stop_lat_event'].notna() & 
            df['stop_lon_event'].notna()
        ].copy()
    
    print(f"   ✓ {len(df):,} registros carregados do CSV")
    
else:
    print("   ☁️  Carregando do BigQuery...")
    client = bigquery.Client(project='proj-ml-469320')
    
    # Amostra estratificada APENAS HORÁRIO COMERCIAL (5h-23h) + DIAS ÚTEIS
    query = """
    WITH sampled_data AS (
        SELECT *,
               ROW_NUMBER() OVER (PARTITION BY time_hour ORDER BY RAND()) as rn,
               COUNT(*) OVER (PARTITION BY gtfs_stop_id) as stop_frequency
        FROM `proj-ml-469320.app_cittamobi.dataset-updated`
        WHERE target IS NOT NULL
          AND stop_lat_event IS NOT NULL
          AND stop_lon_event IS NOT NULL
          AND time_hour >= 5            -- Horário comercial (5h às 23h)
          AND time_hour <= 23
          AND time_day_of_week <= 4     -- Segunda a Sexta (0=segunda, 4=sexta)
    )
    SELECT * EXCEPT(rn)
    FROM sampled_data
    WHERE rn <= 700                     -- Mais amostras por hora
      AND stop_frequency >= 50          -- Apenas paradas com tráfego significativo
    """
    
    df = client.query(query).to_dataframe()
    
    if 'stop_frequency' in df.columns:
        df = df.drop('stop_frequency', axis=1)
    
    print(f"   ✓ {len(df):,} registros carregados do BigQuery")

print(f"   - Paradas únicas: {df['gtfs_stop_id'].nunique() if 'gtfs_stop_id' in df.columns else 'N/A'}")
print(f"   - Horas: {sorted(df['time_hour'].unique()) if 'time_hour' in df.columns else 'N/A'}")

# ===========================================================================
# 3. CRIAR FEATURES (PHASE 1: GEOGRAPHIC)
# ===========================================================================
print("\n[3/6] Criando features geográficas...")

def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# Stop Historical Conversion
stop_conversion = df.groupby('gtfs_stop_id')['target'].mean().to_dict()
df['stop_historical_conversion'] = df['gtfs_stop_id'].map(stop_conversion)

# Stop Density
coords_df = df[['stop_lat_event', 'stop_lon_event']].drop_duplicates().dropna()
if len(coords_df) > 1:
    nn = NearestNeighbors(n_neighbors=min(11, len(coords_df)), metric='euclidean')
    nn.fit(coords_df)
    distances, _ = nn.kneighbors(df[['stop_lat_event', 'stop_lon_event']].values)
    df['stop_density'] = 1 / (distances.mean(axis=1) + 0.001)

# Distance to CBD
cbd_coords = [
    (-23.5505, -46.6333), (-22.9068, -43.1729), (-19.9167, -43.9345),
    (-25.4284, -49.2733), (-30.0346, -51.2177),
]
min_distances = []
for cbd_lat, cbd_lon in cbd_coords:
    dist = haversine_vectorized(df['stop_lat_event'], df['stop_lon_event'], cbd_lat, cbd_lon)
    min_distances.append(dist)
df['dist_to_nearest_cbd'] = np.minimum.reduce(min_distances)

# Stop Clustering
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

# Stop Volatility
stop_volatility = df.groupby('gtfs_stop_id')['target'].std().fillna(0).to_dict()
df['stop_volatility'] = df['gtfs_stop_id'].map(stop_volatility)

print(f"✅ 6 features geográficas criadas")

# ===========================================================================
# 4. CRIAR FEATURES (PHASE 2A: DYNAMIC)
# ===========================================================================
print("\n[4/6] Criando features dinâmicas...")

df['hour_conversion_rate'] = df.groupby('time_hour')['target'].transform('mean')
df['dow_conversion_rate'] = df.groupby('time_day_of_week')['target'].transform('mean')
df['stop_hour_conversion'] = df.groupby(['gtfs_stop_id', 'time_hour'])['target'].transform('mean')

df['geo_temporal'] = df['dist_to_nearest_cbd'] * df['is_peak_hour']
df['density_peak'] = df['stop_density'] * df['is_peak_hour']

if 'device_id' in df.columns:
    user_conversion = df.groupby('device_id')['target'].mean().to_dict()
    df['user_conversion_rate'] = df['device_id'].map(user_conversion)
    user_stop_ratio = (df.groupby('device_id')['gtfs_stop_id'].nunique() / df.groupby('device_id').size()).to_dict()
    df['user_vs_stop_ratio'] = df['device_id'].map(user_stop_ratio)
else:
    df['user_conversion_rate'] = df['stop_historical_conversion']
    df['user_vs_stop_ratio'] = 0.5

stop_counts = df.groupby('gtfs_stop_id').size().to_dict()
df['stop_event_count'] = df['gtfs_stop_id'].map(stop_counts)
df['stop_rarity'] = 1 / (df['stop_event_count'] + 1)

if 'device_id' in df.columns:
    user_counts = df.groupby('device_id').size().to_dict()
    df['user_frequency'] = df['device_id'].map(user_counts)
    df['user_rarity'] = 1 / (df['user_frequency'] + 1)
else:
    df['user_rarity'] = 0.01

# Distance deviation
if 'device_id' in df.columns:
    stop_device_agg = df.groupby(['gtfs_stop_id', 'device_id']).agg({
        'stop_lat_event': 'mean', 'stop_lon_event': 'mean'
    }).reset_index()
    stop_device_agg.columns = ['gtfs_stop_id', 'device_id', 'stop_lat_mean', 'stop_lon_mean']
    
    stop_agg = df.groupby('gtfs_stop_id').agg({
        'stop_lat_event': ['mean', 'std'], 'stop_lon_event': ['mean', 'std']
    }).reset_index()
    stop_agg.columns = ['gtfs_stop_id', 'stop_lat_mean_all', 'stop_lat_std', 'stop_lon_mean_all', 'stop_lon_std']
    
    merged = stop_device_agg.merge(stop_agg, on='gtfs_stop_id', how='left')
    merged['stop_dist_std'] = merged['stop_lat_std'].fillna(0) + merged['stop_lon_std'].fillna(0)
    
    df = df.merge(merged[['gtfs_stop_id', 'device_id', 'stop_dist_std']], on=['gtfs_stop_id', 'device_id'], how='left')
    df['stop_dist_std'].fillna(0, inplace=True)
else:
    df['stop_dist_std'] = 0.0

print(f"✅ 10 features dinâmicas criadas")

# ===========================================================================
# 5. GERAR PREDIÇÕES COM O MODELO
# ===========================================================================
print("\n[5/6] Gerando predições com o modelo V8...")

# Preparar features
exclude_cols = ['target', 'gtfs_stop_id', 'timestamp_converted', 'device_id',
                'stop_lat_event', 'stop_lon_event', 'stop_event_count',
                'user_frequency', 'event_timestamp', 'date', 'user_pseudo_id',
                'ctm_service_route', 'direction', 'lotacao_proxy_binaria',
                'y_pred_proba', 'y_pred']

feature_cols = [col for col in df.columns if col not in exclude_cols]
X = df[feature_cols].copy()
X = X.select_dtypes(include=[np.number])

# Limpar nomes
X.columns = X.columns.str.replace('[', '_', regex=False).str.replace(']', '_', regex=False)

# Verificar features faltantes
missing_features = set(selected_features) - set(X.columns)
if missing_features:
    for feat in missing_features:
        X[feat] = 0

X = X[selected_features].copy()
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(0, inplace=True)

# Normalizar e predizer
X_scaled = scaler.transform(X)

lgb_pred = lgb_model.predict(X_scaled)
X_df = pd.DataFrame(X_scaled, columns=selected_features)
xgb_pred = xgb_model.predict(xgb.DMatrix(X_df))

# Ensemble
df['lotacao_prevista'] = (
    ensemble_weights['lightgbm'] * lgb_pred +
    ensemble_weights['xgboost'] * xgb_pred
)

print(f"✅ Predições geradas!")
print(f"   - Lotação média prevista: {df['lotacao_prevista'].mean():.2%}")
print(f"   - Mínima: {df['lotacao_prevista'].min():.2%}")
print(f"   - Máxima: {df['lotacao_prevista'].max():.2%}")

# ===========================================================================
# 6. CRIAR MAPA ANIMADO (COM TODAS AS COMBINAÇÕES PARADA × HORA)
# ===========================================================================
print("\n[6/6] Criando mapa animado...")

# PASSO 1: Calcular coordenada média por gtfs_stop_id (chave única)
print("   🔍 Calculando coordenada média por parada (gtfs_stop_id)...")
coordenadas_por_parada = df.groupby('gtfs_stop_id').agg({
    'stop_lat_event': 'mean',
    'stop_lon_event': 'mean'
}).reset_index()

print(f"   ✓ {len(coordenadas_por_parada)} paradas únicas identificadas por gtfs_stop_id")

# PASSO 2: Agregar predições por gtfs_stop_id e hora
print("   📊 Agregando predições por parada e hora...")
df_agg = df.groupby(['gtfs_stop_id', 'time_hour']).agg({
    'lotacao_prevista': 'mean'
}).reset_index()

# PASSO 3: Selecionar apenas paradas com boa cobertura temporal
stop_hour_count = df_agg.groupby('gtfs_stop_id')['time_hour'].nunique()
valid_stops = stop_hour_count[stop_hour_count >= 10].index.tolist()

df_filtered = df_agg[df_agg['gtfs_stop_id'].isin(valid_stops)].copy()
print(f"   ✓ {len(valid_stops)} paradas com cobertura temporal >= 10 horas")

# PASSO 4: Adicionar coordenadas médias às predições
df_filtered = df_filtered.merge(coordenadas_por_parada, on='gtfs_stop_id', how='inner')

# PASSO 5: Criar lista única de paradas (garantida sem duplicatas)
paradas_unicas = coordenadas_por_parada[
    coordenadas_por_parada['gtfs_stop_id'].isin(valid_stops)
].copy()

print(f"   ✓ {len(paradas_unicas)} paradas finais selecionadas")

# VERIFICAÇÃO: Garantir que não há duplicatas de gtfs_stop_id
assert paradas_unicas['gtfs_stop_id'].nunique() == len(paradas_unicas), "❌ ERRO: Duplicatas de gtfs_stop_id!"
print(f"   ✅ Verificação OK: Cada gtfs_stop_id aparece uma única vez")

# PASSO 6: Criar TODAS as combinações de parada × hora (APENAS HORÁRIO COMERCIAL 5h-23h)
all_hours = list(range(5, 24))  # 5h às 23h (horário de operação real)
all_combinations = []

for _, parada in paradas_unicas.iterrows():
    for hour in all_hours:
        all_combinations.append({
            'gtfs_stop_id': parada['gtfs_stop_id'],
            'time_hour': hour,
            'stop_lat_event': parada['stop_lat_event'],
            'stop_lon_event': parada['stop_lon_event']
        })

df_complete = pd.DataFrame(all_combinations)

# PASSO 7: Fazer merge com predições existentes
df_map = df_complete.merge(
    df_filtered[['gtfs_stop_id', 'time_hour', 'lotacao_prevista']],
    on=['gtfs_stop_id', 'time_hour'],
    how='left'
)

# PASSO 8: Para horas sem dados, usar a MÉDIA da parada ao longo do dia
media_por_parada = df_filtered.groupby('gtfs_stop_id')['lotacao_prevista'].mean().to_dict()
df_map['lotacao_prevista'] = df_map.apply(
    lambda row: row['lotacao_prevista'] if pd.notna(row['lotacao_prevista']) 
                else media_por_parada.get(row['gtfs_stop_id'], 0.05),
    axis=1
)

# VERIFICAÇÃO FINAL: Garantir que cada (gtfs_stop_id, time_hour) aparece uma única vez
duplicates = df_map.groupby(['gtfs_stop_id', 'time_hour']).size()
assert duplicates.max() == 1, f"❌ ERRO: Encontradas {(duplicates > 1).sum()} combinações duplicadas!"

print(f"\n   ✅ GRADE COMPLETA CRIADA: {len(df_map):,} pontos")
print(f"   - Paradas únicas (gtfs_stop_id): {df_map['gtfs_stop_id'].nunique()}")
print(f"   - Horas por parada: {len(all_hours)}")
print(f"   - Total esperado: {df_map['gtfs_stop_id'].nunique()} × {len(all_hours)} = {df_map['gtfs_stop_id'].nunique() * len(all_hours)}")
print(f"   - Cobertura: 100% ✅")

def get_color_by_lotacao(lotacao):
    """Cor baseada no nível de lotação prevista"""
    if lotacao < 0.10:
        return '#00ff00'  # Verde claro
    elif lotacao < 0.20:
        return '#7fff00'  # Verde-amarelo
    elif lotacao < 0.30:
        return '#ffff00'  # Amarelo
    elif lotacao < 0.40:
        return '#ffa500'  # Laranja
    elif lotacao < 0.50:
        return '#ff6600'  # Laranja escuro
    elif lotacao < 0.60:
        return '#ff0000'  # Vermelho
    else:
        return '#8b0000'  # Vermelho escuro

# Criar mapa
m = folium.Map(
    location=[df_map['stop_lat_event'].mean(), df_map['stop_lon_event'].mean()],
    zoom_start=12,
    tiles='CartoDB dark_matter'  # Fundo escuro para destacar os pontos
)

# Preparar features GeoJSON
features = []
seen_keys = {}  # Para detectar duplicatas

print("\n   🔍 Gerando features GeoJSON...")
for idx, row in df_map.iterrows():
    hour = int(row['time_hour'])
    lotacao = float(row['lotacao_prevista'])
    stop_id = row['gtfs_stop_id']
    
    # Verificar duplicatas
    key = (stop_id, hour)
    if key in seen_keys:
        print(f"   ⚠️  DUPLICATA DETECTADA: Stop {stop_id}, Hora {hour}")
        print(f"      - Anterior: lotação={seen_keys[key]:.2%}")
        print(f"      - Atual: lotação={lotacao:.2%}")
        continue  # Pular duplicata
    seen_keys[key] = lotacao
    
    # Determinar tamanho baseado em lotação
    radius = 5 + (lotacao * 20)  # Varia de 5 a 25
    
    feature = {
        'type': 'Feature',
        'geometry': {
            'type': 'Point',
            'coordinates': [float(row['stop_lon_event']), float(row['stop_lat_event'])]
        },
        'properties': {
            'time': f"2024-01-01T{hour:02d}:00:00",
            'gtfs_stop_id': str(stop_id),
            'lotacao': lotacao,
            'popup': f"""
                <div style="font-family: Arial; font-size: 12px;">
                    <b style="font-size: 14px;">🚏 Ponto de Ônibus</b><br>
                    <hr style="margin: 5px 0;">
                    <b>🕐 Horário:</b> {hour:02d}:00h<br>
                    <b>📊 Lotação Prevista:</b> <span style="color: {get_color_by_lotacao(lotacao)}; font-weight: bold;">{lotacao:.1%}</span><br>
                    <b>📍 ID:</b> {stop_id}<br>
                </div>
            """,
            'icon': 'circle',
            'iconstyle': {
                'fillColor': get_color_by_lotacao(lotacao),
                'color': 'white',
                'fillOpacity': 0.8,
                'weight': 2,
                'radius': radius
            }
        }
    }
    features.append(feature)

print(f"   ✓ {len(features)} features geradas (de {len(df_map)} linhas no DataFrame)")

# Adicionar TimestampedGeoJson
# IMPORTANTE: duration deve ser menor que period para evitar sobreposição
TimestampedGeoJson(
    {'type': 'FeatureCollection', 'features': features},
    period='PT1H',           # Período de cada timestamp (1 hora)
    duration='PT30M',        # Duração de exibição (30 min) - MENOR que period!
    add_last_point=False,    # Não adicionar último ponto (evita duplicação)
    auto_play=False,
    loop=True,
    max_speed=2,
    loop_button=True,
    date_options='HH:mm',
    time_slider_drag_update=True
).add_to(m)

# Adicionar legenda
legend_html = '''
<div style="position: fixed; 
            bottom: 50px; right: 50px; width: 280px; 
            background-color: white; border: 2px solid grey; 
            z-index: 9999; font-size: 14px; padding: 15px;
            border-radius: 10px; box-shadow: 0 0 15px rgba(0,0,0,0.5);">
    <h4 style="margin-top: 0; text-align: center; color: #333;">
        📊 Legenda de Lotação Prevista
    </h4>
    <hr style="margin: 10px 0;">
    <p><span style="color: #00ff00; font-size: 20px;">●</span> <b>0-10%</b> Muito Baixa</p>
    <p><span style="color: #7fff00; font-size: 20px;">●</span> <b>10-20%</b> Baixa</p>
    <p><span style="color: #ffff00; font-size: 20px;">●</span> <b>20-30%</b> Moderada</p>
    <p><span style="color: #ffa500; font-size: 20px;">●</span> <b>30-40%</b> Média-Alta</p>
    <p><span style="color: #ff6600; font-size: 20px;">●</span> <b>40-50%</b> Alta</p>
    <p><span style="color: #ff0000; font-size: 20px;">●</span> <b>50-60%</b> Muito Alta</p>
    <p><span style="color: #8b0000; font-size: 20px;">●</span> <b>> 60%</b> Crítica</p>
    <hr style="margin: 10px 0;">
    <p style="font-size: 11px; color: #666; text-align: center;">
        🎬 Use o slider para navegar pelas horas<br>
        ⚡ Tamanho do círculo = intensidade
    </p>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# Salvar
output_file = 'predicoes_modelo_v8.html'
m.save(output_file)

# Estatísticas por hora
hourly_stats = df_map.groupby('time_hour')['lotacao_prevista'].agg(['mean', 'std', 'min', 'max', 'count']).round(4)
hourly_stats.columns = ['média', 'desvio', 'mínima', 'máxima', 'n_paradas']

print("\n📊 LOTAÇÃO PREVISTA POR HORA:")
print("="*80)
print(hourly_stats.to_string())

hourly_stats.to_csv('lotacao_prevista_por_hora_v8.csv')

# Encontrar horários de pico
hora_pico = hourly_stats['média'].idxmax()
hora_baixa = hourly_stats['média'].idxmin()

print("\n" + "="*80)
print("✅ MAPA DE PREDIÇÕES CRIADO COM SUCESSO!")
print("="*80)
print(f"📁 Arquivo: {output_file}")
print(f"📊 Total de predições: {len(df_map):,}")
print(f"📍 Paradas analisadas: {df_map['gtfs_stop_id'].nunique()}")
print(f"🕐 Horas cobertas: {df_map['time_hour'].nunique()}")
print(f"\n🔝 Horário de pico: {hora_pico}h (lotação média: {hourly_stats.loc[hora_pico, 'média']:.1%})")
print(f"🔽 Horário mais vazio: {hora_baixa}h (lotação média: {hourly_stats.loc[hora_baixa, 'média']:.1%})")
print("\n🎬 Abra o arquivo HTML para visualizar as predições do modelo!")
print("="*80)
