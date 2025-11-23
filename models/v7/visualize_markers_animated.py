"""
═══════════════════════════════════════════════════════════════════════════════
MAPA ANIMADO COM MARCADORES: PONTOS DE ÔNIBUS COM PREDIÇÃO TEMPORAL
═══════════════════════════════════════════════════════════════════════════════

🗺️ Mapa interativo com animação temporal mostrando:
   - Marcadores individuais para cada parada
   - Cores mudando ao longo do dia conforme lotação
   - Popups clicáveis com informações detalhadas
   - Slider temporal para controlar a animação
   - Todos os recursos do primeiro mapa + animação

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
import warnings

warnings.filterwarnings('ignore')

print("="*80)
print("🗺️  MAPA ANIMADO COM MARCADORES INDIVIDUAIS")
print("="*80)
print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# ===========================================================================
# ETAPA 1: CARREGAR MODELOS TREINADOS
# ===========================================================================
print("\n[1/5] Carregando modelos treinados...")

try:
    lgb_model = lgb.Booster(model_file='lightgbm_model_v7_FINAL.txt')
    xgb_model = xgb.Booster()
    xgb_model.load_model('xgboost_model_v7_FINAL.json')
    scaler = joblib.load('scaler_v7_FINAL.pkl')
    
    with open('model_config_v7_FINAL.json', 'r') as f:
        config = json.load(f)
    
    with open('selected_features_v7_FINAL.txt', 'r') as f:
        selected_features = [line.strip() for line in f if line.strip()]
    
    ensemble_weights = config['ensemble']['weights']
    threshold = config['ensemble']['threshold']
    
    print(f"✅ Modelos carregados!")
    
except Exception as e:
    print(f"❌ Erro: {e}")
    exit(1)

# ===========================================================================
# ETAPA 2: CARREGAR DADOS DAS PARADAS
# ===========================================================================
print("\n[2/5] Carregando dados das paradas...")

project_id = "proj-ml-469320"
client = bigquery.Client(project=project_id)

query = """
WITH stop_stats AS (
    SELECT 
        stop_lat_event,
        stop_lon_event,
        COUNT(*) as total_events,
        SUM(CAST(target AS INT64)) as total_conversions,
        AVG(CAST(target AS INT64)) as conversion_rate,
        AVG(headway_avg_stop_hour) as avg_headway,
        AVG(dist_device_stop) as avg_distance,
        COUNTIF(is_peak_hour = 1) as peak_events
    FROM `proj-ml-469320.app_cittamobi.dataset-updated`
    WHERE stop_lat_event IS NOT NULL 
      AND stop_lon_event IS NOT NULL
    GROUP BY stop_lat_event, stop_lon_event
    HAVING total_events >= 10
)
SELECT * FROM stop_stats
ORDER BY total_events DESC
LIMIT 100  -- 100 paradas para performance otimizada
"""

print("⏳ Carregando top 100 paradas...")
df_stops = client.query(query).to_dataframe()
print(f"✅ {len(df_stops)} paradas carregadas!")

# ===========================================================================
# ETAPA 3: GERAR PREDIÇÕES PARA CADA HORA DO DIA
# ===========================================================================
print("\n[3/5] Gerando predições para cada hora (0h-23h)...")

temporal_data = []

for hour in range(24):
    print(f"   Processando hora {hour:02d}:00...", end='\r')
    
    df_hour = df_stops.copy()
    
    # Features temporais
    df_hour['hour'] = hour
    df_hour['day_of_week'] = 2
    df_hour['is_weekend'] = 0
    df_hour['is_peak'] = 1 if hour in [7, 8, 9, 17, 18, 19] else 0
    
    # Features cíclicas
    df_hour['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df_hour['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df_hour['day_sin'] = np.sin(2 * np.pi * 2 / 7)
    df_hour['day_cos'] = np.cos(2 * np.pi * 2 / 7)
    
    # Features de interação
    df_hour['dist_x_peak'] = df_hour['avg_distance'] * df_hour['is_peak']
    df_hour['headway_x_weekend'] = df_hour['avg_headway'] * df_hour['is_weekend']
    df_hour['conversion_interaction'] = df_hour['conversion_rate'] * df_hour['avg_headway']
    
    # Preparar features
    X_pred = pd.DataFrame()
    for feat in selected_features:
        if feat in df_hour.columns:
            X_pred[feat] = df_hour[feat]
        else:
            X_pred[feat] = 0
    
    # Normalizar e prever
    X_pred_scaled = scaler.transform(X_pred)
    pred_lgb = lgb_model.predict(X_pred_scaled)
    pred_xgb = xgb_model.predict(xgb.DMatrix(X_pred, feature_names=selected_features))
    pred_ensemble = (ensemble_weights['lightgbm'] * pred_lgb + 
                     ensemble_weights['xgboost'] * pred_xgb)
    
    # Adicionar à lista temporal
    df_hour['hour'] = hour
    df_hour['prob_conversao'] = pred_ensemble
    df_hour['nivel_lotacao'] = pd.cut(pred_ensemble, 
                                       bins=[0, 0.3, 0.5, 0.7, 1.0],
                                       labels=['Baixa', 'Média', 'Alta', 'Muito Alta'])
    
    temporal_data.append(df_hour[['stop_lat_event', 'stop_lon_event', 'hour', 
                                   'prob_conversao', 'nivel_lotacao', 'total_events',
                                   'conversion_rate', 'avg_headway', 'avg_distance']])

print(f"\n✅ Predições geradas para 24 horas!")

# Concatenar todos os dados
df_temporal = pd.concat(temporal_data, ignore_index=True)

# ===========================================================================
# ETAPA 4: CRIAR MAPA BASE
# ===========================================================================
print("\n[4/5] Criando mapa com marcadores animados...")

center_lat = df_stops['stop_lat_event'].mean()
center_lon = df_stops['stop_lon_event'].mean()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=12,
    tiles='OpenStreetMap'
)

folium.TileLayer('CartoDB positron', name='Light Mode').add_to(m)
folium.TileLayer('CartoDB dark_matter', name='Dark Mode').add_to(m)

# ===========================================================================
# ETAPA 5: CRIAR FEATURES GEOJSON PARA ANIMAÇÃO TEMPORAL
# ===========================================================================
print("\n[5/5] Criando animação com marcadores...")

def get_color(prob):
    """Retorna cor baseada na probabilidade"""
    if prob >= 0.7:
        return 'red'
    elif prob >= 0.5:
        return 'orange'
    elif prob >= 0.3:
        return 'yellow'
    else:
        return 'green'

def get_icon(prob):
    """Retorna ícone baseado na lotação"""
    if prob >= 0.7:
        return 'exclamation-sign'
    elif prob >= 0.5:
        return 'warning-sign'
    elif prob >= 0.3:
        return 'info-sign'
    else:
        return 'ok-sign'

# Preparar dados no formato GeoJSON para TimestampedGeoJson
features = []

for _, stop in df_stops.iterrows():
    # Pegar dados desta parada ao longo do dia
    stop_timeline = df_temporal[
        (df_temporal['stop_lat_event'] == stop['stop_lat_event']) &
        (df_temporal['stop_lon_event'] == stop['stop_lon_event'])
    ].sort_values('hour')
    
    # Calcular estatísticas
    peak_hour = stop_timeline.loc[stop_timeline['prob_conversao'].idxmax(), 'hour']
    peak_prob = stop_timeline['prob_conversao'].max()
    min_prob = stop_timeline['prob_conversao'].min()
    avg_prob = stop_timeline['prob_conversao'].mean()
    
    # Criar feature para cada timestamp (hora)
    for _, row in stop_timeline.iterrows():
        # Timestamp no formato ISO
        base_date = datetime(2025, 11, 23)
        timestamp = (base_date + timedelta(hours=int(row['hour']))).isoformat()
        
        # Cor e ícone para esta hora
        color = get_color(row['prob_conversao'])
        icon = get_icon(row['prob_conversao'])
        
        # Criar popup HTML
        popup_html = f"""
        <div style="font-family: Arial; width: 320px;">
            <h4 style="margin: 0; color: #2c3e50;">🚏 Parada</h4>
            <p style="margin: 5px 0; color: #7f8c8d; font-size: 11px;">
                📍 {stop['stop_lat_event']:.4f}, {stop['stop_lon_event']:.4f}
            </p>
            <hr style="margin: 10px 0;">
            
            <div style="background: #ecf0f1; padding: 10px; border-radius: 5px; margin: 10px 0;">
                <h5 style="margin: 0 0 10px 0; color: #e74c3c;">🕐 HORÁRIO: {int(row['hour']):02d}:00</h5>
                <p style="margin: 5px 0; font-size: 14px;">
                    <b>Probabilidade de Conversão:</b> 
                    <span style="color: {color}; font-size: 18px; font-weight: bold;">{row['prob_conversao']:.1%}</span>
                </p>
                <p style="margin: 5px 0; font-size: 14px;">
                    <b>Nível de Lotação:</b> <span style="color: {color};">{row['nivel_lotacao']}</span>
                </p>
            </div>
            
            <div style="background: #e8f4f8; padding: 10px; border-radius: 5px; margin: 10px 0;">
                <h5 style="margin: 0 0 10px 0; color: #3498db;">📊 PERFIL DO DIA</h5>
                <p style="margin: 5px 0; font-size: 13px;">
                    <b>Média diária:</b> {avg_prob:.1%}
                </p>
                <p style="margin: 5px 0; font-size: 13px;">
                    <b>Pico:</b> {peak_hour:02d}:00 ({peak_prob:.1%})
                </p>
                <p style="margin: 5px 0; font-size: 13px;">
                    <b>Variação:</b> {min_prob:.1%} - {peak_prob:.1%}
                </p>
            </div>
            
            <div style="background: #fff3cd; padding: 10px; border-radius: 5px;">
                <h5 style="margin: 0 0 10px 0; color: #856404;">📈 ESTATÍSTICAS</h5>
                <p style="margin: 5px 0; font-size: 12px;">
                    <b>Total eventos:</b> {stop['total_events']:,}
                </p>
                <p style="margin: 5px 0; font-size: 12px;">
                    <b>Taxa conversão:</b> {stop['conversion_rate']:.1%}
                </p>
                <p style="margin: 5px 0; font-size: 12px;">
                    <b>Intervalo médio:</b> {stop['avg_headway']:.0f} min
                </p>
            </div>
            
            <div style="margin-top: 10px; padding: 8px; background: {color}; border-radius: 3px;">
                <p style="margin: 0; font-size: 11px; color: white; font-weight: bold; text-align: center;">
                    {'🔥 ALTA DEMANDA!' if row['prob_conversao'] > 0.7 
                     else '⚠️ ATENÇÃO' if row['prob_conversao'] > 0.5
                     else '✅ NORMAL' if row['prob_conversao'] > 0.3
                     else '💚 TRANQUILO'}
                </p>
            </div>
        </div>
        """
        
        # Feature GeoJSON
        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [stop['stop_lon_event'], stop['stop_lat_event']]
            },
            'properties': {
                'time': timestamp,
                'popup': popup_html,
                'icon': 'circle',
                'iconstyle': {
                    'fillColor': color,
                    'fillOpacity': 0.8,
                    'stroke': 'true',
                    'color': 'white',
                    'weight': 2,
                    'radius': 8
                },
                'style': {
                    'color': color,
                    'weight': 2,
                    'fillColor': color,
                    'fillOpacity': 0.7
                }
            }
        }
        
        features.append(feature)

print(f"✅ {len(features)} marcadores temporais criados!")

# Criar TimestampedGeoJson
TimestampedGeoJson(
    {
        'type': 'FeatureCollection',
        'features': features
    },
    period='PT1H',  # Período de 1 hora
    add_last_point=True,
    auto_play=True,
    loop=True,
    max_speed=2,
    loop_button=True,
    date_options='HH:mm',
    time_slider_drag_update=True,
    duration='PT1H'
).add_to(m)

# ===========================================================================
# ADICIONAR CONTROLES E LEGENDA
# ===========================================================================

folium.LayerControl(position='topright', collapsed=False).add_to(m)

# Título
title_html = '''
<div style="position: fixed; 
            top: 10px; 
            left: 50%; 
            transform: translateX(-50%);
            width: 600px; 
            background-color: rgba(0,0,0,0.85); 
            border: 2px solid #3498db; 
            z-index: 9999; 
            font-size: 16px;
            padding: 15px;
            text-align: center;
            border-radius: 10px;
            box-shadow: 0 0 30px rgba(52,152,219,0.5);
            color: white;">
    <h3 style="margin: 0; color: #3498db;">🚍 CITTAMOBI - ANIMAÇÃO TEMPORAL COM MARCADORES</h3>
    <p style="margin: 5px 0 0 0; font-size: 14px; color: #bdc3c7;">
        100 Paradas | 24 Horas | 2.400 Predições | Modelo V7 Ensemble
    </p>
</div>
'''
m.get_root().html.add_child(folium.Element(title_html))

# Legenda
legend_html = f'''
<div style="position: fixed; 
            bottom: 50px; 
            left: 50px; 
            width: 280px; 
            background-color: white; 
            border: 2px solid grey; 
            z-index: 9999; 
            font-size: 14px;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 0 15px rgba(0,0,0,0.2);">
    <h4 style="margin: 0 0 10px 0; color: #2c3e50;">🗺️ LEGENDA - LOTAÇÃO</h4>
    <p style="margin: 5px 0;"><span style="color: green;">●</span> <b>Baixa</b> (0-30%)</p>
    <p style="margin: 5px 0;"><span style="color: yellow;">●</span> <b>Média</b> (30-50%)</p>
    <p style="margin: 5px 0;"><span style="color: orange;">●</span> <b>Alta</b> (50-70%)</p>
    <p style="margin: 5px 0;"><span style="color: red;">●</span> <b>Muito Alta</b> (70-100%)</p>
    <hr>
    <p style="margin: 5px 0; font-size: 12px; color: #7f8c8d;">
        <b>Total de paradas:</b> {len(df_stops)}<br>
        <b>Horas simuladas:</b> 24h<br>
    </p>
    <p style="margin: 10px 0 0 0; font-size: 11px; color: #95a5a6; text-align: center;">
        🖱️ Clique nos marcadores para detalhes
    </p>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# Instruções
instructions_html = '''
<div style="position: fixed; 
            bottom: 50px; 
            right: 20px; 
            width: 280px; 
            background-color: rgba(255,255,255,0.95); 
            border: 2px solid #3498db; 
            z-index: 9999; 
            font-size: 13px;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.3);">
    <h4 style="margin: 0 0 10px 0; color: #3498db;">🎮 COMO USAR</h4>
    <ul style="margin: 0; padding-left: 20px; line-height: 1.8;">
        <li><b>▶️ Play:</b> Inicia animação automática</li>
        <li><b>⏸️ Pause:</b> Pausa para análise</li>
        <li><b>🎚️ Slider:</b> Navegue entre as horas</li>
        <li><b>🖱️ Clique:</b> Veja detalhes da parada</li>
        <li><b>🔄 Loop:</b> Repete continuamente</li>
        <li><b>🗺️ Camadas:</b> Troque o estilo do mapa</li>
    </ul>
    <div style="margin-top: 15px; padding: 10px; background: #e8f4f8; border-left: 4px solid #3498db; border-radius: 3px;">
        <p style="margin: 0; font-size: 11px; color: #2c3e50;">
            <b>💡 Dica:</b> Observe como os marcadores mudam de cor ao longo do dia!
        </p>
    </div>
</div>
'''
m.get_root().html.add_child(folium.Element(instructions_html))

# ===========================================================================
# SALVAR MAPA
# ===========================================================================
output_file = 'mapa_marcadores_animado_temporal.html'
m.save(output_file)

print("\n" + "="*80)
print("✅ MAPA ANIMADO COM MARCADORES CRIADO COM SUCESSO!")
print("="*80)
print(f"\n📂 Arquivo: {output_file}")
print(f"🌐 Abrindo no navegador...")

# Estatísticas
print("\n📊 ESTATÍSTICAS:")
print(f"   - Paradas: {len(df_stops)}")
print(f"   - Horas: 24")
print(f"   - Total marcadores: {len(features)}")
print(f"   - Predições: {len(df_temporal):,}")

# Análise temporal resumida
print("\n🔥 VARIAÇÃO TEMPORAL:")
for hour in [0, 7, 8, 12, 17, 18, 23]:
    hour_data = df_temporal[df_temporal['hour'] == hour]
    avg = hour_data['prob_conversao'].mean()
    status = "🔴" if hour in [7,8,17,18] else "🟢"
    print(f"   {status} {hour:02d}:00 - {avg:.1%} lotação média")

print("\n" + "="*80)
print("🎬 Abra o arquivo para ver marcadores animados!")
print("   - Cada marcador muda de cor conforme a hora")
print("   - Clique para ver informações completas")
print("   - Use o slider para controlar o tempo")
print("="*80)

# Abrir no navegador
import subprocess
subprocess.run(['open', output_file])
