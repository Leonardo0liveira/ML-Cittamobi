"""
═══════════════════════════════════════════════════════════════════════════════
VISUALIZAÇÃO ANIMADA: PONTOS DE ÔNIBUS COM PREDIÇÃO TEMPORAL
═══════════════════════════════════════════════════════════════════════════════

🎬 Animação temporal mostrando:
   - Como a lotação varia ao longo do dia (0h às 23h)
   - Predição hora a hora para cada ponto
   - Visualização dinâmica das mudanças
   - Slider interativo para controlar o tempo

📊 Baseado no Modelo V7 Ensemble treinado

═══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
from google.cloud import bigquery
import folium
from folium.plugins import TimestampedGeoJson, HeatMapWithTime
import lightgbm as lgb
import xgboost as xgb
import joblib
import json
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

print("="*80)
print("🎬 VISUALIZAÇÃO ANIMADA: PREDIÇÃO TEMPORAL DE LOTAÇÃO")
print("="*80)
print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# ===========================================================================
# ETAPA 1: CARREGAR MODELOS TREINADOS
# ===========================================================================
print("\n[1/6] Carregando modelos treinados...")

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
# ETAPA 2: CARREGAR DADOS DAS PARADAS (SAMPLE REPRESENTATIVO)
# ===========================================================================
print("\n[2/6] Carregando dados das paradas...")

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
        COUNTIF(is_peak_hour = 1) as peak_events,
        
        -- Agregações por hora para criar perfil temporal
        COUNTIF(time_hour BETWEEN 0 AND 5) as events_dawn,
        COUNTIF(time_hour BETWEEN 6 AND 11) as events_morning,
        COUNTIF(time_hour BETWEEN 12 AND 17) as events_afternoon,
        COUNTIF(time_hour BETWEEN 18 AND 23) as events_night
        
    FROM `proj-ml-469320.app_cittamobi.dataset-updated`
    WHERE stop_lat_event IS NOT NULL 
      AND stop_lon_event IS NOT NULL
    GROUP BY stop_lat_event, stop_lon_event
    HAVING total_events >= 50  -- Apenas paradas com bom volume
)
SELECT * FROM stop_stats
ORDER BY total_events DESC
LIMIT 200  -- Top 200 paradas (para performance da animação)
"""

print("⏳ Carregando top 200 paradas...")
df_stops = client.query(query).to_dataframe()
print(f"✅ {len(df_stops)} paradas carregadas!")

# ===========================================================================
# ETAPA 3: GERAR PREDIÇÕES PARA CADA HORA DO DIA
# ===========================================================================
print("\n[3/6] Gerando predições para cada hora (0h-23h)...")

# Lista para armazenar todas as predições temporais
temporal_data = []

# Para cada hora do dia (0-23)
for hour in range(24):
    print(f"   Processando hora {hour:02d}:00...", end='\r')
    
    # Criar features para esta hora
    df_hour = df_stops.copy()
    
    # Features temporais
    df_hour['hour'] = hour
    df_hour['day_of_week'] = 2  # Terça-feira (dia típico)
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
    
    # Preparar features (preencher faltantes com 0)
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
                                   'conversion_rate', 'avg_headway']])

print(f"\n✅ Predições geradas para 24 horas!")

# Concatenar todos os dados
df_temporal = pd.concat(temporal_data, ignore_index=True)

# ===========================================================================
# ETAPA 4: CRIAR MAPA BASE
# ===========================================================================
print("\n[4/6] Criando mapa base...")

center_lat = df_stops['stop_lat_event'].mean()
center_lon = df_stops['stop_lon_event'].mean()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=12,
    tiles='CartoDB dark_matter'
)

folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(m)
folium.TileLayer('CartoDB positron', name='Light Mode').add_to(m)

# ===========================================================================
# ETAPA 5: CRIAR ANIMAÇÃO COM HEATMAPWITHTIME
# ===========================================================================
print("\n[5/6] Criando animação temporal (heatmap)...")

# Preparar dados para HeatMapWithTime
# Formato: lista de listas, uma para cada timestamp
heatmap_data = []
timestamps = []

for hour in range(24):
    # Filtrar dados desta hora
    df_hour = df_temporal[df_temporal['hour'] == hour]
    
    # Criar lista de pontos [lat, lon, intensidade]
    hour_data = [
        [row['stop_lat_event'], row['stop_lon_event'], row['prob_conversao']]
        for _, row in df_hour.iterrows()
    ]
    
    heatmap_data.append(hour_data)
    timestamps.append(f"{hour:02d}:00")

# Adicionar HeatMapWithTime
HeatMapWithTime(
    heatmap_data,
    index=timestamps,
    name='Lotação ao Longo do Dia',
    auto_play=True,
    max_opacity=0.8,
    radius=20,
    blur=25,
    gradient={
        0.0: 'green',
        0.3: 'yellow',
        0.5: 'orange',
        0.7: 'red',
        1.0: 'darkred'
    },
    display_index=True,
    scale_radius=True,
    position='bottomleft'
).add_to(m)

print("✅ Heatmap temporal criado!")

# ===========================================================================
# ETAPA 6: ADICIONAR MARCADORES COM INFO
# ===========================================================================
print("\n[6/6] Adicionando marcadores interativos...")

# Adicionar marcadores para as top 50 paradas mais movimentadas
top_stops = df_stops.nlargest(50, 'total_events')

for idx, row in top_stops.iterrows():
    # Pegar predições desta parada ao longo do dia
    stop_timeline = df_temporal[
        (df_temporal['stop_lat_event'] == row['stop_lat_event']) &
        (df_temporal['stop_lon_event'] == row['stop_lon_event'])
    ].sort_values('hour')
    
    # Calcular estatísticas temporais
    peak_hour = stop_timeline.loc[stop_timeline['prob_conversao'].idxmax(), 'hour']
    peak_prob = stop_timeline['prob_conversao'].max()
    min_prob = stop_timeline['prob_conversao'].min()
    avg_prob = stop_timeline['prob_conversao'].mean()
    
    # Criar gráfico sparkline (mini-gráfico de tendência)
    sparkline_values = stop_timeline['prob_conversao'].values
    sparkline_html = ""
    for i, val in enumerate(sparkline_values):
        height = int(val * 50)  # Escalar para pixels
        color = 'green' if val < 0.3 else 'yellow' if val < 0.5 else 'orange' if val < 0.7 else 'red'
        sparkline_html += f'<div style="display:inline-block; width:3px; height:{height}px; background:{color}; margin-right:1px; vertical-align:bottom;"></div>'
    
    popup_html = f"""
    <div style="font-family: Arial; width: 350px;">
        <h4 style="margin: 0; color: #2c3e50;">🚏 Parada Top {idx+1}</h4>
        <p style="margin: 5px 0; color: #7f8c8d; font-size: 11px;">
            📍 {row['stop_lat_event']:.4f}, {row['stop_lon_event']:.4f}
        </p>
        <hr style="margin: 10px 0;">
        
        <div style="background: #1a1a1a; padding: 10px; border-radius: 5px; margin: 10px 0;">
            <h5 style="margin: 0 0 10px 0; color: #3498db;">📈 PERFIL TEMPORAL (24h)</h5>
            <div style="background: #2c2c2c; padding: 5px; border-radius: 3px; margin-bottom: 10px;">
                {sparkline_html}
            </div>
            <p style="margin: 5px 0; font-size: 13px; color: #ecf0f1;">
                <b>Pico:</b> <span style="color: #e74c3c;">{peak_hour:02d}:00 ({peak_prob:.0%})</span>
            </p>
            <p style="margin: 5px 0; font-size: 13px; color: #ecf0f1;">
                <b>Média diária:</b> {avg_prob:.0%}
            </p>
            <p style="margin: 5px 0; font-size: 13px; color: #ecf0f1;">
                <b>Variação:</b> {min_prob:.0%} - {peak_prob:.0%}
            </p>
        </div>
        
        <div style="background: #e8f4f8; padding: 10px; border-radius: 5px;">
            <h5 style="margin: 0 0 10px 0; color: #2c3e50;">📊 ESTATÍSTICAS</h5>
            <p style="margin: 5px 0; font-size: 13px;">
                <b>Total eventos:</b> {row['total_events']:,}
            </p>
            <p style="margin: 5px 0; font-size: 13px;">
                <b>Taxa conversão:</b> {row['conversion_rate']:.1%}
            </p>
            <p style="margin: 5px 0; font-size: 13px;">
                <b>Intervalo médio:</b> {row['avg_headway']:.0f} min
            </p>
        </div>
        
        <div style="margin-top: 10px; padding: 10px; background: {'#ff4444' if peak_prob > 0.7 else '#ffaa00' if peak_prob > 0.5 else '#44ff44'}; border-radius: 3px;">
            <p style="margin: 0; font-size: 12px; color: white; font-weight: bold; text-align: center;">
                {'🔥 ALTA DEMANDA NO PICO!' if peak_prob > 0.7 
                 else '⚠️ DEMANDA MODERADA' if peak_prob > 0.5
                 else '✅ DEMANDA CONTROLADA'}
            </p>
        </div>
    </div>
    """
    
    # Cor baseada na média
    color = 'red' if avg_prob > 0.7 else 'orange' if avg_prob > 0.5 else 'yellow' if avg_prob > 0.3 else 'green'
    
    folium.CircleMarker(
        location=[row['stop_lat_event'], row['stop_lon_event']],
        radius=8,
        popup=folium.Popup(popup_html, max_width=400),
        tooltip=f"🚏 Parada Top {idx+1} - Pico: {peak_hour:02d}:00 ({peak_prob:.0%})",
        color=color,
        fill=True,
        fillColor=color,
        fillOpacity=0.7,
        weight=2
    ).add_to(m)

print("✅ Marcadores adicionados!")

# ===========================================================================
# ADICIONAR LEGENDA E CONTROLES
# ===========================================================================

folium.LayerControl(position='topright', collapsed=False).add_to(m)

legend_html = f'''
<div style="position: fixed; 
            top: 10px; 
            left: 50%; 
            transform: translateX(-50%);
            width: 700px; 
            background-color: rgba(0,0,0,0.85); 
            border: 2px solid #3498db; 
            z-index: 9999; 
            font-size: 16px;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 0 30px rgba(52,152,219,0.5);
            color: white;">
    <h3 style="margin: 0 0 10px 0; color: #3498db; text-align: center;">
        🎬 CITTAMOBI - ANIMAÇÃO TEMPORAL DE LOTAÇÃO
    </h3>
    <div style="display: flex; justify-content: space-around; margin-bottom: 10px;">
        <div style="text-align: center;">
            <p style="margin: 0; font-size: 24px; font-weight: bold; color: #2ecc71;">{len(df_stops)}</p>
            <p style="margin: 0; font-size: 12px; color: #bdc3c7;">Paradas</p>
        </div>
        <div style="text-align: center;">
            <p style="margin: 0; font-size: 24px; font-weight: bold; color: #e74c3c;">24h</p>
            <p style="margin: 0; font-size: 12px; color: #bdc3c7;">Predições</p>
        </div>
        <div style="text-align: center;">
            <p style="margin: 0; font-size: 24px; font-weight: bold; color: #f39c12;">{len(df_temporal):,}</p>
            <p style="margin: 0; font-size: 12px; color: #bdc3c7;">Data Points</p>
        </div>
    </div>
    <hr style="border-color: #34495e; margin: 10px 0;">
    <div style="display: flex; justify-content: space-around; font-size: 12px;">
        <span><span style="color: green;">●</span> Baixa (0-30%)</span>
        <span><span style="color: yellow;">●</span> Média (30-50%)</span>
        <span><span style="color: orange;">●</span> Alta (50-70%)</span>
        <span><span style="color: red;">●</span> Muito Alta (70-100%)</span>
    </div>
    <p style="margin: 10px 0 0 0; font-size: 11px; color: #95a5a6; text-align: center;">
        ▶️ Use o slider no canto inferior esquerdo para navegar no tempo | 🖱️ Clique nos marcadores para detalhes
    </p>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# Instruções de uso
instructions_html = '''
<div style="position: fixed; 
            bottom: 80px; 
            right: 20px; 
            width: 300px; 
            background-color: rgba(255,255,255,0.95); 
            border: 2px solid #e74c3c; 
            z-index: 9999; 
            font-size: 13px;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.3);">
    <h4 style="margin: 0 0 10px 0; color: #e74c3c;">🎮 CONTROLES</h4>
    <ul style="margin: 0; padding-left: 20px; line-height: 1.8;">
        <li><b>▶️ Play/Pause:</b> Inicia/pausa animação</li>
        <li><b>🎚️ Slider:</b> Navega entre as horas</li>
        <li><b>⏩ Velocidade:</b> Ajusta velocidade da animação</li>
        <li><b>🖱️ Zoom:</b> Aproxima para ver detalhes</li>
        <li><b>📍 Marcadores:</b> Clique para perfil temporal</li>
    </ul>
    <div style="margin-top: 15px; padding: 10px; background: #fff3cd; border-left: 4px solid #ffc107; border-radius: 3px;">
        <p style="margin: 0; font-size: 11px; color: #856404;">
            <b>💡 Dica:</b> Observe como a lotação aumenta nos horários de pico (7h-9h e 17h-19h)!
        </p>
    </div>
</div>
'''
m.get_root().html.add_child(folium.Element(instructions_html))

# ===========================================================================
# SALVAR MAPA
# ===========================================================================
output_file = 'mapa_lotacao_animado_temporal.html'
m.save(output_file)

print("\n" + "="*80)
print("✅ ANIMAÇÃO TEMPORAL CRIADA COM SUCESSO!")
print("="*80)
print(f"\n📂 Arquivo: {output_file}")
print(f"🌐 Abrindo no navegador...")

# Estatísticas finais
print("\n📊 ESTATÍSTICAS DA ANIMAÇÃO:")
print(f"   - Paradas analisadas: {len(df_stops)}")
print(f"   - Marcadores top: {len(top_stops)}")
print(f"   - Frames temporais: 24 horas")
print(f"   - Total de predições: {len(df_temporal):,}")

# Análise de variação temporal
print("\n🔥 ANÁLISE TEMPORAL:")
for hour in [0, 6, 7, 8, 12, 17, 18, 19, 23]:
    hour_data = df_temporal[df_temporal['hour'] == hour]
    avg = hour_data['prob_conversao'].mean()
    status = "🔴 PICO" if hour in [7,8,17,18,19] else "🟢 NORMAL" if hour in [0,1,2,3,4,5,23] else "🟡 MODERADO"
    print(f"   {hour:02d}:00 - {avg:.1%} lotação média {status}")

print("\n" + "="*80)
print("🎬 Abra o arquivo para ver a animação em ação!")
print("="*80)

# Abrir no navegador
import subprocess
subprocess.run(['open', output_file])
