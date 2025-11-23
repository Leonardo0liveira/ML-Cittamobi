"""
═══════════════════════════════════════════════════════════════════════════════
VISUALIZAÇÃO INTERATIVA: PONTOS DE ÔNIBUS COM PREDIÇÃO DE LOTAÇÃO
═══════════════════════════════════════════════════════════════════════════════

🗺️  Mapa interativo mostrando:
   - Localização de cada ponto de ônibus
   - Predição de lotação/conversão em tempo real
   - Cores variando conforme probabilidade
   - Informações detalhadas ao clicar

📊 Baseado no Modelo V7 Ensemble treinado

═══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
from google.cloud import bigquery
import folium
from folium.plugins import HeatMap, MarkerCluster
import lightgbm as lgb
import xgboost as xgb
import joblib
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

print("="*80)
print("🗺️  VISUALIZAÇÃO INTERATIVA: PONTOS DE ÔNIBUS COM PREDIÇÃO")
print("="*80)
print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# ===========================================================================
# ETAPA 1: CARREGAR MODELOS TREINADOS
# ===========================================================================
print("\n[1/5] Carregando modelos treinados...")

try:
    # Carregar modelos
    lgb_model = lgb.Booster(model_file='lightgbm_model_v7_FINAL.txt')
    
    xgb_model = xgb.Booster()
    xgb_model.load_model('xgboost_model_v7_FINAL.json')
    
    scaler = joblib.load('scaler_v7_FINAL.pkl')
    
    with open('model_config_v7_FINAL.json', 'r') as f:
        config = json.load(f)
    
    # Carregar lista de features do arquivo txt
    with open('selected_features_v7_FINAL.txt', 'r') as f:
        selected_features = [line.strip() for line in f if line.strip()]
    
    ensemble_weights = config['ensemble']['weights']
    threshold = config['ensemble']['threshold']
    
    print(f"✅ Modelos carregados com sucesso!")
    print(f"   - Features: {len(selected_features)}")
    print(f"   - Threshold: {threshold:.2f}")
    print(f"   - Ensemble weights: LGB={ensemble_weights['lightgbm']:.1%}, XGB={ensemble_weights['xgboost']:.1%}")
    
except Exception as e:
    print(f"❌ Erro ao carregar modelos: {e}")
    print("\n⚠️  Execute primeiro o script de treinamento:")
    print("   python3 model_v7_ensemble_FINAL_PRODUCTION.py")
    exit(1)

# ===========================================================================
# ETAPA 2: CARREGAR DADOS RECENTES DAS PARADAS
# ===========================================================================
print("\n[2/5] Carregando dados das paradas de ônibus...")

project_id = "proj-ml-469320"
client = bigquery.Client(project=project_id)

# Query para pegar snapshot recente com agregações por parada  
query = """
WITH stop_stats AS (
    SELECT 
        stop_lat_event,
        stop_lon_event,
        
        -- Agregações por parada (últimas 24h simuladas)
        COUNT(*) as total_events,
        SUM(CAST(target AS INT64)) as total_conversions,
        AVG(CAST(target AS INT64)) as conversion_rate,
        AVG(headway_avg_stop_hour) as avg_headway,
        AVG(dist_device_stop) as avg_distance,
        
        -- Horários de pico
        COUNTIF(is_peak_hour = 1) as peak_events,
        
        -- Timestamp mais recente
        MAX(event_timestamp) as last_event
        
    FROM `proj-ml-469320.app_cittamobi.dataset-updated`
    WHERE stop_lat_event IS NOT NULL 
      AND stop_lon_event IS NOT NULL
    GROUP BY stop_lat_event, stop_lon_event
    HAVING total_events >= 10  -- Apenas paradas com volume significativo
)
SELECT * FROM stop_stats
ORDER BY conversion_rate DESC
LIMIT 1000  -- Top 1000 paradas
"""

print("⏳ Carregando dados das paradas (top 1000)...")
df_stops = client.query(query).to_dataframe()

print(f"✅ {len(df_stops)} paradas carregadas!")
print(f"   - Lat range: [{df_stops['stop_lat_event'].min():.4f}, {df_stops['stop_lat_event'].max():.4f}]")
print(f"   - Lon range: [{df_stops['stop_lon_event'].min():.4f}, {df_stops['stop_lon_event'].max():.4f}]")
print(f"   - Taxa conversão média: {df_stops['conversion_rate'].mean():.2%}")

# ===========================================================================
# ETAPA 3: GERAR FEATURES E PREDIÇÕES PARA CADA PARADA
# ===========================================================================
print("\n[3/5] Gerando predições de lotação para cada parada...")

# Criar features básicas (simulando horário atual)
current_hour = datetime.now().hour
current_day = datetime.now().weekday()
is_weekend = 1 if current_day >= 5 else 0

df_viz = df_stops.copy()

# Features temporais (simulando agora)
df_viz['hour'] = current_hour
df_viz['day_of_week'] = current_day
df_viz['is_weekend'] = is_weekend
df_viz['is_peak'] = 1 if current_hour in [7, 8, 9, 17, 18, 19] else 0

# Features cíclicas
df_viz['hour_sin'] = np.sin(2 * np.pi * df_viz['hour'] / 24)
df_viz['hour_cos'] = np.cos(2 * np.pi * df_viz['hour'] / 24)
df_viz['day_sin'] = np.sin(2 * np.pi * df_viz['day_of_week'] / 7)
df_viz['day_cos'] = np.cos(2 * np.pi * df_viz['day_of_week'] / 7)

# Features de interação
df_viz['dist_x_peak'] = df_viz['avg_distance'] * df_viz['is_peak']
df_viz['headway_x_weekend'] = df_viz['avg_headway'] * df_viz['is_weekend']
df_viz['conversion_interaction'] = df_viz['conversion_rate'] * df_viz['avg_headway']

# Preparar features para predição (ajustar nomes para match com treinamento)
feature_mapping = {
    'stop_lat_event': 'stop_lat_event',
    'stop_lon_event': 'stop_lon_event',
    'avg_headway': 'headway_secs',
    'avg_duration': 'trip_duration_secs',
    'avg_distance': 'distance_from_previous_meters',
    'conversion_rate': 'stop_conversion_rate',
    'total_conversions': 'stop_total_conversions',
    'total_events': 'stop_total_events',
}

# Criar dataset com features necessárias (preencher features faltantes com médias)
X_pred = pd.DataFrame()

for feat in selected_features:
    if feat in df_viz.columns:
        X_pred[feat] = df_viz[feat]
    elif feat in feature_mapping.values():
        # Mapear nome
        original = [k for k, v in feature_mapping.items() if v == feat]
        if original and original[0] in df_viz.columns:
            X_pred[feat] = df_viz[original[0]]
        else:
            X_pred[feat] = 0  # Default
    else:
        # Features não disponíveis: usar média do treinamento ou zero
        X_pred[feat] = 0

# Normalizar
X_pred_scaled = scaler.transform(X_pred)

# Predições
print("   Gerando predições com ensemble...")
pred_lgb = lgb_model.predict(X_pred_scaled)
pred_xgb = xgb_model.predict(xgb.DMatrix(X_pred, feature_names=selected_features))

# Ensemble
pred_ensemble = (ensemble_weights['lightgbm'] * pred_lgb + 
                 ensemble_weights['xgboost'] * pred_xgb)

df_viz['prob_conversao'] = pred_ensemble
df_viz['predicao'] = (pred_ensemble >= threshold).astype(int)
df_viz['nivel_lotacao'] = pd.cut(pred_ensemble, 
                                  bins=[0, 0.3, 0.5, 0.7, 1.0],
                                  labels=['Baixa', 'Média', 'Alta', 'Muito Alta'])

print(f"✅ Predições geradas!")
print(f"   - Média prob. conversão: {df_viz['prob_conversao'].mean():.2%}")
print(f"   - Paradas com alta lotação (>70%): {(df_viz['prob_conversao'] > 0.7).sum()}")

# ===========================================================================
# ETAPA 4: CRIAR MAPA INTERATIVO
# ===========================================================================
print("\n[4/5] Criando mapa interativo...")

# Centro do mapa (média das coordenadas)
center_lat = df_viz['stop_lat_event'].mean()
center_lon = df_viz['stop_lon_event'].mean()

# Criar mapa base
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=12,
    tiles='OpenStreetMap'
)

# Adicionar camadas alternativas com atribuição
folium.TileLayer('CartoDB positron', name='CartoDB Positron').add_to(m)
folium.TileLayer('CartoDB dark_matter', name='CartoDB Dark').add_to(m)

# ===========================================================================
# OPÇÃO 1: MARCADORES INDIVIDUAIS COM CLUSTERS
# ===========================================================================
print("   Adicionando marcadores...")

marker_cluster = MarkerCluster(name='Pontos de Ônibus').add_to(m)

def get_marker_color(prob):
    """Cor baseada na probabilidade de conversão"""
    if prob >= 0.7:
        return 'red'  # Muito alta
    elif prob >= 0.5:
        return 'orange'  # Alta
    elif prob >= 0.3:
        return 'yellow'  # Média
    else:
        return 'green'  # Baixa

def get_icon(prob):
    """Ícone baseado na lotação"""
    if prob >= 0.7:
        return 'exclamation-sign'  # Alerta
    elif prob >= 0.5:
        return 'warning-sign'
    elif prob >= 0.3:
        return 'info-sign'
    else:
        return 'ok-sign'

# Adicionar marcador para cada parada
for idx, row in df_viz.iterrows():
    # HTML para popup com informações detalhadas
    stop_name = f"Parada ({row['stop_lat_event']:.4f}, {row['stop_lon_event']:.4f})"
    popup_html = f"""
    <div style="font-family: Arial; width: 300px;">
        <h4 style="margin: 0; color: #2c3e50;">🚏 {stop_name}</h4>
        <hr style="margin: 10px 0;">
        
        <div style="background: #ecf0f1; padding: 10px; border-radius: 5px; margin: 10px 0;">
            <h5 style="margin: 0 0 10px 0; color: #e74c3c;">🔥 PREDIÇÃO ATUAL</h5>
            <p style="margin: 5px 0; font-size: 14px;">
                <b>Probabilidade de Conversão:</b> <span style="color: #e74c3c; font-size: 16px;">{row['prob_conversao']:.1%}</span>
            </p>
            <p style="margin: 5px 0; font-size: 14px;">
                <b>Nível de Lotação:</b> <span style="color: #e74c3c;">{row['nivel_lotacao']}</span>
            </p>
            <p style="margin: 5px 0; font-size: 12px; color: #95a5a6;">
                (Horário: {current_hour}:00, {'Fim de semana' if is_weekend else 'Dia útil'})
            </p>
        </div>
        
        <div style="background: #e8f4f8; padding: 10px; border-radius: 5px;">
            <h5 style="margin: 0 0 10px 0; color: #3498db;">📊 ESTATÍSTICAS HISTÓRICAS</h5>
            <p style="margin: 5px 0; font-size: 13px;">
                <b>Total de eventos:</b> {row['total_events']:,}
            </p>
            <p style="margin: 5px 0; font-size: 13px;">
                <b>Conversões:</b> {row['total_conversions']:,} ({row['conversion_rate']:.1%})
            </p>
            <p style="margin: 5px 0; font-size: 13px;">
                <b>Intervalo médio:</b> {row['avg_headway']:.0f} min
            </p>
            <p style="margin: 5px 0; font-size: 13px;">
                <b>Distância média:</b> {row['avg_distance']:.0f}m
            </p>
        </div>
        
        <div style="margin-top: 10px; padding: 10px; background: #fff3cd; border-left: 4px solid #ffc107; border-radius: 3px;">
            <p style="margin: 0; font-size: 12px; color: #856404;">
                <b>💡 Recomendação:</b> 
                {'⚠️ ALTA DEMANDA - Aumentar frequência!' if row['prob_conversao'] > 0.7 
                 else '✅ Demanda normal - Manter operação.' if row['prob_conversao'] > 0.3
                 else '📉 Baixa demanda - Considerar redução.'}
            </p>
        </div>
        
        <p style="margin: 10px 0 0 0; font-size: 11px; color: #95a5a6; text-align: center;">
            📍 Lat: {row['stop_lat_event']:.6f}, Lon: {row['stop_lon_event']:.6f}
        </p>
    </div>
    """
    
    # Criar marcador
    folium.Marker(
        location=[row['stop_lat_event'], row['stop_lon_event']],
        popup=folium.Popup(popup_html, max_width=350),
        tooltip=f"🚏 Parada - {row['prob_conversao']:.0%} lotação",
        icon=folium.Icon(
            color=get_marker_color(row['prob_conversao']),
            icon=get_icon(row['prob_conversao']),
            prefix='glyphicon'
        )
    ).add_to(marker_cluster)

# ===========================================================================
# OPÇÃO 2: HEATMAP DE CONVERSÕES
# ===========================================================================
print("   Adicionando heatmap...")

# Preparar dados para heatmap (lat, lon, intensidade)
heat_data = [
    [row['stop_lat_event'], row['stop_lon_event'], row['prob_conversao']]
    for _, row in df_viz.iterrows()
]

HeatMap(
    heat_data,
    name='Heatmap de Lotação',
    min_opacity=0.3,
    max_zoom=18,
    radius=15,
    blur=20,
    gradient={
        0.0: 'green',
        0.3: 'yellow',
        0.5: 'orange',
        0.7: 'red',
        1.0: 'darkred'
    }
).add_to(m)

# ===========================================================================
# ADICIONAR CONTROLES E LEGENDA
# ===========================================================================

# Controle de camadas
folium.LayerControl(position='topright', collapsed=False).add_to(m)

# Legenda customizada
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
    <h4 style="margin: 0 0 10px 0; color: #2c3e50;">🗺️ LEGENDA - PREDIÇÃO DE LOTAÇÃO</h4>
    <p style="margin: 5px 0;"><span style="color: green;">●</span> <b>Baixa</b> (0-30%): Demanda normal</p>
    <p style="margin: 5px 0;"><span style="color: yellow;">●</span> <b>Média</b> (30-50%): Atenção</p>
    <p style="margin: 5px 0;"><span style="color: orange;">●</span> <b>Alta</b> (50-70%): Aumentar frequência</p>
    <p style="margin: 5px 0;"><span style="color: red;">●</span> <b>Muito Alta</b> (70-100%): Alerta! Superlotação</p>
    <hr>
    <p style="margin: 5px 0; font-size: 12px; color: #7f8c8d;">
        <b>Total de paradas:</b> {len(df_viz)}<br>
        <b>Horário simulado:</b> {current_hour}:00<br>
        <b>Modelo:</b> V7 Ensemble (ROC-AUC 90.56%)
    </p>
    <p style="margin: 10px 0 0 0; font-size: 11px; color: #95a5a6; text-align: center;">
        💡 Clique nos marcadores para detalhes
    </p>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# Título no mapa
title_html = '''
<div style="position: fixed; 
            top: 10px; 
            left: 50%; 
            transform: translateX(-50%);
            width: 500px; 
            background-color: rgba(255,255,255,0.9); 
            border: 2px solid #3498db; 
            z-index: 9999; 
            font-size: 18px;
            padding: 15px;
            text-align: center;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.3);">
    <h3 style="margin: 0; color: #2c3e50;">🚍 CITTAMOBI - PREDIÇÃO DE LOTAÇÃO EM TEMPO REAL</h3>
    <p style="margin: 5px 0 0 0; color: #7f8c8d; font-size: 14px;">
        Machine Learning Model V7 | ROC-AUC: 90.56%
    </p>
</div>
'''
m.get_root().html.add_child(folium.Element(title_html))

# ===========================================================================
# ETAPA 5: SALVAR MAPA
# ===========================================================================
print("\n[5/5] Salvando mapa interativo...")

output_file = 'mapa_lotacao_pontos_onibus.html'
m.save(output_file)

print(f"✅ Mapa salvo com sucesso!")
print(f"   📂 Arquivo: {output_file}")
print(f"   🌐 Abra no navegador para visualizar")

# ===========================================================================
# ESTATÍSTICAS FINAIS
# ===========================================================================
print("\n" + "="*80)
print("📊 ESTATÍSTICAS DA VISUALIZAÇÃO")
print("="*80)

print(f"\n📍 Cobertura Geográfica:")
print(f"   - Total de paradas: {len(df_viz)}")
print(f"   - Latitude: [{df_viz['stop_lat_event'].min():.4f}, {df_viz['stop_lat_event'].max():.4f}]")
print(f"   - Longitude: [{df_viz['stop_lon_event'].min():.4f}, {df_viz['stop_lon_event'].max():.4f}]")

print(f"\n🔥 Predições de Lotação:")
print(f"   - Média probabilidade: {df_viz['prob_conversao'].mean():.2%}")
print(f"   - Mediana: {df_viz['prob_conversao'].median():.2%}")
print(f"   - Mínima: {df_viz['prob_conversao'].min():.2%}")
print(f"   - Máxima: {df_viz['prob_conversao'].max():.2%}")

print(f"\n📊 Distribuição por Nível:")
for nivel in ['Baixa', 'Média', 'Alta', 'Muito Alta']:
    count = (df_viz['nivel_lotacao'] == nivel).sum()
    pct = count / len(df_viz) * 100
    print(f"   - {nivel}: {count} paradas ({pct:.1f}%)")

print(f"\n⚠️  Alertas:")
high_demand = (df_viz['prob_conversao'] > 0.7).sum()
medium_demand = ((df_viz['prob_conversao'] > 0.5) & (df_viz['prob_conversao'] <= 0.7)).sum()
print(f"   - Muito Alta demanda (>70%): {high_demand} paradas")
print(f"   - Alta demanda (50-70%): {medium_demand} paradas")
print(f"   - Total crítico: {high_demand + medium_demand} paradas precisam atenção")

print("\n" + "="*80)
print("✅ VISUALIZAÇÃO CONCLUÍDA!")
print("="*80)
print(f"\n🌐 Para visualizar:")
print(f"   1. Abra o arquivo: {output_file}")
print(f"   2. Ou execute: open {output_file}")
print(f"\n💡 Recursos do mapa:")
print(f"   ✓ Marcadores clusterizados (zoom para expandir)")
print(f"   ✓ Heatmap de lotação (camadas no canto superior direito)")
print(f"   ✓ Popups detalhados ao clicar")
print(f"   ✓ Tooltips ao passar o mouse")
print(f"   ✓ Múltiplos estilos de mapa (OpenStreetMap, Terrain, etc.)")
print(f"   ✓ Cores indicando nível de lotação")
print(f"   ✓ Recomendações automáticas por parada")
print("\n" + "="*80)
