"""
═══════════════════════════════════════════════════════════════════════════════
MAPA DE PARADAS COM ALTA CONVERSÃO REAL (DADOS HISTÓRICOS)
═══════════════════════════════════════════════════════════════════════════════

🎯 Mostra paradas com TAXA REAL de conversão alta no histórico
   Compara: Taxa Real vs Predição do Modelo

═══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
from google.cloud import bigquery
import folium
from folium.plugins import MarkerCluster, HeatMap
import lightgbm as lgb
import xgboost as xgb
import joblib
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

print("="*80)
print("🎯 ANÁLISE: TAXA REAL vs PREDIÇÃO DO MODELO")
print("="*80)

# Carregar modelos
print("\n[1/4] Carregando modelos...")
lgb_model = lgb.Booster(model_file='lightgbm_model_v7_FINAL.txt')
xgb_model = xgb.Booster()
xgb_model.load_model('xgboost_model_v7_FINAL.json')
scaler = joblib.load('scaler_v7_FINAL.pkl')

with open('model_config_v7_FINAL.json', 'r') as f:
    config = json.load(f)

with open('selected_features_v7_FINAL.txt', 'r') as f:
    selected_features = [line.strip() for line in f if line.strip()]

ensemble_weights = config['ensemble']['weights']
print("✅ Modelos carregados!")

# Carregar dados REAIS
print("\n[2/4] Carregando dados reais de conversão...")
client = bigquery.Client(project='proj-ml-469320')

query = """
SELECT 
    stop_lat_event,
    stop_lon_event,
    COUNT(*) as total_events,
    SUM(CAST(target AS INT64)) as conversoes,
    AVG(CAST(target AS INT64)) as taxa_conversao_real,
    AVG(headway_avg_stop_hour) as avg_headway,
    AVG(dist_device_stop) as avg_distance,
    COUNTIF(is_peak_hour = 1) as peak_events,
    -- Análise por período
    COUNTIF(is_peak_hour = 1 AND target = 1) as conversoes_peak,
    COUNTIF(is_peak_hour = 0 AND target = 1) as conversoes_offpeak
FROM `proj-ml-469320.app_cittamobi.dataset-updated`
WHERE stop_lat_event IS NOT NULL 
  AND stop_lon_event IS NOT NULL
GROUP BY stop_lat_event, stop_lon_event
HAVING total_events >= 50  -- Paradas com volume significativo
ORDER BY taxa_conversao_real DESC
LIMIT 200
"""

df_stops = client.query(query).to_dataframe()
print(f"✅ {len(df_stops)} paradas carregadas!")
print(f"   Taxa de conversão real: {df_stops['taxa_conversao_real'].min():.1%} - {df_stops['taxa_conversao_real'].max():.1%}")

# Gerar predições do modelo
print("\n[3/4] Gerando predições do modelo para comparação...")

hour = 17  # Horário de pico
df_stops['hour'] = hour
df_stops['day_of_week'] = 2
df_stops['is_weekend'] = 0
df_stops['is_peak'] = 1

# Features cíclicas
df_stops['hour_sin'] = np.sin(2 * np.pi * hour / 24)
df_stops['hour_cos'] = np.cos(2 * np.pi * hour / 24)
df_stops['day_sin'] = np.sin(2 * np.pi * 2 / 7)
df_stops['day_cos'] = np.cos(2 * np.pi * 2 / 7)

# Features de interação
df_stops['dist_x_peak'] = df_stops['avg_distance'] * df_stops['is_peak']
df_stops['headway_x_weekend'] = df_stops['avg_headway'] * df_stops['is_weekend']
df_stops['conversion_interaction'] = df_stops['taxa_conversao_real'] * df_stops['avg_headway']

# Preparar features
X_pred = pd.DataFrame()
for feat in selected_features:
    if feat in df_stops.columns:
        X_pred[feat] = df_stops[feat]
    else:
        X_pred[feat] = 0

# Prever
X_pred_scaled = scaler.transform(X_pred)
pred_lgb = lgb_model.predict(X_pred_scaled)
pred_xgb = xgb_model.predict(xgb.DMatrix(X_pred, feature_names=selected_features))
pred_ensemble = (ensemble_weights['lightgbm'] * pred_lgb + 
                 ensemble_weights['xgboost'] * pred_xgb)

df_stops['pred_modelo'] = pred_ensemble
df_stops['diferenca'] = df_stops['taxa_conversao_real'] - df_stops['pred_modelo']
df_stops['erro_abs'] = np.abs(df_stops['diferenca'])

print(f"✅ Predições geradas!")

# Análise
print("\n📊 COMPARAÇÃO:")
print(f"   Taxa real média: {df_stops['taxa_conversao_real'].mean():.1%}")
print(f"   Predição média: {df_stops['pred_modelo'].mean():.1%}")
print(f"   Erro médio absoluto: {df_stops['erro_abs'].mean():.1%}")
print(f"   Correlação: {df_stops['taxa_conversao_real'].corr(df_stops['pred_modelo']):.3f}")

# Categorizar paradas
df_stops['categoria_real'] = pd.cut(df_stops['taxa_conversao_real'], 
                                     bins=[0, 0.1, 0.3, 0.5, 1.0],
                                     labels=['Baixa', 'Média', 'Alta', 'Muito Alta'])

df_stops['categoria_pred'] = pd.cut(df_stops['pred_modelo'],
                                     bins=[0, 0.1, 0.3, 0.5, 1.0],
                                     labels=['Baixa', 'Média', 'Alta', 'Muito Alta'])

print("\n🎯 DISTRIBUIÇÃO REAL:")
print(df_stops['categoria_real'].value_counts().sort_index())

print("\n🤖 DISTRIBUIÇÃO PREDITA:")
print(df_stops['categoria_pred'].value_counts().sort_index())

# Criar mapa
print("\n[4/4] Criando mapa comparativo...")

center_lat = df_stops['stop_lat_event'].mean()
center_lon = df_stops['stop_lon_event'].mean()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=12,
    tiles='OpenStreetMap'
)

folium.TileLayer('CartoDB dark_matter', name='Dark Mode').add_to(m)

# Função de cor baseada na TAXA REAL
def get_color_real(taxa):
    if taxa >= 0.5:
        return 'red'
    elif taxa >= 0.3:
        return 'orange'
    elif taxa >= 0.1:
        return 'yellow'
    else:
        return 'green'

# Adicionar marcadores
for _, row in df_stops.iterrows():
    color_real = get_color_real(row['taxa_conversao_real'])
    
    # Tamanho do marcador proporcional à taxa real
    radius = 5 + (row['taxa_conversao_real'] * 20)
    
    popup_html = f"""
    <div style="font-family: Arial; width: 350px;">
        <h4 style="margin: 0; color: #2c3e50;">🚏 Análise de Conversão</h4>
        <p style="margin: 5px 0; color: #7f8c8d; font-size: 11px;">
            📍 {row['stop_lat_event']:.4f}, {row['stop_lon_event']:.4f}
        </p>
        <hr style="margin: 10px 0;">
        
        <div style="background: #ffe6e6; padding: 10px; border-radius: 5px; margin: 10px 0; border-left: 4px solid {color_real};">
            <h5 style="margin: 0 0 10px 0; color: #c0392b;">📊 TAXA REAL (Histórico)</h5>
            <p style="margin: 5px 0; font-size: 18px;">
                <b style="color: {color_real}; font-size: 24px;">{row['taxa_conversao_real']:.1%}</b>
            </p>
            <p style="margin: 5px 0; font-size: 13px;">
                <b>Categoria:</b> <span style="color: {color_real};">{row['categoria_real']}</span>
            </p>
            <p style="margin: 5px 0; font-size: 12px; color: #7f8c8d;">
                {row['conversoes']:.0f} conversões em {row['total_events']:.0f} eventos
            </p>
        </div>
        
        <div style="background: #e3f2fd; padding: 10px; border-radius: 5px; margin: 10px 0;">
            <h5 style="margin: 0 0 10px 0; color: #1976d2;">🤖 PREDIÇÃO DO MODELO</h5>
            <p style="margin: 5px 0; font-size: 18px;">
                <b style="color: #1976d2; font-size: 20px;">{row['pred_modelo']:.1%}</b>
            </p>
            <p style="margin: 5px 0; font-size: 13px;">
                <b>Categoria:</b> {row['categoria_pred']}
            </p>
        </div>
        
        <div style="background: {'#ffebee' if row['erro_abs'] > 0.2 else '#e8f5e9'}; padding: 10px; border-radius: 5px;">
            <h5 style="margin: 0 0 10px 0; color: {'#c62828' if row['erro_abs'] > 0.2 else '#2e7d32'};">
                ⚖️ DIFERENÇA
            </h5>
            <p style="margin: 5px 0; font-size: 16px;">
                <b>Erro:</b> <span style="color: {'red' if row['diferenca'] > 0 else 'green'}; font-weight: bold;">
                    {row['diferenca']:+.1%}
                </span>
            </p>
            <p style="margin: 5px 0; font-size: 12px;">
                {'🔴 Modelo SUBESTIMOU' if row['diferenca'] > 0.2 
                 else '🟡 Pequena diferença' if abs(row['diferenca']) <= 0.2
                 else '🟢 Modelo SUPERESTIMOU'}
            </p>
        </div>
        
        <div style="background: #fff3cd; padding: 8px; border-radius: 5px; margin: 10px 0;">
            <p style="margin: 5px 0; font-size: 12px;">
                <b>📈 Análise temporal:</b><br>
                • Conversões no pico: {row['conversoes_peak']:.0f}<br>
                • Conversões fora pico: {row['conversoes_offpeak']:.0f}<br>
                • Intervalo médio: {row['avg_headway']:.0f} min
            </p>
        </div>
    </div>
    """
    
    folium.CircleMarker(
        location=[row['stop_lat_event'], row['stop_lon_event']],
        radius=radius,
        popup=folium.Popup(popup_html, max_width=400),
        color='white',
        weight=2,
        fill=True,
        fillColor=color_real,
        fillOpacity=0.8,
        tooltip=f"Real: {row['taxa_conversao_real']:.1%} | Modelo: {row['pred_modelo']:.1%}"
    ).add_to(m)

# Adicionar heatmap da taxa REAL
heat_data = [[row['stop_lat_event'], row['stop_lon_event'], row['taxa_conversao_real']] 
             for _, row in df_stops.iterrows()]
HeatMap(heat_data, radius=15, blur=20, max_zoom=13).add_to(m)

# Controles
folium.LayerControl().add_to(m)

# Título
title_html = '''
<div style="position: fixed; 
            top: 10px; 
            left: 50%; 
            transform: translateX(-50%);
            width: 700px; 
            background-color: rgba(0,0,0,0.9); 
            border: 3px solid #e74c3c; 
            z-index: 9999; 
            padding: 15px;
            text-align: center;
            border-radius: 10px;
            color: white;">
    <h3 style="margin: 0; color: #e74c3c;">🎯 TAXA REAL vs PREDIÇÃO DO MODELO</h3>
    <p style="margin: 5px 0; font-size: 13px; color: #bdc3c7;">
        Comparação: Conversão Histórica Real vs Predição ML (Horário: 17h)
    </p>
</div>
'''
m.get_root().html.add_child(folium.Element(title_html))

# Legenda
legend_html = f'''
<div style="position: fixed; 
            bottom: 50px; 
            left: 50px; 
            width: 320px; 
            background-color: white; 
            border: 2px solid grey; 
            z-index: 9999; 
            padding: 15px;
            border-radius: 5px;">
    <h4 style="margin: 0 0 10px 0; color: #2c3e50;">🎯 TAXA DE CONVERSÃO REAL</h4>
    <p style="margin: 5px 0;"><span style="color: green;">●</span> <b>Baixa</b> (0-10%)</p>
    <p style="margin: 5px 0;"><span style="color: yellow;">●</span> <b>Média</b> (10-30%)</p>
    <p style="margin: 5px 0;"><span style="color: orange;">●</span> <b>Alta</b> (30-50%)</p>
    <p style="margin: 5px 0;"><span style="color: red;">●</span> <b>Muito Alta</b> (50-100%)</p>
    <hr>
    <h5 style="margin: 10px 0 5px 0; color: #e74c3c;">📊 ESTATÍSTICAS</h5>
    <p style="margin: 3px 0; font-size: 12px;">
        <b>Taxa real média:</b> {df_stops['taxa_conversao_real'].mean():.1%}<br>
        <b>Predição média:</b> {df_stops['pred_modelo'].mean():.1%}<br>
        <b>Erro médio:</b> {df_stops['erro_abs'].mean():.1%}<br>
        <b>Correlação:</b> {df_stops['taxa_conversao_real'].corr(df_stops['pred_modelo']):.3f}
    </p>
    <hr>
    <p style="margin: 5px 0; font-size: 11px; color: #7f8c8d;">
        💡 Tamanho do círculo = Taxa real<br>
        🖱️ Clique para ver comparação detalhada
    </p>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# Salvar
output_file = 'mapa_comparacao_real_vs_modelo.html'
m.save(output_file)

print(f"\n✅ Mapa salvo: {output_file}")

# Top 10 maiores erros
print("\n🔴 TOP 10 PARADAS ONDE O MODELO MAIS ERROU:")
print("="*80)
top_errors = df_stops.nlargest(10, 'erro_abs')[['stop_lat_event', 'stop_lon_event', 
                                                   'taxa_conversao_real', 'pred_modelo', 
                                                   'diferenca', 'total_events']]
print(top_errors.to_string(index=False))

# Top 10 maiores taxas reais
print("\n🔥 TOP 10 PARADAS COM MAIOR TAXA REAL:")
print("="*80)
top_real = df_stops.nlargest(10, 'taxa_conversao_real')[['stop_lat_event', 'stop_lon_event', 
                                                           'taxa_conversao_real', 'pred_modelo',
                                                           'conversoes', 'total_events']]
print(top_real.to_string(index=False))

print("\n" + "="*80)
print("✅ ANÁLISE COMPLETA!")
print("="*80)

import subprocess
subprocess.run(['open', output_file])
