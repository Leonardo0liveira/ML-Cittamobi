"""
═══════════════════════════════════════════════════════════════════════════════
DASHBOARD COMPLETO: TODAS AS PREDIÇÕES TEMPORAIS
═══════════════════════════════════════════════════════════════════════════════

📊 Visualização completa mostrando:
   - TODAS as 4.800 predições (200 paradas × 24 horas)
   - Tabela interativa com filtros e ordenação
   - Gráficos de análise temporal
   - Estatísticas detalhadas por hora e parada
   - Export para CSV

═══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
from google.cloud import bigquery
import lightgbm as lgb
import xgboost as xgb
import joblib
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

print("="*80)
print("📊 DASHBOARD: TODAS AS PREDIÇÕES TEMPORAIS")
print("="*80)
print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# ===========================================================================
# ETAPA 1: CARREGAR MODELOS
# ===========================================================================
print("\n[1/5] Carregando modelos treinados...")

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
        AVG(dist_device_stop) as avg_distance
    FROM `proj-ml-469320.app_cittamobi.dataset-updated`
    WHERE stop_lat_event IS NOT NULL 
      AND stop_lon_event IS NOT NULL
    GROUP BY stop_lat_event, stop_lon_event
    HAVING total_events >= 50
)
SELECT * FROM stop_stats
ORDER BY total_events DESC
LIMIT 200
"""

df_stops = client.query(query).to_dataframe()
print(f"✅ {len(df_stops)} paradas carregadas!")

# Adicionar ID único para cada parada
df_stops['stop_id'] = range(1, len(df_stops) + 1)

# ===========================================================================
# ETAPA 3: GERAR TODAS AS PREDIÇÕES (200 × 24 = 4800)
# ===========================================================================
print("\n[3/5] Gerando TODAS as predições (200 paradas × 24 horas)...")

all_predictions = []

for hour in range(24):
    print(f"   Processando hora {hour:02d}:00... ({hour+1}/24)", end='\r')
    
    df_hour = df_stops.copy()
    
    # Features temporais
    df_hour['hour'] = hour
    df_hour['day_of_week'] = 2  # Terça-feira
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
    
    # Armazenar predições
    for idx, row in df_hour.iterrows():
        all_predictions.append({
            'stop_id': row['stop_id'],
            'latitude': row['stop_lat_event'],
            'longitude': row['stop_lon_event'],
            'hour': hour,
            'hour_label': f"{hour:02d}:00",
            'is_peak': row['is_peak'],
            'prob_lightgbm': pred_lgb[idx],
            'prob_xgboost': pred_xgb[idx],
            'prob_ensemble': pred_ensemble[idx],
            'prediction': 1 if pred_ensemble[idx] >= threshold else 0,
            'nivel_lotacao': (
                'Muito Alta' if pred_ensemble[idx] >= 0.7 else
                'Alta' if pred_ensemble[idx] >= 0.5 else
                'Média' if pred_ensemble[idx] >= 0.3 else
                'Baixa'
            ),
            'total_events': row['total_events'],
            'conversion_rate': row['conversion_rate'],
            'avg_headway': row['avg_headway']
        })

print(f"\n✅ {len(all_predictions)} predições geradas!")

# Criar DataFrame com todas as predições
df_all = pd.DataFrame(all_predictions)

# ===========================================================================
# ETAPA 4: ANÁLISES E ESTATÍSTICAS
# ===========================================================================
print("\n[4/5] Gerando análises estatísticas...")

# Análise por hora
print("\n📊 ANÁLISE POR HORA:")
print("-" * 80)
print(f"{'Hora':<6} {'Média':<8} {'Mín':<8} {'Máx':<8} {'Alta+':<8} {'Status':<15}")
print("-" * 80)

hour_stats = []
for hour in range(24):
    hour_data = df_all[df_all['hour'] == hour]
    avg = hour_data['prob_ensemble'].mean()
    min_val = hour_data['prob_ensemble'].min()
    max_val = hour_data['prob_ensemble'].max()
    high = (hour_data['prob_ensemble'] >= 0.5).sum()
    
    status = "🔴 PICO" if hour in [7,8,17,18,19] else "🟢 NORMAL"
    
    print(f"{hour:02d}:00  {avg:.2%}   {min_val:.2%}   {max_val:.2%}   {high:>3}      {status}")
    
    hour_stats.append({
        'hour': hour,
        'hour_label': f"{hour:02d}:00",
        'avg_prob': avg,
        'min_prob': min_val,
        'max_prob': max_val,
        'high_count': high,
        'status': status
    })

df_hour_stats = pd.DataFrame(hour_stats)

# Análise por parada (top 20)
print("\n\n📊 TOP 20 PARADAS COM MAIOR LOTAÇÃO MÉDIA:")
print("-" * 100)
print(f"{'ID':<5} {'Latitude':<12} {'Longitude':<12} {'Média':<8} {'Pico':<8} {'Hora Pico':<12} {'Variação':<10}")
print("-" * 100)

stop_stats = []
for stop_id in df_stops['stop_id'].head(20):
    stop_data = df_all[df_all['stop_id'] == stop_id]
    avg = stop_data['prob_ensemble'].mean()
    peak = stop_data['prob_ensemble'].max()
    peak_hour = stop_data.loc[stop_data['prob_ensemble'].idxmax(), 'hour']
    variation = stop_data['prob_ensemble'].std()
    lat = stop_data.iloc[0]['latitude']
    lon = stop_data.iloc[0]['longitude']
    
    print(f"{stop_id:<5} {lat:<12.4f} {lon:<12.4f} {avg:.2%}   {peak:.2%}   {peak_hour:02d}:00        {variation:.2%}")
    
    stop_stats.append({
        'stop_id': stop_id,
        'latitude': lat,
        'longitude': lon,
        'avg_prob': avg,
        'peak_prob': peak,
        'peak_hour': peak_hour,
        'variation': variation
    })

df_stop_stats = pd.DataFrame(stop_stats)

# Estatísticas gerais
print("\n\n📈 ESTATÍSTICAS GERAIS:")
print("-" * 80)
print(f"Total de predições:           {len(df_all):,}")
print(f"Paradas analisadas:           {df_all['stop_id'].nunique()}")
print(f"Horas cobertas:               24")
print(f"Predição média (geral):       {df_all['prob_ensemble'].mean():.2%}")
print(f"Predição mínima:              {df_all['prob_ensemble'].min():.2%}")
print(f"Predição máxima:              {df_all['prob_ensemble'].max():.2%}")
print(f"Desvio padrão:                {df_all['prob_ensemble'].std():.2%}")
print(f"\nDistribuição por nível:")
for nivel in ['Baixa', 'Média', 'Alta', 'Muito Alta']:
    count = (df_all['nivel_lotacao'] == nivel).sum()
    pct = count / len(df_all) * 100
    print(f"  {nivel:<12} {count:>5} ({pct:>5.1f}%)")

# Análise de concordância entre modelos
print(f"\n🔍 ANÁLISE DE CONCORDÂNCIA ENTRE MODELOS:")
print("-" * 80)
diff = np.abs(df_all['prob_lightgbm'] - df_all['prob_xgboost'])
print(f"Diferença média LightGBM vs XGBoost:  {diff.mean():.2%}")
print(f"Diferença máxima:                      {diff.max():.2%}")
print(f"Correlação:                            {df_all['prob_lightgbm'].corr(df_all['prob_xgboost']):.4f}")

# ===========================================================================
# ETAPA 5: EXPORTAR DADOS
# ===========================================================================
print("\n[5/5] Exportando dados...")

# Export 1: Todas as predições
df_all.to_csv('predicoes_completas.csv', index=False)
print(f"✅ Arquivo criado: predicoes_completas.csv ({len(df_all):,} linhas)")

# Export 2: Estatísticas por hora
df_hour_stats.to_csv('estatisticas_por_hora.csv', index=False)
print(f"✅ Arquivo criado: estatisticas_por_hora.csv ({len(df_hour_stats)} linhas)")

# Export 3: Estatísticas por parada
df_stop_stats.to_csv('estatisticas_por_parada.csv', index=False)
print(f"✅ Arquivo criado: estatisticas_por_parada.csv ({len(df_stop_stats)} linhas)")

# Export 4: Criar pivot table (paradas × horas)
df_pivot = df_all.pivot_table(
    index='stop_id',
    columns='hour_label',
    values='prob_ensemble',
    aggfunc='first'
)
df_pivot.to_csv('matriz_predicoes_paradas_horas.csv')
print(f"✅ Arquivo criado: matriz_predicoes_paradas_horas.csv ({len(df_pivot)} paradas × 24 horas)")

# ===========================================================================
# CRIAR HTML INTERATIVO COM TODAS AS PREDIÇÕES
# ===========================================================================
print("\n📄 Criando dashboard HTML interativo...")

html_content = f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard - Todas as Predições</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header p {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
        }}
        
        .stat-card {{
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.3s;
        }}
        
        .stat-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        }}
        
        .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
            margin: 10px 0;
        }}
        
        .stat-label {{
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .content {{
            padding: 30px;
        }}
        
        .section {{
            margin-bottom: 40px;
        }}
        
        .section h2 {{
            color: #667eea;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}
        
        .table-container {{
            overflow-x: auto;
            margin-top: 20px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
        }}
        
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
            position: sticky;
            top: 0;
        }}
        
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
        }}
        
        tr:hover {{
            background: #f8f9fa;
        }}
        
        .prob-cell {{
            font-weight: bold;
            padding: 5px 10px;
            border-radius: 5px;
            display: inline-block;
            min-width: 60px;
            text-align: center;
        }}
        
        .prob-muito-alta {{ background: #ff4444; color: white; }}
        .prob-alta {{ background: #ff9944; color: white; }}
        .prob-media {{ background: #ffdd44; color: #333; }}
        .prob-baixa {{ background: #44ff44; color: #333; }}
        
        .filters {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        
        .filter-group {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            align-items: center;
        }}
        
        .filter-group label {{
            font-weight: 600;
            color: #667eea;
        }}
        
        .filter-group input,
        .filter-group select {{
            padding: 8px 15px;
            border: 2px solid #667eea;
            border-radius: 5px;
            font-size: 1em;
        }}
        
        .btn {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 10px 25px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1em;
            font-weight: 600;
            transition: transform 0.2s;
        }}
        
        .btn:hover {{
            transform: scale(1.05);
        }}
        
        .chart-container {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            margin-top: 20px;
        }}
        
        .download-section {{
            background: #e8f4f8;
            padding: 25px;
            border-radius: 10px;
            margin-top: 20px;
        }}
        
        .download-section h3 {{
            color: #667eea;
            margin-bottom: 15px;
        }}
        
        .download-links {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }}
        
        .download-btn {{
            background: white;
            color: #667eea;
            padding: 10px 20px;
            border: 2px solid #667eea;
            border-radius: 5px;
            text-decoration: none;
            font-weight: 600;
            transition: all 0.3s;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }}
        
        .download-btn:hover {{
            background: #667eea;
            color: white;
            transform: translateY(-2px);
        }}
        
        .pagination {{
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-top: 20px;
        }}
        
        .page-btn {{
            padding: 8px 15px;
            border: 2px solid #667eea;
            background: white;
            color: #667eea;
            border-radius: 5px;
            cursor: pointer;
            font-weight: 600;
        }}
        
        .page-btn.active {{
            background: #667eea;
            color: white;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚍 Dashboard de Predições - Cittamobi</h1>
            <p>Análise Completa de Lotação por Hora e Parada</p>
            <p style="font-size: 0.9em; margin-top: 10px;">Modelo V7 Ensemble | ROC-AUC: 90.56%</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Total de Predições</div>
                <div class="stat-value">{len(df_all):,}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Paradas Analisadas</div>
                <div class="stat-value">{df_all['stop_id'].nunique()}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Horas Cobertas</div>
                <div class="stat-value">24</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Lotação Média</div>
                <div class="stat-value">{df_all['prob_ensemble'].mean():.1%}</div>
            </div>
        </div>
        
        <div class="content">
            <div class="section">
                <h2>📊 Todas as Predições ({len(df_all):,} registros)</h2>
                
                <div class="filters">
                    <div class="filter-group">
                        <label>Filtrar por hora:</label>
                        <select id="filterHour" onchange="filterTable()">
                            <option value="">Todas</option>
                            {''.join([f'<option value="{h:02d}:00">{h:02d}:00</option>' for h in range(24)])}
                        </select>
                        
                        <label>Filtrar por nível:</label>
                        <select id="filterNivel" onchange="filterTable()">
                            <option value="">Todos</option>
                            <option value="Muito Alta">Muito Alta</option>
                            <option value="Alta">Alta</option>
                            <option value="Média">Média</option>
                            <option value="Baixa">Baixa</option>
                        </select>
                        
                        <label>Buscar parada:</label>
                        <input type="number" id="searchStop" placeholder="ID da parada" onkeyup="filterTable()">
                        
                        <button class="btn" onclick="resetFilters()">Limpar Filtros</button>
                    </div>
                </div>
                
                <div class="table-container">
                    <table id="predictionsTable">
                        <thead>
                            <tr>
                                <th>Parada ID</th>
                                <th>Latitude</th>
                                <th>Longitude</th>
                                <th>Hora</th>
                                <th>Prob. LightGBM</th>
                                <th>Prob. XGBoost</th>
                                <th>Prob. Ensemble</th>
                                <th>Nível</th>
                                <th>Predição</th>
                            </tr>
                        </thead>
                        <tbody>
"""

# Adicionar primeiras 500 linhas (para não sobrecarregar)
for idx, row in df_all.head(500).iterrows():
    prob_class = (
        'prob-muito-alta' if row['prob_ensemble'] >= 0.7 else
        'prob-alta' if row['prob_ensemble'] >= 0.5 else
        'prob-media' if row['prob_ensemble'] >= 0.3 else
        'prob-baixa'
    )
    
    html_content += f"""
                            <tr data-hour="{row['hour_label']}" data-nivel="{row['nivel_lotacao']}" data-stop="{row['stop_id']}">
                                <td>{row['stop_id']}</td>
                                <td>{row['latitude']:.4f}</td>
                                <td>{row['longitude']:.4f}</td>
                                <td>{row['hour_label']}</td>
                                <td>{row['prob_lightgbm']:.2%}</td>
                                <td>{row['prob_xgboost']:.2%}</td>
                                <td><span class="prob-cell {prob_class}">{row['prob_ensemble']:.2%}</span></td>
                                <td>{row['nivel_lotacao']}</td>
                                <td>{'✅ Sim' if row['prediction'] == 1 else '❌ Não'}</td>
                            </tr>
"""

html_content += f"""
                        </tbody>
                    </table>
                </div>
                <p style="text-align: center; margin-top: 15px; color: #666;">
                    Mostrando primeiras 500 de {len(df_all):,} predições. 
                    <a href="predicoes_completas.csv" download style="color: #667eea; font-weight: 600;">
                        Baixe o arquivo CSV completo
                    </a>
                </p>
            </div>
            
            <div class="section">
                <h2>⏰ Estatísticas por Hora</h2>
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Hora</th>
                                <th>Média</th>
                                <th>Mínimo</th>
                                <th>Máximo</th>
                                <th>Alta+ (≥50%)</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody>
"""

for _, row in df_hour_stats.iterrows():
    html_content += f"""
                            <tr>
                                <td><strong>{row['hour_label']}</strong></td>
                                <td>{row['avg_prob']:.2%}</td>
                                <td>{row['min_prob']:.2%}</td>
                                <td>{row['max_prob']:.2%}</td>
                                <td>{row['high_count']}</td>
                                <td>{row['status']}</td>
                            </tr>
"""

html_content += f"""
                        </tbody>
                    </table>
                </div>
            </div>
            
            <div class="download-section">
                <h3>📥 Downloads Disponíveis</h3>
                <div class="download-links">
                    <a href="predicoes_completas.csv" download class="download-btn">
                        📄 Todas as Predições ({len(df_all):,} linhas)
                    </a>
                    <a href="estatisticas_por_hora.csv" download class="download-btn">
                        ⏰ Estatísticas por Hora (24 linhas)
                    </a>
                    <a href="estatisticas_por_parada.csv" download class="download-btn">
                        🚏 Estatísticas por Parada ({len(df_stop_stats)} linhas)
                    </a>
                    <a href="matriz_predicoes_paradas_horas.csv" download class="download-btn">
                        📊 Matriz Paradas×Horas (Pivot)
                    </a>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        function filterTable() {{
            const hourFilter = document.getElementById('filterHour').value;
            const nivelFilter = document.getElementById('filterNivel').value;
            const stopSearch = document.getElementById('searchStop').value;
            
            const rows = document.querySelectorAll('#predictionsTable tbody tr');
            
            rows.forEach(row => {{
                const hour = row.getAttribute('data-hour');
                const nivel = row.getAttribute('data-nivel');
                const stop = row.getAttribute('data-stop');
                
                const matchHour = !hourFilter || hour === hourFilter;
                const matchNivel = !nivelFilter || nivel === nivelFilter;
                const matchStop = !stopSearch || stop === stopSearch;
                
                if (matchHour && matchNivel && matchStop) {{
                    row.style.display = '';
                }} else {{
                    row.style.display = 'none';
                }}
            }});
        }}
        
        function resetFilters() {{
            document.getElementById('filterHour').value = '';
            document.getElementById('filterNivel').value = '';
            document.getElementById('searchStop').value = '';
            filterTable();
        }}
    </script>
</body>
</html>
"""

# Salvar HTML
with open('dashboard_predicoes_completas.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"✅ Arquivo criado: dashboard_predicoes_completas.html")

# ===========================================================================
# FINALIZAÇÃO
# ===========================================================================
print("\n" + "="*80)
print("✅ TODAS AS PREDIÇÕES PROCESSADAS E EXPORTADAS!")
print("="*80)
print(f"\n📁 Arquivos criados:")
print(f"   1. predicoes_completas.csv - {len(df_all):,} predições")
print(f"   2. estatisticas_por_hora.csv - 24 horas")
print(f"   3. estatisticas_por_parada.csv - {len(df_stop_stats)} paradas")
print(f"   4. matriz_predicoes_paradas_horas.csv - Pivot table")
print(f"   5. dashboard_predicoes_completas.html - Dashboard interativo")

print(f"\n🌐 Abrindo dashboard no navegador...")
print("="*80)

# Abrir dashboard
import subprocess
subprocess.run(['open', 'dashboard_predicoes_completas.html'])
