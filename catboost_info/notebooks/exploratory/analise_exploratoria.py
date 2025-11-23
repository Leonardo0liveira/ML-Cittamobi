import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from google.cloud import bigquery
import warnings
import os
warnings.filterwarnings('ignore')

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

os.makedirs('visualizations/eda', exist_ok=True)

# ===========================================================================
# ANÁLISE EXPLORATÓRIA DE DADOS - CITTAMOBI
# ===========================================================================
print(f"\n{'='*80}")
print(f"ANÁLISE EXPLORATÓRIA DE DADOS - CITTAMOBI")
print(f"{'='*80}\n")

# ===========================================================================
# ETAPA 1: CARREGAR DADOS
# ===========================================================================
print(f"{'='*70}")
print(f"ETAPA 1: CARREGAR DADOS")
print(f"{'='*70}")

project_id = "proj-ml-469320"
client = bigquery.Client(project=project_id)

query = """
    SELECT * FROM `proj-ml-469320.app_cittamobi.dataset-updated` 
    TABLESAMPLE SYSTEM (20 PERCENT)
    LIMIT 50000
"""

print("Carregando dados do BigQuery...")
df = client.query(query).to_dataframe()
print(f"✓ Dados carregados: {len(df):,} registros")
print(f"✓ Features: {len(df.columns)} colunas\n")

# ===========================================================================
# ETAPA 2: INFORMAÇÕES GERAIS
# ===========================================================================
print(f"{'='*70}")
print(f"ETAPA 2: INFORMAÇÕES GERAIS DO DATASET")
print(f"{'='*70}\n")

print(f"📊 DIMENSÕES:")
print(f"   Linhas:    {df.shape[0]:,}")
print(f"   Colunas:   {df.shape[1]}")
print(f"   Memória:   {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")

print(f"📋 TIPOS DE DADOS:")
type_counts = df.dtypes.value_counts()
for dtype, count in type_counts.items():
    print(f"   {str(dtype):15s}: {count:3d} colunas")

print(f"\n🔍 VALORES AUSENTES:")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'Missing': missing,
    'Percentage': missing_pct
}).sort_values('Missing', ascending=False)

missing_cols = missing_df[missing_df['Missing'] > 0]
if len(missing_cols) > 0:
    print(f"   Colunas com valores ausentes: {len(missing_cols)}")
    for col, row in missing_cols.head(10).iterrows():
        print(f"   - {col:40s}: {int(row['Missing']):6,} ({row['Percentage']:5.2f}%)")
else:
    print(f"   ✓ Nenhum valor ausente!")

# ===========================================================================
# ETAPA 3: ANÁLISE DO TARGET
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 3: ANÁLISE DA VARIÁVEL TARGET")
print(f"{'='*70}\n")

target = 'target'
if target in df.columns:
    target_dist = df[target].value_counts().sort_index()
    target_pct = (df[target].value_counts(normalize=True) * 100).sort_index()
    
    print(f"📊 DISTRIBUIÇÃO DO TARGET:")
    for val in target_dist.index:
        count = target_dist[val]
        pct = target_pct[val]
        bar = '█' * int(pct / 2)
        print(f"   Classe {val}: {count:6,} ({pct:5.2f}%) {bar}")
    
    # Calcular desbalanceamento
    imbalance_ratio = target_dist.max() / target_dist.min()
    print(f"\n⚖️  DESBALANCEAMENTO:")
    print(f"   Razão: {imbalance_ratio:.2f}:1")
    if imbalance_ratio > 3:
        print(f"   ⚠️  Dataset ALTAMENTE DESBALANCEADO!")
    elif imbalance_ratio > 1.5:
        print(f"   ⚠️  Dataset MODERADAMENTE DESBALANCEADO")
    else:
        print(f"   ✓ Dataset BALANCEADO")
    
    # Visualização
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Contagem
    ax1 = axes[0]
    colors = ['#2ecc71', '#e74c3c']
    bars = ax1.bar(target_dist.index, target_dist.values, color=colors, alpha=0.7)
    ax1.set_xlabel('Classe', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Contagem', fontweight='bold', fontsize=12)
    ax1.set_title('Distribuição do Target - Contagem', fontweight='bold', fontsize=14)
    ax1.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, target_dist.values):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 200, f'{val:,}', 
                ha='center', va='bottom', fontweight='bold')
    
    # Percentual
    ax2 = axes[1]
    wedges, texts, autotexts = ax2.pie(target_pct.values, labels=[f'Classe {i}' for i in target_pct.index],
                                         autopct='%1.2f%%', colors=colors, startangle=90,
                                         textprops={'fontweight': 'bold', 'fontsize': 11})
    ax2.set_title('Distribuição do Target - Percentual', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('visualizations/eda/target_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Visualização salva: visualizations/eda/target_distribution.png")

# ===========================================================================
# ETAPA 4: ANÁLISE TEMPORAL
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 4: ANÁLISE TEMPORAL")
print(f"{'='*70}\n")

if 'event_timestamp' in df.columns:
    df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])
    df = df.sort_values('event_timestamp')
    
    print(f"📅 PERÍODO DOS DADOS:")
    print(f"   Início:    {df['event_timestamp'].min()}")
    print(f"   Fim:       {df['event_timestamp'].max()}")
    print(f"   Duração:   {(df['event_timestamp'].max() - df['event_timestamp'].min()).days} dias")
    
    # Features temporais
    df['hour'] = df['event_timestamp'].dt.hour
    df['day_of_week'] = df['event_timestamp'].dt.dayofweek
    df['day_name'] = df['event_timestamp'].dt.day_name()
    df['month'] = df['event_timestamp'].dt.month
    df['date'] = df['event_timestamp'].dt.date
    
    # Visualizações temporais
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Eventos por hora
    ax1 = axes[0, 0]
    hourly = df.groupby('hour').size()
    ax1.plot(hourly.index, hourly.values, marker='o', linewidth=2, color='#3498db', markersize=8)
    ax1.fill_between(hourly.index, hourly.values, alpha=0.3, color='#3498db')
    ax1.set_xlabel('Hora do Dia', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Número de Eventos', fontweight='bold', fontsize=11)
    ax1.set_title('Distribuição de Eventos por Hora', fontweight='bold', fontsize=13)
    ax1.grid(alpha=0.3)
    ax1.set_xticks(range(0, 24))
    
    peak_hour = hourly.idxmax()
    ax1.axvline(peak_hour, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax1.text(peak_hour, hourly.max(), f'Pico: {peak_hour}h', ha='center', va='bottom',
             fontweight='bold', color='red')
    
    # 2. Eventos por dia da semana
    ax2 = axes[0, 1]
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily = df.groupby('day_name').size().reindex(day_order)
    colors_days = ['#e74c3c' if day in ['Saturday', 'Sunday'] else '#3498db' for day in day_order]
    bars = ax2.bar(range(len(daily)), daily.values, color=colors_days, alpha=0.7)
    ax2.set_xlabel('Dia da Semana', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Número de Eventos', fontweight='bold', fontsize=11)
    ax2.set_title('Distribuição de Eventos por Dia da Semana', fontweight='bold', fontsize=13)
    ax2.set_xticks(range(len(daily)))
    ax2.set_xticklabels(['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom'], rotation=0)
    ax2.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, daily.values):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 50, f'{val:,}', 
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 3. Taxa de conversão por hora
    ax3 = axes[1, 0]
    if target in df.columns:
        hourly_conv = df.groupby('hour')[target].mean() * 100
        ax3.plot(hourly_conv.index, hourly_conv.values, marker='s', linewidth=2, 
                color='#2ecc71', markersize=8)
        ax3.fill_between(hourly_conv.index, hourly_conv.values, alpha=0.3, color='#2ecc71')
        ax3.set_xlabel('Hora do Dia', fontweight='bold', fontsize=11)
        ax3.set_ylabel('Taxa de Conversão (%)', fontweight='bold', fontsize=11)
        ax3.set_title('Taxa de Conversão por Hora', fontweight='bold', fontsize=13)
        ax3.grid(alpha=0.3)
        ax3.set_xticks(range(0, 24))
        
        best_hour = hourly_conv.idxmax()
        ax3.axvline(best_hour, color='green', linestyle='--', linewidth=2, alpha=0.7)
        ax3.text(best_hour, hourly_conv.max(), f'{best_hour}h\n{hourly_conv.max():.2f}%', 
                ha='center', va='bottom', fontweight='bold', color='green')
    
    # 4. Série temporal
    ax4 = axes[1, 1]
    daily_events = df.groupby('date').size()
    ax4.plot(daily_events.index, daily_events.values, linewidth=2, color='#9b59b6')
    ax4.fill_between(daily_events.index, daily_events.values, alpha=0.3, color='#9b59b6')
    ax4.set_xlabel('Data', fontweight='bold', fontsize=11)
    ax4.set_ylabel('Número de Eventos', fontweight='bold', fontsize=11)
    ax4.set_title('Série Temporal de Eventos', fontweight='bold', fontsize=13)
    ax4.grid(alpha=0.3)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig('visualizations/eda/temporal_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Visualização salva: visualizations/eda/temporal_analysis.png")
    
    # Estatísticas temporais
    print(f"\n⏰ PADRÕES TEMPORAIS:")
    print(f"   Hora de pico:              {peak_hour}h ({hourly[peak_hour]:,} eventos)")
    if target in df.columns:
        print(f"   Melhor hora (conversão):   {best_hour}h ({hourly_conv[best_hour]:.2f}%)")
    print(f"   Dia mais movimentado:      {daily.idxmax()} ({daily.max():,} eventos)")
    print(f"   Dia menos movimentado:     {daily.idxmin()} ({daily.min():,} eventos)")

# ===========================================================================
# ETAPA 5: ANÁLISE DE FEATURES NUMÉRICAS
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 5: ANÁLISE DE FEATURES NUMÉRICAS")
print(f"{'='*70}\n")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if target in numeric_cols:
    numeric_cols.remove(target)

# Remover IDs
id_cols = ['user_pseudo_id', 'gtfs_stop_id', 'gtfs_route_id', 'session_id']
numeric_cols = [col for col in numeric_cols if col not in id_cols]

print(f"📊 FEATURES NUMÉRICAS: {len(numeric_cols)}")

# Estatísticas descritivas das principais features
main_features = ['user_frequency', 'stop_event_rate', 'stop_event_count', 
                 'stop_total_samples', 'dist_device_stop']
main_features = [col for col in main_features if col in numeric_cols]

if len(main_features) > 0:
    print(f"\n📈 ESTATÍSTICAS DESCRITIVAS (PRINCIPAIS FEATURES):")
    stats = df[main_features].describe().T
    stats['missing'] = df[main_features].isnull().sum()
    stats['missing_pct'] = (stats['missing'] / len(df) * 100).round(2)
    
    print(f"\n{stats.to_string()}")
    
    # Visualizar distribuições
    n_features = min(len(main_features), 6)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(main_features[:n_features]):
        ax = axes[i]
        data = df[col].dropna()
        
        # Histograma
        ax.hist(data, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
        ax.set_xlabel(col, fontweight='bold', fontsize=10)
        ax.set_ylabel('Frequência', fontweight='bold', fontsize=10)
        ax.set_title(f'Distribuição: {col}', fontweight='bold', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        # Estatísticas no gráfico
        mean_val = data.mean()
        median_val = data.median()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Média: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Mediana: {median_val:.2f}')
        ax.legend(fontsize=8)
    
    # Remover eixos vazios
    for i in range(n_features, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig('visualizations/eda/numeric_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Visualização salva: visualizations/eda/numeric_distributions.png")

# ===========================================================================
# ETAPA 6: CORRELAÇÃO COM O TARGET
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 6: CORRELAÇÃO COM O TARGET")
print(f"{'='*70}\n")

if target in df.columns and len(numeric_cols) > 0:
    # Calcular correlações
    correlations = df[numeric_cols + [target]].corr()[target].drop(target).sort_values(ascending=False)
    
    print(f"🔗 TOP 15 CORRELAÇÕES POSITIVAS:")
    for col, corr in correlations.head(15).items():
        bar = '█' * int(abs(corr) * 50)
        print(f"   {col:40s}: {corr:+.4f} {bar}")
    
    print(f"\n🔗 TOP 15 CORRELAÇÕES NEGATIVAS:")
    for col, corr in correlations.tail(15).items():
        bar = '█' * int(abs(corr) * 50)
        print(f"   {col:40s}: {corr:+.4f} {bar}")
    
    # Visualização
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Top 30 correlações (positivas e negativas)
    top_corr = pd.concat([correlations.head(15), correlations.tail(15)]).sort_values()
    colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in top_corr.values]
    
    ax.barh(range(len(top_corr)), top_corr.values, color=colors, alpha=0.7)
    ax.set_yticks(range(len(top_corr)))
    ax.set_yticklabels(top_corr.index, fontsize=9)
    ax.set_xlabel('Correlação com Target', fontweight='bold', fontsize=12)
    ax.set_title('Top 30 Features - Correlação com Target', fontweight='bold', fontsize=14)
    ax.axvline(0, color='black', linewidth=1)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/eda/target_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Visualização salva: visualizations/eda/target_correlations.png")

# ===========================================================================
# ETAPA 7: ANÁLISE GEOESPACIAL
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 7: ANÁLISE GEOESPACIAL")
print(f"{'='*70}\n")

if 'device_lat' in df.columns and 'device_lon' in df.columns:
    geo_df = df[['device_lat', 'device_lon']].dropna()
    geo_df = geo_df[(geo_df['device_lat'] != 0) & (geo_df['device_lon'] != 0)]
    
    print(f"📍 COORDENADAS VÁLIDAS: {len(geo_df):,}")
    print(f"\n📊 ESTATÍSTICAS GEOESPACIAIS:")
    print(f"   Latitude  - Min: {geo_df['device_lat'].min():.6f}, Max: {geo_df['device_lat'].max():.6f}")
    print(f"   Longitude - Min: {geo_df['device_lon'].min():.6f}, Max: {geo_df['device_lon'].max():.6f}")
    
    if 'dist_device_stop' in df.columns:
        dist_stats = df['dist_device_stop'].describe()
        print(f"\n📏 DISTÂNCIA DISPOSITIVO-PARADA:")
        print(f"   Média:    {dist_stats['mean']:.2f}")
        print(f"   Mediana:  {dist_stats['50%']:.2f}")
        print(f"   P95:      {df['dist_device_stop'].quantile(0.95):.2f}")
        print(f"   P99:      {df['dist_device_stop'].quantile(0.99):.2f}")
    
    # Visualização
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Mapa de calor
    ax1 = axes[0]
    sample = geo_df.sample(min(10000, len(geo_df)))
    ax1.hexbin(sample['device_lon'], sample['device_lat'], gridsize=50, cmap='YlOrRd', mincnt=1)
    ax1.set_xlabel('Longitude', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Latitude', fontweight='bold', fontsize=11)
    ax1.set_title('Densidade Geoespacial de Eventos', fontweight='bold', fontsize=13)
    
    # Distância
    if 'dist_device_stop' in df.columns:
        ax2 = axes[1]
        dist_data = df['dist_device_stop'].dropna()
        dist_data = dist_data[dist_data < dist_data.quantile(0.99)]  # Remover outliers
        ax2.hist(dist_data, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Distância (metros)', fontweight='bold', fontsize=11)
        ax2.set_ylabel('Frequência', fontweight='bold', fontsize=11)
        ax2.set_title('Distribuição da Distância Dispositivo-Parada', fontweight='bold', fontsize=13)
        ax2.grid(axis='y', alpha=0.3)
        ax2.axvline(dist_data.median(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mediana: {dist_data.median():.0f}m')
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig('visualizations/eda/geospatial_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Visualização salva: visualizations/eda/geospatial_analysis.png")

# ===========================================================================
# ETAPA 8: ANÁLISE DE USUÁRIOS E PARADAS
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 8: ANÁLISE DE USUÁRIOS E PARADAS")
print(f"{'='*70}\n")

if 'user_pseudo_id' in df.columns:
    user_events = df['user_pseudo_id'].value_counts()
    print(f"👥 USUÁRIOS:")
    print(f"   Total único:           {df['user_pseudo_id'].nunique():,}")
    print(f"   Eventos/usuário (med): {user_events.median():.0f}")
    print(f"   Eventos/usuário (med): {user_events.mean():.0f}")
    print(f"   Max eventos/usuário:   {user_events.max():,}")

if 'gtfs_stop_id' in df.columns:
    stop_events = df['gtfs_stop_id'].value_counts()
    print(f"\n🚏 PARADAS:")
    print(f"   Total único:          {df['gtfs_stop_id'].nunique():,}")
    print(f"   Eventos/parada (med): {stop_events.median():.0f}")
    print(f"   Eventos/parada (méd): {stop_events.mean():.0f}")
    print(f"   Max eventos/parada:   {stop_events.max():,}")

if 'gtfs_route_id' in df.columns:
    route_events = df['gtfs_route_id'].value_counts()
    print(f"\n🚌 LINHAS:")
    print(f"   Total único:         {df['gtfs_route_id'].nunique():,}")
    print(f"   Eventos/linha (med): {route_events.median():.0f}")
    print(f"   Eventos/linha (méd): {route_events.mean():.0f}")
    print(f"   Max eventos/linha:   {route_events.max():,}")

# Visualização
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

if 'user_pseudo_id' in df.columns:
    ax1 = axes[0]
    user_sample = user_events.head(20)
    ax1.barh(range(len(user_sample)), user_sample.values, color='#3498db', alpha=0.7)
    ax1.set_yticks(range(len(user_sample)))
    ax1.set_yticklabels([f'User {i+1}' for i in range(len(user_sample))], fontsize=8)
    ax1.set_xlabel('Número de Eventos', fontweight='bold')
    ax1.set_title('Top 20 Usuários Mais Ativos', fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)

if 'gtfs_stop_id' in df.columns:
    ax2 = axes[1]
    stop_sample = stop_events.head(20)
    ax2.barh(range(len(stop_sample)), stop_sample.values, color='#2ecc71', alpha=0.7)
    ax2.set_yticks(range(len(stop_sample)))
    ax2.set_yticklabels([f'Stop {i+1}' for i in range(len(stop_sample))], fontsize=8)
    ax2.set_xlabel('Número de Eventos', fontweight='bold')
    ax2.set_title('Top 20 Paradas Mais Movimentadas', fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

if 'gtfs_route_id' in df.columns:
    ax3 = axes[2]
    route_sample = route_events.head(20)
    ax3.barh(range(len(route_sample)), route_sample.values, color='#e74c3c', alpha=0.7)
    ax3.set_yticks(range(len(route_sample)))
    ax3.set_yticklabels([f'Route {i+1}' for i in range(len(route_sample))], fontsize=8)
    ax3.set_xlabel('Número de Eventos', fontweight='bold')
    ax3.set_title('Top 20 Linhas Mais Movimentadas', fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/eda/users_stops_routes.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"\n✓ Visualização salva: visualizations/eda/users_stops_routes.png")

# ===========================================================================
# ETAPA 9: OUTLIERS
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 9: DETECÇÃO DE OUTLIERS")
print(f"{'='*70}\n")

outlier_features = ['stop_event_rate', 'stop_event_count', 'user_frequency', 'dist_device_stop']
outlier_features = [col for col in outlier_features if col in df.columns]

if len(outlier_features) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(outlier_features[:4]):
        ax = axes[i]
        data = df[col].dropna()
        
        # Boxplot
        bp = ax.boxplot([data], vert=True, patch_artist=True, widths=0.6)
        for patch in bp['boxes']:
            patch.set_facecolor('#3498db')
            patch.set_alpha(0.7)
        
        ax.set_ylabel(col, fontweight='bold', fontsize=11)
        ax.set_title(f'Boxplot: {col}', fontweight='bold', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        # Estatísticas
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = data[(data < lower) | (data > upper)]
        
        ax.text(1.15, q3, f'Q3: {q3:.2f}', fontsize=9)
        ax.text(1.15, data.median(), f'Med: {data.median():.2f}', fontsize=9)
        ax.text(1.15, q1, f'Q1: {q1:.2f}', fontsize=9)
        
        outlier_pct = (len(outliers) / len(data)) * 100
        print(f"📊 {col}:")
        print(f"   Outliers: {len(outliers):,} ({outlier_pct:.2f}%)")
        print(f"   Range válido: [{lower:.2f}, {upper:.2f}]")
    
    plt.tight_layout()
    plt.savefig('visualizations/eda/outliers_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Visualização salva: visualizations/eda/outliers_analysis.png")

# ===========================================================================
# ETAPA 10: MATRIZ DE CORRELAÇÃO
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 10: MATRIZ DE CORRELAÇÃO")
print(f"{'='*70}\n")

if len(numeric_cols) > 0:
    # Selecionar top features para correlação
    top_features = correlations.abs().nlargest(15).index.tolist()
    if target in df.columns:
        top_features.append(target)
    
    corr_matrix = df[top_features].corr()
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Matriz de Correlação - Top 15 Features', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig('visualizations/eda/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Visualização salva: visualizations/eda/correlation_matrix.png")
    
    # Identificar multicolinearidade
    print(f"\n🔗 MULTICOLINEARIDADE DETECTADA (|corr| > 0.8):")
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                high_corr.append({
                    'Feature 1': corr_matrix.columns[i],
                    'Feature 2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })
    
    if len(high_corr) > 0:
        for item in high_corr:
            print(f"   {item['Feature 1']:30s} <-> {item['Feature 2']:30s}: {item['Correlation']:+.4f}")
    else:
        print(f"   ✓ Nenhuma multicolinearidade forte detectada!")

# ===========================================================================
# RESUMO FINAL
# ===========================================================================
print(f"\n{'='*80}")
print(f"RESUMO DA ANÁLISE EXPLORATÓRIA")
print(f"{'='*80}\n")

print(f"✅ ANÁLISE COMPLETA!")
print(f"\n📁 Visualizações geradas:")
print(f"   - visualizations/eda/target_distribution.png")
print(f"   - visualizations/eda/temporal_analysis.png")
print(f"   - visualizations/eda/numeric_distributions.png")
print(f"   - visualizations/eda/target_correlations.png")
print(f"   - visualizations/eda/geospatial_analysis.png")
print(f"   - visualizations/eda/users_stops_routes.png")
print(f"   - visualizations/eda/outliers_analysis.png")
print(f"   - visualizations/eda/correlation_matrix.png")

print(f"\n📊 PRINCIPAIS INSIGHTS:")
print(f"   1️⃣  Dataset com {len(df):,} registros e {len(df.columns)} features")
print(f"   2️⃣  Target desbalanceado: {imbalance_ratio:.1f}:1 (requer class_weight='balanced')")
if 'event_timestamp' in df.columns:
    print(f"   3️⃣  Padrão temporal: Pico às {peak_hour}h")
if target in df.columns and 'hour' in df.columns:
    print(f"   4️⃣  Melhor hora para conversão: {best_hour}h ({hourly_conv[best_hour]:.2f}%)")
if len(correlations) > 0:
    top_feat = correlations.abs().idxmax()
    top_corr = correlations[top_feat]
    print(f"   5️⃣  Feature mais correlacionada: {top_feat} ({top_corr:+.4f})")
if 'user_pseudo_id' in df.columns:
    print(f"   6️⃣  {df['user_pseudo_id'].nunique():,} usuários únicos")
if 'gtfs_stop_id' in df.columns:
    print(f"   7️⃣  {df['gtfs_stop_id'].nunique():,} paradas únicas")

print(f"\n{'='*80}")
print(f"✅ ANÁLISE EXPLORATÓRIA CONCLUÍDA!")
print(f"{'='*80}\n")
