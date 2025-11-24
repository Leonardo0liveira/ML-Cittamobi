DOCUMENTAÇÃO ACADÊMICA - PROJETO ML CITTAMOBI


1. INTRODUÇÃO

Este documento apresenta a metodologia completa de preparação de dados e engenharia de features para o projeto de predição de lotação de ônibus da Cittamobi. O objetivo principal é desenvolver um modelo de Machine Learning capaz de prever a lotação de ônibus urbanos com base em dados históricos de eventos de usuários e informações de transporte público (GTFS - General Transit Feed Specification).


2. EXTRAÇÃO DE DADOS

2.1 Fonte de Dados

Os dados foram extraídos do Google BigQuery, especificamente da tabela proj-ml-469320.app_cittamobi.dataset-updated, que contém registros de eventos de usuários do aplicativo Cittamobi. O BigQuery é uma plataforma de data warehouse da Google Cloud que permite armazenamento e análise de grandes volumes de dados de forma escalável e eficiente.

2.2 Processo de Extração

A extração foi realizada utilizando a biblioteca google-cloud-bigquery em Python, que fornece uma interface programática para executar queries SQL e recuperar resultados diretamente em estruturas de dados do pandas. O processo envolveu as seguintes etapas:

Código de extração:
from google.cloud import bigquery
import pandas as pd

Configuração do cliente BigQuery:
project_id = "proj-ml-469320"
client = bigquery.Client(project=project_id)

Query SQL para extração:
query = "SELECT * FROM proj-ml-469320.app_cittamobi.dataset-updated LIMIT 200000"

Execução da query e conversão para DataFrame:
df = client.query(query).to_dataframe()

Justificativa da Amostragem: Devido ao volume massivo de dados (milhões de registros), foi realizada uma amostragem de 200.000 eventos para viabilizar o processamento computacional e análise exploratória inicial. Esta amostra foi considerada representativa para o desenvolvimento do modelo, garantindo diversidade temporal, geográfica e de padrões de uso.

2.3 Estrutura dos Dados Brutos

Os dados extraídos contêm os seguintes campos principais:

Campo | Tipo | Descrição
| `event_timestamp` | TIMESTAMP | Data e hora do evento do usuário |
| `user_pseudo_id` | STRING | Identificador anônimo do usuário |
| `event_name` | STRING | Tipo de evento (ex: `bstop_open`) |
| `device_lat` | STRING | Coordenada POINT com latitude do dispositivo |
| `device_lon` | STRING | Coordenada POINT com longitude do dispositivo |
| `stop_lat` | STRING | Coordenada POINT da parada de ônibus |
| `stop_lon` | STRING | Coordenada POINT da parada de ônibus |
| `gtfs_stop_id` | STRING | ID da parada no sistema GTFS |
| `route_short_name` | STRING | Nome/número da linha de ônibus |
| `platform` | STRING | Plataforma do dispositivo (Android/iOS) |

### 2.4 Dados Complementares: GTFS (General Transit Feed Specification)

Para enriquecer o dataset com informações sobre o sistema de transporte público, foram utilizados arquivos GTFS da SPTrans (São Paulo Transporte). O GTFS é um formato padronizado internacionalmente para disponibilização de dados de transporte público, permitindo a integração entre diferentes sistemas e aplicações.

Arquivos GTFS utilizados:
1. routes.txt: Informações sobre as linhas de ônibus (identificadores, nomes, operadoras)
2. trips.txt: Viagens programadas por linha (trajetos, horários, sentidos)
3. stops.txt: Localização e detalhes das paradas (coordenadas geográficas, nomes, códigos)
4. stop_times.txt: Horários de passagem dos ônibus em cada parada das viagens
5. frequencies.txt: Frequência (headway/intervalo) dos ônibus por período do dia

Processamento dos arquivos GTFS:
gtfs_files = ['sptrans/routes.txt', 'sptrans/trips.txt', 'sptrans/stops.txt', 'sptrans/stop_times.txt', 'sptrans/frequencies.txt']
gtfs_data = {}
for file_path in gtfs_files:
    key = file_path.split('/')[-1].replace('.txt', '')
    gtfs_data[key] = pd.read_csv(file_path, dtype=str)

Todos os arquivos foram carregados como strings para preservar códigos com zeros à esquerda e permitir processamento posterior adequado.


3. PREPARAÇÃO E LIMPEZA DOS DADOS

### 3.1 Filtragem de Eventos Relevantes

O dataset original contém diversos tipos de eventos. Para este estudo, foram filtrados apenas os eventos do tipo `bstop_open`, que representam a abertura da tela de paradas de ônibus no aplicativo, indicando interesse do usuário em uma parada específica.

```python
df_events = df_events[df_events['event_name'] == 'bstop_open'].copy()
```

**Justificativa**: Eventos `bstop_open` são indicadores diretos de demanda por transporte em uma determinada parada e horário, servindo como proxy para a lotação esperada.

### 3.2 Conversão de Timestamps e Fuso Horário

Os timestamps foram convertidos para o fuso horário de São Paulo (America/Sao_Paulo) para garantir a precisão das análises temporais. Esta conversão é essencial pois os dados brutos podem estar armazenados em UTC (Coordinated Universal Time), e as análises de padrões temporais (horários de pico, períodos do dia) precisam refletir o horário local real dos usuários.

Código de conversão:
df_events['event_timestamp'] = pd.to_datetime(df_events['event_timestamp'], format='mixed').dt.tz_convert('America/Sao_Paulo')

O parâmetro format='mixed' permite que o pandas interprete automaticamente diferentes formatos de timestamp presentes nos dados, aumentando a robustez do processo de conversão.

3.3 Extração de Coordenadas Geográficas

As coordenadas geográficas estavam armazenadas no formato textual POINT(longitude latitude), que é um padrão do PostGIS (extensão espacial do PostgreSQL) usado para representar geometrias pontuais. Foi necessário extrair os valores numéricos de latitude e longitude para possibilitar cálculos geoespaciais.

Função de extração:
import re

def extract_coords(point_str):
    if pd.isna(point_str):
        return None, None
    match = re.search(r'POINT\s*\(\s*(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s*\)', str(point_str))
    if match:
        lon = float(match.group(1))
        lat = float(match.group(2))
        return lat, lon
    return None, None

Aplicação da extração:
df_events['device_lat'], df_events['device_lon'] = zip(*df_events['device_lat'].apply(extract_coords))
df_events['stop_lat'], df_events['stop_lon'] = zip(*df_events['stop_lat'].apply(extract_coords))

A função utiliza expressões regulares (regex) para identificar o padrão POINT e extrair os dois números (longitude e latitude) contidos nos parênteses. O uso de expressões regulares garante flexibilidade no parsing, tratando variações de espaçamento e formatação.

3.4 Spatial Join com GTFS

Para vincular eventos do aplicativo a paradas oficiais do sistema GTFS, foi realizado um spatial join (junção espacial) utilizando k-d tree (estrutura de dados para busca eficiente em espaços multidimensionais) para identificar a parada GTFS mais próxima de cada evento.

Processamento do spatial join:
from scipy.spatial import cKDTree

Coordenadas das paradas GTFS:
gtfs_coords = df_stops[['stop_lat', 'stop_lon']].astype(float).values
tree = cKDTree(gtfs_coords)

Coordenadas dos eventos:
event_coords = df_events[['stop_lat', 'stop_lon']].values

Buscar parada mais próxima:
distances, indices = tree.query(event_coords, k=1)

Atribuir gtfs_stop_id:
df_events['gtfs_stop_id'] = df_stops.iloc[indices]['stop_id'].values
df_events['dist_device_stop'] = distances

Justificativa: O spatial join permite associar eventos do aplicativo às paradas oficiais do sistema GTFS, possibilitando o cruzamento com dados estruturados de frequência, horários programados e características das paradas. O algoritmo k-d tree foi escolhido por sua eficiência computacional O(log n) em buscas de vizinhos mais próximos, essencial para processar grandes volumes de eventos. A distância calculada (em graus de latitude/longitude) também foi preservada como feature, pois pode indicar imprecisão na localização ou eventos registrados longe das paradas oficiais.


4. CRIAÇÃO DA VARIÁVEL ALVO (TARGET)

### 4.1 Definição do Problema

O objetivo é prever a **lotação de ônibus** em uma parada específica em um dado momento. Como não há dados diretos de lotação, foi criado um **proxy de lotação** baseado na quantidade de usuários únicos que acessaram a parada no aplicativo em uma janela temporal.

### 4.2 Agregação Temporal

Foi utilizada uma **janela de agregação de 2 minutos** para contar usuários únicos por parada:

```python
agg_window = '2T'  # 2 minutos

# Agrupar por parada e janela temporal
df_counts = df_events.set_index('event_timestamp').groupby(
    ['gtfs_stop_id', pd.Grouper(freq=agg_window)]
)['user_pseudo_id'].nunique()

df_proxy = df_counts.to_frame(name='user_count_2min')
```

**Justificativa**: Janelas de 2 minutos capturam picos de demanda em curto prazo, refletindo a dinâmica de chegada de passageiros nas paradas.

### 4.3 Discretização em Classes

Inicialmente, a variável alvo foi dividida em **3 classes** de lotação:

```python
bins = [0, 1, 2, np.inf]
labels = ['Baixa', 'Média', 'Alta']

df_proxy['lotacao_proxy'] = pd.cut(
    df_proxy['user_count_2min'], 
    bins=bins, 
    labels=labels, 
    right=True
)
```

- **Baixa**: 0-1 usuários (baixa demanda)
- **Média**: 2 usuários (demanda moderada)
- **Alta**: 3+ usuários (alta demanda)

### 4.4 Conversão para Classificação Binária

Devido ao **desbalanceamento severo** das classes (classe "Baixa" representava >90% dos casos), o problema foi reformulado como **detecção de evento raro** (classificação binária):

```python
df_final['lotacao_proxy_binaria'] = df_final['lotacao_proxy'].map({
    'Baixa': 'Baixa',           # Classe minoritária (evento raro)
    'Média': 'Nao_Baixa',       # Classe majoritária
    'Alta': 'Nao_Baixa'         # Classe majoritária
})
```

**Justificativa**: A classificação binária simplifica o problema e melhora o desempenho do modelo em identificar situações de baixa lotação (eventos raros), que são críticas para o planejamento operacional.

### 4.5 Distribuição da Variável Alvo

Após a conversão para classificação binária, a distribuição ficou:

| Classe | Percentual | Contagem Absoluta |
|--------|-----------|-------------------|
| Nao_Baixa | ~75% | ~150.000 |
| Baixa | ~25% | ~50.000 |

**Observação**: Mesmo com a simplificação, ainda há desbalanceamento, requerendo técnicas de balanceamento como SMOTE, class weights ou undersampling durante o treinamento dos modelos.

---

## 5. Engenharia de Features

### 5.1 Features Temporais

Features temporais capturam padrões de demanda por transporte ao longo do tempo:

```python
# Features básicas extraídas do timestamp
df_final['time_hour'] = df_final['event_timestamp'].dt.hour
df_final['time_day_of_week'] = df_final['event_timestamp'].dt.dayofweek
df_final['time_day_of_month'] = df_final['event_timestamp'].dt.day
df_final['time_month'] = df_final['event_timestamp'].dt.month

# Features categóricas binárias
df_final['is_holiday'] = df_final.index.date.apply(
    lambda x: x in holidays.Brazil(state='SP')
).astype(int)

df_final['is_weekend'] = (df_final['time_day_of_week'] >= 5).astype(int)

df_final['is_peak_hour'] = df_final['time_hour'].apply(
    lambda h: 1 if (6 <= h < 9) or (17 <= h < 19) else 0
)
```

**Features Criadas**:
- `time_hour` (0-23): Hora do dia
- `time_day_of_week` (0-6): Dia da semana (0=Segunda)
- `time_day_of_month` (1-31): Dia do mês
- `time_month` (1-12): Mês
- `is_holiday` (0/1): Indica se é feriado
- `is_weekend` (0/1): Indica se é fim de semana
- `is_peak_hour` (0/1): Indica horário de pico (6h-9h, 17h-19h)

**Justificativa**: Padrões temporais são fundamentais em transporte público. Horários de pico, fins de semana e feriados apresentam demandas distintas.

### 5.2 Features Geoespaciais

Features baseadas em localização e distâncias:

```python
from geopy.distance import geodesic

# Calcular distância entre dispositivo e parada
df_final['dist_device_stop'] = df_final.apply(
    lambda row: geodesic(
        (row['device_lat'], row['device_lon']),
        (row['stop_lat'], row['stop_lon'])
    ).meters,
    axis=1
)
```

**Features Criadas**:
- `device_lat`, `device_lon`: Coordenadas do dispositivo
- `stop_lat`, `stop_lon`: Coordenadas da parada
- `dist_device_stop`: Distância em metros entre dispositivo e parada
- `gtfs_stop_id`: Identificador da parada no GTFS

**Justificativa**: A proximidade do usuário à parada e as características geográficas influenciam a demanda local.

### 5.3 Features de Serviço (GTFS - Headway)

O **headway** (intervalo entre ônibus consecutivos) é um indicador-chave da qualidade do serviço:

```python
# Calcular headway médio por parada e hora
df_freq = df_frequencies.copy()
df_freq['headway_secs'] = pd.to_numeric(df_freq['headway_secs'])
df_freq['start_hour'] = df_freq['start_time'].str.split(':').str[0].astype(int)

df_headway_avg = df_freq.groupby(
    ['stop_id', 'start_hour']
)['headway_secs'].mean().reset_index()

# Juntar ao dataset principal
df_final = df_final.merge(
    df_headway_avg,
    left_on=['gtfs_stop_id', 'time_hour'],
    right_on=['stop_id', 'start_hour'],
    how='left'
)

# Preencher ausências com 3600s (baixa frequência)
df_final['headway_avg_stop_hour'].fillna(3600, inplace=True)
```

**Feature Criada**:
- `headway_avg_stop_hour`: Headway médio (em segundos) para a parada e hora específicas

**Justificativa**: Headways menores (maior frequência) tendem a reduzir aglomerações e melhorar a distribuição de passageiros ao longo do tempo.

### 5.4 Features Cíclicas

Features cíclicas capturam a natureza periódica de variáveis temporais:

```python
# Codificação cíclica para hora do dia
df_final['hour_sin'] = np.sin(2 * np.pi * df_final['time_hour'] / 24)
df_final['hour_cos'] = np.cos(2 * np.pi * df_final['time_hour'] / 24)

# Codificação cíclica para dia da semana
df_final['day_sin'] = np.sin(2 * np.pi * df_final['time_day_of_week'] / 7)
df_final['day_cos'] = np.cos(2 * np.pi * df_final['time_day_of_week'] / 7)
```

**Features Criadas**:
- `hour_sin`, `hour_cos`: Codificação cíclica da hora
- `day_sin`, `day_cos`: Codificação cíclica do dia da semana

**Justificativa**: Codificações cíclicas preservam a continuidade de variáveis temporais (ex: 23h está próxima de 0h), melhorando o desempenho de modelos baseados em árvore.

### 5.5 Features de Interação

Features de interação capturam relações complexas entre variáveis:

```python
df_final['headway_x_hour'] = df_final['headway_avg_stop_hour'] * df_final['time_hour']
df_final['headway_x_weekend'] = df_final['headway_avg_stop_hour'] * df_final['is_weekend']
df_final['dist_x_peak'] = df_final['dist_device_stop'] * df_final['is_peak_hour']
df_final['dist_x_weekend'] = df_final['dist_device_stop'] * df_final['is_weekend']
```

**Features Criadas**:
- `headway_x_hour`: Interação entre headway e hora
- `headway_x_weekend`: Interação entre headway e fim de semana
- `dist_x_peak`: Interação entre distância e horário de pico
- `dist_x_weekend`: Interação entre distância e fim de semana

**Justificativa**: Interações revelam efeitos combinados (ex: baixa frequência em horários de pico é mais crítica que em horários normais).

### 5.6 Features de Agregação por Parada

Features agregadas capturam características históricas das paradas:

```python
# Taxa de eventos por parada
stop_event_rate = df_final.groupby('gtfs_stop_id')['lotacao_proxy_binaria'].agg([
    ('stop_event_rate', lambda x: (x == 'Baixa').mean()),
    ('stop_event_count', 'count'),
    ('stop_total_samples', 'size')
])

# Estatísticas de distância por parada
stop_dist_stats = df_final.groupby('gtfs_stop_id')['dist_device_stop'].agg([
    ('stop_dist_mean', 'mean'),
    ('stop_dist_std', 'std')
]).fillna(0)

# Estatísticas de headway por parada
stop_headway_stats = df_final.groupby('gtfs_stop_id')['headway_avg_stop_hour'].agg([
    ('stop_headway_mean', 'mean'),
    ('stop_headway_std', 'std')
]).fillna(3600)

# Juntar ao dataset
df_final = df_final.merge(stop_event_rate, on='gtfs_stop_id', how='left')
df_final = df_final.merge(stop_dist_stats, on='gtfs_stop_id', how='left')
df_final = df_final.merge(stop_headway_stats, on='gtfs_stop_id', how='left')
```

**Features Criadas**:
- `stop_event_rate`: Taxa histórica de eventos "Baixa" na parada
- `stop_event_count`: Quantidade de eventos na parada
- `stop_total_samples`: Total de amostras da parada
- `stop_dist_mean`, `stop_dist_std`: Média e desvio padrão da distância
- `stop_headway_mean`, `stop_headway_std`: Média e desvio padrão do headway

**Justificativa**: Algumas paradas têm características intrínsecas (ex: terminais vs. paradas de bairro) que influenciam a lotação. Features agregadas capturam essas diferenças.

---

## 6. Codificação de Variáveis Categóricas

### 6.1 Label Encoding

Variáveis categóricas foram codificadas numericamente:

```python
from sklearn.preprocessing import LabelEncoder

categorical_cols = ['gtfs_stop_id', 'route_short_name']

for col in categorical_cols:
    if col in df_final.columns:
        le = LabelEncoder()
        df_final[col] = le.fit_transform(df_final[col].astype(str))
```

### 6.2 Target Encoding

A variável alvo binária foi mapeada para valores numéricos:

```python
df_final['target'] = df_final['lotacao_proxy_binaria'].map({
    'Baixa': 1,      # Evento raro (classe positiva)
    'Nao_Baixa': 0   # Evento comum (classe negativa)
})
```

---

## 7. Exportação do Dataset Final

O dataset processado foi exportado para uso nos modelos:

```python
df_final.to_csv('dataset_cittamobi_final.csv', index=False)

print(f"✅ Dataset final exportado com {len(df_final)} registros")
print(f"📊 Total de features: {df_final.shape[1]}")
print(f"🎯 Variável alvo: 'target' (0=Nao_Baixa, 1=Baixa)")
```

---

## 8. Resumo das Features Finais

### 8.1 Categorias de Features

| Categoria | Quantidade | Exemplos |
|-----------|-----------|----------|
| **Temporais** | 11 | `time_hour`, `is_peak_hour`, `hour_sin`, `day_cos` |
| **Geoespaciais** | 5 | `device_lat`, `device_lon`, `dist_device_stop` |
| **Serviço (GTFS)** | 2 | `gtfs_stop_id`, `headway_avg_stop_hour` |
| **Interação** | 4 | `headway_x_hour`, `dist_x_peak` |
| **Agregação** | 7 | `stop_event_rate`, `stop_dist_mean` |
| **Total** | **29** | - |

### 8.2 Variável Alvo

- **Nome**: `target`
- **Tipo**: Binária (0/1)
- **Significado**:
  - **1 (Baixa)**: Evento raro - Baixa lotação (< 2 usuários em 2 min)
  - **0 (Nao_Baixa)**: Evento comum - Lotação normal/alta (≥ 2 usuários em 2 min)

---

## 9. Considerações Finais

Este processo de preparação de dados estabeleceu a base para o desenvolvimento de modelos preditivos de lotação. As principais contribuições metodológicas incluem:

1. **Criação de proxy de lotação**: Abordagem inovadora utilizando contagem de usuários únicos em janelas temporais
2. **Reformulação como problema binário**: Tratamento de desbalanceamento e foco em eventos raros
3. **Engenharia de features abrangente**: 29 features cobrindo aspectos temporais, geoespaciais, de serviço e históricos
4. **Integração com GTFS**: Enriquecimento com dados oficiais de transporte público

Os próximos passos incluem:
- Análise exploratória de dados (EDA)
- Seleção de features
- Treinamento e otimização de modelos
- Validação com métricas apropriadas para classes desbalanceadas (AUC-ROC, F1-Score, Precision-Recall)

---

## Referências

- **Google Cloud BigQuery**: https://cloud.google.com/bigquery
- **GTFS Specification**: https://gtfs.org/
- **SPTrans Open Data**: http://www.sptrans.com.br/desenvolvedores/
- **Geopy Documentation**: https://geopy.readthedocs.io/
- **Scikit-learn Preprocessing**: https://scikit-learn.org/stable/modules/preprocessing.html
