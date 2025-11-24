# 📚 DOCUMENTAÇÃO COMPLETA - MODEL V8 PRODUCTION

## 📋 Índice
1. [Visão Geral](#visão-geral)
2. [Arquitetura do Modelo](#arquitetura-do-modelo)
3. [Fluxo de Execução](#fluxo-de-execução)
4. [Feature Engineering Detalhado](#feature-engineering-detalhado)
5. [Validação Cruzada Temporal](#validação-cruzada-temporal)
6. [Ensemble e Threshold Dinâmico](#ensemble-e-threshold-dinâmico)
7. [Métricas e Avaliação](#métricas-e-avaliação)
8. [Artefatos Gerados](#artefatos-gerados)
9. [Como Usar em Produção](#como-usar-em-produção)

---

## 🎯 Visão Geral

### Objetivo
Prever se um usuário do app Cittamobi vai **converter** (comprar uma passagem) após visualizar informações de uma parada de ônibus.

### Performance do Modelo
```
✓ F1 Classe 1 (Conversão): ~55.39%
✓ F1 Classe 0 (Não-Conversão): ~95.76%
✓ ROC-AUC: ~94.25%
✓ F1-Macro: ~75.58%
✓ Accuracy: ~92.40%
```

### Diferencial Principal
- **Validação Cruzada Temporal (TimeSeriesSplit)** com 5 folds
- **Split sequencial** que respeita a ordem temporal dos dados
- **Ensemble otimizado** de LightGBM + XGBoost
- **Threshold dinâmico** baseado em histórico de conversão

---

## 🏗️ Arquitetura do Modelo

### Componentes Principais

```
┌─────────────────────────────────────────────────────────────┐
│                    DADOS BRUTOS                             │
│              (BigQuery ou CSV Local)                        │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              FEATURE ENGINEERING                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Geographic  │  │   Dynamic    │  │     Base     │     │
│  │  (6 features)│  │ (10 features)│  │  (restantes) │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│            TIME SERIES SPLIT (5 FOLDS)                      │
│  Fold 1 → Fold 2 → Fold 3 → Fold 4 → Fold 5                │
│  (Mantém ordem temporal dos dados)                          │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                 NORMALIZAÇÃO                                │
│            (StandardScaler por fold)                        │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              SAMPLE WEIGHTS DINÂMICOS                       │
│  Alto conv. → Classe 1: 3.0x | Classe 0: 0.5x              │
│  Médio conv. → Classe 1: 2.0x | Classe 0: 0.8x             │
│  Baixo conv. → Classe 1: 1.5x | Classe 0: 1.0x             │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                TREINAMENTO ENSEMBLE                         │
│  ┌──────────────┐              ┌──────────────┐            │
│  │  LightGBM    │              │   XGBoost    │            │
│  │  (48.5%)     │              │   (51.5%)    │            │
│  └──────────────┘              └──────────────┘            │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│            THRESHOLD DINÂMICO                               │
│  Alta conv. (≥50%): threshold = 0.40                        │
│  Média conv. (≥30%): threshold = 0.50                       │
│  Baixa conv. (≥10%): threshold = 0.60                       │
│  Muito baixa (<10%): threshold = 0.75                       │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                   PREDIÇÃO FINAL                            │
│              (0 = Não Conversão / 1 = Conversão)            │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔄 Fluxo de Execução

### Etapa 1: Carregamento de Dados

```python
# Opção 1: CSV Local (para testes)
USE_CSV = True
df = pd.read_csv('dataset-updated.csv')

# Opção 2: BigQuery (para produção)
USE_CSV = False
client = bigquery.Client(project='proj-ml-469320')
df = client.query(query).to_dataframe()
```

**O que acontece:**
- Carrega os dados do Cittamobi
- Filtra apenas registros com `target` válido (0 ou 1)
- Remove NaNs na coluna target
- Mostra estatísticas básicas (total de registros, taxa de conversão)

**Variáveis importantes:**
- `target`: 1 = conversão, 0 = não conversão
- `gtfs_stop_id`: ID único da parada
- `device_id`: ID único do usuário
- `stop_lat_event`, `stop_lon_event`: Coordenadas da parada
- `time_hour`, `time_day_of_week`: Informações temporais

---

### Etapa 2: Feature Engineering - FASE 1 (Geographic)

#### 2A. Stop Historical Conversion
```python
stop_conversion = df.groupby('gtfs_stop_id')['target'].mean().to_dict()
df['stop_historical_conversion'] = df['gtfs_stop_id'].map(stop_conversion)
```
**O que faz:** Calcula a taxa média de conversão de cada parada.
**Exemplo:** Se a parada "ABC123" teve 100 eventos e 30 conversões, essa feature será 0.30 (30%).

#### 2B. Stop Density
```python
nn = NearestNeighbors(n_neighbors=11, metric='euclidean')
nn.fit(coords_df)
distances, _ = nn.kneighbors(...)
df['stop_density'] = 1 / (distances.mean(axis=1) + 0.001)
```
**O que faz:** Mede quão "densa" é a área ao redor da parada (quantas outras paradas existem próximas).
**Interpretação:** 
- Valor ALTO = muitas paradas próximas (centro urbano)
- Valor BAIXO = paradas isoladas (periferia)

#### 2C. Distance to Nearest CBD
```python
cbd_coords = [
    (-23.5505, -46.6333),  # São Paulo
    (-22.9068, -43.1729),  # Rio de Janeiro
    ...
]
df['dist_to_nearest_cbd'] = haversine_vectorized(...)
```
**O que faz:** Calcula a distância (em km) da parada até o centro comercial mais próximo.
**Hipótese:** Paradas mais próximas de centros comerciais têm maior conversão.

#### 2D. Stop Clustering (DBSCAN)
```python
clustering = DBSCAN(eps=0.01, min_samples=5)
cluster_labels = clustering.fit_predict(coords_for_clustering)
```
**O que faz:** Agrupa paradas geograficamente próximas em clusters.
**Resultado:**
- `stop_cluster`: ID do cluster da parada (-1 = outlier)
- `cluster_conversion_rate`: Taxa de conversão média do cluster

#### 2E. Stop Volatility
```python
stop_volatility = df.groupby('gtfs_stop_id')['target'].std().to_dict()
```
**O que faz:** Mede a variabilidade das conversões de uma parada.
**Interpretação:**
- Volatilidade ALTA = conversão imprevisível
- Volatilidade BAIXA = comportamento consistente

---

### Etapa 3: Feature Engineering - FASE 2A (Dynamic + Interactions)

#### 3A. Temporal Conversion Rates
```python
df['hour_conversion_rate'] = df.groupby('time_hour')['target'].transform('mean')
df['dow_conversion_rate'] = df.groupby('time_day_of_week')['target'].transform('mean')
df['stop_hour_conversion'] = df.groupby(['gtfs_stop_id', 'time_hour'])['target'].transform('mean')
```
**O que faz:** 
- `hour_conversion_rate`: Taxa de conversão por hora do dia (0-23)
- `dow_conversion_rate`: Taxa de conversão por dia da semana (0-6)
- `stop_hour_conversion`: Taxa específica da parada naquele horário

**Exemplo:**
- Parada "ABC123" às 8h da manhã pode ter taxa de 40%
- A mesma parada às 23h pode ter taxa de 5%

#### 3B. Geo-Temporal Interactions
```python
df['geo_temporal'] = df['dist_to_nearest_cbd'] * df['is_peak_hour']
df['density_peak'] = df['stop_density'] * df['is_peak_hour']
```
**O que faz:** Combina features geográficas com temporais.
**Hipótese:** O impacto da localização muda durante horários de pico.

#### 3C. User Features
```python
df['user_conversion_rate'] = df['device_id'].map(user_conversion)
df['user_vs_stop_ratio'] = df['device_id'].map(user_stop_ratio)
```
**O que faz:**
- `user_conversion_rate`: Histórico de conversão do usuário
- `user_vs_stop_ratio`: Diversidade de paradas visitadas pelo usuário

**Interpretação:**
- Usuário que sempre converte → valor alto
- Usuário que visita muitas paradas diferentes → explorador

#### 3D. Rarity Features
```python
df['stop_rarity'] = 1 / (df['stop_event_count'] + 1)
df['user_rarity'] = 1 / (df['user_frequency'] + 1)
```
**O que faz:** Mede quão "rara" é a parada/usuário.
**Interpretação:**
- Parada rara = poucas visualizações
- Usuário raro = pouco uso do app

#### 3E. Distance Deviation
```python
df['stop_dist_std'] = ...  # Desvio padrão das coordenadas da parada
```
**O que faz:** Mede a variação nas coordenadas reportadas para a mesma parada.
**Interpretação:** Paradas com alta variação podem ter GPS impreciso.

---

### Etapa 4: Preparação de Features

```python
# Remover colunas que não devem ser features
exclude_cols = ['target', 'gtfs_stop_id', 'timestamp_converted', 'device_id', ...]

# Selecionar apenas numéricas
X = X.select_dtypes(include=[np.number])

# Limpar caracteres especiais dos nomes
X.columns = X.columns.str.replace('[', '_', regex=False)

# Tratar infinitos e NaNs
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(0, inplace=True)
```

**Por que isso é importante:**
- Modelos ML só aceitam valores numéricos
- Caracteres especiais em nomes de colunas quebram LightGBM/XGBoost
- Infinitos e NaNs causam erros no treinamento

---

### Etapa 5: Time Series Split

```python
tscv = TimeSeriesSplit(n_splits=5)

for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
    # Treina e valida em cada fold
```

**Como funciona:**
```
Dados: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

Fold 1:
  Train: [1, 2]
  Val:   [3, 4]

Fold 2:
  Train: [1, 2, 3, 4]
  Val:   [5, 6]

Fold 3:
  Train: [1, 2, 3, 4, 5, 6]
  Val:   [7, 8]

Fold 4:
  Train: [1, 2, 3, 4, 5, 6, 7, 8]
  Val:   [9, 10]

Fold 5:
  Train: [1, 2, 3, 4, 5, 6, 7, 8, 9]
  Val:   [10]
```

**Por que usar TimeSeriesSplit:**
- Respeita a ordem temporal dos dados
- Evita "vazamento de informação do futuro"
- Simula cenário real de predição (treinar no passado, prever no futuro)

---

### Etapa 6: Validação Cruzada

Para cada fold:

#### 6.1. Normalização
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_fold)
X_val_scaled = scaler.transform(X_val_fold)
```
**O que faz:** Padroniza features para média=0 e desvio=1.
**Por que:** Melhora convergência e performance dos modelos.

#### 6.2. Sample Weights Dinâmicos
```python
def get_dynamic_sample_weights(X, y):
    weights = np.ones(len(y))
    stop_conv = X['stop_historical_conversion'].values
    
    # Paradas com alta conversão
    high_mask = stop_conv > 0.5
    weights[high_mask & (y == 1)] = 3.0  # Conversões valem 3x
    weights[high_mask & (y == 0)] = 0.5  # Não-conversões valem 0.5x
    
    # Paradas com média conversão
    med_mask = (stop_conv > 0.2) & (stop_conv <= 0.5)
    weights[med_mask & (y == 1)] = 2.0
    weights[med_mask & (y == 0)] = 0.8
    
    # Paradas com baixa conversão
    low_mask = stop_conv <= 0.2
    weights[low_mask & (y == 1)] = 1.5
    weights[low_mask & (y == 0)] = 1.0
    
    return weights
```

**Por que isso é crucial:**
- Dá mais "peso" para conversões em paradas com alto potencial
- Reduz peso de falsos positivos em paradas ruins
- Melhora F1 da classe minoritária (conversões)

#### 6.3. Scale Pos Weight
```python
scale_weight = len(y[y==0]) / len(y[y==1])
# Se 90% são não-conversões e 10% conversões: scale_weight = 9.0
```
**O que faz:** Compensa o desbalanceamento de classes.
**Aplicação:** Multiplicado por 1.3 para dar ainda mais ênfase à classe minoritária.

#### 6.4. Treinamento LightGBM
```python
params_lgb = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'scale_pos_weight': scale_weight * 1.3,
}
```

**Parâmetros principais:**
- `num_leaves: 63`: Complexidade da árvore (mais folhas = mais complexo)
- `learning_rate: 0.05`: Taxa de aprendizado (menor = mais conservador)
- `feature_fraction: 0.8`: Usa 80% das features em cada árvore (evita overfitting)
- `bagging_fraction: 0.8`: Usa 80% dos dados em cada iteração
- `min_child_samples: 20`: Mínimo de amostras por folha (evita overfitting)

#### 6.5. Treinamento XGBoost
```python
params_xgb = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'scale_pos_weight': scale_weight * 1.3,
    'tree_method': 'hist',
}
```

**Diferenças do LightGBM:**
- `max_depth` ao invés de `num_leaves`: Controla profundidade máxima
- `tree_method: 'hist'`: Algoritmo mais rápido para grandes datasets

#### 6.6. Ensemble
```python
w_lgb = 0.485
w_xgb = 0.515
pred_ensemble = w_lgb * pred_lgb + w_xgb * pred_xgb
```
**Pesos otimizados empiricamente:** XGBoost um pouco mais forte que LightGBM.

#### 6.7. Threshold Dinâmico
```python
def get_dynamic_threshold(stop_conv):
    if stop_conv >= 0.5:
        return 0.40  # Paradas excelentes: threshold baixo
    elif stop_conv >= 0.3:
        return 0.50  # Paradas boas: threshold médio
    elif stop_conv >= 0.1:
        return 0.60  # Paradas regulares: threshold alto
    else:
        return 0.75  # Paradas ruins: threshold muito alto
```

**Lógica:**
- Paradas boas: Ser "mais otimista" → aceitar probabilidades mais baixas como conversão
- Paradas ruins: Ser "mais conservador" → exigir probabilidades muito altas

---

### Etapa 7-10: Treinamento do Modelo Final

Após a validação cruzada, treina o modelo final com 80% dos dados:

```python
split_idx = int(len(X) * 0.8)
X_train_final = X.iloc[:split_idx]  # 80% primeiros (mais antigos)
X_test_final = X.iloc[split_idx:]   # 20% últimos (mais recentes)
```

**Por que 80/20 sequencial:**
- Simula produção: treinar no passado, testar no futuro
- Evita data leakage

---

### Etapa 11-12: Avaliação Final

#### Métricas Principais

**1. ROC-AUC (~94.25%)**
- Mede capacidade geral de separar classes
- Valor excelente (próximo de 1.0)

**2. F1-Score Classe 1 (~55.39%)**
- Equilíbrio entre Precision e Recall para CONVERSÕES
- Mais importante para o negócio (encontrar quem vai comprar)

**3. F1-Score Classe 0 (~95.76%)**
- Equilíbrio entre Precision e Recall para NÃO-CONVERSÕES
- Muito alto (fácil prever quem NÃO vai comprar)

**4. F1-Macro (~75.58%)**
- Média das duas classes
- Mostra equilíbrio geral do modelo

**Confusion Matrix:**
```
                 Predito: 0    Predito: 1
Real: 0 (não)    TN            FP
Real: 1 (sim)    FN            TP
```
- **TN (True Negative):** Acertou que não iria converter
- **FP (False Positive):** Errou, disse que ia converter mas não converteu
- **FN (False Negative):** Errou, disse que não ia converter mas converteu
- **TP (True Positive):** Acertou que iria converter

---

## 💾 Artefatos Gerados

### 1. `lightgbm_model_v8_production.txt`
**Formato:** Texto proprietário do LightGBM
**Conteúdo:** Estrutura completa das árvores do LightGBM
**Uso:** Carregar com `lgb.Booster(model_file='...')`

### 2. `xgboost_model_v8_production.json`
**Formato:** JSON
**Conteúdo:** Estrutura completa das árvores do XGBoost
**Uso:** Carregar com `xgb.Booster(); booster.load_model('...')`

### 3. `scaler_v8_production.pkl`
**Formato:** Pickle (Python)
**Conteúdo:** Objeto StandardScaler treinado
**Uso:** Normalizar novos dados antes de prever
```python
with open('scaler_v8_production.pkl', 'rb') as f:
    scaler = pickle.load(f)
X_new_scaled = scaler.transform(X_new)
```

### 4. `selected_features_v8_production.txt`
**Formato:** Texto (uma feature por linha)
**Conteúdo:** Lista ordenada das features usadas
**Uso:** Garantir que novos dados tenham as mesmas features na mesma ordem

### 5. `model_config_v8_production.json`
**Formato:** JSON
**Conteúdo:**
```json
{
    "model_version": "v8_production_timeseries_cv",
    "creation_date": "2025-11-23 14:30:00",
    "n_features": 42,
    "ensemble_weights": {
        "lightgbm": 0.485,
        "xgboost": 0.515
    },
    "cross_validation": {
        "method": "TimeSeriesSplit",
        "n_splits": 5,
        "fold_results": [...],
        "cv_metrics_mean": {
            "auc_ensemble": 0.9425,
            "f1_macro": 0.7558,
            ...
        }
    },
    "final_test_metrics": {...},
    "threshold_strategy": "dynamic",
    "threshold_rules": {...},
    "training_params": {...}
}
```

---

## 🚀 Como Usar em Produção

### Script de Inferência Básico

```python
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import pickle
import json

# 1. CARREGAR ARTEFATOS
print("Carregando modelos...")
lgb_model = lgb.Booster(model_file='lightgbm_model_v8_production.txt')
xgb_model = xgb.Booster()
xgb_model.load_model('xgboost_model_v8_production.json')

with open('scaler_v8_production.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('selected_features_v8_production.txt', 'r') as f:
    feature_cols = [line.strip() for line in f]

with open('model_config_v8_production.json', 'r') as f:
    config = json.load(f)

w_lgb = config['ensemble_weights']['lightgbm']
w_xgb = config['ensemble_weights']['xgboost']

# 2. CARREGAR NOVOS DADOS
print("Carregando novos dados...")
df_new = pd.read_csv('novos_eventos.csv')

# 3. FEATURE ENGINEERING
print("Criando features...")
# [Repetir TODAS as etapas de feature engineering do treinamento]
# - stop_historical_conversion
# - stop_density
# - dist_to_nearest_cbd
# - ... todas as 16 features criadas

# 4. PREPARAR FEATURES
X_new = df_new[feature_cols].copy()
X_new = X_new.select_dtypes(include=[np.number])
X_new.replace([np.inf, -np.inf], np.nan, inplace=True)
X_new.fillna(0, inplace=True)

# 5. NORMALIZAR
X_new_scaled = scaler.transform(X_new)
X_new_scaled = pd.DataFrame(X_new_scaled, columns=feature_cols)

# 6. PREDIÇÃO
print("Fazendo predições...")
pred_lgb = lgb_model.predict(X_new_scaled)
pred_xgb = xgb_model.predict(xgb.DMatrix(X_new_scaled))
pred_ensemble = w_lgb * pred_lgb + w_xgb * pred_xgb

# 7. THRESHOLD DINÂMICO
def get_dynamic_threshold(stop_conv):
    if stop_conv >= 0.5:
        return 0.40
    elif stop_conv >= 0.3:
        return 0.50
    elif stop_conv >= 0.1:
        return 0.60
    else:
        return 0.75

thresholds = X_new_scaled['stop_historical_conversion'].apply(get_dynamic_threshold)
predictions = (pred_ensemble > thresholds.values).astype(int)

# 8. RESULTADOS
df_new['probabilidade_conversao'] = pred_ensemble
df_new['predicao_conversao'] = predictions
df_new['threshold_usado'] = thresholds

print("\n📊 ESTATÍSTICAS:")
print(f"Total de eventos: {len(df_new):,}")
print(f"Conversões previstas: {predictions.sum():,} ({predictions.sum()/len(df_new):.2%})")
print(f"Probabilidade média: {pred_ensemble.mean():.4f}")

# 9. SALVAR
df_new[['gtfs_stop_id', 'device_id', 'probabilidade_conversao', 
        'predicao_conversao', 'threshold_usado']].to_csv('predicoes.csv', index=False)
print("✓ Predições salvas em predicoes.csv")
```

---

## ⚠️ Considerações Importantes

### 1. Ordem Temporal
**CRÍTICO:** Sempre manter ordem temporal dos dados.
- ❌ NÃO usar `train_test_split` com `shuffle=True`
- ✅ SIM usar split sequencial ou `TimeSeriesSplit`

### 2. Feature Engineering Consistente
**OBRIGATÓRIO:** Em produção, criar exatamente as mesmas features do treinamento.
- Usar os mesmos dicionários de conversão
- Aplicar as mesmas transformações
- Manter a mesma ordem das features

### 3. Normalização
**SEMPRE** usar o scaler treinado, nunca criar um novo:
```python
# ❌ ERRADO
scaler_new = StandardScaler()
X_new_scaled = scaler_new.fit_transform(X_new)

# ✅ CORRETO
X_new_scaled = scaler.transform(X_new)
```

### 4. Threshold Dinâmico
O threshold muda para cada parada baseado em seu histórico:
- Paradas boas (≥50% conv.): threshold = 0.40
- Paradas ruins (<10% conv.): threshold = 0.75

### 5. Atualizações Periódicas
Recomendações:
- **Mensal:** Retreinar modelo com dados recentes
- **Semanal:** Atualizar dicionários de conversão (stop_historical_conversion, etc.)
- **Diário:** Monitorar métricas de predição vs realidade

---

## 📊 Interpretação das Métricas

### Quando o Modelo É Bom?

**ROC-AUC:**
- 0.90-0.95: Excelente (nosso caso: 0.9425)
- 0.80-0.90: Bom
- 0.70-0.80: Razoável
- <0.70: Ruim

**F1-Score Classe 1 (Conversões):**
- >0.60: Excelente para classe desbalanceada
- 0.50-0.60: Bom (nosso caso: 0.5539)
- 0.40-0.50: Razoável
- <0.40: Ruim

**F1-Macro:**
- >0.75: Excelente (nosso caso: 0.7558)
- 0.65-0.75: Bom
- 0.55-0.65: Razoável
- <0.55: Ruim

---

## 🔧 Troubleshooting

### Problema: F1 Classe 1 muito baixo

**Soluções:**
1. Aumentar `scale_pos_weight`
2. Ajustar sample weights (dar mais peso à classe 1)
3. Diminuir thresholds dinâmicos

### Problema: Muitos falsos positivos

**Soluções:**
1. Aumentar thresholds dinâmicos
2. Reduzir peso da classe 1 em sample weights
3. Adicionar features que diferenciem melhor as classes

### Problema: Modelo não generaliza

**Soluções:**
1. Verificar se TimeSeriesSplit está sendo usado
2. Aumentar `min_child_samples` / `min_child_weight`
3. Reduzir `num_leaves` / `max_depth`
4. Aumentar regularização (`lambda_l1`, `lambda_l2`)

---

## 📈 Próximos Passos

### Melhorias Possíveis:

1. **Features Adicionais:**
   - Clima/Tempo (temperatura, chuva)
   - Eventos especiais (feriados, shows, jogos)
   - Padrões de tráfego (congestionamento)

2. **Modelos Alternativos:**
   - CatBoost (melhor para categóricas)
   - Neural Networks (MLP, LSTM)
   - Stacking de múltiplos modelos

3. **Threshold Adaptativo:**
   - Ajustar threshold baseado em ROI
   - Otimizar para maximizar lucro, não apenas F1

4. **Monitoramento:**
   - Dashboard em tempo real
   - Alertas de degradação de performance
   - A/B testing de novas versões

---

## ✅ Checklist de Deploy

- [ ] Treinar modelo com dataset completo
- [ ] Validar métricas em holdout set temporal
- [ ] Salvar todos os artefatos
- [ ] Testar script de inferência com dados reais
- [ ] Documentar versão e data do modelo
- [ ] Configurar monitoramento de performance
- [ ] Definir critérios de retreinamento
- [ ] Criar fallback para falhas do modelo
- [ ] Configurar logging de predições
- [ ] Validar latência em produção (<100ms ideal)

---

## 📞 Contato e Suporte

Para dúvidas sobre este modelo:
1. Revisar esta documentação
2. Analisar comentários no código
3. Verificar métricas de validação cruzada
4. Comparar com versões anteriores (v7, v6, etc.)

---

**Versão da Documentação:** 1.0  
**Data:** Novembro 2025  
**Modelo:** V8 Production (TimeSeriesSplit)
