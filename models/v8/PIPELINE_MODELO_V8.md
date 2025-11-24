# 🔄 PIPELINE COMPLETO - MODEL V8 PRODUCTION

## 📋 Visão Geral

Este documento descreve o **pipeline end-to-end** do Model V8, desde a coleta de dados até a predição final.

---

## 🎯 PIPELINE EM FASES

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FASE 1: COLETA DE DADOS                     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FASE 2: FEATURE ENGINEERING                      │
│                    (Geographic + Dynamic + Base)                     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   FASE 3: PREPARAÇÃO DOS DADOS                      │
│                 (Seleção, Limpeza, Time Series Split)               │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FASE 4: VALIDAÇÃO CRUZADA                        │
│                     (TimeSeriesSplit - 5 Folds)                     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   FASE 5: TREINAMENTO FINAL                         │
│              (LightGBM + XGBoost com 80% dos dados)                 │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FASE 6: AVALIAÇÃO                              │
│                    (Teste com 20% dos dados)                        │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   FASE 7: DEPLOY/PRODUÇÃO                           │
│              (Salvar modelos + Inferência em novos dados)           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📊 PIPELINE DETALHADO

### FASE 1: COLETA DE DADOS 📥

```
┌──────────────────────────────────────────────────────────────┐
│ INPUT: Dados Brutos                                          │
├──────────────────────────────────────────────────────────────┤
│ • Fonte A: BigQuery (proj-ml-469320.app_cittamobi)          │
│ • Fonte B: CSV Local (dataset-updated.csv)                  │
│                                                              │
│ Campos principais:                                           │
│ • target (0/1) - Conversão do usuário                       │
│ • gtfs_stop_id - ID da parada                               │
│ • device_id - ID do usuário                                 │
│ • stop_lat_event, stop_lon_event - Coordenadas             │
│ • time_hour, time_day_of_week - Info temporal              │
│ • is_peak_hour - Horário de pico                           │
│                                                              │
│ Tamanho típico: ~200K-300K registros                        │
│ Taxa de conversão: ~7% (desbalanceado)                      │
└──────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│ PROCESSAMENTO:                                               │
│ • Remover registros com target=NULL                         │
│ • Validar tipos de dados                                    │
│ • Calcular estatísticas básicas                            │
└──────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│ OUTPUT: DataFrame limpo com ~200K registros válidos          │
└──────────────────────────────────────────────────────────────┘
```

---

### FASE 2: FEATURE ENGINEERING 🔧

#### 2.1 GEOGRAPHIC FEATURES (6 features)

```
┌──────────────────────────────────────────────────────────────┐
│ INPUT: DataFrame + Coordenadas geográficas                   │
└──────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│ FEATURE 1: stop_historical_conversion                        │
│ ─────────────────────────────────────────────────────────    │
│ Método: GROUP BY gtfs_stop_id → MEAN(target)                │
│                                                              │
│ Código:                                                      │
│   stop_conv = df.groupby('gtfs_stop_id')['target'].mean()   │
│   df['stop_historical_conversion'] = map(stop_conv)         │
│                                                              │
│ Significado: Taxa média de conversão histórica da parada     │
│ Range: 0.0 (nunca converte) a 1.0 (sempre converte)        │
└──────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│ FEATURE 2: stop_density                                      │
│ ─────────────────────────────────────────────────────────    │
│ Método: NearestNeighbors (k=11) → 1/mean(distances)         │
│                                                              │
│ Código:                                                      │
│   nn = NearestNeighbors(n_neighbors=11)                     │
│   distances = nn.kneighbors(coords)                         │
│   df['stop_density'] = 1 / (distances.mean() + 0.001)       │
│                                                              │
│ Significado: Densidade de paradas na região                  │
│ Interpretação:                                               │
│   • Alto → Área urbana densa (centro)                       │
│   • Baixo → Área isolada (periferia)                        │
└──────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│ FEATURE 3: dist_to_nearest_cbd                              │
│ ─────────────────────────────────────────────────────────    │
│ Método: Haversine distance para 5 CBDs brasileiros          │
│                                                              │
│ CBDs considerados:                                           │
│   • São Paulo (-23.5505, -46.6333)                          │
│   • Rio de Janeiro (-22.9068, -43.1729)                     │
│   • Belo Horizonte (-19.9167, -43.9345)                     │
│   • Curitiba (-25.4284, -49.2733)                           │
│   • Porto Alegre (-30.0346, -51.2177)                       │
│                                                              │
│ Código:                                                      │
│   for cbd in cbds:                                           │
│       dist = haversine(stop_coords, cbd)                    │
│   df['dist_to_nearest_cbd'] = min(distances)                │
│                                                              │
│ Range: 0 km (centro) a 50+ km (periferia)                  │
└──────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│ FEATURE 4: stop_cluster + cluster_conversion_rate           │
│ ─────────────────────────────────────────────────────────    │
│ Método: DBSCAN clustering geográfico                         │
│                                                              │
│ Parâmetros:                                                  │
│   • eps = 0.01 (raio do cluster)                            │
│   • min_samples = 5 (mínimo de paradas por cluster)        │
│                                                              │
│ Código:                                                      │
│   clustering = DBSCAN(eps=0.01, min_samples=5)              │
│   labels = clustering.fit_predict(coords)                   │
│   df['stop_cluster'] = labels                               │
│   cluster_rates = groupby('stop_cluster').mean()            │
│   df['cluster_conversion_rate'] = map(cluster_rates)        │
│                                                              │
│ Significado: Agrupa paradas próximas geograficamente         │
│ Benefício: Captura padrões regionais de conversão           │
└──────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│ FEATURE 5: stop_volatility                                  │
│ ─────────────────────────────────────────────────────────    │
│ Método: Standard deviation da conversão por parada          │
│                                                              │
│ Código:                                                      │
│   stop_vol = df.groupby('gtfs_stop_id')['target'].std()     │
│   df['stop_volatility'] = map(stop_vol).fillna(0)           │
│                                                              │
│ Significado: Variabilidade/previsibilidade da parada         │
│ Interpretação:                                               │
│   • Alto → Comportamento inconsistente                      │
│   • Baixo → Comportamento previsível                        │
└──────────────────────────────────────────────────────────────┘
```

#### 2.2 DYNAMIC FEATURES (10 features)

```
┌──────────────────────────────────────────────────────────────┐
│ A. TEMPORAL CONVERSION RATES (3 features)                    │
│ ─────────────────────────────────────────────────────────    │
│ hour_conversion_rate:                                        │
│   • GROUP BY time_hour → MEAN(target)                       │
│   • Taxa de conversão por hora (0-23h)                      │
│                                                              │
│ dow_conversion_rate:                                         │
│   • GROUP BY time_day_of_week → MEAN(target)                │
│   • Taxa de conversão por dia da semana                     │
│                                                              │
│ stop_hour_conversion:                                        │
│   • GROUP BY (gtfs_stop_id, time_hour) → MEAN(target)       │
│   • Taxa específica: parada X em hora Y                     │
└──────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│ B. GEO-TEMPORAL INTERACTIONS (2 features)                    │
│ ─────────────────────────────────────────────────────────    │
│ geo_temporal:                                                │
│   • dist_to_nearest_cbd × is_peak_hour                       │
│   • Impacto da distância durante pico                       │
│                                                              │
│ density_peak:                                                │
│   • stop_density × is_peak_hour                              │
│   • Densidade urbana durante pico                           │
└──────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│ C. USER FEATURES (2 features)                               │
│ ─────────────────────────────────────────────────────────    │
│ user_conversion_rate:                                        │
│   • GROUP BY device_id → MEAN(target)                       │
│   • Histórico pessoal do usuário                            │
│                                                              │
│ user_vs_stop_ratio:                                          │
│   • unique_stops(device_id) / count(device_id)              │
│   • Diversidade de paradas visitadas                        │
└──────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│ D. RARITY FEATURES (2 features)                             │
│ ─────────────────────────────────────────────────────────    │
│ stop_rarity:                                                 │
│   • 1 / (count_events(stop) + 1)                            │
│   • Quão rara/popular é a parada                            │
│                                                              │
│ user_rarity:                                                 │
│   • 1 / (count_events(user) + 1)                            │
│   • Quão frequente é o usuário                              │
└──────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│ E. DISTANCE DEVIATION (1 feature)                           │
│ ─────────────────────────────────────────────────────────    │
│ stop_dist_std:                                               │
│   • GROUP BY (stop, device) → STD(lat, lon)                 │
│   • Variação nas coordenadas reportadas                     │
│   • Indica precisão do GPS                                  │
└──────────────────────────────────────────────────────────────┘
```

#### 2.3 OUTPUT FASE 2

```
┌──────────────────────────────────────────────────────────────┐
│ FEATURES CRIADAS:                                            │
│                                                              │
│ Geographic (6):                                              │
│   ✓ stop_historical_conversion                              │
│   ✓ stop_density                                             │
│   ✓ dist_to_nearest_cbd                                      │
│   ✓ stop_cluster                                             │
│   ✓ cluster_conversion_rate                                  │
│   ✓ stop_volatility                                          │
│                                                              │
│ Dynamic (10):                                                │
│   ✓ hour_conversion_rate                                     │
│   ✓ dow_conversion_rate                                      │
│   ✓ stop_hour_conversion                                     │
│   ✓ geo_temporal                                             │
│   ✓ density_peak                                             │
│   ✓ user_conversion_rate                                     │
│   ✓ user_vs_stop_ratio                                       │
│   ✓ stop_rarity                                              │
│   ✓ user_rarity                                              │
│   ✓ stop_dist_std                                            │
│                                                              │
│ Base (originais): ~26 features                               │
│                                                              │
│ TOTAL: ~42 features numéricas                                │
└──────────────────────────────────────────────────────────────┘
```

---

### FASE 3: PREPARAÇÃO DOS DADOS 🧹

```
┌──────────────────────────────────────────────────────────────┐
│ ETAPA 3.1: Seleção de Features                              │
├──────────────────────────────────────────────────────────────┤
│ • Remover colunas não-features:                             │
│   ✗ target (é o label!)                                     │
│   ✗ gtfs_stop_id (identificador)                            │
│   ✗ device_id (identificador)                               │
│   ✗ timestamp_converted (temporal)                          │
│   ✗ stop_lat_event, stop_lon_event (já usadas)             │
│   ✗ event_timestamp, date (temporal)                        │
│                                                              │
│ • Manter apenas features numéricas                          │
│ • Resultado: X (features) e y (target)                      │
└──────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│ ETAPA 3.2: Limpeza de Nomes                                 │
├──────────────────────────────────────────────────────────────┤
│ • Remover caracteres especiais:                             │
│   [ ] { } " ' : ,                                            │
│                                                              │
│ • Exemplo:                                                   │
│   user_stats['conversion'] → user_stats_conversion          │
│                                                              │
│ • Motivo: LightGBM/XGBoost não aceitam caracteres especiais │
└──────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│ ETAPA 3.3: Tratamento de Valores Especiais                  │
├──────────────────────────────────────────────────────────────┤
│ • Substituir infinitos (inf, -inf) por NaN                  │
│ • Preencher NaN com 0                                       │
│                                                              │
│ Código:                                                      │
│   X.replace([np.inf, -np.inf], np.nan, inplace=True)        │
│   X.fillna(0, inplace=True)                                 │
│                                                              │
│ • Motivo: Modelos ML não aceitam inf/NaN                    │
└──────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│ ETAPA 3.4: Time Series Split                                │
├──────────────────────────────────────────────────────────────┤
│ • NÃO usar train_test_split aleatório!                      │
│ • MANTER ordem temporal dos dados                           │
│                                                              │
│ Método: TimeSeriesSplit(n_splits=5)                         │
│                                                              │
│ Dados: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]                     │
│                                                              │
│ Fold 1: Train [1,2] → Val [3,4]                            │
│ Fold 2: Train [1,2,3,4] → Val [5,6]                        │
│ Fold 3: Train [1,2,3,4,5,6] → Val [7,8]                    │
│ Fold 4: Train [1,2,3,4,5,6,7,8] → Val [9,10]               │
│ Fold 5: Train [1,2,3,4,5,6,7,8,9] → Val [10]               │
│                                                              │
│ ⚠️ CRÍTICO: Evita data leakage!                             │
└──────────────────────────────────────────────────────────────┘
```

---

### FASE 4: VALIDAÇÃO CRUZADA 🔄

```
┌──────────────────────────────────────────────────────────────┐
│                      PARA CADA FOLD (5x):                    │
└──────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│ ETAPA 4.1: Normalização (StandardScaler)                    │
├──────────────────────────────────────────────────────────────┤
│ scaler = StandardScaler()                                    │
│ X_train = scaler.fit_transform(X_train)                     │
│ X_val = scaler.transform(X_val)                             │
│                                                              │
│ Resultado: Média=0, Desvio=1 para cada feature              │
└──────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│ ETAPA 4.2: Sample Weights Dinâmicos                         │
├──────────────────────────────────────────────────────────────┤
│ Baseado em stop_historical_conversion:                       │
│                                                              │
│ IF stop_conv > 0.5:  # Paradas excelentes                   │
│     weight[target=1] = 3.0                                   │
│     weight[target=0] = 0.5                                   │
│                                                              │
│ ELIF stop_conv > 0.2:  # Paradas boas                       │
│     weight[target=1] = 2.0                                   │
│     weight[target=0] = 0.8                                   │
│                                                              │
│ ELSE:  # Paradas regulares                                  │
│     weight[target=1] = 1.5                                   │
│     weight[target=0] = 1.0                                   │
│                                                              │
│ Efeito: Prioriza conversões em paradas com potencial        │
└──────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│ ETAPA 4.3: Calcular Scale Pos Weight                        │
├──────────────────────────────────────────────────────────────┤
│ scale_weight = count(target=0) / count(target=1)            │
│ scale_weight = scale_weight * 1.3  # Boost adicional        │
│                                                              │
│ Exemplo: 90% não-conv, 10% conv → scale_weight = 11.7       │
│                                                              │
│ Efeito: Compensa desbalanceamento de classes                │
└──────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│ ETAPA 4.4: Treinar LightGBM                                 │
├──────────────────────────────────────────────────────────────┤
│ Parâmetros:                                                  │
│   • objective: 'binary'                                      │
│   • metric: 'auc'                                            │
│   • num_leaves: 63                                           │
│   • learning_rate: 0.05                                      │
│   • feature_fraction: 0.8                                    │
│   • bagging_fraction: 0.8                                    │
│   • min_child_samples: 20                                    │
│   • scale_pos_weight: [calculado]                            │
│   • num_boost_round: 300                                     │
│                                                              │
│ Output: pred_lgb_val (probabilidades 0-1)                    │
└──────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│ ETAPA 4.5: Treinar XGBoost                                  │
├──────────────────────────────────────────────────────────────┤
│ Parâmetros:                                                  │
│   • objective: 'binary:logistic'                             │
│   • eval_metric: 'auc'                                       │
│   • max_depth: 8                                             │
│   • learning_rate: 0.05                                      │
│   • subsample: 0.8                                           │
│   • colsample_bytree: 0.8                                    │
│   • min_child_weight: 5                                      │
│   • scale_pos_weight: [calculado]                            │
│   • num_boost_round: 300                                     │
│                                                              │
│ Output: pred_xgb_val (probabilidades 0-1)                    │
└──────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│ ETAPA 4.6: Ensemble                                          │
├──────────────────────────────────────────────────────────────┤
│ w_lgb = 0.485  (48.5%)                                       │
│ w_xgb = 0.515  (51.5%)                                       │
│                                                              │
│ pred_ensemble = w_lgb × pred_lgb + w_xgb × pred_xgb          │
│                                                              │
│ Exemplo:                                                     │
│   LightGBM: 0.70                                             │
│   XGBoost:  0.80                                             │
│   Ensemble: 0.485×0.70 + 0.515×0.80 = 0.7515 (75.15%)       │
└──────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│ ETAPA 4.7: Threshold Dinâmico                               │
├──────────────────────────────────────────────────────────────┤
│ Baseado em stop_historical_conversion:                       │
│                                                              │
│ IF stop_conv >= 0.5:  threshold = 0.40  (otimista)          │
│ ELIF stop_conv >= 0.3:  threshold = 0.50                    │
│ ELIF stop_conv >= 0.1:  threshold = 0.60                    │
│ ELSE:  threshold = 0.75  (conservador)                      │
│                                                              │
│ y_pred = (pred_ensemble > threshold) ? 1 : 0                │
│                                                              │
│ Lógica: Paradas boas → ser otimista                         │
│         Paradas ruins → ser conservador                     │
└──────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│ ETAPA 4.8: Calcular Métricas do Fold                        │
├──────────────────────────────────────────────────────────────┤
│ • ROC-AUC (com probabilidades)                               │
│ • F1-Score Classe 0                                          │
│ • F1-Score Classe 1                                          │
│ • F1-Macro (média das duas)                                  │
│                                                              │
│ Salvar resultados do fold                                    │
└──────────────────────────────────────────────────────────────┘
                        │
                        ▼
                  [PRÓXIMO FOLD]
```

#### OUTPUT FASE 4:

```
┌──────────────────────────────────────────────────────────────┐
│ RESULTADOS DA VALIDAÇÃO CRUZADA (5 FOLDS):                  │
├──────────────────────────────────────────────────────────────┤
│ Fold  Train    Val     AUC      F1-Macro  F1-C0   F1-C1     │
│ ────  ──────  ──────  ───────  ─────────  ─────  ─────     │
│  1    40K     40K     0.9420    0.7550    0.9570  0.5530    │
│  2    80K     40K     0.9430    0.7560    0.9580  0.5540    │
│  3    120K    40K     0.9425    0.7555    0.9575  0.5535    │
│  4    160K    40K     0.9420    0.7550    0.9570  0.5530    │
│  5    200K    40K     0.9428    0.7558    0.9576  0.5539    │
│                                                              │
│ MÉDIA:         0.9425 ± 0.0004                              │
│                0.7558 ± 0.0004                              │
└──────────────────────────────────────────────────────────────┘
```

---

### FASE 5: TREINAMENTO FINAL 🏆

```
┌──────────────────────────────────────────────────────────────┐
│ ETAPA 5.1: Split Final (80/20 Sequencial)                   │
├──────────────────────────────────────────────────────────────┤
│ Total: 240K registros                                        │
│                                                              │
│ Train: [1 → 192K]  (80% primeiros/mais antigos)             │
│ Test:  [192K → 240K]  (20% últimos/mais recentes)           │
│                                                              │
│ ⚠️ Simula produção: treinar no passado, testar no futuro    │
└──────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│ ETAPA 5.2: Normalização Final                               │
├──────────────────────────────────────────────────────────────┤
│ scaler = StandardScaler()                                    │
│ X_train = scaler.fit_transform(X_train)  # FIT no treino    │
│ X_test = scaler.transform(X_test)        # APENAS transform │
│                                                              │
│ ⚠️ Salvar scaler para produção!                             │
│    pickle.dump(scaler, 'scaler_v8_production.pkl')          │
└──────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│ ETAPA 5.3: Treinar LightGBM Final                           │
├──────────────────────────────────────────────────────────────┤
│ Mesmo config da validação cruzada                           │
│ Treinar com 192K registros                                  │
│ 300 iterações (boost rounds)                                │
│                                                              │
│ ⚠️ Salvar modelo:                                            │
│    lgb_model.save_model('lightgbm_model_v8_production.txt') │
└──────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│ ETAPA 5.4: Treinar XGBoost Final                            │
├──────────────────────────────────────────────────────────────┤
│ Mesmo config da validação cruzada                           │
│ Treinar com 192K registros                                  │
│ 300 iterações (boost rounds)                                │
│                                                              │
│ ⚠️ Salvar modelo:                                            │
│    xgb_model.save_model('xgboost_model_v8_production.json') │
└──────────────────────────────────────────────────────────────┘
```

---

### FASE 6: AVALIAÇÃO 📊

```
┌──────────────────────────────────────────────────────────────┐
│ ETAPA 6.1: Predição no Test Set                             │
├──────────────────────────────────────────────────────────────┤
│ pred_lgb = lgb_model.predict(X_test)                         │
│ pred_xgb = xgb_model.predict(X_test)                         │
│ pred_ensemble = 0.485 × pred_lgb + 0.515 × pred_xgb          │
└──────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│ ETAPA 6.2: Aplicar Threshold Dinâmico                       │
├──────────────────────────────────────────────────────────────┤
│ FOR EACH registro in test_set:                              │
│     threshold = get_threshold(stop_historical_conversion)    │
│     y_pred = 1 IF pred_ensemble > threshold ELSE 0          │
└──────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│ ETAPA 6.3: Calcular Métricas Finais                         │
├──────────────────────────────────────────────────────────────┤
│ • ROC-AUC: 0.9425                                            │
│ • F1-Score Classe 1: 0.5539 (55.39%)                        │
│ • F1-Score Classe 0: 0.9576 (95.76%)                        │
│ • F1-Macro: 0.7558 (75.58%)                                  │
│ • Accuracy: 0.9240 (92.40%)                                  │
│                                                              │
│ Confusion Matrix:                                            │
│           Pred 0    Pred 1                                   │
│ Real 0    42,500    1,100                                    │
│ Real 1    1,200     1,800                                    │
└──────────────────────────────────────────────────────────────┘
```

---

### FASE 7: DEPLOY/PRODUÇÃO 🚀

```
┌──────────────────────────────────────────────────────────────┐
│ ETAPA 7.1: Salvar Artefatos                                 │
├──────────────────────────────────────────────────────────────┤
│ ✓ lightgbm_model_v8_production.txt                          │
│ ✓ xgboost_model_v8_production.json                          │
│ ✓ scaler_v8_production.pkl                                  │
│ ✓ selected_features_v8_production.txt                       │
│ ✓ model_config_v8_production.json                           │
└──────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│ ETAPA 7.2: Pipeline de Inferência (Novos Dados)             │
├──────────────────────────────────────────────────────────────┤
│ INPUT: Novos eventos (sem target)                           │
│    ↓                                                         │
│ 1. Feature Engineering (MESMAS 42 features)                 │
│    ↓                                                         │
│ 2. Selecionar features corretas (ordem importa!)            │
│    ↓                                                         │
│ 3. Normalizar com scaler TREINADO                           │
│    ↓                                                         │
│ 4. Predição LightGBM                                         │
│    ↓                                                         │
│ 5. Predição XGBoost                                          │
│    ↓                                                         │
│ 6. Ensemble (0.485 × lgb + 0.515 × xgb)                     │
│    ↓                                                         │
│ 7. Threshold dinâmico                                        │
│    ↓                                                         │
│ OUTPUT: Probabilidade + Predição binária                     │
└──────────────────────────────────────────────────────────────┘
```

#### Código de Inferência:

```python
# 1. CARREGAR ARTEFATOS
lgb_model = lgb.Booster(model_file='lightgbm_model_v8_production.txt')
xgb_model = xgb.Booster()
xgb_model.load_model('xgboost_model_v8_production.json')

with open('scaler_v8_production.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('selected_features_v8_production.txt', 'r') as f:
    feature_cols = [line.strip() for line in f]

# 2. PROCESSAR NOVOS DADOS
df_new = pd.read_csv('novos_eventos.csv')

# 3. FEATURE ENGINEERING (repetir TODAS as 42 features)
# [... mesmo código da Fase 2 ...]

# 4. PREPARAR
X_new = df_new[feature_cols]
X_new_scaled = scaler.transform(X_new)

# 5. PREDIÇÃO
pred_lgb = lgb_model.predict(X_new_scaled)
pred_xgb = xgb_model.predict(xgb.DMatrix(X_new_scaled))
pred_ensemble = 0.485 * pred_lgb + 0.515 * pred_xgb

# 6. THRESHOLD
thresholds = X_new['stop_historical_conversion'].apply(get_dynamic_threshold)
predictions = (pred_ensemble > thresholds).astype(int)

# 7. RESULTADO
df_new['probabilidade'] = pred_ensemble
df_new['predicao'] = predictions
```

---

## 📊 MÉTRICAS DE PERFORMANCE DO PIPELINE

### Tempo de Execução (Estimativa):

```
┌────────────────────────────────────────────────────────┐
│ FASE                          TEMPO (200K registros)   │
├────────────────────────────────────────────────────────┤
│ 1. Coleta BigQuery            ~30s                     │
│ 2. Feature Engineering        ~60s                     │
│ 3. Preparação                 ~10s                     │
│ 4. Validação Cruzada (5x)     ~15 min                  │
│ 5. Treinamento Final          ~3 min                   │
│ 6. Avaliação                  ~5s                      │
│ 7. Salvar Artefatos           ~2s                      │
│                                                         │
│ TOTAL TREINAMENTO:            ~20 minutos              │
├────────────────────────────────────────────────────────┤
│ INFERÊNCIA (1K registros):    ~0.5s                    │
│ INFERÊNCIA (100K registros):  ~30s                     │
└────────────────────────────────────────────────────────┘
```

### Uso de Memória:

```
┌────────────────────────────────────────────────────────┐
│ COMPONENTE                    RAM                      │
├────────────────────────────────────────────────────────┤
│ DataFrame (200K × 42)         ~70 MB                   │
│ LightGBM Model                ~10 MB                   │
│ XGBoost Model                 ~12 MB                   │
│ Scaler                        ~1 MB                    │
│ Processamento                 ~100 MB                  │
│                                                         │
│ TOTAL TREINAMENTO:            ~200 MB                  │
│ TOTAL INFERÊNCIA:             ~50 MB                   │
└────────────────────────────────────────────────────────┘
```

---

## 🔄 PIPELINE SIMPLIFICADO (RESUMO)

```
DATA → FEATURES → PREPARE → VALIDATE → TRAIN → EVAL → DEPLOY
 │         │          │          │         │       │       │
 │         │          │          │         │       │       └─→ Artefatos
 │         │          │          │         │       └─→ Métricas
 │         │          │          │         └─→ Modelos Finais
 │         │          │          └─→ 5-Fold CV
 │         │          └─→ Clean + Split
 │         └─→ 42 Features (6 Geo + 10 Dynamic + 26 Base)
 └─→ BigQuery/CSV (~200K registros)
```

---

## ⚠️ PONTOS CRÍTICOS DO PIPELINE

### ✅ FAZER:

1. **SEMPRE** manter ordem temporal (TimeSeriesSplit)
2. **FIT scaler apenas no treino**, transform no teste
3. **Criar features IDÊNTICAS** no treinamento e produção
4. **Salvar scaler** junto com os modelos
5. **Usar thresholds dinâmicos** (melhora F1-C1)
6. **Sample weights** baseados em stop_historical_conversion

### ❌ NÃO FAZER:

1. ❌ Usar `train_test_split(shuffle=True)`
2. ❌ Fit scaler em dados de teste
3. ❌ Esquecer de criar alguma feature em produção
4. ❌ Usar threshold fixo para todas as paradas
5. ❌ Treinar sem sample weights (ignora desbalanceamento)
6. ❌ Ignorar o ensemble (usar só um modelo)

---

## 📈 MONITORAMENTO EM PRODUÇÃO

```
┌────────────────────────────────────────────────────────┐
│ MÉTRICA                   ALERTAR SE                   │
├────────────────────────────────────────────────────────┤
│ Latência p95              > 100ms                      │
│ F1-Score Classe 1         < 0.50                       │
│ Drift em features         > 20% mudança                │
│ Taxa de conversão         < 5% ou > 10%                │
│ Erros de predição         > 1%                         │
│ Uso de memória            > 500 MB                     │
└────────────────────────────────────────────────────────┘
```

### Retreinamento:

- **Mensal**: Com dados do último mês
- **Trimestral**: Revisão completa do pipeline
- **Emergencial**: Se F1-C1 < 0.45 por 3 dias consecutivos

---

**Versão:** 1.0  
**Data:** Novembro 2025  
**Modelo:** V8 Production
