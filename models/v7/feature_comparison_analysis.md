# 🔬 ANÁLISE COMPARATIVA: Features V7 vs OFICIAL.ipynb

## 📊 RESUMO EXECUTIVO

| Aspecto | MODEL V7 | OFICIAL.ipynb | Vencedor |
|---------|----------|---------------|----------|
| **Features Totais** | 53 → 49 selecionadas | 9 features fixas | ✅ V7 (5.4x mais) |
| **Seleção de Features** | ✅ Automática (XGBoost importance) | ❌ Manual | ✅ V7 |
| **Agregações por Usuário** | ✅ 9 features | ❌ Não tem | ✅ V7 |
| **Agregações por Parada** | ✅ 7 features | ❌ Não tem | ✅ V7 |
| **Features Temporais** | ✅ 13 features (cíclicas + contexto) | ✅ 5 features | ✅ V7 |
| **Features GTFS** | ✅ 2 features (headway) | ✅ 1 feature (headway) | ≈ Empate |
| **Interações** | ✅ 6 features (2ª ordem) | ✅ 4 features | ✅ V7 |
| **Features Geoespaciais** | ✅ 8 features | ✅ 1 feature | ✅ V7 |

---

## 📋 INVENTÁRIO COMPLETO DE FEATURES

### **1. FEATURES TEMPORAIS**

#### **V7 (13 features)**
```python
✅ time_hour                    # Hora do dia (0-23)
✅ time_day_of_week             # Dia da semana (0-6)
✅ time_day_of_month            # Dia do mês (1-31)
✅ time_month                   # Mês (1-12) [NÃO selecionada no top 49]
✅ week_of_year                 # Semana do ano [NÃO selecionada]

# Features Cíclicas (sin/cos)
✅ hour_sin, hour_cos           # Hora como círculo
✅ day_sin, day_cos             # Dia da semana como círculo
✅ month_sin, month_cos         # Mês como círculo [month_cos selecionada]

# Features de Contexto Urbano
✅ is_holiday                   # É feriado? (Brasil-SP)
✅ is_weekend                   # É fim de semana?
✅ is_peak_hour                 # É hora de pico? (6-9h, 17-19h)
```

#### **OFICIAL.ipynb (5 features)**
```python
✅ time_hour                    # Hora do dia
✅ time_day_of_week             # Dia da semana
✅ is_holiday                   # É feriado?
✅ is_weekend                   # É fim de semana?
✅ is_peak_hour                 # É hora de pico?
```

**DIFERENÇAS:**
- ❌ OFICIAL.ipynb **NÃO tem** features cíclicas (sin/cos) na lista final
- ❌ OFICIAL.ipynb **NÃO tem** time_day_of_month, time_month, week_of_year
- ✅ V7 tem **8 features extras** (cíclicas + granularidade temporal)

---

### **2. FEATURES DE AGREGAÇÃO POR USUÁRIO**

#### **V7 (9 features) - CRÍTICAS!**
```python
✅ user_conversion_rate         # Taxa histórica de conversão do usuário
✅ user_total_conversions       # Total de conversões do usuário
✅ user_frequency               # Frequência de uso (count de eventos)
✅ user_avg_dist                # Distância média percorrida
✅ user_std_dist                # Desvio padrão da distância
✅ user_min_dist                # Distância mínima
✅ user_max_dist                # Distância máxima
✅ user_avg_hour                # Hora média de uso
✅ user_std_hour                # Desvio padrão da hora
```

**IMPORTÂNCIA:**
- `user_conversion_rate` é a **2ª feature mais importante** (gain: 162.28)!
- `user_total_conversions` é a **4ª mais importante** (gain: 56.31)!
- `user_max_dist` é a **9ª mais importante** (gain: 50.58)!

#### **OFICIAL.ipynb (0 features)**
```python
❌ NÃO TEM agregações por usuário
```

**IMPACTO:**
- 🔥 **MAIOR DIFERENÇA**: V7 captura **comportamento individual** do usuário
- Permite identificar usuários "convertedores" vs "navegadores"
- Explica por que V7 tem recall 73.6% vs ~50% do OFICIAL

---

### **3. FEATURES DE AGREGAÇÃO POR PARADA**

#### **V7 (7 features)**
```python
✅ stop_conversion_rate         # Taxa histórica de conversão na parada
✅ stop_total_conversions       # Total de conversões na parada
✅ stop_event_count_agg         # Total de eventos na parada
✅ stop_avg_dist                # Distância média dos usuários
✅ stop_dist_std                # Desvio padrão da distância
✅ stop_lat_agg                 # Latitude agregada da parada
✅ stop_lon_agg                 # Longitude agregada da parada
```

**IMPORTÂNCIA:**
- `stop_lon_agg` é a **3ª feature mais importante** (gain: 62.65)!
- `stop_total_conversions` é a **6ª mais importante** (gain: 53.35)!
- `stop_lat_agg` é a **8ª mais importante** (gain: 50.85)!

#### **OFICIAL.ipynb (7 features criadas, mas NÃO usadas no modelo final!)**
```python
# Criadas na Célula 14, mas NÃO incluídas na lista FEATURES:
❌ stop_event_rate              # Taxa de eventos na parada (criada mas não usada)
❌ stop_event_count             # Contagem de eventos (criada mas não usada)
❌ stop_total_samples           # Total de amostras (criada mas não usada)
❌ stop_dist_mean               # Distância média (criada mas não usada)
❌ stop_dist_std                # Desvio padrão (criada mas não usada)
❌ stop_headway_mean            # Headway médio (criada mas não usada)
❌ stop_headway_std             # Headway std (criada mas não usada)
```

**PROBLEMA NO OFICIAL.ipynb:**
```python
# Célula 14 - Cria as features
df_final = df_final.merge(stop_event_rate, on='gtfs_stop_id', how='left')

# Célula 15 - Define features para o modelo
FEATURES = [
    'time_hour',
    'time_day_of_week',
    'is_holiday',
    'is_weekend',
    'is_peak_hour',
    'dist_device_stop',
    'headway_avg_stop_hour',
    'gtfs_stop_id'  # ❌ Usa ID, mas NÃO as agregações!
]
```

🚨 **BUG CRÍTICO**: OFICIAL.ipynb **cria** 7 agregações por parada mas **não as usa** no modelo!

---

### **4. FEATURES DE INTERAÇÃO (2ª ORDEM)**

#### **V7 (6 features)**
```python
# Interações User × Stop
✅ conversion_interaction       # user_conversion_rate × stop_conversion_rate
✅ distance_interaction         # user_avg_dist × dist_device_stop
✅ frequency_interaction        # user_frequency × stop_event_count_agg

# Interações Temporais
✅ dist_x_peak                  # dist_device_stop × is_peak_hour
✅ dist_x_weekend               # dist_device_stop × is_weekend
✅ headway_x_hour               # headway × time_hour [NÃO selecionada no top 49]
✅ headway_x_weekend            # headway × is_weekend
```

**IMPORTÂNCIA:**
- 🔥 `conversion_interaction` é a **FEATURE MAIS IMPORTANTE** (gain: 4328.72)!
- `dist_x_peak` é a **5ª mais importante** (gain: 53.69)!
- `distance_interaction` é a **10ª mais importante** (gain: 48.47)!

#### **OFICIAL.ipynb (4 features criadas na Célula 13)**
```python
✅ headway_x_hour               # headway × time_hour
✅ headway_x_weekend            # headway × is_weekend
✅ dist_x_peak                  # dist_device_stop × is_peak_hour
✅ dist_x_weekend               # dist_device_stop × is_weekend
```

**DIFERENÇAS:**
- ❌ OFICIAL.ipynb **NÃO tem** interações User × Stop (não tem features de usuário!)
- ❌ `conversion_interaction` (a mais importante!) **não existe** no OFICIAL
- ✅ V7 tem interações mais ricas porque tem agregações de usuário

---

### **5. FEATURES GEOESPACIAIS**

#### **V7 (8 features)**
```python
✅ device_lat                   # Latitude do dispositivo
✅ device_lon                   # Longitude do dispositivo
✅ stop_lat_event               # Latitude da parada do evento
✅ stop_lon_event               # Longitude da parada do evento
✅ stop_lat_agg                 # Latitude agregada (por gtfs_stop_id)
✅ stop_lon_agg                 # Longitude agregada
✅ dist_device_stop             # Distância euclidiana (metros)
✅ gtfs_stop_id                 # ID da parada GTFS (categórica)
```

#### **OFICIAL.ipynb (1 feature)**
```python
✅ dist_device_stop             # Distância (calculada com geodesic)
✅ gtfs_stop_id                 # ID da parada (usado como categórica)
```

**DIFERENÇAS:**
- ✅ OFICIAL.ipynb calcula `dist_device_stop` com **geopy.geodesic** (mais preciso)
- ✅ V7 usa coordenadas **brutas** (lat/lon) como features
- ✅ V7 tem `stop_lat_agg` e `stop_lon_agg` (agregações geográficas)
- 📍 `stop_lon_agg` (#3) e `stop_lat_agg` (#8) são **muito importantes** no V7

---

### **6. FEATURES DE SERVIÇO (GTFS)**

#### **V7 (2 features + agregações)**
```python
✅ headway_avg_stop_hour        # Headway médio por parada/hora (do dataset)
✅ stop_headway_mean            # Headway médio agregado por parada
✅ stop_headway_std             # Desvio padrão do headway
```

#### **OFICIAL.ipynb (1 feature)**
```python
✅ headway_avg_stop_hour        # Headway médio por parada/hora
                                 # (calculado via merge GTFS: stop_times + frequencies)
```

**DIFERENÇAS:**
- ✅ OFICIAL.ipynb **calcula do zero** usando arquivos GTFS (stops.txt, frequencies.txt)
- ✅ V7 **assume que já existe** no BigQuery dataset
- ⚠️ OFICIAL.ipynb tem cálculo **mais preciso** (direto da fonte GTFS)
- ✅ V7 tem agregações extras (mean, std) por parada

---

### **7. FEATURES NÃO USADAS / PROBLEMAS**

#### **V7 - Features com problemas de nome:**
```python
⚠️ Unnamed: 0                   # Coluna de índice do pandas (lixo)
⚠️ int64_field_0                # Campo desconhecido do BigQuery
⚠️ user_frequency_x             # Duplicata? (merge issue)
⚠️ user_frequency_y             # Duplicata? (merge issue)
⚠️ stop_dist_std_x              # Duplicata? (merge issue)
⚠️ stop_dist_std_y              # Duplicata? (merge issue)
⚠️ stop_event_count             # Duplicata de stop_event_count_agg?
⚠️ stop_total_samples           # Feature não documentada
```

**PROBLEMA**: Possíveis **merges duplicados** criando features "_x" e "_y"

#### **OFICIAL.ipynb - Features criadas mas não usadas:**
```python
❌ hour_sin, hour_cos           # Criadas na Célula 13, NÃO usadas no modelo!
❌ day_sin, day_cos             # Criadas na Célula 13, NÃO usadas no modelo!
❌ stop_event_rate              # Criada na Célula 14, NÃO usada!
❌ stop_event_count             # Criada na Célula 14, NÃO usada!
❌ stop_total_samples           # Criada na Célula 14, NÃO usada!
❌ stop_dist_mean               # Criada na Célula 14, NÃO usada!
❌ stop_dist_std                # Criada na Célula 14, NÃO usada!
❌ stop_headway_mean            # Criada na Célula 14, NÃO usada!
❌ stop_headway_std             # Criada na Célula 14, NÃO usada!
```

🚨 **BUG**: OFICIAL.ipynb desperdiça **11 features** que foram criadas mas não incluídas!

---

## 🎯 COMPARAÇÃO LADO A LADO

### **Features Compartilhadas (Ambos Têm)**
| Feature | V7 | OFICIAL | Observações |
|---------|----|------------|-------------|
| `time_hour` | ✅ | ✅ | Idêntico |
| `time_day_of_week` | ✅ | ✅ | Idêntico |
| `is_holiday` | ✅ | ✅ | Idêntico (biblioteca holidays) |
| `is_weekend` | ✅ | ✅ | Idêntico |
| `is_peak_hour` | ✅ | ✅ | Idêntico (6-9h, 17-19h) |
| `dist_device_stop` | ✅ | ✅ | OFICIAL usa geodesic (melhor) |
| `headway_avg_stop_hour` | ✅ | ✅ | OFICIAL calcula do GTFS |
| `gtfs_stop_id` | ✅ | ✅ | Categórica em ambos |

**Total**: 8 features compartilhadas

---

### **Features EXCLUSIVAS do V7**
| Categoria | Quantidade | Features |
|-----------|------------|----------|
| **Agregações por Usuário** | 9 | user_conversion_rate, user_total_conversions, user_frequency, user_avg_dist, user_std_dist, user_min_dist, user_max_dist, user_avg_hour, user_std_hour |
| **Agregações por Parada** | 7 | stop_conversion_rate, stop_total_conversions, stop_event_count_agg, stop_avg_dist, stop_dist_std, stop_lat_agg, stop_lon_agg |
| **Interações User×Stop** | 3 | conversion_interaction, distance_interaction, frequency_interaction |
| **Features Temporais Extras** | 6 | time_day_of_month, hour_sin, hour_cos, day_sin, day_cos, month_cos |
| **Coordenadas Geográficas** | 4 | device_lat, device_lon, stop_lat_event, stop_lon_event |
| **GTFS Agregados** | 2 | stop_headway_mean, stop_headway_std |
| **Interações Temporais** | 2 | headway_x_hour, headway_x_weekend |

**Total**: 33 features exclusivas do V7

---

### **Features CRIADAS mas NÃO USADAS no OFICIAL.ipynb**
```python
# Criadas na Célula 13:
❌ hour_sin, hour_cos
❌ day_sin, day_cos
❌ headway_x_hour
❌ headway_x_weekend
❌ dist_x_peak
❌ dist_x_weekend

# Criadas na Célula 14:
❌ stop_event_rate
❌ stop_event_count
❌ stop_total_samples
❌ stop_dist_mean
❌ stop_dist_std
❌ stop_headway_mean
❌ stop_headway_std
```

**Total**: 13 features desperdiçadas!

---

## 📈 IMPACTO DAS FEATURES NA PERFORMANCE

### **Top 10 Features do V7 (por Gain)**
```
1. conversion_interaction (4328.72)     ← USER×STOP (NÃO existe no OFICIAL!)
2. user_conversion_rate (162.28)       ← USER (NÃO existe no OFICIAL!)
3. stop_lon_agg (62.65)                ← STOP (NÃO existe no OFICIAL!)
4. user_total_conversions (56.31)      ← USER (NÃO existe no OFICIAL!)
5. dist_x_peak (53.69)                 ← TEMPORAL (criada mas NÃO usada no OFICIAL!)
6. stop_total_conversions (53.35)      ← STOP (NÃO existe no OFICIAL!)
7. device_lon (51.21)                  ← GEO (NÃO explícita no OFICIAL!)
8. stop_lat_agg (50.85)                ← STOP (NÃO existe no OFICIAL!)
9. user_max_dist (50.58)               ← USER (NÃO existe no OFICIAL!)
10. distance_interaction (48.47)       ← USER×STOP (NÃO existe no OFICIAL!)
```

**ANÁLISE:**
- 🔥 **8 das top 10** são features que **NÃO EXISTEM** no OFICIAL.ipynb!
- 🔥 Top 1 (`conversion_interaction`) é **26x mais importante** que a #2!
- 🔥 Agregações USER dominam: #2, #4, #9
- 🔥 Agregações STOP dominam: #3, #6, #8
- 🔥 Interações dominam: #1, #5, #10

---

## 🎓 LIÇÕES E RECOMENDAÇÕES

### ✅ **O QUE V7 FAZ MELHOR:**

1. **Agregações por Usuário** (9 features)
   - Captura comportamento individual
   - `user_conversion_rate` (#2 mais importante!)
   - Explica recall superior (73.6% vs ~50%)

2. **Interações User×Stop** (3 features)
   - `conversion_interaction` é **DOMINANTE** (4328 gain!)
   - Captura sinergia entre usuário e local

3. **Seleção Automática** (XGBoost)
   - 53 features → 49 selecionadas
   - Remove features redundantes automaticamente

4. **Features Cíclicas** (sin/cos)
   - `hour_sin`, `hour_cos`, `day_sin`, `day_cos`
   - Melhor que one-hot para variáveis temporais

5. **Coordenadas Explícitas**
   - `device_lat`, `device_lon`, `stop_lat_agg`, `stop_lon_agg`
   - Permite capturar padrões geográficos

---

### ❌ **PROBLEMAS NO OFICIAL.ipynb:**

1. **Features Desperdiçadas** (13 features)
   - Cria `hour_sin/cos`, `day_sin/cos` mas **não usa**!
   - Cria 7 agregações por parada mas **não usa**!
   - Desperdiça esforço computacional

2. **Falta Agregações de Usuário** (0 features)
   - Não captura comportamento individual
   - Limita recall (~50% vs 73.6%)

3. **Features Fixas** (9 apenas)
   - Não usa seleção automática
   - Pode incluir features irrelevantes

4. **Dataset Muito Desbalanceado**
   - 99.96% classe 0 vs 93% no V7
   - Dificulta treinamento

---

### 🔧 **MELHORIAS SUGERIDAS PARA OFICIAL.ipynb:**

```python
# 1. USAR as features já criadas!
FEATURES_IMPROVED = [
    # Temporais
    'time_hour', 'time_day_of_week', 
    'is_holiday', 'is_weekend', 'is_peak_hour',
    
    # ✅ ADICIONAR: Features cíclicas (já criadas na Célula 13!)
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    
    # Geoespaciais
    'dist_device_stop',
    
    # GTFS
    'headway_avg_stop_hour',
    
    # ✅ ADICIONAR: Interações temporais (já criadas na Célula 13!)
    'headway_x_hour', 'headway_x_weekend',
    'dist_x_peak', 'dist_x_weekend',
    
    # ✅ ADICIONAR: Agregações por parada (já criadas na Célula 14!)
    'stop_event_rate', 'stop_event_count', 'stop_total_samples',
    'stop_dist_mean', 'stop_dist_std',
    'stop_headway_mean', 'stop_headway_std',
    
    # Parada ID
    'gtfs_stop_id'
]

# Total: 9 → 25 features (178% de aumento!)
```

### 🔧 **MELHORIAS SUGERIDAS PARA V7:**

```python
# 1. Limpar features duplicadas
features_to_remove = [
    'Unnamed: 0',          # Índice do pandas
    'int64_field_0',       # Campo desconhecido
    'user_frequency_x',    # Escolher _x ou _y, não ambos
    'stop_dist_std_x'      # Escolher _x ou _y
]

# 2. Criar features de janela temporal (últimos N dias)
user_last_7d_conversion_rate   # Taxa dos últimos 7 dias
user_last_30d_conversion_rate  # Taxa dos últimos 30 dias
stop_last_7d_conversion_rate   # Taxa da parada nos últimos 7 dias

# 3. Features de tendência
user_conversion_trend          # Usuário está melhorando/piorando?
stop_conversion_trend          # Parada está ficando mais/menos popular?

# 4. Melhorar cálculo de distância
# Usar geopy.geodesic ao invés de coordenadas brutas
```

---

## 🏆 RESUMO FINAL

| Métrica | V7 | OFICIAL.ipynb |
|---------|----|----|
| **Features Usadas** | 49 | 9 |
| **Agregações User** | ✅ 9 | ❌ 0 |
| **Agregações Stop** | ✅ 7 | ❌ 0 (criadas mas não usadas) |
| **Interações** | ✅ 6 | ❌ 0 (criadas mas não usadas) |
| **Features Cíclicas** | ✅ 6 | ❌ 0 (criadas mas não usadas) |
| **Seleção Automática** | ✅ Sim | ❌ Não |
| **ROC-AUC** | ✅ 0.9749 | ~0.15-0.25 (AUCPR) |
| **Recall** | ✅ 73.6% | ~50% |
| **Tempo Treino** | ✅ 6.46s | ~Não especificado |

**VENCEDOR**: V7 por **larga margem** em riqueza de features!

---

## 💡 INSIGHT PRINCIPAL

A feature **`conversion_interaction`** (user_conversion_rate × stop_conversion_rate) é **4328 gain** - **26x mais importante** que a segunda colocada!

**Por que?**
- Captura **sinergia** entre usuário convertedor e parada popular
- Usuário com alta taxa de conversão + Parada com alta taxa = **Alta probabilidade**
- É um **multiplicador** de comportamento

**Implicação**: Interações User×Stop são **críticas** e o OFICIAL.ipynb **não tem** porque não tem agregações de usuário!

---

**Data**: 2025-11-12  
**Versões Comparadas**: model_v7_comparison.py vs OFICIAL.ipynb
