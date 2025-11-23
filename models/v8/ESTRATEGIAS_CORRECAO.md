# 🎯 ESTRATÉGIAS PARA CORRIGIR SUBESTIMAÇÃO DE ALTA CONVERSÃO

## 📊 Problema Identificado

**Sintomas:**
- Modelo prevê ~21% para TODAS as paradas
- Dataset possui paradas com 20% até **98.5%** de conversão real
- Modelo não captura a variação: erro médio de 19.1%
- 100% das predições ficam na categoria "Média" (10-30%)
- 57 paradas com conversão 50-100% → modelo prevê 0 nessa faixa

**Causa Raiz:**
O modelo está aprendendo a **média geral** (~20%) mas não as **características específicas** que diferenciam paradas de alta conversão.

---

## 🔧 SOLUÇÕES PROPOSTAS

### 1️⃣ **ESTRATÉGIA 1: Adicionar Features Geográficas Específicas** ⭐ MAIS IMPORTANTE

**Problema:** Paradas com alta conversão podem estar em locais específicos (terminais, áreas centrais, etc.)

**Solução:**
```python
# A. Agregar conversão POR PARADA (não só média geral)
stop_conversion_rate = df.groupby('gtfs_stop_id')['target'].mean()
df['stop_historical_conversion'] = df['gtfs_stop_id'].map(stop_conversion_rate)

# B. Densidade de paradas (áreas centrais têm mais paradas)
from sklearn.neighbors import NearestNeighbors
coords = df[['stop_lat_event', 'stop_lon_event']].values
nn = NearestNeighbors(n_neighbors=10)
nn.fit(coords)
distances, _ = nn.kneighbors(coords)
df['stop_density'] = 1 / distances[:, 1:].mean(axis=1)  # Paradas próximas

# C. Distância ao centro (CBD - Central Business District)
centro_sp = (-23.550520, -46.633308)  # Praça da Sé
df['dist_to_cbd'] = haversine(df['stop_lat_event'], df['stop_lon_event'], 
                               centro_sp[0], centro_sp[1])

# D. Região/Cluster de paradas
from sklearn.cluster import DBSCAN
clustering = DBSCAN(eps=0.01, min_samples=5)
df['stop_cluster'] = clustering.fit_predict(coords)
```

**Por que funciona:** Paradas de alta conversão geralmente estão em locais específicos (terminais, áreas comerciais). Essa feature ensina o modelo a reconhecer esses lugares.

---

### 2️⃣ **ESTRATÉGIA 2: Balanceamento por Binning** ⭐ CRÍTICO

**Problema:** Dataset desbalanceado: 92.5% classe 0, 7.5% classe 1

**Solução:**
```python
# Converter em problema multi-classe
df['target_binned'] = pd.cut(df['conversion_rate'], 
                              bins=[0, 0.1, 0.3, 0.5, 1.0],
                              labels=[0, 1, 2, 3])  # Baixa, Média, Alta, Muito Alta

# Usar scale_pos_weight mais agressivo
scale_weight = len(df[df['target']==0]) / len(df[df['target']==1])  # ~12.3

lgb_params = {
    'scale_pos_weight': scale_weight * 1.5,  # Aumentar peso das conversões
    'class_weight': 'balanced'
}
```

---

### 3️⃣ **ESTRATÉGIA 3: Focal Loss** (XGBoost Custom)

**Problema:** Cross-entropy padrão trata todos os exemplos igualmente

**Solução:**
```python
# Focal Loss: penaliza mais erros em exemplos "difíceis" (alta conversão)
def focal_loss(y_pred, dtrain, alpha=0.25, gamma=2.0):
    y_true = dtrain.get_label()
    p = 1 / (1 + np.exp(-y_pred))
    
    # Focal loss formula
    loss = -alpha * (1 - p)**gamma * y_true * np.log(p + 1e-8) \
           - (1 - alpha) * p**gamma * (1 - y_true) * np.log(1 - p + 1e-8)
    
    grad = alpha * (gamma * (1 - p)**(gamma - 1) * y_true * np.log(p) + 
                    (1 - p)**gamma * y_true / p) - \
           (1 - alpha) * (gamma * p**(gamma - 1) * (1 - y_true) * np.log(1 - p) + 
                          p**gamma * (1 - y_true) / (1 - p))
    
    hess = np.ones_like(grad)  # Aproximação
    return grad, hess

xgb_model = xgb.train(params, dtrain, obj=focal_loss)
```

---

### 4️⃣ **ESTRATÉGIA 4: Threshold Dinâmico por Parada**

**Problema:** Threshold global (0.45) não funciona para todas as paradas

**Solução:**
```python
# Calibrar threshold específico por faixa de conversão histórica
def get_dynamic_threshold(stop_historical_conversion):
    if stop_historical_conversion > 0.7:
        return 0.3  # Threshold mais baixo para paradas de alta conversão
    elif stop_historical_conversion > 0.4:
        return 0.4
    else:
        return 0.5  # Threshold mais alto para baixa conversão

df['threshold'] = df['stop_historical_conversion'].apply(get_dynamic_threshold)
df['prediction'] = (df['prob_ensemble'] > df['threshold']).astype(int)
```

---

### 5️⃣ **ESTRATÉGIA 5: Feature de Volume por Parada**

**Problema:** Paradas com muitos eventos podem ter comportamento diferente

**Solução:**
```python
# Agregações avançadas por parada
stop_stats = df.groupby('gtfs_stop_id').agg({
    'target': ['mean', 'sum', 'std'],  # Taxa, total, variação
    'user_pseudo_id': 'nunique',        # Usuários únicos
    'event_timestamp': 'count',         # Volume total
    'is_peak_hour': 'mean'              # % eventos no pico
})

stop_stats.columns = ['stop_conversion_rate', 'stop_total_conversions', 
                       'stop_conversion_std', 'stop_unique_users',
                       'stop_event_volume', 'stop_peak_ratio']

df = df.merge(stop_stats, left_on='gtfs_stop_id', right_index=True)

# Feature de volatilidade
df['stop_volatility'] = df['stop_conversion_std'] / (df['stop_conversion_rate'] + 0.01)
```

---

### 6️⃣ **ESTRATÉGIA 6: Oversampling Estratificado**

**Problema:** Poucos exemplos de alta conversão no treino

**Solução:**
```python
from imblearn.over_sampling import SMOTE

# SMOTE apenas em paradas de alta conversão
high_conversion = df[df['stop_conversion_rate'] > 0.5]
low_conversion = df[df['stop_conversion_rate'] <= 0.5]

# Oversample as de alta conversão
smote = SMOTE(sampling_strategy=0.3, random_state=42)
X_high_resampled, y_high_resampled = smote.fit_resample(
    high_conversion[features], 
    high_conversion[target]
)

# Combinar
df_balanced = pd.concat([low_conversion, 
                         pd.DataFrame(X_high_resampled, columns=features)])
```

---

### 7️⃣ **ESTRATÉGIA 7: Ensemble com Modelo Especializado**

**Problema:** Um modelo só não captura todos os padrões

**Solução:**
```python
# Treinar 2 modelos:
# Modelo A: Geral (todas as paradas)
# Modelo B: Especializado (só paradas > 30% conversão)

# Modelo especializado
df_high = df[df['stop_historical_conversion'] > 0.3]
model_specialist = lgb.train(params, lgb.Dataset(X_high, y_high))

# Predição combinada
def predict_ensemble(row):
    pred_general = model_general.predict(row)
    
    if row['stop_historical_conversion'] > 0.3:
        pred_specialist = model_specialist.predict(row)
        # Dar mais peso ao especialista
        return 0.3 * pred_general + 0.7 * pred_specialist
    else:
        return pred_general
```

---

## 📋 PLANO DE IMPLEMENTAÇÃO (ORDEM DE PRIORIDADE)

### ✅ **FASE 1: Quick Wins (1-2 dias)**
1. ✅ Adicionar `stop_historical_conversion` como feature
2. ✅ Adicionar `stop_density` (densidade de paradas)
3. ✅ Aumentar `scale_pos_weight` para 15-20
4. ✅ Calibrar threshold dinâmico

**Expectativa:** Erro cair de 19% para 12-15%

---

### ✅ **FASE 2: Melhorias Médias (3-5 dias)**
5. ✅ Implementar clustering de paradas (DBSCAN)
6. ✅ Adicionar distância ao CBD
7. ✅ Features de volume/volatilidade por parada
8. ✅ Validação por faixa de conversão

**Expectativa:** Erro cair para 8-10%, começar a prever algumas paradas de alta conversão

---

### 🔄 **FASE 3: Avançado (1-2 semanas)**
9. 🔄 Implementar Focal Loss
10. 🔄 SMOTE estratificado
11. 🔄 Modelo especializado (ensemble híbrido)
12. 🔄 Hyperparameter tuning com Optuna

**Expectativa:** Erro < 5%, capturar 70%+ das paradas de alta conversão

---

## 🎯 MÉTRICAS DE SUCESSO

**Antes (V7):**
- Taxa real: 40.2% (média das top 200)
- Predição: 21.1%
- Erro: 19.1%
- Correlação: 0.484
- Paradas >50% previstas: **0** (0%)

**Meta V8 (Fase 1):**
- Predição: 30-35%
- Erro: <15%
- Correlação: >0.60
- Paradas >50% previstas: **15+** (26%)

**Meta V8 (Fase 2):**
- Predição: 35-40%
- Erro: <10%
- Correlação: >0.75
- Paradas >50% previstas: **35+** (61%)

**Meta V8 (Fase 3):**
- Predição: 38-42%
- Erro: <5%
- Correlação: >0.85
- Paradas >50% previstas: **45+** (79%)

---

## 🚀 PRÓXIMOS PASSOS

1. **Criar model_v8_improved.py** com Fase 1
2. **Testar** no mesmo dataset de validação
3. **Comparar** métricas V7 vs V8
4. **Iterar** com Fase 2 se Fase 1 funcionar
5. **Gerar** novos mapas com predições V8

Qual fase você quer que eu implemente primeiro? 💪
