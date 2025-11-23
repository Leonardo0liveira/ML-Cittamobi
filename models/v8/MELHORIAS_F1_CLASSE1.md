# 🎯 ESTRATÉGIAS PARA MELHORAR F1 CLASSE 1 (CONVERSÃO)

## 📊 Situação Atual (V8 Fase 1)
- **F1 Classe 1**: 0.4206
- **Precision**: 0.3023 (30%) ❌ - Muitos falsos positivos
- **Recall**: 0.6910 (69%) ✅ - Bom!
- **Problema**: Modelo prevê conversão em 13,700 casos, mas apenas 4,141 são reais

---

## 🔧 ESTRATÉGIAS (ORDEM DE IMPACTO)

### 1️⃣ **SMOTE/ADASYN - Oversampling Inteligente** ⭐⭐⭐
**Impacto esperado:** +10-15% F1

**Problema:** Dataset desbalanceado (9:1) - modelo vê poucos exemplos positivos

**Solução:**
```python
from imblearn.over_sampling import SMOTENC, ADASYN

# SMOTENC: SMOTE para features mistas (numéricas + categóricas)
smote = SMOTENC(
    categorical_features=[...],  # Índices de features categóricas
    sampling_strategy=0.3,        # Balancear para 30% classe 1
    k_neighbors=5,
    random_state=42
)

X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
```

**Por que funciona:**
- Cria exemplos sintéticos da classe minoritária
- Modelo aprende melhor os padrões de conversão
- Reduz viés para classe majoritária

---

### 2️⃣ **Focal Loss (Custom Objective)** ⭐⭐⭐
**Impacto esperado:** +8-12% F1

**Problema:** Cross-entropy trata todos os erros igualmente

**Solução:**
```python
def focal_loss_lgb(y_pred, dtrain):
    y_true = dtrain.get_label()
    gamma = 2.0  # Foco em exemplos difíceis
    alpha = 0.75  # Peso para classe positiva
    
    p = 1 / (1 + np.exp(-y_pred))
    
    # Focal loss: reduz peso de exemplos "fáceis"
    loss = -alpha * (1 - p)**gamma * y_true * np.log(p + 1e-8) \
           - (1 - alpha) * p**gamma * (1 - y_true) * np.log(1 - p + 1e-8)
    
    grad = ...  # Derivada
    hess = ...  # Segunda derivada
    
    return grad, hess

lgb_model = lgb.train(params, dtrain, fobj=focal_loss_lgb)
```

**Por que funciona:**
- Penaliza mais erros em exemplos "difíceis" (conversões raras)
- Reduz peso de exemplos "fáceis" (não-conversões óbvias)
- Melhora precision sem perder recall

---

### 3️⃣ **Class Weight Dinâmico** ⭐⭐
**Impacto esperado:** +5-8% F1

**Problema:** `scale_pos_weight=11.6` é genérico para todo dataset

**Solução:**
```python
# Calcular peso por faixa de conversão
def get_sample_weights(X, y):
    weights = np.ones(len(y))
    
    # Peso maior para paradas de alta conversão
    high_conv_mask = X['stop_historical_conversion'] > 0.5
    weights[high_conv_mask & (y == 1)] = 3.0  # Conversões em paradas altas
    weights[high_conv_mask & (y == 0)] = 0.5  # Não-conversões em paradas altas
    
    # Peso médio para paradas médias
    med_conv_mask = (X['stop_historical_conversion'] > 0.2) & \
                    (X['stop_historical_conversion'] <= 0.5)
    weights[med_conv_mask & (y == 1)] = 2.0
    
    return weights

sample_weights = get_sample_weights(X_train, y_train)
dtrain = lgb.Dataset(X_train_scaled, y_train, weight=sample_weights)
```

**Por que funciona:**
- Dá mais importância a conversões em paradas relevantes
- Reduz peso de não-conversões em paradas de baixa conversão
- Melhora precision focando no que importa

---

### 4️⃣ **Threshold por Faixa de Conversão** ⭐⭐
**Impacto esperado:** +3-5% F1

**Problema:** Threshold único (0.60) não funciona para todas as paradas

**Solução:**
```python
def get_dynamic_threshold(stop_conv):
    if stop_conv > 0.7:
        return 0.40  # Threshold baixo para paradas muito altas
    elif stop_conv > 0.5:
        return 0.50
    elif stop_conv > 0.3:
        return 0.60
    else:
        return 0.75  # Threshold alto para paradas baixas

X_test['threshold_custom'] = X_test['stop_historical_conversion'].apply(
    get_dynamic_threshold
)
X_test['y_pred'] = (X_test['y_pred_prob'] > X_test['threshold_custom']).astype(int)
```

**Por que funciona:**
- Paradas de alta conversão: mais agressivo (threshold baixo)
- Paradas de baixa conversão: mais conservador (threshold alto)
- Reduz falsos positivos em paradas de baixa conversão

---

### 5️⃣ **Features de Contexto Temporal** ⭐⭐
**Impacto esperado:** +4-6% F1

**Problema:** Falta contexto temporal detalhado

**Solução:**
```python
# A. Janelas temporais (rolling features)
df['conversions_last_hour'] = df.groupby('gtfs_stop_id')['target'].shift(1).rolling(6).sum()
df['conversions_last_day'] = df.groupby('gtfs_stop_id')['target'].shift(1).rolling(24*6).sum()

# B. Tendência temporal
df['conversion_trend'] = df.groupby('gtfs_stop_id')['target'].transform(
    lambda x: x.rolling(24, min_periods=1).mean().diff()
)

# C. Sazonalidade
df['hour_conversion_rate'] = df.groupby('time_hour')['target'].transform('mean')
df['dow_conversion_rate'] = df.groupby('time_day_of_week')['target'].transform('mean')

# D. Interação hora x parada
df['stop_hour_conversion'] = df.groupby(['gtfs_stop_id', 'time_hour'])['target'].transform('mean')
```

**Por que funciona:**
- Captura padrões temporais específicos de cada parada
- Modelo aprende quando conversões são mais prováveis
- Reduz falsos positivos fora de horários típicos

---

### 6️⃣ **Ensemble com Modelo Especialista** ⭐⭐⭐
**Impacto esperado:** +7-10% F1

**Problema:** Modelo único tenta aprender tudo

**Solução:**
```python
# Modelo 1: Geral (todas as paradas)
model_general = lgb.train(params_general, dtrain_all)

# Modelo 2: Especialista em alta conversão
high_conv_mask = X_train['stop_historical_conversion'] > 0.3
X_train_specialist = X_train[high_conv_mask]
y_train_specialist = y_train[high_conv_mask]

# Treinar com mais ênfase em precision
params_specialist = {
    **params_general,
    'scale_pos_weight': 5.0,  # Menos agressivo
    'max_depth': 10,          # Mais profundo
}
model_specialist = lgb.train(params_specialist, dtrain_specialist)

# Predição híbrida
def predict_hybrid(row):
    pred_general = model_general.predict(row)
    
    if row['stop_historical_conversion'] > 0.3:
        pred_specialist = model_specialist.predict(row)
        # Weighted average: mais peso ao especialista
        return 0.3 * pred_general + 0.7 * pred_specialist
    else:
        return pred_general
```

**Por que funciona:**
- Modelo especialista foca apenas em paradas relevantes
- Aprende padrões específicos de alta conversão
- Reduz falsos positivos em paradas de baixa conversão

---

### 7️⃣ **Calibração de Probabilidades** ⭐
**Impacto esperado:** +2-4% F1

**Problema:** Probabilidades não calibradas (modelo confiante demais)

**Solução:**
```python
from sklearn.calibration import CalibratedClassifierCV

# Wrapper para LightGBM
class LGBMWrapper:
    def __init__(self, model):
        self.model = model
    
    def predict_proba(self, X):
        pred = self.model.predict(X)
        return np.vstack([1-pred, pred]).T

# Calibrar
lgb_wrapper = LGBMWrapper(lgb_model)
calibrated = CalibratedClassifierCV(lgb_wrapper, method='isotonic', cv=3)
calibrated.fit(X_val_scaled, y_val)

# Predições calibradas
y_pred_calibrated = calibrated.predict_proba(X_test_scaled)[:, 1]
```

**Por que funciona:**
- Ajusta probabilidades para refletir frequências reais
- Reduz overconfidence em predições incorretas
- Melhora precision sem afetar recall

---

### 8️⃣ **Feature Engineering Adicional** ⭐
**Impacto esperado:** +3-5% F1

**Solução:**
```python
# A. Ratio de conversão usuário vs parada
df['user_vs_stop_ratio'] = df['user_conversion_rate'] / (df['stop_conversion_rate'] + 0.01)

# B. Desvio da média
df['dist_deviation'] = (df['dist_device_stop'] - df['stop_avg_dist']) / (df['stop_dist_std'] + 0.01)

# C. Frequência relativa
df['user_frequency_rank'] = df.groupby('gtfs_stop_id')['user_frequency'].rank(pct=True)

# D. Interação geográfica + temporal
df['geo_temporal'] = df['dist_to_nearest_cbd'] * df['is_peak_hour']
df['density_peak'] = df['stop_density'] * df['is_peak_hour']

# E. Features de raridade
df['stop_rarity'] = 1 / (df['stop_event_count'] + 1)  # Paradas raras
df['user_rarity'] = 1 / (df['user_frequency'] + 1)    # Usuários raros
```

---

## 📋 PLANO DE IMPLEMENTAÇÃO

### **FASE 2A: Quick Wins (1-2 dias)** ⚡
1. ✅ **Threshold dinâmico** (implementação simples)
2. ✅ **Class weight dinâmico** (poucas linhas)
3. ✅ **Feature engineering adicional** (4-5 features)

**Meta:** F1 Classe 1 de 0.42 → **0.50+** (+8%)

---

### **FASE 2B: Melhorias Médias (3-5 dias)** 🔧
4. ✅ **SMOTE/ADASYN** (balanceamento)
5. ✅ **Features temporais** (rolling, tendência)
6. ✅ **Calibração** (ajuste de probabilidades)

**Meta:** F1 Classe 1 de 0.50 → **0.58+** (+8%)

---

### **FASE 2C: Avançado (1 semana)** 🚀
7. ✅ **Focal Loss** (custom objective)
8. ✅ **Ensemble especialista** (2 modelos)
9. ✅ **Hyperparameter tuning** (Optuna)

**Meta:** F1 Classe 1 de 0.58 → **0.65+** (+7%)

---

## 🎯 METAS FINAIS

| Métrica | V8 Atual | Fase 2A | Fase 2B | Fase 2C |
|---------|----------|---------|---------|---------|
| F1 Classe 1 | 0.42 | 0.50 | 0.58 | **0.65+** |
| Precision | 0.30 | 0.38 | 0.48 | **0.55+** |
| Recall | 0.69 | 0.69 | 0.70 | **0.78+** |
| F1-Macro | 0.65 | 0.69 | 0.74 | **0.78+** |

---

## 💡 RECOMENDAÇÃO

**Começar com FASE 2A** (threshold dinâmico + class weight + features):
- Implementação rápida (2-3 horas)
- Alto impacto (+8% F1)
- Sem overhead computacional

Depois avaliar se precisa FASE 2B/2C baseado nos resultados.

Quer que eu implemente a **Fase 2A** agora? 🚀
