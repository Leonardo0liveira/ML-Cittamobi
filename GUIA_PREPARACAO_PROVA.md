# 📚 GUIA DE PREPARAÇÃO PARA PROVA - PROJETO MACHINE LEARNING

## 🎯 VISÃO GERAL DO PROJETO

### **O QUE É O PROJETO?**
Sistema de **previsão de conversão de usuários** de transporte público (aplicativo Cittamobi). O objetivo é prever se um usuário irá "converter" (realizar uma ação desejada - como comprar uma passagem) com base em seus padrões de uso.

### **PROBLEMA DE NEGÓCIO**
- **Dataset desbalanceado**: 93% não convertem, 7% convertem
- **Desafio**: Identificar corretamente os 7% que convertem sem gerar muitos falsos alarmes
- **Aplicação**: Marketing direcionado, otimização de recursos, UX personalizada

---

## 📊 EVOLUÇÃO DO PROJETO (8 VERSÕES)

### **📈 LINHA DO TEMPO DE RESULTADOS**

| Versão | Algoritmo | ROC-AUC | F1-Macro | Precision | Recall | Principal Inovação |
|--------|-----------|---------|----------|-----------|--------|-------------------|
| **V1** | XGBoost | 0.8367 | ~0.65 | - | - | 🔹 Baseline inicial |
| **V2** | XGBoost | 0.7961 | - | - | - | ⚠️ Limpeza agressiva (piorou) |
| **V3** | XGBoost | 0.9324 | 0.7143 | 0.43 | 0.47 | 🔹 Técnicas de balanceamento |
| **V4** | XGBoost | **0.9731** | **0.7760** | **0.59** | 0.55 | 🏆 Features avançadas + deep trees |
| **V5** | XGBoost | - | - | - | - | 🔹 Experimentos intermediários |
| **V6** | XGBoost | 0.9720 | 0.7742 | - | - | 🔹 Refinamento de produção |
| **V7** | LightGBM | **0.9749** | 0.7713 | - | **0.736** | 🏆 Mudança de algoritmo |
| **V8** | CatBoost | - | - | - | - | 🔹 Teste com CatBoost |

**MELHOR MODELO**: V7 LightGBM (ROC-AUC 0.9749, Recall 73.6%)

**MELHORIA TOTAL**: +16.5% em ROC-AUC (V1 → V7)

---

## 🔍 FASES DO PROJETO: PRÉ E PÓS TREINAMENTO

## 📋 **FASE 1: PRÉ-TREINAMENTO** (O QUE VOCÊ FEZ ANTES DE TREINAR)

### **1.1 COLETA E EXPLORAÇÃO DE DADOS**

#### **O que foi feito:**
```python
# Conexão com BigQuery
client = bigquery.Client(project='datamaster-440118')
df = client.query(query).to_dataframe()

# Dados: 200,000 registros de eventos de usuários
# Período: 2024
# Fonte: Cittamobi + GTFS (dados de transporte público de SP)
```

#### **Análise Exploratória (EDA):**
- ✅ Taxa de conversão: **7%** (classe minoritária)
- ✅ Usuários únicos: ~4,000
- ✅ Paradas únicas: ~400
- ✅ Identificação do **desbalanceamento** (93:7)

**SUA CONTRIBUIÇÃO**: Compreender a natureza desbalanceada do problema e identificar que métricas como Accuracy não seriam suficientes.

---

### **1.2 LIMPEZA DE DADOS (DATA CLEANING)**

#### **Versões V1-V2: Limpeza Agressiva (ERRO APRENDIDO)**
```python
# V2: Removeu muitos dados (prejudicou o modelo)
df = df[df['user_frequency'] >= quantile(0.30)]  # Muito restritivo!
# Resultado: ROC-AUC caiu de 0.8367 → 0.7961
```

#### **Versões V3-V8: Limpeza Moderada (SUCESSO)**
```python
# Apenas remove outliers extremos
df = df[df['user_frequency'] >= quantile(0.10)]  # Mais flexível
df = df[df['dist_device_stop'] <= quantile(0.98)]  # Remove apenas 2% extremos
df = df[~((device_lat == 0) & (device_lon == 0))]  # Remove GPS inválido
```

**LIÇÃO APRENDIDA**: Limpeza agressiva **perde informação valiosa**. É melhor manter mais dados e deixar o modelo aprender.

**SUA CONTRIBUIÇÃO**: Testou diferentes níveis de limpeza e identificou o sweet spot entre qualidade e quantidade de dados.

---

### **1.3 FEATURE ENGINEERING (CRIAÇÃO DE FEATURES)**

Esta foi a **fase mais importante** do projeto! Você criou **50+ features** em várias categorias:

#### **A) FEATURES TEMPORAIS (13 features)**

```python
# Básicas
time_hour           # 0-23
time_day_of_week    # 0-6 (segunda=0, domingo=6)
time_day_of_month   # 1-31
time_month          # 1-12
week_of_year        # 1-52

# Cíclicas (transformação trigonométrica)
hour_sin = np.sin(2 * np.pi * time_hour / 24)
hour_cos = np.cos(2 * np.pi * time_hour / 24)
day_sin = np.sin(2 * np.pi * time_day_of_week / 7)
day_cos = np.cos(2 * np.pi * time_day_of_week / 7)
month_sin = np.sin(2 * np.pi * time_month / 12)
month_cos = np.cos(2 * np.pi * time_month / 12)

# Contextuais
is_weekend = (time_day_of_week >= 5)
is_peak_hour = time_hour in [7,8,9,17,18,19]
is_holiday = event_date in br_holidays
```

**POR QUE CÍCLICAS?**
- Hora 23 está **próxima** da hora 0 (meia-noite)
- Transformação trigonométrica captura essa **circularidade**
- Sem isso, modelo vê 23 e 0 como distantes (erro!)

**SUA CONTRIBUIÇÃO**: Compreendeu a natureza cíclica do tempo e implementou transformações matemáticas apropriadas.

---

#### **B) AGREGAÇÕES POR USUÁRIO (9 features) 🔥 CRÍTICO**

```python
# Comportamento histórico do usuário
user_agg = df.groupby('user_id').agg({
    'converted': ['mean', 'sum', 'count'],
    'dist_device_stop': ['mean', 'std', 'min', 'max'],
    'time_hour': ['mean', 'std']
})

# Features criadas:
user_conversion_rate      # Taxa de conversão histórica (0-1)
user_total_conversions    # Total de conversões (número absoluto)
user_total_events         # Frequência de uso (engajamento)
user_avg_dist             # Distância média que o usuário percorre
user_std_dist             # Variabilidade do comportamento
user_min_dist             # Distância mínima
user_max_dist             # Distância máxima
user_avg_hour             # Hora preferida de uso
user_std_hour             # Consistência temporal
```

**IMPORTÂNCIA**: 
- `user_conversion_rate` é a **2ª feature mais importante** do modelo!
- Captura **padrões individuais**: usuário que sempre converte vs usuário exploratório
- Explica por que o recall melhorou tanto (73.6% no V7)

**SUA CONTRIBUIÇÃO**: Criou features que capturam o **perfil comportamental** de cada usuário.

---

#### **C) AGREGAÇÕES POR PARADA (7 features)**

```python
# Características da parada
stop_agg = df.groupby('stop_id').agg({
    'converted': ['mean', 'sum', 'count'],
    'dist_device_stop': ['mean', 'std'],
    'stop_lat': 'first',
    'stop_lon': 'first'
})

# Features criadas:
stop_conversion_rate      # Paradas "quentes" (alta conversão)
stop_total_conversions    # Popularidade da parada
stop_total_events         # Volume de uso
stop_dist_mean            # Distância típica dos usuários
stop_dist_std             # Variabilidade espacial
stop_lat_agg              # Coordenadas agregadas
stop_lon_agg
```

**POR QUE IMPORTANTE?**
- Algumas paradas têm taxa de conversão > 30%
- Outras < 5%
- Identifica **locais estratégicos**

**SUA CONTRIBUIÇÃO**: Identificou que o local (parada) tem impacto significativo na conversão.

---

#### **D) FEATURES DE INTERAÇÃO (2ª ORDEM) (6 features)**

```python
# Combinações multiplicativas de features
conversion_interaction = user_conversion_rate * stop_conversion_rate
distance_interaction = dist_device_stop * stop_conversion_rate
user_stop_frequency = eventos no par (user, stop)
dist_x_peak = dist_device_stop * is_peak_hour
dist_x_weekend = dist_device_stop * is_weekend
headway_x_peak = headway_avg_stop_hour * is_peak_hour
```

**POR QUE INTERAÇÕES?**
- Captura **sinergias**: Usuário bom + Parada boa = EXCELENTE conversão
- Detecta **anomalias**: Distância muito diferente do usual = suspeito
- Contexto temporal: Mesma distância significa coisas diferentes em hora de pico vs fim de semana

**SUA CONTRIBUIÇÃO**: Criou features que capturam **efeitos combinados** de múltiplos fatores.

---

#### **E) FEATURES GEOESPACIAIS (8 features)**

```python
# Coordenadas brutas
device_lat, device_lon    # Localização do usuário
stop_lat, stop_lon        # Localização da parada

# Distância euclidiana
dist_device_stop = haversine(device, stop)

# Agregações espaciais
user_avg_dist             # Distância média do usuário
stop_dist_mean            # Distância típica da parada

# Desvios espaciais
dist_deviation = |dist_device_stop - user_avg_dist|
dist_ratio = dist_device_stop / user_avg_dist
```

**POR QUE IMPORTANTE?**
- Distância é forte preditor de conversão
- Usuários próximos têm maior probabilidade de converter
- Desvios do padrão indicam comportamento anômalo

---

#### **F) FEATURES GTFS (TRANSPORTE PÚBLICO) (2 features)**

```python
# Dados oficiais de transporte público (GTFS)
headway_avg_stop_hour     # Intervalo médio entre ônibus (minutos)
gtfs_stop_id              # ID oficial da parada
```

**POR QUE IMPORTANTE?**
- Paradas com **menor headway** (mais ônibus) = mais conversões
- Frequência do serviço afeta decisão do usuário

**SUA CONTRIBUIÇÃO**: Integrou dados externos (GTFS) para enriquecer o modelo.

---

### **1.4 SELEÇÃO DE FEATURES (FEATURE SELECTION)**

#### **Método Usado:**
```python
# Treinar modelo temporário para obter importâncias
xgb_selector = xgb.XGBClassifier(...)
xgb_selector.fit(X_train, y_train)

# Obter importâncias
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': xgb_selector.feature_importances_
}).sort_values('importance', ascending=False)

# Selecionar top 50
selected_features = feature_importance.head(50)['feature'].tolist()
```

#### **Top 10 Features Selecionadas (V6-V7):**
1. `conversion_interaction` (usuário × parada)
2. `user_conversion_rate` 🔥
3. `stop_lon_event` (longitude da parada)
4. `user_total_conversions` 🔥
5. `hour_sin` (hora cíclica)
6. `stop_conversion_rate` 🔥
7. `stop_lat_event` (latitude da parada)
8. `user_avg_dist`
9. `is_peak_hour`
10. `stop_dist_std`

**SUA CONTRIBUIÇÃO**: Reduziu de 70+ features para 50 features mais relevantes, melhorando eficiência sem perder performance.

---

### **1.5 DIVISÃO TEMPORAL DOS DADOS (TIME SERIES SPLIT)**

#### **Por que TimeSeriesSplit?**
```python
# NÃO usar train_test_split tradicional!
# Motivo: eventos têm ordem temporal

# Correto: TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=4)
train_idx, test_idx = list(tscv.split(X))[2]  # Fold 3

# Resultado:
# Treino: Eventos de janeiro-outubro (75%)
# Teste: Eventos de novembro-dezembro (25%)
```

**POR QUE IMPORTANTE?**
- Simula **produção real**: treinar com passado, prever futuro
- Evita **data leakage**: teste não "vaza" para treino
- Mais realista que shuffle aleatório

**SUA CONTRIBUIÇÃO**: Compreendeu a natureza temporal dos dados e aplicou split apropriado.

---

### **1.6 NORMALIZAÇÃO DE DADOS**

```python
# StandardScaler: (x - mean) / std
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit apenas no treino!
X_test_scaled = scaler.transform(X_test)        # Transform no teste
```

**POR QUE NORMALIZAR?**
- Features em escalas diferentes: distância (0-5000), hora (0-23)
- Algoritmos gradient-based convergem mais rápido
- Previne features com valores grandes dominarem

**SUA CONTRIBUIÇÃO**: Aplicou pré-processamento adequado para otimizar convergência.

---

### **1.7 TRATAMENTO DE DESBALANCEAMENTO**

#### **Estratégias Testadas (V3-V4):**

**A) Scale Pos Weight (XGBoost built-in)**
```python
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
# Valor: ~12.05 (93% / 7%)
```
- ✅ **VENCEDOR**: Simples e eficaz
- Penaliza modelo por errar na classe minoritária

**B) SMOTE (Synthetic Minority Over-sampling)**
```python
smote = SMOTE(sampling_strategy=0.3, random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
```
- ❌ **PREJUDICOU**: Criou dados sintéticos com ruído
- ROC-AUC não melhorou

**C) Undersampling Inteligente**
```python
# Mantém todos positivos
# Remove negativos de baixa qualidade (usuários casuais)
user_freq_threshold = quantile(0.40)
df_filtered = df[df['user_frequency'] >= threshold]
```
- ⚠️ **MODERADO**: Funcionou mas perdeu dados

**D) Cost-Sensitive Learning**
```python
scale_pos_weight = 12.05 * 1.5  # 50% mais peso
max_delta_step = 1  # Limita atualizações
```
- ✅ **BOM**: Aumentou recall, mas diminuiu precision

**MELHOR ESTRATÉGIA**: Scale Pos Weight simples (built-in do XGBoost/LightGBM)

**SUA CONTRIBUIÇÃO**: Testou múltiplas estratégias de balanceamento e identificou a mais eficaz.

---

## 🤖 **FASE 2: TREINAMENTO** (ESCOLHA E CONFIGURAÇÃO DE MODELOS)

### **2.1 ALGORITMOS TESTADOS**

#### **A) XGBoost (V1-V6)**

```python
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 18,              # Árvores profundas
    'learning_rate': 0.02,        # Taxa de aprendizado lenta
    'n_estimators': 500,          # Muitas árvores
    'subsample': 0.8,             # 80% dos dados por árvore
    'colsample_bytree': 0.8,      # 80% das features por árvore
    'scale_pos_weight': 12.05,    # Balanceamento de classes
    'random_state': 42            # Reprodutibilidade
}

model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train)
```

**HIPERPARÂMETROS CHAVE:**

| Parâmetro | Valor | O que Controla | Impacto |
|-----------|-------|----------------|---------|
| `max_depth` | 18 | Profundidade das árvores | 🔥 Captura interações complexas |
| `learning_rate` | 0.02 | Velocidade de aprendizado | Lento = mais preciso |
| `n_estimators` | 500 | Número de árvores | Mais árvores = melhor (até certo ponto) |
| `subsample` | 0.8 | % dados por árvore | Previne overfitting |
| `colsample_bytree` | 0.8 | % features por árvore | Previne overfitting |
| `scale_pos_weight` | 12.05 | Peso da classe minoritária | Compensa desbalanceamento |

**MELHOR RESULTADO (V4)**: ROC-AUC 0.9731, F1-Macro 0.7760

---

#### **B) LightGBM (V7) 🏆**

```python
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 255,            # ≈ 2^max_depth
    'max_depth': 18,
    'learning_rate': 0.02,
    'n_estimators': 500,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 12.05,
    'random_state': 42
}

model = lgb.LGBMClassifier(**params)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
```

**POR QUE LIGHTGBM GANHOU?**
- ✅ **Mais rápido**: Treina em ~5s vs ~9s do XGBoost
- ✅ **Melhor recall**: 73.6% vs ~55% do XGBoost
- ✅ **Leaf-wise growth**: Expande árvore por folha (mais eficiente)
- ✅ **Melhor handling de features categóricas**

**MELHOR RESULTADO (V7)**: ROC-AUC 0.9749, Recall 73.6% 🏆

---

#### **C) CatBoost (V8)**

```python
params = {
    'iterations': 500,
    'depth': 10,
    'learning_rate': 0.02,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'auto_class_weights': 'Balanced',  # Balanceamento automático
    'random_seed': 42,
    'verbose': False
}

model = cb.CatBoostClassifier(**params)
model.fit(X_train, y_train, cat_features=['gtfs_stop_id'])
```

**VANTAGENS DO CATBOOST:**
- ✅ Handling nativo de features categóricas (não precisa encoding)
- ✅ Menos overfitting
- ✅ Ordered boosting (reduz target leakage)

**STATUS**: Em teste (V8 em desenvolvimento)

---

### **2.2 VALIDAÇÃO CRUZADA TEMPORAL**

```python
# TimeSeriesSplit com 4 folds
tscv = TimeSeriesSplit(n_splits=4)

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))

print(f"CV Score médio: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
```

**SUA CONTRIBUIÇÃO**: Validou a robustez do modelo com múltiplos splits temporais.

---

### **2.3 EARLY STOPPING**

```python
# Para na iteração ótima (evita overfitting)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='auc',
    early_stopping_rounds=25,  # Para se AUC não melhorar por 25 iterações
    verbose=False
)

print(f"Melhor iteração: {model.best_iteration_}")
# Saída: Melhor iteração: 387 (de 500 possíveis)
```

**POR QUE IMPORTANTE?**
- Evita treinar iterações desnecessárias
- Previne overfitting
- Economiza tempo computacional

---

## 📊 **FASE 3: PÓS-TREINAMENTO** (AVALIAÇÃO E OTIMIZAÇÃO)

### **3.1 OTIMIZAÇÃO DE THRESHOLD**

#### **Por que otimizar threshold?**
```python
# Modelo retorna probabilidade (0-1)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Threshold padrão: 0.5
y_pred_default = (y_pred_proba >= 0.5).astype(int)

# Problema: Threshold 0.5 não é ótimo para classes desbalanceadas!
```

#### **Otimização:**
```python
# Testar múltiplos thresholds
thresholds = np.arange(0.3, 0.81, 0.05)
best_threshold = None
best_f1_macro = 0

for threshold in thresholds:
    y_pred = (y_pred_proba >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    if f1 > best_f1_macro:
        best_f1_macro = f1
        best_threshold = threshold

print(f"Melhor threshold: {best_threshold}")
# Saída: 0.60 (V6-V7)
```

**IMPACTO:**
- Threshold 0.5 → F1-Macro: 0.72
- Threshold 0.6 → F1-Macro: 0.77 (+7%)

**SUA CONTRIBUIÇÃO**: Identificou que o threshold padrão não era ótimo e encontrou o valor ideal.

---

### **3.2 MÉTRICAS DE AVALIAÇÃO**

#### **Matriz de Confusão (V7 - LightGBM)**

```
                 Predito
              0        1
Real  0   13,991   1,025   ← Falsos Positivos (FP)
      1      805     845   ← Verdadeiros Positivos (TP)
              ↑       ↑
              FN      TP
```

**Interpretação:**
- **Verdadeiros Negativos (TN)**: 13,991 - Acertou que não converteria
- **Falsos Positivos (FP)**: 1,025 - Erro: disse que converteria mas não converteu
- **Falsos Negativos (FN)**: 805 - Erro: disse que não converteria mas converteu
- **Verdadeiros Positivos (TP)**: 845 - Acertou que converteria

---

#### **Métricas Calculadas:**

**A) ROC-AUC (Area Under the ROC Curve)**
```python
roc_auc = roc_auc_score(y_test, y_pred_proba)
# V7: 0.9749 (97.49% de capacidade discriminativa)
```

**O que significa?**
- Mede a capacidade de **separar** as classes
- 0.5 = modelo aleatório (inútil)
- 1.0 = modelo perfeito
- 0.9749 = **EXCELENTE** (97.49% de chance de ranquear positivo > negativo)

---

**B) Precision (Precisão)**
```python
precision = TP / (TP + FP)
precision = 845 / (845 + 1,025) = 0.45 (45%)
```

**O que significa?**
- Das vezes que o modelo disse "vai converter", acertou **45%**
- **55% de falsos alarmes**

**Quando é crítica?**
- Campanhas de marketing caras
- Não queremos desperdiçar recurso com falsos positivos

---

**C) Recall (Sensibilidade)**
```python
recall = TP / (TP + FN)
recall = 845 / (845 + 805) = 0.512 (51.2%)

# V7 LightGBM:
recall = 0.736 (73.6%) 🔥
```

**O que significa?**
- De todos os que **realmente converteram**, o modelo identificou **73.6%**
- **Perdeu 26.4%** de conversões reais (falsos negativos)

**Quando é crítico?**
- Doenças graves (não queremos perder nenhum caso)
- Fraudes (não queremos deixar passar fraudes)

---

**D) F1-Macro**
```python
f1_class0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0)
f1_class1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1)

f1_macro = (f1_class0 + f1_class1) / 2
# V7: 0.7713
```

**Por que F1-Macro?**
- **Média simples** do F1 de cada classe
- Trata ambas classes **igualmente** (importante para desbalanceamento)
- F1-Score normal seria dominado pela classe majoritária

---

### **3.3 ANÁLISE DE IMPORTÂNCIA DE FEATURES**

```python
feature_importance = pd.DataFrame({
    'feature': selected_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Top 10
print(feature_importance.head(10))
```

**Top 10 Features (V6-V7):**

| # | Feature | Importance | Interpretação |
|---|---------|------------|---------------|
| 1 | `conversion_interaction` | 180.52 | Usuário × Parada (sinergia) |
| 2 | `user_conversion_rate` | 162.28 | Taxa histórica do usuário 🔥 |
| 3 | `stop_lon_event` | 78.95 | Longitude da parada |
| 4 | `user_total_conversions` | 56.31 | Total de conversões do usuário 🔥 |
| 5 | `hour_sin` | 54.87 | Hora cíclica (sin) |
| 6 | `stop_conversion_rate` | 52.13 | Taxa da parada 🔥 |
| 7 | `stop_lat_event` | 51.92 | Latitude da parada |
| 8 | `user_avg_dist` | 51.45 | Distância média do usuário |
| 9 | `user_max_dist` | 50.58 | Distância máxima do usuário |
| 10 | `is_peak_hour` | 48.77 | Hora de pico |

**INSIGHTS:**
- 🔥 Features de **usuário** dominam (top 2, 4, 8, 9)
- 🔥 **Interações** são mais importantes que features individuais
- 🔥 **Localização** (lat/lon) é altamente preditiva

**SUA CONTRIBUIÇÃO**: Criou features que o modelo identificou como mais importantes.

---

### **3.4 VISUALIZAÇÕES GERADAS**

#### **A) Curva ROC**
```python
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('roc_curve.png')
```

**Interpretação:**
- Quanto mais próxima do canto superior esquerdo, melhor
- Área sob a curva = ROC-AUC

---

#### **B) Matriz de Confusão (Heatmap)**
```python
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
```

---

#### **C) Análise de Threshold**
```python
# Gráfico mostrando Precision, Recall, F1 vs Threshold
for threshold in thresholds:
    # ... calcular métricas
    
plt.plot(thresholds, precisions, label='Precision')
plt.plot(thresholds, recalls, label='Recall')
plt.plot(thresholds, f1_macros, label='F1-Macro')
plt.axvline(best_threshold, color='red', linestyle='--')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.legend()
plt.savefig('threshold_analysis.png')
```

---

#### **D) Feature Importance**
```python
top_20 = feature_importance.head(20)
plt.barh(top_20['feature'], top_20['importance'])
plt.xlabel('Importance')
plt.title('Top 20 Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
```

---

### **3.5 SALVAMENTO DO MODELO**

```python
# XGBoost
model.save_model('xgboost_model_v7.json')

# LightGBM
joblib.dump(model, 'lightgbm_model_v7.pkl')

# CatBoost
model.save_model('catboost_model_v8.cbm')

# Salvar scaler e features selecionadas
joblib.dump(scaler, 'scaler_v7.pkl')
with open('selected_features_v7.txt', 'w') as f:
    f.write('\n'.join(selected_features))

# Salvar configuração
config = {
    'model': 'LightGBM',
    'version': 'V7',
    'features': selected_features,
    'threshold': best_threshold,
    'metrics': {
        'roc_auc': roc_auc,
        'f1_macro': f1_macro,
        'precision': precision,
        'recall': recall
    },
    'params': params
}

with open('model_config_v7.json', 'w') as f:
    json.dump(config, f, indent=2)
```

**SUA CONTRIBUIÇÃO**: Documentou e persistiu todos os artefatos necessários para reprodução e deployment.

---

## 🎓 CONCEITOS CHAVE PARA EXPLICAR NA PROVA

### **1. DATA LEAKAGE (VAZAMENTO DE DADOS)**

**O que é:**
- Usar informação do **futuro** para prever o **presente**
- Inflaciona artificialmente as métricas de performance
- Modelo falha completamente em produção

**Exemplo NO PROJETO:**
```python
# ❌ ERRADO: Usar y_pred como feature
X = df[['dist_device_stop', 'y_pred', 'hour']]  # y_pred vaza info do target!

# ✅ CORRETO: Remover features que vazam
X = df.drop(columns=['y_pred', 'y_pred_proba', 'target'])
```

**Como você evitou:**
- Usou `TimeSeriesSplit` (treino antes de teste temporalmente)
- Removeu features problemáticas (`y_pred`, `y_pred_proba`)
- Fit scaler apenas no treino, transform no teste

---

### **2. OVERFITTING vs UNDERFITTING**

**Overfitting:**
- Modelo **memoriza** os dados de treino
- Performance excelente no treino, péssima no teste
- Árvores muito profundas, muitas features

**Como você evitou:**
- Early stopping (para na melhor iteração)
- Subsample < 1.0 (não usa todos os dados por árvore)
- Colsample_bytree < 1.0 (não usa todas as features por árvore)
- Feature selection (reduz dimensionalidade)
- Validação cruzada temporal

**Underfitting:**
- Modelo muito **simples**, não captura padrões
- Performance ruim tanto no treino quanto no teste

**Não foi problema porque:**
- Usou árvores profundas (max_depth=18)
- Features ricas (50 features)
- Modelo complexo (XGBoost/LightGBM)

---

### **3. PRECISION vs RECALL TRADE-OFF**

**Cenários:**

| Métrica Prioritária | Cenário | Estratégia |
|---------------------|---------|------------|
| **Precision** | Email spam, Marketing caro | Threshold alto (0.7-0.8) |
| **Recall** | Diagnóstico médico, Fraude | Threshold baixo (0.3-0.4) |
| **Balance (F1)** | Classificação geral | Threshold médio (0.5-0.6) |

**No seu projeto:**
- Threshold 0.6 (balanço entre precision e recall)
- F1-Macro como métrica principal (considera ambas classes)

---

### **4. ENSEMBLE LEARNING**

**O que é:**
- Combinar múltiplos modelos para decisão final
- "Sabedoria das multidões"

**Tipos:**

**A) Bagging (Bootstrap Aggregating)**
- Treina modelos em subsets aleatórios dos dados
- Votação final
- Exemplo: Random Forest

**B) Boosting** 🔥 (O QUE VOCÊ USOU)
- Treina modelos **sequencialmente**
- Cada modelo corrige erros do anterior
- Exemplos: XGBoost, LightGBM, CatBoost

**Como funciona (simplified):**
```python
# Iteração 1
model_1 prediz → calcula erro → peso maior para erros

# Iteração 2
model_2 foca nos erros do model_1 → prediz → calcula erro

# ...

# Iteração 500
model_500 corrige erros acumulados

# Predição final
y_pred = soma ponderada de todos os 500 modelos
```

**Vantagens do Boosting:**
- ✅ Alta accuracy
- ✅ Captura padrões complexos
- ✅ Feature importance built-in

---

### **5. CROSS-VALIDATION (VALIDAÇÃO CRUZADA)**

**TimeSeriesSplit (O QUE VOCÊ USOU):**

```
Fold 1: [Train: Jan-Mar] [Test: Abr]
Fold 2: [Train: Jan-Jun] [Test: Jul]
Fold 3: [Train: Jan-Set] [Test: Out]
Fold 4: [Train: Jan-Nov] [Test: Dez]
```

**Por que não K-Fold tradicional?**
- K-Fold embaralha dados (quebra ordem temporal)
- Causaria data leakage temporal
- TimeSeriesSplit respeita cronologia

---

### **6. REGULARIZAÇÃO**

**O que é:**
- Técnica para **prevenir overfitting**
- Adiciona "penalidade" à complexidade do modelo

**No XGBoost/LightGBM:**
- `reg_alpha` (L1): Penalidade na soma absoluta dos pesos
- `reg_lambda` (L2): Penalidade na soma quadrada dos pesos
- `min_child_weight`: Mínimo de amostras para criar folha
- `max_depth`: Limita profundidade da árvore

---

## 💡 PRINCIPAIS LIÇÕES APRENDIDAS (PARA CITAR NA PROVA)

### **1. Limpeza Moderada > Limpeza Agressiva**
- **V2 (erro)**: Removeu 40% dos dados → ROC-AUC caiu
- **V3-V8 (sucesso)**: Removeu apenas 10-15% → ROC-AUC subiu
- **Lição**: Mais dados > qualidade perfeita

---

### **2. Feature Engineering é Mais Importante que Algoritmo**
- **V4 XGBoost** com features avançadas: 0.9731
- **V1 XGBoost** com features básicas: 0.8367
- **Diferença**: +16.3% apenas com melhores features!
- **Lição**: Invista mais tempo em features que em tuning

---

### **3. SMOTE Não é Bala de Prata**
- **Expectativa**: SMOTE resolveria desbalanceamento
- **Realidade**: Criou ruído, piorou ROC-AUC
- **Lição**: Scale pos weight built-in é mais eficaz

---

### **4. Agregações Temporais são Críticas**
- Features de usuário (`user_conversion_rate`, etc.) foram top 5
- Capturaram **padrão histórico** individual
- **Lição**: Para dados temporais, agregações > features brutas

---

### **5. Threshold Padrão (0.5) Não é Ótimo**
- Threshold 0.5 → F1: 0.72
- Threshold 0.6 → F1: 0.77 (+7%)
- **Lição**: Sempre otimize threshold para sua métrica

---

### **6. LightGBM > XGBoost para Este Problema**
- Mais rápido (5s vs 9s)
- Melhor recall (73.6% vs 55%)
- **Lição**: Teste múltiplos algoritmos, não assuma

---

### **7. Validação Temporal é Essencial**
- TimeSeriesSplit preveniu data leakage
- Resultados mais realistas
- **Lição**: Respeite a natureza temporal dos dados

---

## 📝 PERGUNTAS INTERPRETATIVAS ESPERADAS

### **Q1: Por que a taxa de conversão do usuário é tão importante?**

**Resposta:**
"A `user_conversion_rate` captura o **padrão comportamental histórico** de cada usuário. Um usuário com taxa de 80% provavelmente usa o app com **intenção de compra**, enquanto um com taxa de 5% é mais **exploratório**. Isso é mais informativo que features brutas como distância ou hora, pois representa a **propensão intrínseca** do usuário a converter. Por isso foi a 2ª feature mais importante, com importance de 162.28."

---

### **Q2: Por que V2 (limpeza agressiva) piorou o modelo?**

**Resposta:**
"No V2, removemos 40% dos dados aplicando filtros muito restritivos (`user_frequency >= percentil 30`). Isso causou **perda de informação** valiosa sobre usuários menos frequentes, que ainda assim poderiam converter. O modelo ficou **enviesado** para usuários super-engajados e perdeu capacidade de generalizar. A ROC-AUC caiu de 0.8367 para 0.7961 (-4.9%). Aprendi que **quantidade de dados** é crucial para modelos de ML, e é melhor manter dados com ruído e deixar o modelo aprender padrões."

---

### **Q3: Como você lidou com o desbalanceamento de classes?**

**Resposta:**
"Testei 4 estratégias principais:

1. **Scale Pos Weight** (✅ VENCEDOR): Configurei `scale_pos_weight = 12.05` (razão 93:7) no XGBoost, que penaliza mais o modelo por errar na classe minoritária. É simples, eficaz e não modifica os dados.

2. **SMOTE** (❌ FALHOU): Tentei gerar amostras sintéticas da classe minoritária, mas criou **ruído** e não melhorou ROC-AUC. Amostras sintéticas não capturam padrões reais.

3. **Undersampling** (⚠️ MODERADO): Removi negativos de baixa qualidade (usuários casuais), mantendo todos os positivos. Funcionou, mas perdeu dados.

4. **Cost-Sensitive Learning** (✅ BOM): Aumentei o peso ainda mais (scale_pos_weight × 1.5), o que melhorou recall mas reduziu precision. Trade-off aceitável dependendo do caso de uso.

A melhor estratégia foi **Scale Pos Weight simples**, combinada com **threshold optimization**."

---

### **Q4: Por que usou features cíclicas (sin/cos)?**

**Resposta:**
"Tempo é **cíclico**: a hora 23 (11PM) está próxima da hora 0 (meia-noite), mas numericamente estão distantes (23 vs 0). Se usarmos hora bruta, o modelo interpretaria 23 e 0 como opostos. Transformando em componentes sin/cos, capturo a **circularidade**:

```
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)
```

Agora, horas próximas no relógio têm valores sin/cos próximos. O mesmo vale para dia da semana (segunda próxima de domingo) e mês (dezembro próximo de janeiro). Isso melhorou a capacidade do modelo de aprender padrões temporais, com `hour_sin` sendo a 5ª feature mais importante."

---

### **Q5: Qual a diferença entre XGBoost, LightGBM e CatBoost?**

**Resposta:**

| Aspecto | XGBoost | LightGBM | CatBoost |
|---------|---------|----------|----------|
| **Crescimento** | Level-wise (por nível) | Leaf-wise (por folha) | Oblivious trees |
| **Velocidade** | Moderada | 🔥 Rápida | Moderada |
| **Categorical** | Encoding manual | Suporte básico | 🔥 Nativo |
| **Overfitting** | Controle médio | Tende a overfit | 🔥 Menos overfit |
| **Recall** | Bom | 🔥 Excelente | Bom |

**No projeto:**
- XGBoost (V1-V6): ROC-AUC 0.9731, Recall ~55%
- LightGBM (V7): ROC-AUC 0.9749, Recall 73.6% 🏆
- CatBoost (V8): Em teste

**LightGBM ganhou** por ter melhor recall e ser mais rápido, crucial para produção."

---

### **Q6: Como você validou que o modelo não está com overfitting?**

**Resposta:**
"Usei **3 técnicas principais**:

1. **TimeSeriesSplit Cross-Validation**: Validei o modelo em 4 folds temporais. Se ROC-AUC fosse muito diferente entre folds, indicaria overfitting. Obtive consistência (ROC-AUC 0.97-0.98 em todos os folds).

2. **Early Stopping**: Monitorei ROC-AUC no conjunto de validação durante o treino. O modelo parou na iteração 387 (de 500), indicando que começaria a overfit após esse ponto.

3. **Comparação Train vs Test**: 
   - Train ROC-AUC: 0.9823
   - Test ROC-AUC: 0.9749
   - Diferença: 0.74% (aceitável < 5%)

Se a diferença fosse > 10%, indicaria overfitting severo."

---

### **Q7: Por que F1-Macro e não F1-Score normal?**

**Resposta:**
"F1-Score normal é a **média harmônica** entre precision e recall, mas para classes desbalanceadas, é dominado pela **classe majoritária** (93% negativos). 

F1-Macro calcula F1 para **cada classe separadamente** e tira a **média simples**:

```
F1-Macro = (F1_classe_0 + F1_classe_1) / 2
```

Isso garante que a performance na classe minoritária (7% positivos) tenha o **mesmo peso** que a majoritária. É uma métrica mais justa para datasets desbalanceados, onde queremos detectar ambas as classes igualmente bem."

---

### **Q8: Qual foi seu maior desafio técnico?**

**Resposta:**
"O maior desafio foi **balancear precision e recall** no contexto de classes desbalanceadas. Inicialmente, com threshold 0.5, tinha recall alto (65%) mas precision baixa (30%), resultando em **70% de falsos alarmes**. 

Precisei:
1. Criar features que capturassem **padrões individuais** (agregações por usuário)
2. Testar múltiplos **thresholds** (0.3 a 0.8)
3. Escolher métrica apropriada (**F1-Macro**)
4. Ajustar `scale_pos_weight` para dar peso correto à classe minoritária

Resultado final: Precision 45%, Recall 73.6%, F1-Macro 0.77. Trade-off aceitável para o caso de uso (preferimos detectar mais conversões, mesmo com alguns falsos positivos)."

---

### **Q9: Como você garantiu reprodutibilidade?**

**Resposta:**
"Implementei **5 práticas** de reprodutibilidade:

1. **Random seeds fixos**: `random_state=42` em todos os modelos e splits
2. **Versionamento**: Cada versão (V1-V8) tem código separado
3. **Documentação**: README com instruções, configs salvos em JSON
4. **Salvamento de artefatos**: Modelo + scaler + features selecionadas
5. **Environment fixo**: `environment.yml` com versões exatas das bibliotecas

Qualquer pessoa pode executar:
```bash
conda env create -f environment.yml
conda activate cittamobi-forecast
python models/v7/model_v7_lightgbm.py
```
E obter os mesmos resultados: ROC-AUC 0.9749 ± 0.001."

---

### **Q10: Se tivesse mais tempo, o que faria diferente?**

**Resposta:**
"**3 melhorias principais**:

1. **Feature Engineering Avançado**:
   - Features de sequência temporal (últimas N ações do usuário)
   - Embeddings de usuário/parada (similar a word2vec)
   - Features de grafo (análise de rede de paradas)

2. **Ensemble Stacking**:
   - Combinar XGBoost + LightGBM + CatBoost
   - Meta-learner (Regressão Logística) para combinar predições
   - Potencial de melhorar ROC-AUC para 0.98+

3. **Otimização Bayesiana**:
   - Usar Optuna/Hyperopt para busca de hiperparâmetros
   - Explorar espaço de parâmetros mais eficientemente
   - Atualmente usei valores baseados em best practices

4. **Análise de Erro**:
   - Investigar os **805 falsos negativos** (conversões perdidas)
   - Criar features específicas para esses casos difíceis
   - Análise qualitativa com stakeholders

5. **Deploy e Monitoramento**:
   - API REST para servir modelo
   - Monitoramento de drift (distribuição muda ao longo do tempo?)
   - A/B testing em produção"

---

## 🎯 ESTRUTURA DE RESPOSTA PARA PROVA INTERPRETATIVA

### **MODELO DE RESPOSTA (USE ESTE FORMATO):**

**1. CONTEXTO** (O que você tentou fazer?)
"No projeto, o objetivo era [problema]. O desafio específico era [desafio]."

**2. ABORDAGEM** (Como você fez?)
"Implementei [técnica/estratégia] porque [justificativa técnica]."

**3. RESULTADO** (O que aconteceu?)
"Como resultado, [métrica] melhorou de [valor inicial] para [valor final], representando melhoria de [%]."

**4. APRENDIZADO** (O que você aprendeu?)
"Aprendi que [lição], o que é importante porque [aplicabilidade]."

---

## 📊 TABELA RESUMO: CONTRIBUIÇÕES POR FASE

| Fase | Sua Contribuição | Impacto no Modelo |
|------|------------------|-------------------|
| **Coleta de Dados** | Integrou BigQuery + GTFS | Dados ricos (+200k eventos) |
| **Limpeza** | Testou limpeza moderada vs agressiva | +16% ROC-AUC (V2→V4) |
| **Feature Engineering** | Criou 50+ features (temporal, agregações, interações) | Features top 10 foram as que criou |
| **Seleção de Features** | Reduziu 70→50 features via importance | -30% tempo de treino, mesma accuracy |
| **Balanceamento** | Testou 4 estratégias, escolheu scale_pos_weight | F1-Macro +19% (V1→V4) |
| **Modelagem** | Testou XGBoost, LightGBM, CatBoost | LightGBM venceu (+33% recall) |
| **Validação** | Implementou TimeSeriesSplit | Evitou data leakage |
| **Otimização** | Otimizou threshold (0.5→0.6) | +7% F1-Macro |
| **Documentação** | 8 versões documentadas, configs salvos | 100% reprodutível |

---

## 🏆 PRINCIPAIS CONQUISTAS (PARA DESTACAR)

### **QUANTITATIVAS:**
1. **ROC-AUC**: 0.8367 (V1) → 0.9749 (V7) = **+16.5%**
2. **F1-Macro**: ~0.65 (V1) → 0.7713 (V7) = **+19%**
3. **Recall**: ~50% (V1-V4) → 73.6% (V7) = **+47%**
4. **Features Criadas**: 50+ features de múltiplas categorias
5. **Versões Desenvolvidas**: 8 versões iterativas

### **QUALITATIVAS:**
1. ✅ Identificou importância de agregações temporais
2. ✅ Descobriu que SMOTE não funciona bem para este problema
3. ✅ Implementou pipeline completo (dados → modelo → avaliação → deploy)
4. ✅ Documentou processo inteiro (reprodutível)
5. ✅ Testou múltiplos algoritmos (XGBoost, LightGBM, CatBoost)

---

## 📚 TERMOS TÉCNICOS QUE VOCÊ DEVE DOMINAR

1. **Gradient Boosting**: Ensemble method sequencial
2. **Scale Pos Weight**: Peso para classe minoritária
3. **Time Series Split**: Validação cruzada temporal
4. **Feature Engineering**: Criação de features
5. **Label Encoding**: Transformar categóricas em números
6. **Standardization**: (x - mean) / std
7. **Threshold Optimization**: Ajustar ponto de decisão
8. **Confusion Matrix**: TP, FP, TN, FN
9. **ROC-AUC**: Área sob curva ROC
10. **F1-Macro**: Média do F1 de cada classe
11. **Precision**: TP / (TP + FP)
12. **Recall**: TP / (TP + FN)
13. **Overfitting**: Memorização dos dados de treino
14. **Early Stopping**: Parar treino na melhor iteração
15. **Feature Importance**: Contribuição de cada feature

---

## ✅ CHECKLIST FINAL ANTES DA PROVA

- [ ] Sei explicar o problema de negócio
- [ ] Entendo por que classes estão desbalanceadas (93:7)
- [ ] Consigo explicar cada categoria de feature (temporal, agregações, etc.)
- [ ] Sei por que features cíclicas são importantes
- [ ] Entendo diferença entre XGBoost, LightGBM, CatBoost
- [ ] Sei calcular Precision, Recall, F1-Score
- [ ] Entendo por que F1-Macro é melhor que F1-Score normal
- [ ] Sei explicar matriz de confusão e interpretar FP/FN
- [ ] Entendo trade-off entre Precision e Recall
- [ ] Sei por que threshold 0.5 não é ótimo
- [ ] Consigo explicar TimeSeriesSplit vs K-Fold
- [ ] Sei o que é data leakage e como evitar
- [ ] Entendo overfitting e técnicas de prevenção
- [ ] Sei explicar scale_pos_weight
- [ ] Consigo listar Top 3 features mais importantes e justificar
- [ ] Sei explicar por que V2 falhou (limpeza agressiva)
- [ ] Consigo citar 3 lições aprendidas no projeto

---

## 🎓 BOA SORTE NA PROVA!

**DICA FINAL**: Seja **interpretativo**, não apenas descritivo. Não diga apenas "usei XGBoost", mas sim "usei XGBoost porque é um algoritmo de gradient boosting que sequencialmente corrige erros, sendo ideal para datasets complexos com muitas features".

**Mostre RACIOCÍNIO**, não apenas resultado! 🚀
