# 🚀 Estratégias Para Melhorar o SGD Classifier

## 📊 Status Atual
- **SGD V2**: 79.17% ROC-AUC com 20 features
- **Baseline**: 78.00% ROC-AUC com 48 features
- **Melhor do projeto**: CatBoost 86.69% (gap de 7.52 pontos)

---

## 🎯 ESTRATÉGIAS DE MELHORIA

### 1️⃣ **FEATURE ENGINEERING AVANÇADO** (Impacto: ++2-4%)

#### A) **Interações Não-Lineares das Top Features**
Atualmente temos 3 features principais:
- `stop_event_rate` (+1.207) - DOMINANTE
- `stop_event_count` (-0.555)
- `stop_total_samples` (+0.284)

**Criar:**
```python
# Razões e proporções
df['stop_rate_per_sample'] = df['stop_event_rate'] / (df['stop_total_samples'] + 1)
df['stop_density'] = df['stop_event_count'] / (df['stop_total_samples'] + 1)

# Bins categóricos (transformar linear em não-linear)
df['stop_tier'] = pd.qcut(df['stop_event_rate'], q=5, labels=[1,2,3,4,5])
df['stop_tier_onehot'] = pd.get_dummies(df['stop_tier'], prefix='tier')

# Polinômios das top features
df['stop_rate_squared'] = df['stop_event_rate'] ** 2
df['stop_rate_log'] = np.log1p(df['stop_event_rate'])
```

#### B) **Features Temporais Mais Inteligentes**
Atualmente: `hour`, `time_hour`, `is_peak_hour`, `is_weekend`

**Criar:**
```python
# Período do dia (mais granular que is_peak_hour)
df['time_period'] = pd.cut(df['hour'], 
                            bins=[0, 6, 9, 12, 14, 17, 19, 24],
                            labels=['madrugada', 'manha_pico', 'manha', 
                                    'almoco', 'tarde', 'tarde_pico', 'noite'])

# Interação hora x dia da semana (segunda 8h != domingo 8h)
df['hour_weekday_interaction'] = df['hour'] * df['day_of_week']
df['is_weekend_peak'] = (df['is_weekend'] == 1) & (df['is_peak_hour'] == 1)

# Distância do horário de pico
df['distance_from_peak'] = df['hour'].apply(lambda h: min(abs(h-8), abs(h-18)))
```

#### C) **Features Geográficas Contextuais**
Atualmente: `device_lat`, `device_lon`, `stop_lat_event`, `stop_lon_event`

**Criar:**
```python
# Distância Euclidiana device <-> stop
df['geo_distance'] = np.sqrt(
    (df['device_lat'] - df['stop_lat_event'])**2 + 
    (df['device_lon'] - df['stop_lon_event'])**2
)

# Quadrante geográfico (região da cidade)
df['lat_quadrant'] = pd.cut(df['device_lat'], bins=4, labels=['sul', 'centro_sul', 'centro_norte', 'norte'])
df['lon_quadrant'] = pd.cut(df['device_lon'], bins=4, labels=['oeste', 'centro_oeste', 'centro_leste', 'leste'])

# Cluster geográfico (K-means com lat/lon)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=8, random_state=42)
df['geo_cluster'] = kmeans.fit_predict(df[['device_lat', 'device_lon']])
```

#### D) **Features de Recência Melhoradas**
Atualmente: `user_recency_days`, `stop_recency_days`

**Criar:**
```python
# Bins de recência (recente vs antigo)
df['user_recency_bin'] = pd.cut(df['user_recency_days'], 
                                  bins=[0, 1, 3, 7, 30, 999],
                                  labels=['hoje', 'recente', 'semana', 'mes', 'antigo'])

# Score de recência (exponential decay)
df['user_recency_score'] = np.exp(-df['user_recency_days'] / 7)  # decay de 7 dias

# Combinação usuário + parada
df['user_stop_recency_interaction'] = df['user_recency_score'] * df['stop_event_rate']
```

---

### 2️⃣ **AJUSTE DE HIPERPARÂMETROS** (Impacto: +1-2%)

#### Atualmente usando:
```python
alpha=0.001  # HIGH_REGULARIZATION
penalty='l2'
learning_rate='optimal'
```

#### Testar Grid Refinado:
```python
param_grid = {
    'alpha': [0.0005, 0.001, 0.002, 0.003],  # Afinar em torno de 0.001
    'l1_ratio': [0.0, 0.1, 0.15, 0.2],  # Elastic Net leve
    'learning_rate': ['optimal', 'invscaling', 'adaptive'],
    'eta0': [0.001, 0.01, 0.1],  # Learning rate inicial
    'max_iter': [1000, 1500, 2000],
    'tol': [1e-3, 1e-4, 1e-5]  # Tolerância de convergência
}
```

**Por que?**
- Alpha=0.001 foi melhor no V1, mas com 20 features pode precisar ajuste
- Elastic Net leve (l1_ratio=0.1-0.2) pode ajudar a selecionar melhor
- Learning rate adaptativo pode convergir melhor

---

### 3️⃣ **ENSEMBLE COM VOTAÇÃO** (Impacto: +1-3%)

#### A) **Ensemble Homogêneo (múltiplos SGDs)**
```python
from sklearn.ensemble import VotingClassifier

# Treinar múltiplos SGD com seeds diferentes
sgd1 = SGDClassifier(..., random_state=42)
sgd2 = SGDClassifier(..., random_state=123)
sgd3 = SGDClassifier(..., random_state=456)

ensemble = VotingClassifier([
    ('sgd1', sgd1),
    ('sgd2', sgd2),
    ('sgd3', sgd3)
], voting='soft')  # soft = média das probabilidades
```

#### B) **Ensemble Heterogêneo (SGD + outros lineares)**
```python
from sklearn.linear_model import LogisticRegression, RidgeClassifier

ensemble = VotingClassifier([
    ('sgd', SGDClassifier(...)),
    ('logreg', LogisticRegression(...)),
    ('ridge', RidgeClassifier(...))
], voting='soft', weights=[2, 1, 1])  # SGD com mais peso
```

---

### 4️⃣ **CALIBRAÇÃO DE PROBABILIDADES** (Impacto: +0.5-1.5%)

```python
from sklearn.calibration import CalibratedClassifierCV

# Calibrar predições do SGD (melhora ROC-AUC)
calibrated_sgd = CalibratedClassifierCV(
    pipeline_final, 
    method='sigmoid',  # ou 'isotonic'
    cv=5
)

calibrated_sgd.fit(X_train, y_train)
y_pred_proba_calibrated = calibrated_sgd.predict_proba(X_test)[:, 1]
```

**Por que?**
- SGD pode ter probabilidades mal calibradas
- Calibração melhora ROC-AUC sem mudar o ranking

---

### 5️⃣ **TRATAMENTO DE DESBALANCEAMENTO AVANÇADO** (Impacto: +1-2%)

Atualmente: `class_weight='balanced'` (90% vs 10%)

#### A) **SMOTE (Synthetic Minority Oversampling)**
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy=0.3, random_state=42)  # Subir para 30%
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

pipeline_final.fit(X_train_balanced, y_train_balanced)
```

#### B) **Threshold Otimizado por Negócio**
Atualmente: threshold=0.70 (otimizado para F1-Macro)

```python
# Otimizar para maximizar recall mantendo precision razoável
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

# Encontrar threshold onde precision >= 0.35 e recall é máximo
optimal_idx = np.where((precision >= 0.35) & (recall == recall[precision >= 0.35].max()))[0][0]
optimal_threshold = thresholds[optimal_idx]
```

#### C) **Pesos de Classe Customizados**
```python
# Ao invés de 'balanced', testar pesos específicos
class_weights = {0: 1.0, 1: 12.0}  # Dar mais peso à classe minoritária

SGDClassifier(..., class_weight=class_weights)
```

---

### 6️⃣ **STACKING COM MODELO NÃO-LINEAR** (Impacto: +2-4%)

```python
from sklearn.ensemble import StackingClassifier

# Usar SGD V2 como base + LightGBM como meta-learner
stacking = StackingClassifier(
    estimators=[
        ('sgd', pipeline_final),
        ('logreg', LogisticRegression(...))
    ],
    final_estimator=LGBMClassifier(n_estimators=50, max_depth=4),
    cv=5
)
```

**Por que?**
- SGD captura relações lineares (rápido)
- LightGBM captura não-linearidades (preciso)
- Combina velocidade do SGD com precisão do LGBM

---

### 7️⃣ **OTIMIZAÇÃO DO PIPELINE COMPLETO** (Impacto: +0.5-1%)

#### A) **Normalização Alternativa**
```python
# Testar outros scalers
from sklearn.preprocessing import RobustScaler, MinMaxScaler

RobustScaler()  # Mais robusto a outliers
MinMaxScaler()  # Escala [0,1]
```

#### B) **Feature Selection Iterativa**
```python
from sklearn.feature_selection import RFE

# Recursive Feature Elimination
rfe = RFE(estimator=SGDClassifier(...), n_features_to_select=15, step=1)
rfe.fit(X_train, y_train)
selected_features = X_train.columns[rfe.support_]
```

#### C) **PCA para Reduzir Colinearidade**
```python
from sklearn.decomposition import PCA

# Aplicar PCA nas features geográficas (lat/lon correlacionadas)
pca_geo = PCA(n_components=2)
geo_features_pca = pca_geo.fit_transform(X[['device_lat', 'device_lon', 
                                               'stop_lat_event', 'stop_lon_event']])
```

---

## 📊 ROADMAP DE IMPLEMENTAÇÃO

### **FASE 1: Quick Wins (1-2 dias)** ⚡
Impacto esperado: +2-3%

1. ✅ **Feature Engineering Básico**
   - Criar `stop_rate_per_sample`
   - Criar `geo_distance`
   - Criar `time_period` (bins de hora)

2. ✅ **Calibração de Probabilidades**
   - Aplicar `CalibratedClassifierCV`
   
3. ✅ **Ajuste de Threshold**
   - Otimizar para precision >= 0.35

**Script**: `sgd_v3_quick_wins.py`

---

### **FASE 2: Feature Engineering Avançado (3-5 dias)** 🔬
Impacto esperado: +2-4%

1. ✅ **Interações Complexas**
   - Polinômios das top 5 features
   - Bins/categorização de features contínuas
   
2. ✅ **Features Temporais Avançadas**
   - `hour_weekday_interaction`
   - `distance_from_peak`
   
3. ✅ **Clusters Geográficos**
   - K-means em lat/lon
   - One-hot encoding de clusters

**Script**: `sgd_v4_advanced_features.py`

---

### **FASE 3: Ensembles e Stacking (5-7 dias)** 🎯
Impacto esperado: +3-5%

1. ✅ **Voting Ensemble**
   - SGD + LogisticRegression + Ridge
   
2. ✅ **Stacking com LightGBM**
   - SGD como base
   - LGBM como meta-learner
   
3. ✅ **Hyperparameter Tuning Grid**
   - Busca refinada em torno de alpha=0.001

**Script**: `sgd_v5_ensemble.py`

---

## 🎯 META REALISTA

### Projeção de Performance:

| Fase | ROC-AUC | Ganho | Tempo |
|------|---------|-------|-------|
| **V2 Atual** | 79.17% | - | - |
| V3 (Quick Wins) | 81-82% | +2-3% | 2 dias |
| V4 (Advanced FE) | 83-84% | +4-5% | 5 dias |
| V5 (Ensemble) | 84-86% | +5-7% | 7 dias |
| **Meta Final** | **~85%** | **+6%** | **7-10 dias** |

**Gap para CatBoost (86.69%)**: Reduzir de 7.5% para ~2%

---

## 💡 RECOMENDAÇÃO ESTRATÉGICA

### **Abordagem Híbrida**: SGD para Velocidade + LightGBM para Precisão

```python
class HybridPredictor:
    def __init__(self):
        self.sgd_fast = pipeline_sgd_v2  # 79% AUC, <10ms
        self.lgbm_precise = model_lgbm    # 86% AUC, ~50ms
        
    def predict(self, X, mode='fast'):
        if mode == 'fast':
            return self.sgd_fast.predict_proba(X)[:, 1]
        elif mode == 'precise':
            return self.lgbm_precise.predict_proba(X)[:, 1]
        elif mode == 'hybrid':
            # SGD filtra "easy cases", LGBM para "hard cases"
            sgd_proba = self.sgd_fast.predict_proba(X)[:, 1]
            
            # Se confiança alta (>0.8 ou <0.2), usar SGD
            easy_mask = (sgd_proba > 0.8) | (sgd_proba < 0.2)
            
            result = sgd_proba.copy()
            result[~easy_mask] = self.lgbm_precise.predict_proba(X[~easy_mask])[:, 1]
            
            return result
```

**Vantagens**:
- 70-80% das predições usam SGD (rápido)
- 20-30% usam LGBM (casos difíceis)
- **Latência média**: ~20ms (vs 50ms LGBM puro)
- **AUC esperado**: ~84-85%

---

## 🚀 PRÓXIMO PASSO RECOMENDADO

**Implementar FASE 1 (Quick Wins)** - Melhor custo-benefício:

```bash
python sgd_v3_quick_wins.py
```

Vai criar:
- 5-7 novas features (interactions + geo_distance)
- Calibração de probabilidades
- Threshold otimizado
- **Ganho esperado**: +2-3% (81-82% ROC-AUC)
- **Tempo**: 1-2 horas de implementação + 5-10min de treino

Quer que eu implemente isso agora? 🚀
