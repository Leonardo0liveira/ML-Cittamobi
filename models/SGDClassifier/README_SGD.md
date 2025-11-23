# 📈 SGD Classifier - Modelo Leak-Free

## 📋 Visão Geral

Modelo **Stochastic Gradient Descent (SGD) Classifier** otimizado para predição de conversão de usuários em aplicativo de transporte público (Cittamobi).

- **Algoritmo**: SGD Classifier (Logistic Regression via SGD)
- **Melhor Config**: HIGH_REGULARIZATION
- **Loss Function**: log_loss (logistic regression)
- **Penalty**: l2
- **ROC-AUC**: 0.7963
- **F1-Macro**: 0.6726
- **Status**: ✅ Leak-Free (sem vazamento de dados)

---

## 🚨 Prevenção de Data Leakage

### ❌ Problema Identificado
Features como `user_conversion_rate` e `stop_conversion_rate` eram calculadas usando o próprio target, causando **vazamento de dados** e ROC-AUC artificialmente alto (>98%).

### ✅ Solução Implementada
1. **Expanding Windows**: Para cada evento em tempo T, usar apenas dados históricos < T
2. **TimeSeriesSplit**: Validação temporal que respeita ordem cronológica
3. **Features Históricas**: Substituição por agregações baseadas apenas no passado
4. **Normalização**: StandardScaler essencial para SGD funcionar corretamente

---

## 📊 Métricas de Performance

| Métrica | Valor |
|---------|-------|
| **ROC-AUC** | **0.7963** |
| Accuracy | 0.8863 |
| Precision | 0.3923 |
| Recall | 0.4253 |
| F1-Score | 0.4081 |
| F1-Macro | 0.6726 |
| Threshold | 0.75 |

### Matriz de Confusão

```
                 Predito
                 0        1
Real  0      10,394      745
      1         650      481
```

- **True Negatives**: 10,394
- **False Positives**: 745
- **False Negatives**: 650
- **True Positives**: 481

---

## 🔍 Comparação de Configurações

| Config | ROC-AUC | F1-Macro | Alpha | Penalty | Tempo (s) |
|--------|---------|----------|-------|---------|----------|
| HIGH_REGULARIZATION 🏆 | 0.7963 | 0.6726 | 0.001 | l2 | 0.2 |
| ELASTIC_NET | 0.7841 | 0.6446 | 0.0001 | elasticnet | 0.3 |
| L1_PENALTY | 0.7669 | 0.6604 | 0.0001 | elasticnet | 0.3 |
| LOW_REGULARIZATION | 0.7108 | 0.5247 | 1e-05 | l2 | 0.2 |
| BASELINE | 0.6997 | 0.5513 | 0.0001 | l2 | 0.3 |

### Insights sobre Configurações
- **BASELINE**: Configuração padrão com alpha=0.0001
- **HIGH_REGULARIZATION**: Maior alpha (0.001) previne overfitting
- **LOW_REGULARIZATION**: Menor alpha (0.00001) permite mais complexidade
- **ELASTIC_NET**: Combina L1 e L2 (l1_ratio=0.5)
- **L1_PENALTY**: Lasso (l1_ratio=1.0) para seleção de features

---

## 🔧 Configuração Técnica

### Parâmetros SGD Classifier
```python
SGDClassifier(
    loss='log_loss',            # Regressão logística
    penalty='l2',           # Regularização
    alpha=0.001,          # Taxa de regularização
    l1_ratio=0.0,            # Elastic Net ratio
    class_weight='balanced',    # Lida com desbalanceamento
    learning_rate='optimal',    # Taxa de aprendizado adaptativa
    max_iter=1000,              # Máximo de épocas
    early_stopping=True,        # Para se não houver melhoria
    validation_fraction=0.1,    # 10% para validação
    n_iter_no_change=5,         # Paciência: 5 épocas
    random_state=42,
    n_jobs=-1                   # Usa todos os cores
)
```

### Pipeline de Pré-processamento
```python
Pipeline([
    ('scaler', StandardScaler()),  # Normalização ESSENCIAL!
    ('sgd', SGDClassifier(...))
])
```

⚠️ **IMPORTANTE**: StandardScaler é **obrigatório** para SGD! Sem normalização, features com escalas diferentes dominam o gradiente.

---

## 📈 Top 20 Features Mais Importantes

*(Baseado em coeficientes do modelo)*

| Rank | Feature | Coeficiente |
|------|---------|-------------|
| 1 | `stop_event_rate` | +1.143173 |
| 2 | `stop_event_count` | -0.487480 |
| 3 | `is_peak_hour` | -0.201753 |
| 4 | `day_of_week` | -0.163586 |
| 5 | `time_day_of_week` | +0.152031 |
| 6 | `headway_x_hour` | +0.151508 |
| 7 | `hour_cos` | +0.151393 |
| 8 | `stop_lon_event` | -0.148723 |
| 9 | `stop_lat_event` | +0.142275 |
| 10 | `time_hour` | +0.142269 |
| 11 | `stop_total_samples` | +0.139098 |
| 12 | `headway_x_weekend` | -0.116230 |
| 13 | `stop_dist_mean` | -0.116183 |
| 14 | `time_day_of_month` | +0.115950 |
| 15 | `day_cos` | -0.100169 |
| 16 | `int64_field_0` | -0.098772 |
| 17 | `device_lon` | -0.090760 |
| 18 | `is_weekend` | +0.086403 |
| 19 | `user_frequency` | -0.085427 |
| 20 | `user_recency_days` | +0.084441 |

- **Coeficiente Positivo**: Aumenta probabilidade de conversão
- **Coeficiente Negativo**: Diminui probabilidade de conversão

---

## 📊 Comparação com Outros Modelos

| Modelo | ROC-AUC | Observações |
|--------|---------|-------------|
| **V6 CatBoost** | **86.69%** | 🏆 Melhor modelo geral |
| **V5 LightGBM** | **86.42%** | Segundo melhor |
| **K-NN (K=31)** | **75.42%** | Mais simples |
| **SGD Classifier** | **79.63%** | Rápido e eficiente |

### 💡 Quando Usar SGD Classifier?

✅ **Vantagens**:
- **Muito rápido**: Treina em mini-batches (ideal para dados grandes)
- **Leve**: Baixo consumo de memória
- **Aprendizado online**: Pode ser atualizado com novos dados sem retreinar tudo
- **Regularização flexível**: L1, L2 ou Elastic Net
- **Interpretável**: Coeficientes mostram importância e direção das features

❌ **Desvantagens**:
- **Modelo linear**: Não captura interações não-lineares automaticamente
- **Performance inferior** a gradient boosting em problemas complexos
- **Sensível à escala**: Requer normalização obrigatória
- **Hiperparâmetros**: Requer tuning de alpha e learning rate

---

## 🗂️ Estrutura de Arquivos

```
SGDClassifier/
├── sgd_leak_free.py               # Script principal
├── README_SGD.md                   # Esta documentação
├── visualizations/
│   ├── config_comparison.png       # Comparação configurações
│   ├── roc_curve_sgd.png           # Curva ROC
│   ├── confusion_matrix_sgd.png    # Matriz de confusão
│   └── feature_coefficients_sgd.png # Coeficientes features
└── reports/
    ├── sgd_leak_free_report.txt    # Relatório detalhado
    └── sgd_config_comparison.csv    # Dados comparação configs
```

---

## 🚀 Como Usar

### 1. Executar o Modelo
```bash
cd SGDClassifier
python sgd_leak_free.py
```

### 2. Ver Resultados
- **Visualizações**: `visualizations/*.png`
- **Relatório Técnico**: `reports/sgd_leak_free_report.txt`
- **Dados Comparação**: `reports/sgd_config_comparison.csv`

### 3. Ajustar Parâmetros
No código `sgd_leak_free.py`, linha ~248:
```python
configs = [
    {'name': 'CUSTOM', 'alpha': 0.0005, 'l1_ratio': 0},
    # Adicionar mais configurações
]
```

---

## ⚙️ Requisitos Técnicos

```
Python >= 3.9
scikit-learn >= 1.0
pandas >= 1.3
numpy >= 1.21
matplotlib >= 3.4
seaborn >= 0.11
google-cloud-bigquery >= 3.0
```

---

## 📝 Metodologia de Desenvolvimento

### 1. Preparação Temporal dos Dados
- Ordenação cronológica por `event_timestamp`
- Features temporais e cíclicas (sin/cos)
- Período: 3 meses de dados

### 2. Expanding Windows (Leak-Free)
Para cada evento em tempo T:
```python
# ✅ CORRETO: Usa apenas histórico < T
hist_data = df.iloc[:i]  # Dados anteriores
user_hist_conversion_rate = hist_data[target].mean()

# ❌ ERRADO: Usa todos os dados (inclui futuro)
user_conversion_rate = df.groupby('user')[target].mean()
```

### 3. Validação Temporal
- **TimeSeriesSplit** com 3 folds
- Treino: 75% dos dados (temporalmente anteriores)
- Teste: 25% dos dados (temporalmente posteriores)

### 4. Otimização de Hiperparâmetros
- Grid search manual em configurações
- Threshold otimizado para maximizar F1-Macro
- StandardScaler aplicado em todas as features

---

## 🎓 Conceitos Importantes

### Stochastic Gradient Descent (SGD)
Algoritmo de otimização que **atualiza pesos iterativamente** usando gradientes calculados em **mini-batches** de dados. Muito mais rápido que gradiente descendente tradicional.

### loss='log_loss'
Usa **log loss** (cross-entropy) como função objetivo:
```
log_loss = -[y*log(p) + (1-y)*log(1-p)]
```
Equivalente a **regressão logística** treinada via SGD.

### Regularização
Previne overfitting penalizando pesos grandes:
- **L2 (Ridge)**: penalty='l2' → minimiza soma dos quadrados dos coeficientes
- **L1 (Lasso)**: penalty='l1' → minimiza soma dos valores absolutos (feature selection)
- **Elastic Net**: combina L1 e L2 (l1_ratio controla proporção)

### class_weight='balanced'
Ajusta pesos das classes automaticamente:
```
weight_class_i = n_samples / (n_classes * n_samples_class_i)
```
**Essencial** para datasets desbalanceados (90% vs 10%).

### early_stopping
Para o treinamento se não houver melhoria:
- Usa 10% dos dados para validação (validation_fraction=0.1)
- Para após 5 épocas sem melhoria (n_iter_no_change=5)
- Previne overfitting e economiza tempo

---

## 🏆 Resultados e Conclusões

### Performance Alcançada
- **ROC-AUC**: 0.7963 (realístico para o problema)
- **F1-Macro**: 0.6726 (bom balanço entre classes)
- **Tempo de treino**: 0.2s (muito rápido)

### Comparação com Gradient Boosting
SGD teve performance **similar ao K-NN** mas **inferior** a CatBoost/LightGBM:
- CatBoost: 86.69% vs SGD: 79.63%
- **Motivo**: SGD é um modelo linear (não captura interações não-lineares)
- **Vantagem**: SGD é **muito mais rápido** (~1s vs ~100s)

### Recomendação Final
- ✅ **Para Produção (Performance)**: CatBoost ou LightGBM
- ✅ **Para Produção (Velocidade)**: SGD Classifier
- ✅ **Para Aprendizado Online**: SGD (pode ser atualizado incrementalmente)
- ✅ **Para Interpretabilidade**: SGD (coeficientes transparentes)

---

## 📚 Referências

- [Scikit-learn SGD Documentation](https://scikit-learn.org/stable/modules/sgd.html)
- [SGD Classifier Theory](https://scikit-learn.org/stable/modules/linear_model.html#sgd)
- [Stochastic Gradient Descent Wikipedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
- [StandardScaler Guide](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
- [TimeSeriesSplit for Temporal Validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)

---

## 👨‍💻 Autor e Contato

**Projeto**: Cittamobi ML - Predição de Conversão de Usuários
**Data**: Novembro 2025
**Status**: ✅ Produção-Ready (Leak-Free)

---

## 📄 Licença

Este projeto é parte do portfólio de Machine Learning Cittamobi.
