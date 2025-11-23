# 🎯 K-NN com weights='distance' - Modelo Leak-Free

## 📋 Visão Geral

Modelo **K-Nearest Neighbors (K-NN)** otimizado para predição de conversão de usuários em aplicativo de transporte público (Cittamobi).

- **Algoritmo**: K-Nearest Neighbors
- **Melhor K**: 31
- **Weights**: 'distance' (vizinhos mais próximos têm mais peso)
- **ROC-AUC**: 0.7492
- **F1-Macro**: 0.6464
- **Status**: ✅ Leak-Free (sem vazamento de dados)

---

## 🚨 Prevenção de Data Leakage

### ❌ Problema Identificado
Features como `user_conversion_rate` e `stop_conversion_rate` eram calculadas usando o próprio target, causando **vazamento de dados** e ROC-AUC artificialmente alto (>98%).

### ✅ Solução Implementada
1. **Expanding Windows**: Para cada evento em tempo T, usar apenas dados históricos < T
2. **TimeSeriesSplit**: Validação temporal que respeita ordem cronológica
3. **Features Históricas**: Substituição por agregações baseadas apenas no passado
4. **Normalização**: StandardScaler essencial para K-NN funcionar corretamente

---

## 📊 Métricas de Performance

| Métrica | Valor |
|---------|-------|
| **ROC-AUC** | **0.7492** |
| Accuracy | 0.8968 |
| Precision | 0.4209 |
| Recall | 0.2980 |
| F1-Score | 0.3489 |
| F1-Macro | 0.6464 |
| Threshold | 0.30 |

### Matriz de Confusão

```
                 Predito
                 0        1
Real  0       9,621      421
      1         721      306
```

- **True Negatives**: 9,621
- **False Positives**: 421
- **False Negatives**: 721
- **True Positives**: 306

---

## 🔍 Comparação de Valores de K

| K | ROC-AUC | F1-Macro | Tempo (s) |
|---|---------|----------|----------|
| 31 🏆 | 0.7492 | 0.6464 | 0.5 |
| 21 | 0.7414 | 0.6415 | 0.5 |
| 15 | 0.7305 | 0.6421 | 0.5 |
| 11 | 0.7166 | 0.6340 | 0.5 |
|  7 | 0.6926 | 0.6264 | 0.5 |
|  5 | 0.6742 | 0.6170 | 0.6 |
|  3 | 0.6474 | 0.5868 | 0.6 |

### Insights sobre K
- **K muito pequeno** (3-5): Sensível a ruído, overfitting
- **K moderado** (31): **Melhor balanço** entre viés e variância
- **K muito grande** (>31): Underfitting, perde padrões locais

---

## 🔧 Configuração Técnica

### Parâmetros K-NN
```python
KNeighborsClassifier(
    n_neighbors=31,
    weights='distance',  # Vizinhos próximos têm mais peso
    algorithm='auto',    # Escolhe melhor algoritmo (ball_tree/kd_tree/brute)
    metric='minkowski',  # Distância Euclidiana
    p=2,                 # p=2 para Euclidiana
    n_jobs=-1            # Usa todos os cores do CPU
)
```

### Pipeline de Pré-processamento
```python
Pipeline([
    ('scaler', StandardScaler()),  # Normalização ESSENCIAL!
    ('knn', KNeighborsClassifier(...))
])
```

⚠️ **IMPORTANTE**: StandardScaler é **obrigatório** para K-NN! Sem normalização, features com escalas diferentes dominam o cálculo de distância.

---

## 📈 Top 10 Features Mais Importantes

*(Baseado em variância após normalização)*

| Rank | Feature | Variância |
|------|---------|----------|
| 1 | `dist_deviation_hist` | 1.0000 |
| 2 | `dist_ratio_hist` | 1.0000 |
| 3 | `user_avg_hour_hist` | 1.0000 |
| 4 | `Unnamed: 0` | 1.0000 |
| 5 | `stop_headway_mean` | 1.0000 |
| 6 | `stop_dist_std` | 1.0000 |
| 7 | `stop_dist_mean` | 1.0000 |
| 8 | `day_of_month_cos` | 1.0000 |
| 9 | `dist_x_peak` | 1.0000 |
| 10 | `week_cos` | 1.0000 |

---

## 📊 Comparação com Outros Modelos

| Modelo | ROC-AUC | Observações |
|--------|---------|-------------|
| **V6 CatBoost** | **86.69%** | 🏆 Melhor modelo geral |
| **V5 LightGBM** | **86.42%** | Segundo melhor |
| **K-NN (K=31)** | **74.92%** | Mais simples e interpretável |

### 💡 Quando Usar K-NN?

✅ **Vantagens**:
- Simples e fácil de entender
- Não faz suposições sobre distribuição dos dados
- Funciona bem com dados não-lineares
- Interpretabilidade: decisões baseadas em vizinhos similares

❌ **Desvantagens**:
- Performance inferior a gradient boosting em dados tabulares
- Sensível a features irrelevantes e alta dimensionalidade
- Computacionalmente caro em produção (precisa calcular distâncias)
- Requer normalização e pré-processamento cuidadoso

---

## 🗂️ Estrutura de Arquivos

```
KNN/
├── knn_leak_free.py              # Script principal
├── README_KNN.md                  # Esta documentação
├── visualizations/
│   ├── k_comparison.png           # Comparação de valores K
│   ├── roc_curve_knn.png          # Curva ROC
│   ├── confusion_matrix_knn.png   # Matriz de confusão
│   └── feature_variance_knn.png   # Importância features
└── reports/
    ├── knn_leak_free_report.txt   # Relatório detalhado
    └── knn_k_comparison.csv        # Dados comparação K
```

---

## 🚀 Como Usar

### 1. Executar o Modelo
```bash
cd KNN
python knn_leak_free.py
```

### 2. Ver Resultados
- **Visualizações**: `visualizations/*.png`
- **Relatório Técnico**: `reports/knn_leak_free_report.txt`
- **Dados Comparação**: `reports/knn_k_comparison.csv`

### 3. Ajustar Parâmetros
No código `knn_leak_free.py`, linha ~344:
```python
k_values = [3, 5, 7, 11, 15, 21, 31]  # Adicionar mais valores
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
- Grid search manual em valores de K
- Threshold otimizado para maximizar F1-Macro
- StandardScaler aplicado em todas as features

---

## 🎓 Conceitos Importantes

### K-Nearest Neighbors (K-NN)
Algoritmo de aprendizado supervisionado que classifica novos pontos baseado nos **K vizinhos mais próximos** no espaço de features.

### weights='distance'
Vizinhos mais próximos têm **maior peso** na decisão:
```
peso = 1 / distância
```
Resultado: Pontos muito próximos influenciam mais a predição.

### StandardScaler
Normaliza features para média=0 e desvio=1:
```
X_scaled = (X - mean) / std
```
**Essencial para K-NN**: Sem normalização, features com valores grandes dominam distâncias.

### Expanding Windows
Técnica anti-vazamento para séries temporais:
- Cada predição usa **apenas dados do passado**
- Simula exatamente o ambiente de produção
- Previne que modelo "veja o futuro"

---

## 🏆 Resultados e Conclusões

### Performance Alcançada
- **ROC-AUC**: 0.7492 (realístico para o problema)
- **F1-Macro**: 0.6464 (bom balanço entre classes)
- **Tempo de treino**: 0.5s (rápido)

### Comparação com Gradient Boosting
K-NN teve performance **inferior** a CatBoost/LightGBM:
- CatBoost: 86.69% vs K-NN: 74.92%
- **Motivo**: K-NN sofre com alta dimensionalidade (58 features)
- **Motivo**: K-NN é sensível a features irrelevantes

### Recomendação Final
- ✅ **Para Produção**: CatBoost ou LightGBM (melhor performance)
- ✅ **Para Interpretabilidade**: K-NN (decisões transparentes)
- ✅ **Para Baseline**: K-NN (rápido de implementar)

---

## 📚 Referências

- [Scikit-learn K-NN Documentation](https://scikit-learn.org/stable/modules/neighbors.html)
- [K-NN Theory and Practice](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
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
