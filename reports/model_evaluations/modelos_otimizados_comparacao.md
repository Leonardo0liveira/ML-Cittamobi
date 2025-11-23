# 📊 COMPARAÇÃO MODELOS OTIMIZADOS (Undersampling + CV)

## Tabela Comparativa - Métricas por Modelo

| Modelo | F1-Score (Class 1) | F1-Macro | ROC-AUC | Accuracy | Precision (Class 1) | Recall (Class 1) | CV ROC-AUC |
|--------|-------------------|----------|---------|----------|-------------------|-----------------|------------|
| **CatBoost** | **0.3887** | **0.6436** | **0.8040** | 0.8084 | 0.2838 | **0.6166** | 0.8020 ± 0.0093 |
| **Random Forest** | **0.4100** | **0.6651** | **0.8083** | **0.8595** | **0.3371** | **0.5231** | 0.7992 ± 0.0135 |
| **SVM (LinearSVC)** | **0.4044** | **0.6623** | **0.7992** | **0.8621** | **0.3371** | **0.5055** | 0.7895 ± 0.0051 |
| **SGD (Hinge)** | 0.3832 | 0.6559 | 0.7747 | **0.8729** | **0.3494** | 0.4243 | 0.7776 ± 0.0142 |
| **Naive Bayes (Bernoulli)** | 0.2897 | 0.5244 | 0.7767 | 0.6395 | 0.1774 | **0.7892** | 0.7651 ± 0.0089 |
| **KNN (K=51)** | 0.3963 | 0.6657 | 0.7533 | **0.8788** | 0.4033 | 0.3893 | 0.7416 ± 0.0086 |
| **Decision Tree (Depth 15)** | 0.3464 | 0.6309 | 0.7502 | 0.8413 | 0.2755 | 0.4664 | 0.7578 ± 0.0094 |

---

## 🏆 Rankings por Métrica

### ROC-AUC (Principal Métrica)
1. 🥇 **Random Forest**: 0.8083
2. 🥈 **CatBoost**: 0.8040
3. 🥉 **SVM**: 0.7992
4. **SGD**: 0.7747
5. **Naive Bayes**: 0.7767
6. **KNN**: 0.7533
7. **Decision Tree**: 0.7502

### Recall (Classe 1 - Conversão)
1. 🥇 **Naive Bayes**: 0.7892 (melhor detecção)
2. 🥈 **CatBoost**: 0.6166
3. 🥉 **Random Forest**: 0.5231
4. **SVM**: 0.5055
5. **Decision Tree**: 0.4664
6. **SGD**: 0.4243
7. **KNN**: 0.3893

### F1-Score (Classe 1 - Balanço)
1. 🥇 **Random Forest**: 0.4100
2. 🥈 **SVM**: 0.4044
3. 🥉 **KNN**: 0.3963
4. **CatBoost**: 0.3887
5. **SGD**: 0.3832
6. **Decision Tree**: 0.3464
7. **Naive Bayes**: 0.2897

### Accuracy (Geral)
1. 🥇 **KNN**: 0.8788
2. 🥈 **SGD**: 0.8729
3. 🥉 **SVM**: 0.8621
4. **Random Forest**: 0.8595
5. **Decision Tree**: 0.8413
6. **CatBoost**: 0.8084
7. **Naive Bayes**: 0.6395

### Precision (Classe 1)
1. 🥇 **KNN**: 0.4033
2. 🥈 **SGD**: 0.3494
3. 🥉 **Random Forest / SVM**: 0.3371
5. **CatBoost**: 0.2838
6. **Decision Tree**: 0.2755
7. **Naive Bayes**: 0.1774

### F1-Macro (Balanceamento Geral)
1. 🥇 **Random Forest**: 0.6651
2. 🥈 **KNN**: 0.6657
3. 🥉 **SVM**: 0.6623
4. **SGD**: 0.6559
5. **CatBoost**: 0.6436
6. **Decision Tree**: 0.6309
7. **Naive Bayes**: 0.5244

---

## 💡 Análise de Trade-offs

### Melhor Geral (ROC-AUC + F1)
- **Random Forest**: Melhor balanço entre todas as métricas
- **CatBoost**: Segundo melhor ROC-AUC, excelente recall

### Melhor para Detecção (High Recall)
- **Naive Bayes**: 78.92% de recall (detecta 4 de cada 5 conversões)
- Trade-off: Baixa precision (muitos falsos positivos)

### Melhor para Precisão (Low False Positives)
- **KNN**: 40.33% precision (menos falsos positivos)
- **SGD**: 34.94% precision

### Melhor para Produção
- **Random Forest** ou **CatBoost**: Melhor ROC-AUC + recall aceitável
- **SVM**: Bom balanço, mais rápido que RF/CatBoost

---

## 📈 Impacto do Undersampling

Todos os modelos utilizaram:
- **Ratio 2:1** (classe_0 : classe_1)
- **Cross-validation**: TimeSeriesSplit com 5 folds
- **Classification reports** completos
- **Código limpo**: Sem prints verbosos

### Melhorias vs Versões Originais:
- **Random Forest**: Recall 16% → 52% (+3.3x)
- **Decision Tree**: Mais estável com CV
- **SVM**: Recall 50% vs 35% original
- **Naive Bayes**: Recall 79% (excelente para classe minoritária)

---

## 🎯 Recomendações

### Para Maximizar Conversões Detectadas:
**Naive Bayes (Bernoulli)** - Detecta 79% das conversões

### Para Aplicação em Produção:
**Random Forest** ou **CatBoost** - Melhor ROC-AUC (80%+), recall aceitável (52-62%)

### Para Sistema Rápido:
**SVM (LinearSVC)** - ROC-AUC 79.92%, treinamento rápido

### Para Máxima Precisão:
**KNN** - 40% precision, menos falsos positivos

---

**Data da análise**: 22 de novembro de 2025  
**Dataset**: 50k registros (BigQuery, TABLESAMPLE 20%)  
**Metodologia**: Expanding windows leak-free + Undersampling 2:1
