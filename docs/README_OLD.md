# 🚀 Projeto Machine Learning - Cittamobi Forecast

## 📋 Índice

1. [Visão Geral](#visão-geral)
2. [Estrutura do Projeto](#estrutura-do-projeto)
3. [Modelos Desenvolvidos](#modelos-desenvolvidos)
4. [Resultados](#resultados)
5. [Como Usar](#como-usar)
6. [Documentação](#documentação)

---

## 🎯 Visão Geral

Projeto de Machine Learning para previsão de eventos usando XGBoost para classificação binária. O projeto passou por múltiplas iterações, incluindo limpeza de data leakage, otimização de hiperparâmetros, feature engineering avançado e comparação de diferentes estratégias de pré-processamento.

**Problema:** Classificação binária desbalanceada (90% classe 0 / 10% classe 1)  
**Algoritmo:** XGBoost (Binary Classification)  
**Dataset:** Google BigQuery (proj-ml-469320.app_cittamobi.dataset-updated)

---

## 📂 Estrutura do Projeto

```
Projeto Machine Learning/
│
├── 📄 Código
│   ├── poc.py                      # Modelo V1 Otimizado (RECOMENDADO)
│   └── model_v2_enhanced.py        # Modelo V2 com limpeza rigorosa
│
├── 📊 Documentação
│   ├── README.md                   # Este arquivo
│   ├── ANALISE_RESULTADOS.md       # Análise detalhada do V1
│   ├── COMPARACAO_V1_V2.md         # Comparação V1 vs V2
│   └── GUIA_DE_USO.md              # Guia completo de uso
│
├── 🤖 Modelos Treinados
│   ├── xgboost_model_optimized.json      # V1 - 4.1 MB (RECOMENDADO)
│   └── xgboost_model_v2_enhanced.json    # V2 - 10 MB
│
└── 📈 Visualizações
    ├── confusion_matrix.png              # V1
    ├── confusion_matrix_v2.png           # V2
    ├── roc_curve.png                     # V1
    ├── roc_curve_v2.png                  # V2
    ├── threshold_analysis.png            # V1
    ├── threshold_analysis_v2.png         # V2
    ├── feature_importance.png            # V1
    └── feature_importance_v2.png         # V2
```

---

## 🏆 Modelos Desenvolvidos

### **V1 - Modelo Otimizado (RECOMENDADO) ✅**

**Características:**
- 50,000 amostras
- 38 features
- Remoção de data leakage básica
- Tuning de hiperparâmetros
- Otimização de threshold (0.6)

**Performance:**
- ROC-AUC: **0.8367** 🥇
- Accuracy: **89.02%**
- Precision: **45.19%**
- Recall: **51.21%**
- F1-Score: **0.4801**

**Arquivo:** `poc.py` | `xgboost_model_optimized.json`

---

### **V2 - Modelo Enhanced (Experimental) ⚗️**

**Características:**
- 500,000 amostras iniciais → 60,498 após filtros (87.9% removido)
- 49 features (+11 novas)
- Limpeza rigorosa de dados:
  - ✓ Usuários com baixa frequência
  - ✓ Localização inválida
  - ✓ Distância muito alta
  - ✓ Paradas com poucos eventos
- Feature engineering avançado
- 4 configurações de tuning testadas

**Performance:**
- ROC-AUC: **0.7961** 
- Accuracy: **86.62%**
- Precision: **41.99%**
- Recall: **48.89%**
- F1-Score: **0.4518**

**Arquivo:** `model_v2_enhanced.py` | `xgboost_model_v2_enhanced.json`

**⚠️ Nota:** Performance inferior ao V1. Veja `COMPARACAO_V1_V2.md` para análise detalhada.

---

## 📊 Resultados Comparativos

| Métrica | V1 (Otimizado) | V2 (Enhanced) | Vencedor |
|---------|----------------|---------------|----------|
| **ROC-AUC** | **0.8367** | 0.7961 | ✅ V1 |
| **Accuracy** | **89.02%** | 86.62% | ✅ V1 |
| **Precision** | **45.19%** | 41.99% | ✅ V1 |
| **Recall** | **51.21%** | 48.89% | ✅ V1 |
| **F1-Score** | **0.4801** | 0.4518 | ✅ V1 |
| **Threshold** | 0.6 | 0.5 | - |
| **Features** | 38 | 49 | ✅ V1 (mais simples) |
| **Amostras** | 50k | 60k | - |
| **Tempo Treino** | Mais rápido | Mais lento | ✅ V1 |

**Conclusão:** V1 é superior em todos os aspectos! 🏆

---

## 🚀 Como Usar

### **1. Instalar Dependências**

```bash
conda create -n cittamobi-forecast python=3.12
conda activate cittamobi-forecast
pip install google-cloud-bigquery pandas numpy scikit-learn xgboost matplotlib seaborn
```

### **2. Autenticar com Google Cloud**

```bash
gcloud auth application-default login
```

### **3. Executar Modelo V1 (Recomendado)**

```bash
cd "/Users/stefano/Documents/Ibmec/Projeto Machine Learning"
python poc.py
```

### **4. Carregar Modelo Treinado**

```python
import xgboost as xgb
import pandas as pd

# Carregar modelo
model = xgb.Booster()
model.load_model('xgboost_model_optimized.json')

# Fazer predições
dmatrix = xgb.DMatrix(X_novo)
probabilidades = model.predict(dmatrix)

# Usar threshold otimizado
THRESHOLD = 0.6
predicoes = (probabilidades >= THRESHOLD).astype(int)
```

**📖 Para instruções detalhadas, consulte:** `GUIA_DE_USO.md`

---

## 📚 Documentação

### **ANALISE_RESULTADOS.md**
- Análise completa do Modelo V1
- Comparação Baseline vs Otimizado
- Métricas detalhadas
- Features mais importantes
- Recomendações de manutenção

### **COMPARACAO_V1_V2.md**
- Comparação detalhada entre V1 e V2
- Análise de por que V2 teve performance inferior
- Hipóteses e insights
- Recomendações para V3
- Lições aprendidas

### **GUIA_DE_USO.md**
- Como carregar e usar o modelo
- Preparação de novos dados
- Interpretação de resultados
- Thresholds recomendados por cenário
- Troubleshooting
- Monitoramento em produção
- Pipeline completo de predição

---

## 🎓 Principais Aprendizados

### **1. Data Leakage é Crítico**
- Identificamos features com correlação perfeita (1.0) com o target
- Remoção de `y_pred`, `y_pred_proba`, e `lotacao_proxy_binaria` foi essencial
- Performance "perfeita" geralmente indica vazamento de dados

### **2. Mais Dados ≠ Sempre Melhor**
- V1 com 50k amostras "sujas" superou V2 com 60k amostras "limpas"
- Limpeza muito rigorosa (87.9% removido) reduziu diversidade necessária
- O "ruído" nos dados pode conter padrões reais de comportamento

### **3. Feature Engineering Deve Ser Validado**
- 11 novas features em V2 não melhoraram a performance
- Complexidade excessiva pode introduzir ruído
- Seleção de features é tão importante quanto criação

### **4. Threshold Optimization é Poderoso**
- Mudar threshold de 0.5 para 0.6 melhorou significativamente
- Precision: 30% → 45% (+50% de melhoria!)
- Trade-off consciente entre Precision e Recall

### **5. Tuning de Hiperparâmetros Vale a Pena**
- ROC-AUC melhorou de 0.8214 → 0.8367 (+1.86%)
- Encontrar configuração ideal entre 4 testadas
- Regularização (min_child_weight, gamma) ajuda muito

---

## 📈 Métricas de Produção (V1)

### **Performance Esperada:**
```
ROC-AUC:   0.8367  (Excelente capacidade discriminativa)
Accuracy:  89.02%  (89 de cada 100 predições corretas)
Precision: 45.19%  (45% de confiança em predições positivas)
Recall:    51.21%  (Captura 51% dos casos positivos reais)
F1-Score:  0.4801  (Bom equilíbrio para classes desbalanceadas)
```

### **Interpretação de Negócio:**
- **Quando prediz Classe 1:** 45% de chance de estar correto
- **Falsos Positivos:** 1,025 casos (reduzidos em 58% vs baseline)
- **Falsos Negativos:** 805 casos
- **Uso Recomendado:** Sistemas de apoio à decisão (não críticos)

---

## ⚙️ Configuração do Modelo V1 (Produção)

```python
{
    'objective': 'binary:logistic',
    'max_depth': 10,
    'learning_rate': 0.03,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'min_child_weight': 5,
    'scale_pos_weight': 9.31,  # Para classes desbalanceadas
    'eval_metric': 'logloss',
    'seed': 42
}
```

**Threshold Otimizado:** 0.6  
**Iterações de Treino:** 200 (com early stopping em 20)

---

## 🔄 Roadmap Futuro

### **V3 - Versão Híbrida (Planejada)**

**Objetivos:**
1. Usar `TABLESAMPLE` para amostragem aleatória (200k amostras)
2. Filtros moderados (remover 30-40%, não 87.9%)
3. Selecionar top 35 features (entre V1 e V2)
4. Implementar SHAP para interpretabilidade
5. Validação cruzada temporal robusta
6. Ensemble de V1 + V2

**Meta:** ROC-AUC > 0.85

---

## 📞 Suporte

Para dúvidas ou problemas:

1. **Consulte a documentação:**
   - `GUIA_DE_USO.md` - Uso básico
   - `ANALISE_RESULTADOS.md` - Métricas e análises
   - `COMPARACAO_V1_V2.md` - Comparações

2. **Verifique os gráficos:**
   - `confusion_matrix.png` - Erros do modelo
   - `roc_curve.png` - Capacidade discriminativa
   - `threshold_analysis.png` - Otimização de threshold
   - `feature_importance.png` - Features mais importantes

3. **Código fonte:**
   - `poc.py` - Modelo V1 completo e comentado
   - `model_v2_enhanced.py` - Modelo V2 experimental

---

## 🏅 Créditos

**Projeto:** Cittamobi Forecast - Machine Learning  
**Dataset:** Google BigQuery (proj-ml-469320.app_cittamobi.dataset-updated)  
**Algoritmo:** XGBoost (Binary Classification)  
**Desenvolvido:** Outubro 2025  
**Ambiente:** Python 3.12 + Conda (cittamobi-forecast)

---

## 📄 Licença

Este projeto é proprietário e destinado ao uso interno da Cittamobi.

---

**Última Atualização:** 29 de Outubro de 2025  
**Versão:** 2.0 (V1 Otimizado + V2 Enhanced)  
**Status:** ✅ Pronto para Produção (V1)
