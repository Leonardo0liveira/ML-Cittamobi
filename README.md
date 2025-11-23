# 🚍 Projeto Machine Learning - Cittamobi Forecast

## 📋 Visão Geral

Projeto de previsão de conversão de eventos de usuários de transporte público utilizando XGBoost.

**Objetivo**: Prever se um usuário irá converter (realizar uma ação desejada) com base em seus padrões de uso e características dos eventos.

---

## 📊 Resultados Finais

| Versão | ROC-AUC | F1-Macro | Precision | Recall | Status |
|--------|---------|----------|-----------|--------|--------|
| V1 Baseline | 0.8367 | ~0.65 | - | - | ✅ Concluído |
| V2 Enhanced | 0.7961 | - | - | - | ✅ Concluído |
| V3 Hybrid | 0.9283 | 0.7050 | 0.43 | 0.71 | ✅ Concluído |
| V3 Enhanced | 0.9324 | 0.7143 | 0.43 | 0.47 | ✅ Concluído |
| **V4 Advanced** 🏆 | **0.9731** | **0.7760** | **0.59** | **0.55** | **✅ RECOMENDADO** |

**Melhoria Total**: +16.3% em ROC-AUC comparado ao V1

---

## 📁 Estrutura do Projeto

```
Projeto Machine Learning/
├── README.md                 # Este arquivo
├── environment.yml          # Ambiente conda
│
├── models/                  # Modelos e código fonte
│   ├── v1/                 # Baseline (poc.py)
│   ├── v2/                 # Enhanced com limpeza agressiva
│   ├── v3/                 # Hybrid + Enhanced (balanceamento)
│   └── v4/                 # Advanced (melhor versão) 🏆
│
├── visualizations/         # Gráficos e análises visuais
│   ├── v1/                # Confusion matrix, ROC, etc.
│   ├── v2/
│   ├── v3/
│   └── v4/
│
├── docs/                   # Documentação técnica
│   ├── ANALISE_RESULTADOS.md
│   ├── COMPARACAO_V1_V2.md
│   ├── GUIA_DE_USO.md
│   ├── V3_ENHANCED_EXPLICACAO.md
│   └── V4_EXPLICACAO.md
│
└── reports/                # Relatórios e features
    ├── features_v3_selected.txt
    └── v3_enhanced_report.txt
```

---

## 🚀 Como Usar

### 1. Configurar Ambiente

```bash
conda env create -f environment.yml
conda activate cittamobi-forecast
```

### 2. Executar Modelo Recomendado (V4)

```bash
cd models/v4
python model_v4_advanced.py
```

### 3. Ver Resultados

- **Modelo treinado**: `models/v4/xgboost_model_v4_advanced.json`
- **Visualizações**: `visualizations/v4/`
- **Documentação**: `docs/V4_EXPLICACAO.md`

---

## 🎯 Características do V4 Advanced (Melhor Modelo)

### Estratégias Testadas:
1. ✅ Baseline Otimizado (V3)
2. ✅ Cost-Sensitive Learning
3. ✅ User Frequency Undersampling
4. ✅ Ensemble Stacking (3 models)
5. 🏆 **Advanced Features + Deep Trees** (VENCEDOR)

### Features Avançadas:
- **9 agregações por usuário**: conversion_rate, total_conversions, avg_distance, etc.
- **4 agregações por parada**: conversion_rate, event_count, user_frequency
- **Interações de 2ª ordem**: conversion_interaction, dist_deviation, user_stop_affinity
- **50 features** selecionadas (vs 40 no V3)

### Hiperparâmetros Vencedores:
```python
{
    'max_depth': 18,           # Árvores profundas para capturar interações complexas
    'learning_rate': 0.02,
    'min_child_weight': 3,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'num_boost_round': 250,
    'early_stopping_rounds': 25
}
```

### Threshold Ótimo:
- **0.65** para maximizar F1-Macro (0.7760)

---

## 📈 Evolução do Projeto

### V1 - Baseline (POC)
- Primeira abordagem
- ROC-AUC: 0.8367
- Problemas: baixa precisão, muitos falsos positivos

### V2 - Enhanced
- Limpeza agressiva dos dados
- **Problema crítico**: removeu 87.9% dos dados
- ROC-AUC caiu para 0.7961

### V3 - Hybrid + Enhanced
- Limpeza moderada (11.6% removido)
- Feature selection (top 40)
- ROC-AUC: 0.9283 (+10.9% vs V1)
- Enhanced: testou 4 técnicas de balanceamento
  - Baseline (scale_pos_weight) foi o melhor

### V4 - Advanced 🏆
- Feature engineering avançado
- 5 estratégias testadas
- **Advanced Features + Deep Trees venceu**
- **ROC-AUC: 0.9731** (+16.3% vs V1)
- **Precision: 0.59** (✅ alcançou meta > 0.50)

---

## 🔬 Tecnologias Utilizadas

- **Python 3.12**
- **XGBoost**: Modelo de gradient boosting
- **Google BigQuery**: Source de dados (TABLESAMPLE 20%)
- **imbalanced-learn**: Técnicas de balanceamento (SMOTE, Tomek, etc.)
- **scikit-learn**: Métricas e validação (TimeSeriesSplit)
- **pandas, numpy**: Manipulação de dados
- **matplotlib, seaborn**: Visualizações

---

## 📚 Documentação

- **[GUIA_DE_USO.md](docs/GUIA_DE_USO.md)**: Como executar cada versão
- **[V4_EXPLICACAO.md](docs/V4_EXPLICACAO.md)**: Detalhes técnicos do V4
- **[V3_ENHANCED_EXPLICACAO.md](docs/V3_ENHANCED_EXPLICACAO.md)**: Técnicas de balanceamento
- **[ANALISE_RESULTADOS.md](docs/ANALISE_RESULTADOS.md)**: Análises detalhadas
- **[COMPARACAO_V1_V2.md](docs/COMPARACAO_V1_V2.md)**: Comparativo inicial

---

## 👨‍💻 Autor

**Stefano**  
Projeto Machine Learning - IBMEC  
Outubro 2025

---

## 📝 Notas

- Dataset: ~200k amostras (20% do total via TABLESAMPLE)
- Classe desbalanceada: ~92% classe 0, ~8% classe 1 (ratio 12:1)
- Validação temporal: TimeSeriesSplit (3 folds)
- Threshold otimizado: 0.65 (vs default 0.50)

**Modelo de produção recomendado**: V4 Advanced - Strategy 5
