# 📑 Índice do Projeto - Cittamobi Forecast

## 🎯 Acesso Rápido

### ⚡ Executar Melhor Modelo
```bash
cd models/v4
python model_v4_advanced.py
```

### 📊 Ver Resultados V4
- Modelo: [`models/v4/xgboost_model_v4_advanced.json`](models/v4/xgboost_model_v4_advanced.json)
- Comparação: [`visualizations/v4/v4_strategies_comparison.png`](visualizations/v4/v4_strategies_comparison.png)
- Confusion Matrix: [`visualizations/v4/confusion_matrix_v4.png`](visualizations/v4/confusion_matrix_v4.png)

---

## 📁 Estrutura Completa

### 🤖 Modelos (`models/`)

#### V1 - Baseline
- **Código**: [`models/v1/poc.py`](models/v1/poc.py)
- **Modelo**: [`models/v1/xgboost_model.json`](models/v1/xgboost_model.json)
- **Resultado**: ROC-AUC 0.8367
- **Status**: ✅ Baseline inicial

#### V2 - Enhanced
- **Código**: [`models/v2/model_v2_enhanced.py`](models/v2/model_v2_enhanced.py)
- **Modelo**: [`models/v2/xgboost_model_v2_enhanced.json`](models/v2/xgboost_model_v2_enhanced.json)
- **Resultado**: ROC-AUC 0.7961
- **Status**: ⚠️ Limpeza agressiva prejudicou desempenho

#### V3 - Hybrid + Enhanced
- **Código**: 
  - [`models/v3/model_v3_hybrid.py`](models/v3/model_v3_hybrid.py)
  - [`models/v3/model_v3_enhanced.py`](models/v3/model_v3_enhanced.py)
- **Modelos**: 
  - [`models/v3/xgboost_model_v3_hybrid.json`](models/v3/xgboost_model_v3_hybrid.json)
  - [`models/v3/xgboost_model_v3_enhanced.json`](models/v3/xgboost_model_v3_enhanced.json)
- **Resultado**: ROC-AUC 0.9324, F1-Macro 0.7143
- **Status**: ✅ Grande melhoria (+11.4% vs V1)

#### V4 - Advanced 🏆
- **Código**: [`models/v4/model_v4_advanced.py`](models/v4/model_v4_advanced.py)
- **Modelo**: [`models/v4/xgboost_model_v4_advanced.json`](models/v4/xgboost_model_v4_advanced.json)
- **Resultado**: ROC-AUC 0.9731, F1-Macro 0.7760, Precision 0.59
- **Status**: 🏆 **MELHOR MODELO - RECOMENDADO**

---

### 📊 Visualizações (`visualizations/`)

#### V1
- [`confusion_matrix.png`](visualizations/v1/confusion_matrix.png)
- [`feature_importance.png`](visualizations/v1/feature_importance.png)
- [`roc_curve.png`](visualizations/v1/roc_curve.png)
- [`threshold_analysis.png`](visualizations/v1/threshold_analysis.png)

#### V2
- [`confusion_matrix_v2.png`](visualizations/v2/confusion_matrix_v2.png)
- [`feature_importance_v2.png`](visualizations/v2/feature_importance_v2.png)
- [`roc_curve_v2.png`](visualizations/v2/roc_curve_v2.png)
- [`threshold_analysis_v2.png`](visualizations/v2/threshold_analysis_v2.png)

#### V3
- [`confusion_matrix_v3.png`](visualizations/v3/confusion_matrix_v3.png)
- [`confusion_matrix_v3_enhanced.png`](visualizations/v3/confusion_matrix_v3_enhanced.png)
- [`feature_importance_v3.png`](visualizations/v3/feature_importance_v3.png)
- [`roc_curve_v3.png`](visualizations/v3/roc_curve_v3.png)
- [`threshold_analysis_v3.png`](visualizations/v3/threshold_analysis_v3.png)
- [`threshold_analysis_v3_enhanced.png`](visualizations/v3/threshold_analysis_v3_enhanced.png)
- [`comparison_v1_v2_v3.png`](visualizations/v3/comparison_v1_v2_v3.png)
- [`balancing_strategies_comparison.png`](visualizations/v3/balancing_strategies_comparison.png)

#### V4 🏆
- [`confusion_matrix_v4.png`](visualizations/v4/confusion_matrix_v4.png)
- [`v4_strategies_comparison.png`](visualizations/v4/v4_strategies_comparison.png)

---

### 📚 Documentação (`docs/`)

#### Guias Principais
- [`GUIA_DE_USO.md`](docs/GUIA_DE_USO.md) - Como executar cada versão
- [`V4_EXPLICACAO.md`](docs/V4_EXPLICACAO.md) - Detalhes técnicos do V4
- [`V3_ENHANCED_EXPLICACAO.md`](docs/V3_ENHANCED_EXPLICACAO.md) - Técnicas de balanceamento

#### Análises
- [`ANALISE_RESULTADOS.md`](docs/ANALISE_RESULTADOS.md) - Análises detalhadas
- [`COMPARACAO_V1_V2.md`](docs/COMPARACAO_V1_V2.md) - Comparativo V1 vs V2
- [`README_OLD.md`](docs/README_OLD.md) - README antigo (histórico)

---

### 📋 Relatórios (`reports/`)

- [`features_v3_selected.txt`](reports/features_v3_selected.txt) - Top 40 features do V3
- [`v3_enhanced_report.txt`](reports/v3_enhanced_report.txt) - Relatório completo V3 Enhanced

---

## 📊 Comparação Rápida

| Métrica | V1 | V2 | V3 Hybrid | V3 Enhanced | V4 Advanced 🏆 |
|---------|----|----|-----------|-------------|----------------|
| ROC-AUC | 0.8367 | 0.7961 | 0.9283 | 0.9324 | **0.9731** |
| F1-Macro | ~0.65 | - | 0.7050 | 0.7143 | **0.7760** |
| Precision | - | - | 0.43 | 0.43 | **0.59** |
| Recall | - | - | 0.71 | 0.47 | **0.55** |
| Features | ~60 | ~60 | 40 | 40 | **50** |
| Samples | 200k | 24k | 176k | 176k | 177k |

---

## 🎓 Aprendizados Principais

1. **Limpeza agressiva prejudica**: V2 removeu 87.9% dos dados e piorou
2. **Feature selection ajuda**: Top 40-50 features é suficiente
3. **Balanceamento simples vence**: scale_pos_weight > SMOTE
4. **Feature engineering é chave**: Agregações por usuário/parada melhoraram muito
5. **Threshold otimizado é crítico**: 0.65 vs 0.50 default faz diferença

---

## ⚙️ Configuração

```bash
# Ativar ambiente
conda activate cittamobi-forecast

# Instalar dependências (se necessário)
conda env create -f environment.yml
```

---

## 🏆 Modelo de Produção

**Use o V4 Advanced - Strategy 5 (Advanced Features + Deep Trees)**

- Melhor ROC-AUC: 0.9731
- Melhor F1-Macro: 0.7760
- Precision > 0.50 ✅
- Threshold ótimo: 0.65
